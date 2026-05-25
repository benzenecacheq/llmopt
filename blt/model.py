import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


class BLTAttention(nn.Module):
    """
    Bilinear attention: replaces Wq and Wk with a single shared M (d_model x d_model).
    Forward: softmax(X @ M @ X^T / scale) @ (X @ Wv) @ Wo
    M is shared across all layers and passed in at construction time.
    """

    def __init__(self, M_shared: nn.Parameter, Wv, Wv_bias, Wo, Wo_bias, config):
        super().__init__()
        self.M = M_shared                          # shared across all layers
        self.scale = math.sqrt(config.n_embd)      # sqrt(768)

        self.Wv = nn.Parameter(Wv)
        self.Wv_bias = nn.Parameter(Wv_bias)
        self.Wo = nn.Parameter(Wo)
        self.Wo_bias = nn.Parameter(Wo_bias)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, **kwargs):
        B, L, D = hidden_states.shape

        # Bilinear attention: (B,L,D) @ (D,D) @ (D,L) → (B,L,L)
        scores = (hidden_states @ self.M) @ hidden_states.transpose(-2, -1) / self.scale

        # Causal mask: upper triangle = -inf
        causal = torch.full((L, L), float('-inf'), device=hidden_states.device,
                            dtype=hidden_states.dtype)
        causal = torch.triu(causal, diagonal=1)
        scores = scores + causal

        # Optional additive mask from caller (e.g. padding), expected shape (B,1,L,L) or (B,1,1,L)
        if attention_mask is not None:
            scores = scores + attention_mask.squeeze(1)  # broadcast over heads dim

        A = F.softmax(scores, dim=-1)              # (B, L, L)

        V = hidden_states @ self.Wv + self.Wv_bias # (B, L, D)
        h = A @ V                                  # (B, L, D)
        out = h @ self.Wo + self.Wo_bias           # (B, L, D)

        return out, None   # (attn_output, past_key_values)


def build_blt_model(pretrained='gpt2'):
    """
    Load pretrained GPT-2 and replace every attention layer with BLT attention.

    One M matrix (768x768) is shared across all 12 layers, initialized as the
    average of Wq @ Wk^T across layers.  Wv and Wo are kept per-layer from the
    pretrained weights.  All other parameters are unchanged.
    """
    model = GPT2LMHeadModel.from_pretrained(pretrained)
    cfg = model.config
    D = cfg.n_embd  # 768

    # Initialize M as average of Wq @ Wk^T across all layers.
    # c_attn.weight shape: (768, 2304) — columns: Wq | Wk | Wv
    M_init = torch.zeros(D, D)
    for layer in model.transformer.h:
        Wq = layer.attn.c_attn.weight[:, :D].detach()
        Wk = layer.attn.c_attn.weight[:, D:2 * D].detach()
        M_init += Wq @ Wk.T
    M_init /= cfg.n_layer

    M_shared = nn.Parameter(M_init)

    # Replace attention in every layer
    for layer in model.transformer.h:
        attn = layer.attn
        Wv = attn.c_attn.weight[:, 2 * D:].detach()   # (768, 768)
        Wv_bias = attn.c_attn.bias[2 * D:].detach()   # (768,)
        Wo = attn.c_proj.weight.detach()               # (768, 768)
        Wo_bias = attn.c_proj.bias.detach()            # (768,)
        layer.attn = BLTAttention(M_shared, Wv, Wv_bias, Wo, Wo_bias, cfg)

    # Register M at the transformer level so it appears in named_parameters
    model.transformer.register_parameter('M_blt', M_shared)

    return model


def parameter_summary(model):
    """Print parameter counts by group."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    unique = {id(p): p for p in model.parameters()}
    unique_total = sum(p.numel() for p in unique.values())
    print(f"Total params (with sharing): {total:>12,}")
    print(f"Unique params:               {unique_total:>12,}")
    print(f"Trainable params:            {trainable:>12,}")
