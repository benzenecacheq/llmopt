import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel


class BLTAttention(nn.Module):
    """
    Bilinear attention: replaces Wq and Wk with a single shared M (d_model x d_model).
    Forward: softmax(X @ M @ X^T / scale) @ (X @ Wv) @ Wo
    M is shared across all layers and passed in at construction time.
    """

    def __init__(self, M_shared: nn.Parameter, Wv, Wv_bias, Wo, Wo_bias, config):
        super().__init__()
        self.M = M_shared                          # shared across all layers
        self.scale = math.sqrt(config.n_embd // config.n_head)  # sqrt(64), matches GPT-2 MHA

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


class BLTMultiMAttention(nn.Module):
    """
    N-group bilinear attention: N shared M matrices (D×D), each governing H/N heads'
    value capacity. Group g: X @ M_g @ X^T applied to Wv_g (D→D/N). Outputs
    concatenated (D) then projected through Wo. N=2 recovers the original 2-group BLT.
    """

    def __init__(self, M_list, Wv_list, Wv_bias_list, Wo, Wo_bias, config):
        super().__init__()
        self.scale = math.sqrt(config.n_embd // config.n_head)
        self._n_groups = len(M_list)
        for i, (M, Wv, Wv_bias) in enumerate(zip(M_list, Wv_list, Wv_bias_list)):
            setattr(self, f'M_{i}', M)                        # shared param reference
            setattr(self, f'Wv_{i}', nn.Parameter(Wv))
            setattr(self, f'Wv_bias_{i}', nn.Parameter(Wv_bias))
        self.Wo = nn.Parameter(Wo)
        self.Wo_bias = nn.Parameter(Wo_bias)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, **kwargs):
        B, L, D = hidden_states.shape
        causal = torch.full((L, L), float('-inf'), device=hidden_states.device,
                            dtype=hidden_states.dtype)
        causal = torch.triu(causal, diagonal=1)

        outputs = []
        for i in range(self._n_groups):
            M       = getattr(self, f'M_{i}')
            Wv      = getattr(self, f'Wv_{i}')
            Wv_bias = getattr(self, f'Wv_bias_{i}')
            scores = (hidden_states @ M) @ hidden_states.transpose(-2, -1) / self.scale
            scores = scores + causal
            if attention_mask is not None:
                scores = scores + attention_mask.squeeze(1)
            A = F.softmax(scores, dim=-1)
            outputs.append(A @ (hidden_states @ Wv + Wv_bias))

        h = torch.cat(outputs, dim=-1)
        return h @ self.Wo + self.Wo_bias, None


class GQAAttention(nn.Module):
    """
    Grouped Query Attention baseline.
    12 query heads, n_kv KV heads (n_head/n_kv queries per KV head).
    Wq/Wo: D×D per layer (full). Wk/Wv: D×(n_kv×d_head) per layer (grouped).
    """

    def __init__(self, Wq, Wq_bias, Wk, Wk_bias, Wv, Wv_bias, Wo, Wo_bias, config, n_kv=2):
        super().__init__()
        self.n_head   = config.n_head
        self.n_kv     = n_kv
        self.d_head   = config.n_embd // config.n_head  # 64
        self.scale    = math.sqrt(self.d_head)
        self.q_per_kv = self.n_head // self.n_kv

        self.Wq      = nn.Parameter(Wq)
        self.Wq_bias = nn.Parameter(Wq_bias)
        self.Wk      = nn.Parameter(Wk)
        self.Wk_bias = nn.Parameter(Wk_bias)
        self.Wv      = nn.Parameter(Wv)
        self.Wv_bias = nn.Parameter(Wv_bias)
        self.Wo      = nn.Parameter(Wo)
        self.Wo_bias = nn.Parameter(Wo_bias)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, **kwargs):
        B, L, D = hidden_states.shape

        Q = (hidden_states @ self.Wq + self.Wq_bias
             ).view(B, L, self.n_head, self.d_head).transpose(1, 2)   # (B, H, L, d)
        K = (hidden_states @ self.Wk + self.Wk_bias
             ).view(B, L, self.n_kv, self.d_head).transpose(1, 2)     # (B, 2, L, d)
        V = (hidden_states @ self.Wv + self.Wv_bias
             ).view(B, L, self.n_kv, self.d_head).transpose(1, 2)     # (B, 2, L, d)

        # Expand KV groups to match query heads
        K = K.repeat_interleave(self.q_per_kv, dim=1)                 # (B, H, L, d)
        V = V.repeat_interleave(self.q_per_kv, dim=1)                 # (B, H, L, d)

        scores = Q @ K.transpose(-2, -1) / self.scale                 # (B, H, L, L)

        causal = torch.full((L, L), float('-inf'), device=hidden_states.device,
                            dtype=hidden_states.dtype)
        causal = torch.triu(causal, diagonal=1)
        scores = scores + causal

        if attention_mask is not None:
            # attention_mask may be (B,1,1,L) or (B,1,L,L) — broadcast over heads
            scores = scores + attention_mask

        A   = F.softmax(scores, dim=-1)                                # (B, H, L, L)
        h   = (A @ V).transpose(1, 2).contiguous().view(B, L, D)      # (B, L, D)
        out = h @ self.Wo + self.Wo_bias

        return out, None


class MHAAttention(nn.Module):
    """
    Standard multi-head attention, matching BLT's interface (returns (out, None)).
    Per-layer Wq, Wk, Wv, Wo — all independent. Used in the hybrid model's early layers.
    """

    def __init__(self, Wq, Wq_bias, Wk, Wk_bias, Wv, Wv_bias, Wo, Wo_bias, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.n_embd // config.n_head
        self.scale  = math.sqrt(self.d_head)

        self.Wq      = nn.Parameter(Wq)
        self.Wq_bias = nn.Parameter(Wq_bias)
        self.Wk      = nn.Parameter(Wk)
        self.Wk_bias = nn.Parameter(Wk_bias)
        self.Wv      = nn.Parameter(Wv)
        self.Wv_bias = nn.Parameter(Wv_bias)
        self.Wo      = nn.Parameter(Wo)
        self.Wo_bias = nn.Parameter(Wo_bias)

    def forward(self, hidden_states, past_key_values=None, attention_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, **kwargs):
        B, L, D = hidden_states.shape

        Q = (hidden_states @ self.Wq + self.Wq_bias
             ).view(B, L, self.n_head, self.d_head).transpose(1, 2)  # (B, H, L, d)
        K = (hidden_states @ self.Wk + self.Wk_bias
             ).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        V = (hidden_states @ self.Wv + self.Wv_bias
             ).view(B, L, self.n_head, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / self.scale                # (B, H, L, L)

        causal = torch.full((L, L), float('-inf'), device=hidden_states.device,
                            dtype=hidden_states.dtype)
        causal = torch.triu(causal, diagonal=1)
        scores = scores + causal

        if attention_mask is not None:
            scores = scores + attention_mask

        A   = F.softmax(scores, dim=-1)
        h   = (A @ V).transpose(1, 2).contiguous().view(B, L, D)
        out = h @ self.Wo + self.Wo_bias

        return out, None


def build_gqa_model(n_kv=2, pretrained=None):
    """
    Build a GPT-2 model with n_kv-group GQA replacing every attention layer.
    n_kv: number of KV heads (must divide n_head). Default 2.
    pretrained: HuggingFace model ID. If provided, Wq/Wo kept from pretrained weights;
                Wk/Wv initialized by averaging pretrained heads within each KV group.
                If None, builds from random weights using GPT2Config() defaults.
    """
    if pretrained:
        model = GPT2LMHeadModel.from_pretrained(pretrained)
    else:
        model = GPT2LMHeadModel(GPT2Config())
    cfg      = model.config
    D        = cfg.n_embd
    n_head   = cfg.n_head
    d_head   = D // n_head
    if n_head % n_kv != 0:
        raise ValueError(f'n_kv ({n_kv}) must evenly divide n_head ({n_head})')
    kv_dim   = n_kv * d_head
    q_per_kv = n_head // n_kv

    for layer in model.transformer.h:
        attn    = layer.attn
        Wq      = attn.c_attn.weight[:, :D].detach()
        Wq_bias = attn.c_attn.bias[:D].detach()
        Wo      = attn.c_proj.weight.detach()
        Wo_bias = attn.c_proj.bias.detach()
        if pretrained:
            Wk_full = attn.c_attn.weight[:, D:2*D].detach().view(D, n_head, d_head)
            Wv_full = attn.c_attn.weight[:, 2*D:].detach().view(D, n_head, d_head)
            Wk = Wk_full.view(D, n_kv, q_per_kv, d_head).mean(dim=2).reshape(D, kv_dim)
            Wv = Wv_full.view(D, n_kv, q_per_kv, d_head).mean(dim=2).reshape(D, kv_dim)
            Wk_bias = torch.zeros(kv_dim)
            Wv_bias = torch.zeros(kv_dim)
        else:
            Wk      = torch.randn(D, kv_dim) * 0.02
            Wk_bias = torch.zeros(kv_dim)
            Wv      = torch.randn(D, kv_dim) * 0.02
            Wv_bias = torch.zeros(kv_dim)
        layer.attn = GQAAttention(Wq, Wq_bias, Wk, Wk_bias, Wv, Wv_bias,
                                  Wo, Wo_bias, cfg, n_kv=n_kv)

    return model


def build_hybrid_model(n_mha=6, pretrained='gpt2'):
    """
    Hybrid: first n_mha layers use standard MHA, remaining layers use BLT with one shared M.
    Always initializes from scratch; `pretrained` only controls the architecture shape
    (e.g. 'gpt2-medium'), not warm-start. Default n_mha=6 gives a 6/6 split on GPT-2 small.

    Tests whether BLT's expressiveness cost is concentrated in early layers.
    M only needs to multi-task across (n_layer - n_mha) layers instead of all of them,
    so the shared matrix has a narrower job and may converge to a more useful solution.
    """
    model = GPT2LMHeadModel(GPT2Config.from_pretrained(pretrained))
    cfg   = model.config
    D     = cfg.n_embd       # 768
    n_head = cfg.n_head      # 12
    d_head = D // n_head     # 64

    M_init   = torch.randn(D, D) / math.sqrt(D)
    M_shared = nn.Parameter(M_init)

    for i, layer in enumerate(model.transformer.h):
        attn     = layer.attn
        Wq       = attn.c_attn.weight[:, :D].detach()
        Wq_bias  = attn.c_attn.bias[:D].detach()
        Wk       = attn.c_attn.weight[:, D:2*D].detach()
        Wk_bias  = attn.c_attn.bias[D:2*D].detach()
        Wv       = attn.c_attn.weight[:, 2*D:].detach()
        Wv_bias  = attn.c_attn.bias[2*D:].detach()
        Wo       = attn.c_proj.weight.detach()
        Wo_bias  = attn.c_proj.bias.detach()

        if i < n_mha:
            layer.attn = MHAAttention(Wq, Wq_bias, Wk, Wk_bias,
                                      Wv, Wv_bias, Wo, Wo_bias, cfg)
        else:
            layer.attn = BLTAttention(M_shared, Wv, Wv_bias, Wo, Wo_bias, cfg)

    model.transformer.register_parameter('M_blt', M_shared)
    return model


def build_blt_model(pretrained='gpt2', num_m_groups=1, layers_per_m=0,
                    random_m=False, from_scratch=False, warmstart_scale=1.0,
                    per_layer_m=False):
    """
    Build a GPT-2 model with BLT attention replacing every attention layer.

    num_m_groups=1: one M (768x768) shared across all layers (original BLT).
    num_m_groups=2: two M matrices, each governing half the value capacity (head-based).

    layers_per_m=N (N>0): strided layer grouping with n_layers//N M matrices,
      each covering N non-adjacent layers. Layer i uses M_params[i % G] where
      G = n_layers // N. Analogous to GQA's heads-per-kv-group.
      N=4 → G=3 for GPT-2 small (12 layers): M_0→{0,3,6,9}, M_1→{1,4,7,10},
      M_2→{2,5,8,11}. N=4 → G=12 for GPT-2 XL (48 layers).
      Mutually exclusive with num_m_groups>1.

    random_m: initialize M with N(0, 1/sqrt(D)) instead of Wq@Wk^T average.
    from_scratch: randomly initialize all weights (do not load pretrained GPT-2).
                  Forces random_m=True since there are no Wq/Wk to average.
    warmstart_scale: blend factor in [0, 1] for the Wq@Wk^T-average M init.
                  1.0 = pure average (original small-model behavior). Lower values
                  shrink the averaged M toward zero, softening the initial
                  perturbation — useful for models with many more layers/heads
                  (e.g. GPT-2 XL: 48 layers x 25 heads = 1200 attention contexts
                  vs GPT-2's 144), where the naive average is a much coarser,
                  more disruptive approximation. No effect when random_m=True.
    per_layer_m: give each layer its own M (768x768) instead of sharing one M
                  across all 12 layers. Still one M per layer shared across all
                  heads within that layer. Warm-start init uses that layer's own
                  Wq @ Wk^T directly (no cross-layer averaging needed, since each
                  M has exactly one layer's worth of Wq/Wk to draw from). 25%
                  fewer attention params than standard MHA (vs. ~48% for the
                  cross-layer-shared M), and loses the "M loaded once for the
                  whole model" bandwidth amortization since each layer's M must
                  be fetched separately. Only supported with num_m_groups=1.
    """
    if num_m_groups < 1:
        raise ValueError(f'num_m_groups must be >= 1, got {num_m_groups}')
    if layers_per_m < 0:
        raise ValueError(f'layers_per_m must be >= 0, got {layers_per_m}')
    if num_m_groups > 1 and layers_per_m > 0:
        raise ValueError('num_m_groups and layers_per_m cannot both be active')
    if per_layer_m and (num_m_groups != 1 or layers_per_m > 0):
        raise ValueError('per_layer_m is only supported with num_m_groups=1 and layers_per_m=0')

    if from_scratch:
        model = GPT2LMHeadModel(GPT2Config.from_pretrained(pretrained))
        random_m = True   # no pretrained Wq/Wk to average
    else:
        model = GPT2LMHeadModel.from_pretrained(pretrained)
    cfg = model.config
    D = cfg.n_embd          # 768
    n_head = cfg.n_head     # 12
    d_head = D // n_head    # 64

    if per_layer_m:
        for layer in model.transformer.h:
            attn = layer.attn
            if random_m:
                M_init = torch.randn(D, D) / math.sqrt(D)
            else:
                Wq = attn.c_attn.weight[:, :D].detach()
                Wk = attn.c_attn.weight[:, D:2 * D].detach()
                M_init = (Wq @ Wk.T) * warmstart_scale
            M_layer = nn.Parameter(M_init)
            Wv      = attn.c_attn.weight[:, 2 * D:].detach()
            Wv_bias = attn.c_attn.bias[2 * D:].detach()
            Wo      = attn.c_proj.weight.detach()
            Wo_bias = attn.c_proj.bias.detach()
            layer.attn = BLTAttention(M_layer, Wv, Wv_bias, Wo, Wo_bias, cfg)

    elif layers_per_m > 0:
        G = cfg.n_layer // layers_per_m
        if cfg.n_layer % layers_per_m != 0:
            raise ValueError(f'layers_per_m={layers_per_m} does not evenly divide n_layer={cfg.n_layer}')

        if random_m:
            M_inits = [torch.randn(D, D) / math.sqrt(D) for _ in range(G)]
        else:
            M_inits = [torch.zeros(D, D) for _ in range(G)]
            counts = [0] * G
            for i, layer in enumerate(model.transformer.h):
                k = i % G
                Wq = layer.attn.c_attn.weight[:, :D].detach()
                Wk = layer.attn.c_attn.weight[:, D:2 * D].detach()
                M_inits[k] += Wq @ Wk.T
                counts[k] += 1
            for k in range(G):
                M_inits[k] = (M_inits[k] / counts[k]) * warmstart_scale

        M_params = [nn.Parameter(M_init) for M_init in M_inits]

        for i, layer in enumerate(model.transformer.h):
            attn = layer.attn
            Wv      = attn.c_attn.weight[:, 2 * D:].detach()
            Wv_bias = attn.c_attn.bias[2 * D:].detach()
            Wo      = attn.c_proj.weight.detach()
            Wo_bias = attn.c_proj.bias.detach()
            layer.attn = BLTAttention(M_params[i % G], Wv, Wv_bias, Wo, Wo_bias, cfg)

        for k, M in enumerate(M_params):
            model.transformer.register_parameter(f'M_layer_{k}', M)

    elif num_m_groups == 1:
        if random_m:
            M_init = torch.randn(D, D) / math.sqrt(D)
        else:
            # Initialize M as average of Wq @ Wk^T across all layers.
            # c_attn.weight shape: (768, 2304) — columns: Wq | Wk | Wv
            M_init = torch.zeros(D, D)
            for layer in model.transformer.h:
                Wq = layer.attn.c_attn.weight[:, :D].detach()
                Wk = layer.attn.c_attn.weight[:, D:2 * D].detach()
                M_init += Wq @ Wk.T
            M_init /= cfg.n_layer
            M_init *= warmstart_scale

        M_shared = nn.Parameter(M_init)

        for layer in model.transformer.h:
            attn = layer.attn
            Wv = attn.c_attn.weight[:, 2 * D:].detach()   # (768, 768)
            Wv_bias = attn.c_attn.bias[2 * D:].detach()   # (768,)
            Wo = attn.c_proj.weight.detach()               # (768, 768)
            Wo_bias = attn.c_proj.bias.detach()            # (768,)
            layer.attn = BLTAttention(M_shared, Wv, Wv_bias, Wo, Wo_bias, cfg)

        model.transformer.register_parameter('M_blt', M_shared)

    else:  # num_m_groups > 1
        G = num_m_groups
        if cfg.n_head % G != 0:
            raise ValueError(f'num_m_groups={G} must divide n_head={cfg.n_head}')
        n_g = n_head // G
        d_g = n_g * d_head

        if random_m:
            M_inits = [torch.randn(D, D) / math.sqrt(D) for _ in range(G)]
        else:
            M_inits = [torch.zeros(D, D) for _ in range(G)]
            for layer in model.transformer.h:
                Wq = layer.attn.c_attn.weight[:, :D].detach()
                Wk = layer.attn.c_attn.weight[:, D:2 * D].detach()
                for g in range(G):
                    M_inits[g] += Wq[:, g * d_g:(g + 1) * d_g] @ Wk[:, g * d_g:(g + 1) * d_g].T
            for g in range(G):
                M_inits[g] /= cfg.n_layer
                M_inits[g] *= warmstart_scale

        M_params = [nn.Parameter(M_init) for M_init in M_inits]

        for layer in model.transformer.h:
            attn = layer.attn
            Wv      = attn.c_attn.weight[:, 2 * D:].detach()
            Wv_bias = attn.c_attn.bias[2 * D:].detach()
            Wo      = attn.c_proj.weight.detach()
            Wo_bias = attn.c_proj.bias.detach()
            Wv_groups      = [Wv[:, g * d_g:(g + 1) * d_g] for g in range(G)]
            Wv_bias_groups = [Wv_bias[g * d_g:(g + 1) * d_g] for g in range(G)]
            layer.attn = BLTMultiMAttention(M_params, Wv_groups, Wv_bias_groups,
                                            Wo, Wo_bias, cfg)

        for g, M in enumerate(M_params):
            model.transformer.register_parameter(f'M_blt_{g}', M)

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
