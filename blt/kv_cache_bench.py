"""
Incremental-decode (KV-cache) implementation and decode-latency benchmark for
BLT, standard MHA (GPT-2), GQA, and the 6+6 hybrid.

None of the training-time attention classes in model.py support caching --
they always recompute full attention over the whole sequence (this is
intentional; model.py is a training harness). This module adds a separate,
inference-only incremental path that reads the same underlying weights and
drives every architecture through the *same* manual decode loop, so timing
differences come from the attention math, not from loop-implementation
differences.

Per-layer cache contents:
  - MHA / GQA: K, V tensors of shape (B, n_kv_heads, T, d_head) -- standard.
  - BLT: raw hidden states X and projected values V, each (B, T, D) -- per
    Section 6 ("KV-cache compatibility"): the implicit key is just x_j, so
    the cache stores x_j itself and x_j @ W_v, with a single (x_t @ M)
    multiply per new token rather than a key projection.
"""

import math
import time
import torch
import torch.nn.functional as F

from model import (BLTAttention, GQAAttention, MHAAttention,
                    build_blt_model, build_gqa_model, build_hybrid_model)
from transformers import GPT2LMHeadModel, GPT2Config


# ---------------------------------------------------------------------------
# Per-layer weight extraction
# ---------------------------------------------------------------------------

def extract_layer_spec(attn, n_head, d_head):
    """Return a dict describing one layer's attention weights, tagged by type."""
    if isinstance(attn, BLTAttention):
        return {
            'type': 'blt',
            'M': attn.M, 'Wv': attn.Wv, 'Wv_b': attn.Wv_bias,
            'Wo': attn.Wo, 'Wo_b': attn.Wo_bias, 'scale': attn.scale,
        }
    if isinstance(attn, (GQAAttention, MHAAttention)):
        n_kv = attn.n_kv if isinstance(attn, GQAAttention) else attn.n_head
        return {
            'type': 'mha', 'n_head': attn.n_head, 'n_kv': n_kv, 'd_head': attn.d_head,
            'scale': attn.scale,
            'Wq': attn.Wq, 'Wq_b': attn.Wq_bias, 'Wk': attn.Wk, 'Wk_b': attn.Wk_bias,
            'Wv': attn.Wv, 'Wv_b': attn.Wv_bias, 'Wo': attn.Wo, 'Wo_b': attn.Wo_bias,
        }
    # Native HF GPT2Attention (used by the unmodified GPT-2 baseline).
    D = n_head * d_head
    c_attn_w, c_attn_b = attn.c_attn.weight, attn.c_attn.bias
    return {
        'type': 'mha', 'n_head': n_head, 'n_kv': n_head, 'd_head': d_head,
        'scale': math.sqrt(d_head),
        'Wq': c_attn_w[:, :D], 'Wq_b': c_attn_b[:D],
        'Wk': c_attn_w[:, D:2 * D], 'Wk_b': c_attn_b[D:2 * D],
        'Wv': c_attn_w[:, 2 * D:], 'Wv_b': c_attn_b[2 * D:],
        'Wo': attn.c_proj.weight, 'Wo_b': attn.c_proj.bias,
    }


def build_decode_model(model):
    """Extract everything needed for incremental decoding from a built model."""
    cfg = model.config
    D, n_head = cfg.n_embd, cfg.n_head
    d_head = D // n_head
    layer_specs = []
    for layer in model.transformer.h:
        spec = extract_layer_spec(layer.attn, n_head, d_head)
        spec['ln_1'] = layer.ln_1
        spec['ln_2'] = layer.ln_2
        spec['mlp'] = layer.mlp
        layer_specs.append(spec)
    return {
        'layers': layer_specs,
        'wte': model.transformer.wte,
        'wpe': model.transformer.wpe,
        'ln_f': model.transformer.ln_f,
        'lm_head': model.lm_head,
    }


# ---------------------------------------------------------------------------
# Cached attention: prefill (whole prompt at once) and step (one new token)
# ---------------------------------------------------------------------------

def _repeat_kv(t, q_per_kv):
    return t if q_per_kv == 1 else t.repeat_interleave(q_per_kv, dim=1)


def mha_prefill(x, spec, cache, causal_mask):
    B, L, D = x.shape
    n_head, n_kv, d_head, scale = spec['n_head'], spec['n_kv'], spec['d_head'], spec['scale']
    q = (x @ spec['Wq'] + spec['Wq_b']).view(B, L, n_head, d_head).transpose(1, 2)
    k = (x @ spec['Wk'] + spec['Wk_b']).view(B, L, n_kv, d_head).transpose(1, 2)
    v = (x @ spec['Wv'] + spec['Wv_b']).view(B, L, n_kv, d_head).transpose(1, 2)
    cache['K'], cache['V'] = k, v
    q_per_kv = n_head // n_kv
    K, V = _repeat_kv(k, q_per_kv), _repeat_kv(v, q_per_kv)
    scores = (q @ K.transpose(-2, -1)) / scale + causal_mask
    A = F.softmax(scores, dim=-1)
    h = (A @ V).transpose(1, 2).contiguous().view(B, L, D)
    return h @ spec['Wo'] + spec['Wo_b']


def mha_step(x_t, spec, cache):
    B = x_t.shape[0]
    n_head, n_kv, d_head, scale = spec['n_head'], spec['n_kv'], spec['d_head'], spec['scale']
    q = (x_t @ spec['Wq'] + spec['Wq_b']).view(B, 1, n_head, d_head).transpose(1, 2)
    k = (x_t @ spec['Wk'] + spec['Wk_b']).view(B, 1, n_kv, d_head).transpose(1, 2)
    v = (x_t @ spec['Wv'] + spec['Wv_b']).view(B, 1, n_kv, d_head).transpose(1, 2)
    cache['K'] = torch.cat([cache['K'], k], dim=2)
    cache['V'] = torch.cat([cache['V'], v], dim=2)
    q_per_kv = n_head // n_kv
    K, V = _repeat_kv(cache['K'], q_per_kv), _repeat_kv(cache['V'], q_per_kv)
    scores = (q @ K.transpose(-2, -1)) / scale
    A = F.softmax(scores, dim=-1)
    h = (A @ V).transpose(1, 2).contiguous().view(B, 1, -1)
    return h @ spec['Wo'] + spec['Wo_b']


def blt_prefill(x, spec, cache, causal_mask):
    v = x @ spec['Wv'] + spec['Wv_b']
    cache['X'], cache['V'] = x, v
    scores = (x @ spec['M']) @ x.transpose(-2, -1) / spec['scale'] + causal_mask
    A = F.softmax(scores, dim=-1)
    h = A @ v
    return h @ spec['Wo'] + spec['Wo_b']


def blt_step(x_t, spec, cache):
    v_t = x_t @ spec['Wv'] + spec['Wv_b']
    cache['X'] = torch.cat([cache['X'], x_t], dim=1)
    cache['V'] = torch.cat([cache['V'], v_t], dim=1)
    q = x_t @ spec['M']
    scores = (q @ cache['X'].transpose(-2, -1)) / spec['scale']
    A = F.softmax(scores, dim=-1)
    h = A @ cache['V']
    return h @ spec['Wo'] + spec['Wo_b']


# ---------------------------------------------------------------------------
# Full-model prefill / decode-step, mixing layer types as needed (hybrid)
# ---------------------------------------------------------------------------

def prefill(dm, input_ids):
    """input_ids: (B, L). Returns logits for the last position and a fresh cache list."""
    B, L = input_ids.shape
    device = input_ids.device
    pos = torch.arange(L, device=device).unsqueeze(0)
    x = dm['wte'](input_ids) + dm['wpe'](pos)

    causal = torch.full((L, L), float('-inf'), device=device, dtype=x.dtype)
    causal = torch.triu(causal, diagonal=1)

    caches = []
    for spec in dm['layers']:
        cache = {}
        residual = x
        h = spec['ln_1'](x)
        if spec['type'] == 'blt':
            attn_out = blt_prefill(h, spec, cache, causal)
        else:
            attn_out = mha_prefill(h, spec, cache, causal)
        x = residual + attn_out
        residual = x
        x = residual + spec['mlp'](spec['ln_2'](x))
        caches.append(cache)

    x = dm['ln_f'](x)
    logits = dm['lm_head'](x[:, -1:, :])
    return logits, caches, L


def decode_step(dm, token_ids, caches, position):
    """token_ids: (B, 1). position: scalar int, the position index of this token."""
    device = token_ids.device
    pos = torch.full((1, 1), position, device=device, dtype=torch.long)
    x = dm['wte'](token_ids) + dm['wpe'](pos)

    for spec, cache in zip(dm['layers'], caches):
        residual = x
        h = spec['ln_1'](x)
        if spec['type'] == 'blt':
            attn_out = blt_step(h, spec, cache)
        else:
            attn_out = mha_step(h, spec, cache)
        x = residual + attn_out
        residual = x
        x = residual + spec['mlp'](spec['ln_2'](x))

    x = dm['ln_f'](x)
    logits = dm['lm_head'](x)
    return logits


# ---------------------------------------------------------------------------
# Correctness check: cached decode must match the model's own (non-cached)
# forward pass, which already exercises the trained-and-tested attention
# classes computing full causal self-attention from scratch.
# ---------------------------------------------------------------------------

def check_correctness(model, name, device, seq_len=12, vocab_size=50257, atol=1e-3):
    model.eval()
    dm = build_decode_model(model)
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (2, seq_len), device=device)

    with torch.no_grad():
        full_logits = model(input_ids).logits  # (B, L, V)

        k = seq_len // 2
        logits_pf, caches, pos = prefill(dm, input_ids[:, :k])
        err = (logits_pf.squeeze(1) - full_logits[:, k - 1, :]).abs().max().item()
        ok = err < atol
        for t in range(k, seq_len):
            logits_step = decode_step(dm, input_ids[:, t:t + 1], caches, t)
            step_err = (logits_step.squeeze(1) - full_logits[:, t, :]).abs().max().item()
            err = max(err, step_err)
            ok = ok and (step_err < atol)

    status = 'OK' if ok else 'MISMATCH'
    print(f'  [{name}] correctness: {status} (max abs logit diff = {err:.2e})')
    return ok


# ---------------------------------------------------------------------------
# Decode-latency benchmark
# ---------------------------------------------------------------------------

def benchmark(model, name, device, batch_size, prompt_len, decode_len,
              vocab_size=50257, n_warmup=3):
    model.eval()
    dm = build_decode_model(model)
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

    def run_once():
        with torch.no_grad():
            logits, caches, pos = prefill(dm, input_ids)
            tok = logits.argmax(dim=-1)
            for i in range(decode_len):
                logits = decode_step(dm, tok, caches, pos + i)
                tok = logits.argmax(dim=-1)

    for _ in range(n_warmup):
        run_once()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        logits, caches, pos = prefill(dm, input_ids)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    tok = logits.argmax(dim=-1)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(decode_len):
            logits = decode_step(dm, tok, caches, pos + i)
            tok = logits.argmax(dim=-1)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_decode = time.perf_counter() - t0

    return {
        'name': name, 'batch_size': batch_size, 'prompt_len': prompt_len, 'decode_len': decode_len,
        'prefill_ms': t_prefill * 1000,
        'decode_ms_per_step': t_decode * 1000 / decode_len,
        'decode_tok_per_sec': (batch_size * decode_len) / t_decode,
    }


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    models = {}
    print('Loading checkpoints...')
    models['GPT-2 (MHA)'] = load_checkpoint(
        GPT2LMHeadModel(GPT2Config()), 'run_gpt2_baseline_seed42.pt', device).to(device)
    models['BLT'] = load_checkpoint(
        build_blt_model(from_scratch=True), 'run_blt_scratch_seed7.pt', device).to(device)
    models['GQA'] = load_checkpoint(
        build_gqa_model(from_scratch=True), 'run_gqa_scratch_seed42.pt', device).to(device)
    models['Hybrid (6+6)'] = load_checkpoint(
        build_hybrid_model(n_mha=6), 'run_hybrid_mha6_scratch_seed42.pt', device).to(device)

    print('\nCorrectness checks (cached decode vs. non-cached full forward):')
    for name, model in models.items():
        check_correctness(model, name, device)

    print('\nDecode-latency benchmark:')
    configs = [
        dict(batch_size=1, prompt_len=128, decode_len=64),
        dict(batch_size=8, prompt_len=128, decode_len=64),
        dict(batch_size=1, prompt_len=900, decode_len=64),
    ]
    results = []
    for cfg in configs:
        for name, model in models.items():
            r = benchmark(model, name, device, **cfg)
            results.append(r)
            print(f"  {r['name']:<14} bs={cfg['batch_size']:<3} prompt={cfg['prompt_len']:<5} "
                  f"prefill={r['prefill_ms']:>8.2f} ms  "
                  f"decode={r['decode_ms_per_step']:>6.3f} ms/step  "
                  f"{r['decode_tok_per_sec']:>9.1f} tok/s")

    import json
    with open('kv_cache_bench_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved results to kv_cache_bench_results.json')
