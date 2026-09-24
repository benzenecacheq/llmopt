"""
Per-GPU-shard decode/prefill benchmark: isolates exactly the piece of an
attention layer that one GPU would hold under N-way tensor parallelism,
for two designs, swept across model width D:

  MHA shard  (N=8-way head sharding): Wq,Wk,Wv each D x (D/N), Wo (D/N) x D.
              Attention computed over the shard's own (D/N)-dim Q/K/V.
  UV shard   (num_uv_groups=8, one group/GPU): U,V each D x r (r fixed=256),
              Wv D x (D/N), Wo (D/N) x D. Attention computed over rank-r Q/K.

Both shards produce a (D/N)-dim partial output that would be all-reduced
across GPUs in a real deployment -- that all-reduce is a one-time, roughly
equal-sized cost for both designs (same output shape (D/N) either way), so
it's deliberately excluded here to isolate the piece that actually differs:
per-shard compute + per-shard KV-cache read/write.

No trained weights needed -- this measures wall-clock/bandwdith, not quality,
matching the same principle kv_cache_bench.py already uses for its cached-
decode correctness and latency checks.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'
torch.manual_seed(0)

N_SHARDS = 8      # tensor-parallel degree = num_uv_groups
UV_RANK = 256     # fixed, matches the project's medium-scale run
D_VALUES = [768, 1024, 2048, 4096, 8192]
BATCH = 1
PROMPT_LEN = 128
DECODE_LEN = 200
WARMUP = 10


class MHAShard(nn.Module):
    """One GPU's slice under N-way head-sharded TP: Q/K/V/O each D x (D/N)."""
    def __init__(self, D, d_shard):
        super().__init__()
        self.Wq = nn.Linear(D, d_shard, bias=False)
        self.Wk = nn.Linear(D, d_shard, bias=False)
        self.Wv = nn.Linear(D, d_shard, bias=False)
        self.Wo = nn.Linear(d_shard, D, bias=False)
        self.scale = d_shard ** 0.5

    def prefill(self, x, causal_mask):
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = (q @ k.transpose(-2, -1)) / self.scale + causal_mask
        a = F.softmax(scores, dim=-1)
        h = a @ v
        return self.Wo(h), {'K': k, 'V': v}

    def step(self, x_t, cache):
        q_t, k_t, v_t = self.Wq(x_t), self.Wk(x_t), self.Wv(x_t)
        cache['K'] = torch.cat([cache['K'], k_t], dim=1)
        cache['V'] = torch.cat([cache['V'], v_t], dim=1)
        scores = (q_t @ cache['K'].transpose(-2, -1)) / self.scale
        a = F.softmax(scores, dim=-1)
        h = a @ cache['V']
        return self.Wo(h)


class UVShard(nn.Module):
    """One GPU's slice under num_uv_groups=N TP (one group/GPU): U,V D x r, Wv D x (D/N)."""
    def __init__(self, D, d_shard, r):
        super().__init__()
        self.U = nn.Linear(D, r, bias=False)
        self.V = nn.Linear(D, r, bias=False)
        self.Wv = nn.Linear(D, d_shard, bias=False)
        self.Wo = nn.Linear(d_shard, D, bias=False)
        self.scale = r ** 0.5

    def prefill(self, x, causal_mask):
        q, k, v = self.U(x), self.V(x), self.Wv(x)
        scores = (q @ k.transpose(-2, -1)) / self.scale + causal_mask
        a = F.softmax(scores, dim=-1)
        h = a @ v
        return self.Wo(h), {'K': k, 'V': v}

    def step(self, x_t, cache):
        q_t, k_t, v_t = self.U(x_t), self.V(x_t), self.Wv(x_t)
        cache['K'] = torch.cat([cache['K'], k_t], dim=1)
        cache['V'] = torch.cat([cache['V'], v_t], dim=1)
        scores = (q_t @ cache['K'].transpose(-2, -1)) / self.scale
        a = F.softmax(scores, dim=-1)
        h = a @ cache['V']
        return self.Wo(h)


def bench_shard(shard, D):
    causal = torch.triu(torch.full((PROMPT_LEN, PROMPT_LEN), float('-inf'), device=device), diagonal=1)
    x_prompt = torch.randn(BATCH, PROMPT_LEN, D, device=device)
    x_step = torch.randn(BATCH, 1, D, device=device)

    def run_once():
        _, cache = shard.prefill(x_prompt, causal)
        for _ in range(DECODE_LEN):
            shard.step(x_step, cache)
        torch.cuda.synchronize()

    for _ in range(3):
        run_once()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, cache = shard.prefill(x_prompt, causal)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(DECODE_LEN):
        shard.step(x_step, cache)
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t0

    return ttft * 1000, DECODE_LEN / decode_time


print(f"N_SHARDS={N_SHARDS}  UV_RANK={UV_RANK}  batch={BATCH}  prompt={PROMPT_LEN}  decode_steps={DECODE_LEN}")
print(f"{'D':>6} | {'d_shard':>8} | {'MHA TTFT(ms)':>13} | {'MHA tok/s':>10} | {'UV TTFT(ms)':>12} | {'UV tok/s':>9} | {'UV/MHA decode':>13}")
for D in D_VALUES:
    d_shard = D // N_SHARDS
    mha = MHAShard(D, d_shard).to(device).eval()
    uv = UVShard(D, d_shard, UV_RANK).to(device).eval()
    with torch.no_grad():
        mha_ttft, mha_tps = bench_shard(mha, D)
        uv_ttft, uv_tps = bench_shard(uv, D)
    ratio = uv_tps / mha_tps
    print(f"{D:>6} | {d_shard:>8} | {mha_ttft:>13.3f} | {mha_tps:>10.1f} | {uv_ttft:>12.3f} | {uv_tps:>9.1f} | {ratio:>13.3f}")
    del mha, uv
    torch.cuda.empty_cache()
