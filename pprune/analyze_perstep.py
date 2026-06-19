"""
analyze_perstep.py
------------------
Analyze and print per-step KL / token-match trajectories from perStep_kl_eval.py output.

Usage:
    python analyze_perstep.py \\
        --input lb_results_base/perstep_kl_longbench.json \\
        --methods naive_65pct,chunk_word128_t20,snapkv,pyramidkv \\
        --max_step 30

    # Combine longbench + synthetic:
    python analyze_perstep.py \\
        --input lb_results_base/perstep_kl_longbench.json \\
               lb_results_base/perstep_kl_synthetic.json \\
        --methods naive_65pct,chunk_word128_t20,snapkv,pyramidkv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Friendly display names
METHOD_LABELS = {
    "naive_65pct":        "Naive",
    "chunk_word128_t20":  "phr128",
    "snapkv":             "SnapKV",
    "pyramidkv":          "PyramidKV",
    "streaming":          "Streaming",
}


def load_ckpt(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def collect_trajectories(
    ckpts: List[dict],
    methods: List[str],
) -> Dict[str, dict]:
    """Collect kl_steps, match_steps, fout, fout_early per method across all checkpoints."""
    result = {m: {"kl": [], "match": [], "fout": [], "fout_early": []} for m in methods}
    for ckpt in ckpts:
        for key, val in ckpt.items():
            if not isinstance(val, dict):
                continue
            parts = key.split("|")
            if len(parts) != 3:
                continue
            _, method, _ = parts
            if method not in result:
                continue
            result[method]["kl"].append(val["kl_steps"])
            result[method]["match"].append([float(b) for b in val["match_steps"]])
            if val.get("fout") is not None:
                result[method]["fout"].append(val["fout"])
            if val.get("fout_early") is not None:
                result[method]["fout_early"].append(val["fout_early"])
    return result


def mean_at_step(trajectories: List[List[float]], t: int) -> Optional[float]:
    """Mean over examples that have at least t+1 steps."""
    vals = [s[t] for s in trajectories if len(s) > t]
    return float(np.mean(vals)) if vals else None


def n_at_step(trajectories: List[List[float]], t: int) -> int:
    return sum(1 for s in trajectories if len(s) > t)


def print_trajectory_table(data: Dict[str, dict], methods: List[str], max_step: int = 20):
    """Print KL(t) and match%(t) tables side-by-side."""
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    col_w  = max(max(len(l) for l in labels), 8)

    # --- KL table ---
    print("\nKL(P_full || P_comp) at each generation step t (lower is better)")
    print(f"{'t':>4}  {'n':>5}  " + "  ".join(f"{l:>{col_w}}" for l in labels))
    print("-" * (4 + 2 + 5 + 2 + (col_w + 2) * len(methods)))
    for t in range(max_step):
        row_parts = []
        n_ref = None
        for m in methods:
            kl_traj = data[m]["kl"]
            v = mean_at_step(kl_traj, t)
            n = n_at_step(kl_traj, t)
            if n_ref is None:
                n_ref = n
            row_parts.append(f"{v:>{col_w}.4f}" if v is not None else f"{'—':>{col_w}}")
        if n_ref == 0:
            break
        print(f"{t:>4}  {n_ref:>5}  " + "  ".join(row_parts))

    # --- Match rate table ---
    print("\nToken-match rate at each generation step t (higher is better, %)")
    print(f"{'t':>4}  {'n':>5}  " + "  ".join(f"{l:>{col_w}}" for l in labels))
    print("-" * (4 + 2 + 5 + 2 + (col_w + 2) * len(methods)))
    for t in range(max_step):
        row_parts = []
        n_ref = None
        for m in methods:
            mt = data[m]["match"]
            v = mean_at_step(mt, t)
            n = n_at_step(mt, t)
            if n_ref is None:
                n_ref = n
            row_parts.append(f"{v*100:>{col_w}.1f}" if v is not None else f"{'—':>{col_w}}")
        if n_ref == 0:
            break
        print(f"{t:>4}  {n_ref:>5}  " + "  ".join(row_parts))


def print_aggregate_summary(data: Dict[str, dict], methods: List[str]):
    """Print summary stats: KL trajectory, F_out, early F_out."""
    print("\nAggregate summary")
    print(f"{'Method':<26}  {'n_ex':>5}  {'KL_mean':>8}  {'KL_t0':>7}  {'KL_t5':>7}  "
          f"{'t0_match%':>10}  {'F_out%':>8}  {'F_early%':>9}")
    print("-" * 90)

    for m in methods:
        kl_traj    = data[m]["kl"]
        mt_traj    = data[m]["match"]
        fout_vals  = data[m]["fout"]
        fearly_vals = data[m]["fout_early"]
        label      = METHOD_LABELS.get(m, m)

        if not kl_traj:
            print(f"{label:<26}  {'—':>5}")
            continue

        mean_kl  = float(np.mean([np.mean(s) for s in kl_traj]))
        kl_t0    = mean_at_step(kl_traj, 0)
        kl_t5    = mean_at_step(kl_traj, 5)
        match_t0 = mean_at_step(mt_traj, 0)
        n_ex     = len(kl_traj)
        fout_str   = f"{np.mean(fout_vals)*100:>7.1f}%" if fout_vals  else f"{'—':>8}"
        fearly_str = f"{np.mean(fearly_vals)*100:>8.1f}%" if fearly_vals else f"{'—':>9}"

        print(
            f"{label:<26}  {n_ex:>5}  {mean_kl:>8.4f}  "
            f"{kl_t0:>7.4f}  "
            f"{(kl_t5 if kl_t5 is not None else float('nan')):>7.4f}  "
            f"{match_t0*100:>9.1f}%  "
            f"{fout_str}  {fearly_str}"
        )

    # T0 vs T5+ ratio for each method
    print("\nKL ratio T5/T0 (PyramidKV should show the largest ratio):")
    for m in methods:
        kl_traj = data[m]["kl"]
        label   = METHOD_LABELS.get(m, m)
        if not kl_traj:
            continue
        kl_t0 = mean_at_step(kl_traj, 0)
        kl_t5 = mean_at_step(kl_traj, 5)
        if kl_t0 and kl_t5 and kl_t0 > 1e-9:
            ratio = kl_t5 / kl_t0
            print(f"  {label:<24}  KL_t5/KL_t0 = {ratio:.2f}x")

    # Cumulative match rate: fraction of examples where match degrades between t0 and t10
    print("\nMatch-rate drop from t=0 to t=10 (PyramidKV should show the largest drop):")
    for m in methods:
        mt_traj = data[m]["match"]
        label   = METHOD_LABELS.get(m, m)
        if not mt_traj:
            continue
        m_t0  = mean_at_step(mt_traj, 0)
        m_t10 = mean_at_step(mt_traj, 10)
        if m_t0 is not None and m_t10 is not None:
            drop = (m_t0 - m_t10) * 100
            print(f"  {label:<24}  t0={m_t0*100:.1f}%  t10={m_t10*100:.1f}%  drop={drop:+.1f}pp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    nargs="+", required=True,
                   help="One or more checkpoint JSON files from perStep_kl_eval.py")
    p.add_argument("--methods",  default="naive_65pct,chunk_word128_t20,snapkv,pyramidkv")
    p.add_argument("--max_step", type=int, default=20,
                   help="Maximum step t to show in trajectory tables")
    return p.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    ckpts = [load_ckpt(Path(p)) for p in args.input]
    total = sum(sum(1 for v in c.values() if isinstance(v, dict)) for c in ckpts)
    print(f"Loaded {len(ckpts)} checkpoint(s), {total} completed entries")

    data = collect_trajectories(ckpts, methods)

    n_per_method = {m: len(data[m]["kl"]) for m in methods}
    print("Examples per method:", {METHOD_LABELS.get(m, m): n for m, n in n_per_method.items()})

    print_aggregate_summary(data, methods)
    print_trajectory_table(data, methods, max_step=args.max_step)


if __name__ == "__main__":
    main()
