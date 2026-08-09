import torch
ckpt = torch.load("run_gpt2_medium_baseline_seed42.pt", map_location="cpu")
print("bender medium baseline final step:", ckpt["step"])
bad = any((not v.isfinite().all()) for v in ckpt["model_state"].values() if hasattr(v, "isfinite"))
print("corrupted" if bad else "checkpoint healthy, all finite")
