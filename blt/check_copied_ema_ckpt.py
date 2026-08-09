import torch
ckpt = torch.load("run_gpt2_medium_ema_blend75_sine_seed42.pt", map_location="cpu")
print("copied checkpoint step:", ckpt["step"])
bad = any((not v.isfinite().all()) for v in ckpt["model_state"].values() if hasattr(v, "isfinite"))
print("corrupted" if bad else "copied checkpoint healthy, all finite")
