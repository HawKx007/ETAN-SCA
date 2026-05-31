# src/utils.py
import json, random, time
from pathlib import Path
import numpy as np
import torch


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_run_id():
    return time.strftime("%Y%m%d-%H%M%S")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def benchmark_latency_ms(model, device, T: int, iters: int = 100, warmup: int = 20):
    model.eval()
    x = torch.randn(1, T, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.time() - t0) * 1000.0 / iters


def save_json(path: Path, obj: dict):
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)