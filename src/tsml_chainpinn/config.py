"""Fail early on missing physical constants and unresolved paper settings."""
import math
from .data import read_json


def load_config(path):
    cfg = read_json(path)
    missing = [f"physics.{k}" for k, v in cfg["physics"].items() if v is None]
    missing += [f"training.{k}" for k, v in cfg["training"].items() if v is None]
    if missing:
        raise ValueError("Required author/CFD settings unresolved: " + ", ".join(missing))
    for k, v in cfg["physics"].items():
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"physics.{k} must be finite and positive")
    t = cfg["training"]
    for k in ("batch_size", "microbatch_size", "steps_per_epoch"):
        if not isinstance(t[k], int) or t[k] < 1:
            raise ValueError(f"training.{k} must be a positive integer")
    if len(t["epochs"]) != 3 or len(t["learning_rates"]) != 3:
        raise ValueError("Three stage schedules are required")
    if any(not isinstance(x, int) or x <= 0 for x in t["epochs"]):
        raise ValueError("Epoch counts must be positive integers")
    if any(not math.isfinite(x) or x <= 0 for x in t["learning_rates"]):
        raise ValueError("Learning rates must be finite and positive")
    if not 0 <= t["warmup_epochs"] <= t["epochs"][2]:
        raise ValueError("Warm-up must fit within Stage 3")
    return cfg
