"""Atomic epoch checkpoints with optimizer and RNG state for continuation."""
import os
import random
from pathlib import Path
import torch


def capture_rng(generator):
    return {"numpy_generator": generator.bit_generator.state,
            "python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    generator.bit_generator.state = state["numpy_generator"]
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "tsml-paper-v2":
        raise ValueError("Expected a paper-version checkpoint; legacy tsml3 weights are incompatible")
    return state
