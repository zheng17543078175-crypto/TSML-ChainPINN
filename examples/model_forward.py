"""Inspect the full paper architecture without a dataset or trained checkpoint."""
import torch
from tsml_chainpinn import ChainPINN

model = ChainPINN()
model.eval()
with torch.no_grad():
    output = model(torch.zeros(4, 4))
print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
print("Output shapes:", {k: tuple(v.shape) for k, v in output.items()})
print("Randomly initialized model: outputs are NOT trained predictions.")
