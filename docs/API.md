# Python and CLI reference

## Model

```python
import torch
from tsml_chainpinn import ChainPINN
model = ChainPINN(width=512, depth=10, bands=64, heads=8)
out = model(torch.zeros(8, 4))
# out['flow']: (8,4), normalized [p,u,v,w]
# out['alpha_logits']: (8,1)
# out['alpha']: (8,1), physical liquid fraction
```

`model.set_stage(1)`, `model.set_stage(2)`, and `model.set_stage(3, warmup=True/False)` set parameter trainability. These are optimization controls, not different checkpoint architectures.

## Dimensional inference

`Surrogate(checkpoint, device='cpu')` loads a completed Stage-3 checkpoint. `predict(coordinates, batch_size=4096)` takes a finite nonempty N×4 SI array and returns copied coordinates, dimensional flow, liquid/vapor fractions, and a Boolean cavitation mask.

Do not feed normalized coordinates to `Surrogate.predict`: normalization is performed internally. Do not assume a checkpoint generalizes to another operating condition.

## Training

`train(config, manifest, output, mode='final', seed=1337, device='cpu', resume=False)` returns the completed Stage-3 checkpoint path. Use `load_config(path)` to validate a config first. New output directories must not exist; resume uses the existing `latest.pt`.

`evaluate(checkpoint, manifest, output, device='cpu', batch_size=4096, save_predictions=False)` returns the temporal-average summary and writes per-snapshot metrics. It enforces the checkpoint's recorded split and manifest.

## Commands

| Command | Purpose |
|---|---|
| `tsml doctor` | Report local environment and optional dependencies |
| `tsml check --config ...` | Validate configuration and count full-model parameters |
| `tsml demo --output ... [--plot]` | Run the small synthetic workflow |
| `tsml prepare --source-manifest ... --output ...` | Convert original CFD volume meshes |
| `tsml inspect-data --manifest ... [--verify-hashes]` | Inspect all 200 prepared snapshots |
| `tsml train ... [--resume]` | Three-stage training/continuation |
| `tsml evaluate ...` | Full-grid held-out metrics and optional predictions |
| `tsml experiment ... --seeds 1337 2024 3407` | Train/evaluate independent runs |
| `tsml aggregate --summaries ... --output ...` | Aggregate completed evaluations |
| `tsml plot --run ... --evaluation ... --output ...` | Plot recorded histories and metrics |

`python -m tsml_chainpinn` is equivalent to `tsml`. Each subcommand provides `--help`.
