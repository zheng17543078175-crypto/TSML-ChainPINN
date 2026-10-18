# Data contract

Use **200 actual chronological snapshots** at 0.005 s intervals (0.995 s span). Start time may be nonzero. IDs are one-based 1…200. Train a separate model for each operating condition. Do not rename, duplicate or interpolate legacy 39/40-frame archives to pretend they meet this contract.

## VTK conversion

Copy `configs/source_manifest.example.json` and supply all 200 records. Its two records only illustrate the schema. File paths are resolved relative to the source manifest. Specify:

- Coordinate, velocity and pressure scales to SI.
- Offset to absolute pressure; zero for an absolute-pressure field. The example's 101325 Pa is not universal.
- Exact velocity/pressure/phase fields and phase kind (`liquid` or `vapor`).
- A block-name/index path ending at one volume dataset; `[]` for a plain volume mesh.
- Actual physical `time_s` from the CFD export.

```bash
tsml prepare --source-manifest source_manifest.json --output data/prepared
tsml inspect-data --manifest data/prepared/manifest.json --verify-hashes
```

Cell fields are averaged onto existing nodes. The converter retains original positions, obtains nodal spatial phase gradients with VTK, and lumps each cell volume equally onto its vertices, preserving total volume. It does not create a uniform grid.

A fixed mesh and stable point ordering are required for temporal phase differences. Changing meshes raise an error instead of silently interpolating.

## Direct NPZ input

| Array | Shape | Meaning / units |
|---|---|---|
| `xyz` | N × 3 | Original positions, m |
| `flow` | N × 4 | **p_abs,u,v,w**, Pa and m/s |
| `alpha_liquid` | N × 1 | Liquid fraction in [0,1] |
| `alpha_gradient` | N × 3 | CFD spatial liquid-fraction gradient, 1/m |
| `volume_weights` | N | Nonnegative nodal volume quadrature, m³ |

Each snapshot has a distinct NPZ. Values must be finite. Evaluation uses every original point. The manifest has `condition`, optional `data_kind`, `units`, and 200 `snapshots` records:

```json
{
  "condition": "C1",
  "data_kind": "user_supplied_CFD",
  "units": {"coordinates": "m", "time": "s", "velocity": "m/s", "pressure": "Pa_absolute"},
  "snapshots": [
    {"id": 1, "time_s": 0.0, "file": "snapshot_001.npz", "sha256": "actual file digest"}
  ]
}
```

This single record illustrates the schema; all 200 are required. The converter writes real hashes. Direct NPZ manifests may omit hashes, but `--verify-hashes` then refuses to claim verification.

## Preventing leakage

Selection: normalize/train on 1–128, evaluate 129–160. Final retraining: reinitialize, normalize/train on 1–160, evaluate 161–200. Stage-1 temporal CFD phase derivatives use only neighboring **fitting** states, with one-sided differences at boundaries.

Normalization is point-weighted population mean/std over fitting snapshots; constant channels receive scale 1. Inference always uses saved statistics.
