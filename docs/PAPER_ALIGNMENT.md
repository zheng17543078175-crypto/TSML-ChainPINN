# Paper alignment and implementation defaults

PyTorch implementation of **Transient cavitation reconstruction in hydrodynamic torque converter stators via physically constrained flow and phase learning**, revision 2026-10-18. Commands below are run from the repository root.

Includes the dual-branch model, three-stage training, mixture-momentum and ZGB vapor-transport losses, chronological data partitions, and original-mesh inference. Original research files and legacy checkpoints remain outside this package. This is a revised implementation, not a claim that old checkpoints were trained with these settings.

## Install

```bash
python -m pip install .
tsml check --config configs/paper.json
```

For VTK conversion or tests:

```bash
python -m pip install ".[vtk,test]"
python -m pytest -q
```

## Run

Prepare an explicit list of the **actual 200 CFD snapshots**, spaced by 0.005 s. Edit `configs/source_manifest.example.json`: replace the two illustrative records with all 200 actual file/time records and verify field names, phase convention, units and pressure offset. The example is not a dataset. For already prepared NPZ data, skip conversion.

```bash
tsml prepare --source-manifest source_manifest.json --output data/prepared
tsml train --config configs/paper.json --manifest data/prepared/manifest.json --output runs/seed1337 --mode final --seed 1337 --device cuda
tsml evaluate --checkpoint runs/seed1337/stage3.pt --manifest data/prepared/manifest.json --output results/seed1337 --device cuda --save-predictions
```

`--mode selection` fits snapshots 1–128 and evaluates 129–160. `--mode final` starts from initialization, fits 1–160 and evaluates 161–200. Normalization uses fitting snapshots only. Run three independent seeds in separate output directories for the paper's three-run protocol; seed 1337 is a software default, not a recovered paper seed. Train each operating condition separately.

## Main settings

| Item | Value |
|---|---|
| Branch depth / width | 10 / 512, both branches |
| Activation / Fourier bands / heads | Softplus / 64 from 2⁰ to 2⁸ / 8 |
| Optimizer / batch size | Adam / **4096** |
| Stage epochs | 80 / 120 / 160 |
| Learning rates | 1e-3 / 5e-4 / 1e-4 |
| Stage-3 phase-only warm-up | First 20 of 160 epochs |
| Gradient clipping | 1.0 |
| Cavitation threshold / enriched fraction | alpha_L < 0.90 / 0.20 |
| Hybrid loss weights / focal gamma | 1.0, 0.5, 2.0 / 2.0 |
| Stage-3 phase weight | 14.0 |
| Stage-1 physics weights | divergence 0.10, momentum 0.05 |
| Stage-3 physics weights | divergence 0.05, momentum 0.05, vapor 0.10 |
| ZGB pv / Fvap / Fcond / alpha_nuc / RB | 3169 Pa / 50 / 0.01 / 5e-4 / 1e-6 m |

Batch size 4096 follows Table S1 and the author's explicit choice over the manuscript's 640. `microbatch_size=64` only limits memory; gradients form one 4096-point Adam update, including correctly weighted masked losses.

## Data and conventions

Each NPZ contains `xyz` (N,3), `flow` (N,4) in **[absolute pressure, u, v, w]**, `alpha_liquid` (N,1), `alpha_gradient` (N,3; spatial CFD derivatives in SI), and `volume_weights` (N,). The JSON manifest holds chronological `id`, `time_s`, and `file` records, plus units `{"coordinates":"m","time":"s","velocity":"m/s","pressure":"Pa_absolute"}`. See the converter for the complete format. The minimal converter requires a fixed volume mesh with stable point ordering.

Inference uses every original mesh point without resampling. Stage-1 temporal phase derivatives use only neighboring fitting snapshots, including one-sided differences at fitting boundaries. CFD phase values and gradients supply the variable mixture properties during Stage 1; predicted phase supplies them during Stage 3. Pressure must be absolute for ZGB; the example's gauge offset must be checked against the CFD export.

## Implementation notes

- Equations (11), (12), and (18) are implemented with 2π Fourier encoding, outer Softplus residual activation, and unweighted binary focal loss. Flow outputs consistently use [p,u,v,w]. Legacy checkpoints are incompatible and intentionally excluded.
- Attention retains one token per coordinate from the original code: its softmax is singleton feature exchange, not spatial attention. The paper does not specify a different tokenization. Parameter count: **15,766,021**, rounding to the reported 1.58×10⁷.
- Figure 2's sampler draws 20% from the cavitation pool and 80% from the full domain; the latter can also contain cavitating points. Empty cavitation pools raise an explicit error.
- Unreported numerical choices are exposed: Softplus beta=2, 1000 updates/epoch, Huber delta=1, smooth positive-part width=1 Pa, flow/Stage-2 outer weights=1, fixed learning rates, FP32. Divergence scale is Uref/Lref; momentum scale is rho_L Uref²/Lref; vapor scale is rho_v Uref/Lref.
- Editable material defaults: rho_L=998.2, rho_v=0.02375 kg/m³; mu_L=0.001003, mu_v=1.34e-5 Pa·s; Uref=18 m/s, Lref=0.0155 m. These are implementation defaults carried forward from local diagnostics, not values fully specified in the paper.
- The package includes no CFD data, trained weights, baseline models, or plotting assets. Synthetic tests validate implementation only; reproduction of numerical paper results requires the actual sequence and new training.

MIT licensed. No publication or dataset redistribution is performed by this package.
