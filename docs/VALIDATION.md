# Local validation record

Date: **2026-10-18**. This records executed implementation checks, not reproduction of the paper's numerical results.

| Check | Outcome |
|---|---|
| Analytic and workflow tests | 19 passed |
| Fourier 2π mapping and unweighted focal loss | Passed |
| Dimensional derivatives / conservative variable-mixture momentum | Passed against analytic fields |
| ZGB source signs and finite gradients | Passed |
| Freezing / memory-microbatch gradient equivalence | Passed for all training phases |
| Temporal split / fitting-only normalization | Passed |
| VTK conversion | Passed on a synthetic 200-state fixed tetrahedral mesh |
| Epoch interruption and continuation | Final CPU weights and history equal uninterrupted run |
| Python inference API | Matches saved full-grid evaluation |
| Data hash checking and aggregate mismatch rejection | Passed |
| Synthetic train → evaluate → plot command | Completed; 7 epochs, 40 held-out snapshots |
| Three-seed experiment orchestration | Completed on synthetic demo; mean/sample SD written |
| Built wheel installed into an isolated target | Version 0.3.0 imported; doctor and full demo/plot completed |
| Source distribution | Built successfully |
| Repository links and bundled defaults | Local release check passed |

The inherited, unchanged full-size model/physics core also passed synthetic Stage-1 and Stage-3 backward checks on an NVIDIA GeForce RTX 5080, with finite losses and gradients. It has 15,766,021 parameters. This is a differentiation check, not a complete GPU training run.

The local test environment uses Python 3.13 and PyTorch 2.9.1+cu128 on Windows. The supplied remote CI matrix targets Python 3.10 and 3.12 on Linux, but remote CI is not claimed to have run locally.

Plotting is executed in a separate process for the one-command demo to avoid PyTorch/Matplotlib OpenMP runtime collisions in the local Windows/conda environment. No duplicate-runtime suppression flag is used.

Recorded demo CSV/JSON are in [examples/reference_run](../examples/reference_run/README.md). Synthetic data and reduced models cannot establish accuracy on real stator cavitation.
