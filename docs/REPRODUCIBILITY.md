# Reproduction status

This repository is a new paper-oriented implementation of the 2026-10-18 revision. Its existence and passing tests do not establish that the archived weights used the revised loss, or that paper scores have been reproduced.

| Item | Release status |
|---|---|
| Model / training / dimensional physics | Implemented |
| CPU synthetic workflow | Executed locally; see validation record |
| Full-size GPU differentiation | Checked separately on synthetic coordinates |
| Real 200-state CFD sequence | Not bundled |
| Weights trained with the revised protocol | Not bundled; require new training |
| Reproduction of published numerical values | Not performed |
| Legacy code/checkpoints | Preserved outside this standalone repository |
| Hosted source and remote CI | See the [repository](https://github.com/zheng17543078175-crypto/TSML-ChainPINN) and [current workflow runs](https://github.com/zheng17543078175-crypto/TSML-ChainPINN/actions) |
| Package-index publishing | Not published to PyPI; install from the source repository |

## Reproduce a new experiment

1. Supply the true per-condition 200-state sequence, physical times, mesh, phase convention and absolute pressure.
2. Inspect material defaults and source unit mappings.
3. Run data validation with hashes and keep the prepared manifest unchanged.
4. During selection, fit only 1–128; compare validation 129–160 without enriched evaluation sampling.
5. Freeze chosen settings, initialize fresh models and fit 1–160; evaluate 161–200.
6. Repeat three independent seeds and retain all run/evaluation records.

The paper config's optimizer batch is 4096, per the author's explicit choice. Seeds, Softplus beta, smoothing width, Huber delta, epoch update count and some material parameters are software choices/defaults rather than recovered historical values; see [alignment notes](PAPER_ALIGNMENT.md).

## What the example means

The synthetic dataset is a small analytic space–time field. It does not solve the governing CFD problem, does not reproduce stator geometry, and is not a valid accuracy benchmark. Its reduced architecture and schedule exercise data preparation, differentiation, optimization, serialization, held-out evaluation and plotting. All demonstration figures and generated metadata identify that scope.

## Data and model distribution

The code is MIT licensed. No rights to redistribute CFD data or publication material are assumed. No repository URL, DOI, dataset download, hosted weights or citation identifier is invented. Add actual release links only when they exist.
