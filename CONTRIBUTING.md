# Contributing

Please keep changes small enough to review and include a minimal reproducer for bugs.

## Development setup

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For a change to physics or normalization, add an analytic test with a known derivative or residual. For training/checkpoint changes, test interrupted continuation. For data changes, verify that original coordinates, units, split boundaries and phase meanings remain explicit.

Do not alter paper defaults to make a smoke test faster; use a separate synthetic configuration. Do not insert fabricated scores, imply that synthetic tests validate paper performance, or silently substitute legacy datasets.

## Reporting a problem

Include the exact command, `tsml doctor` output, the configuration, error traceback, and a small synthetic reproducer when possible. Do not attach private CFD data or credentials to a public issue.

## Pull requests

Explain the concrete problem, resulting behavior and checks run. Mark any change to a paper equation, output channel order, threshold, split, metric definition or checkpoint format explicitly. Run `python tools/check_release.py` before proposing a release.
