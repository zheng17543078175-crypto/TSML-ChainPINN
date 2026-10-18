# Recorded synthetic example

These files are outputs of an actually executed CPU demonstration, not CFD results:

- `demo_config.json`: the separate reduced architecture and short training schedule.
- `history.csv`: seven completed epochs across three stages.
- `metrics.csv`: 40 chronological held-out synthetic snapshots.
- `summary.json`: temporal averages and explicit synthetic-data identity.

Recreate the workflow with `tsml demo --output outputs/demo --plot`. Runtime/library differences may change floating-point results. The fields are analytic synthetic functions and do not solve the physical CFD problem. Scores are useful for checking the software path only.

The homepage demonstration graphic is generated from these recorded logs. No trained weights or synthetic snapshots are included; the command generates them locally.
