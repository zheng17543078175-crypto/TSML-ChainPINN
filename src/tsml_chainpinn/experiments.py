"""Independent-seed experiments and honest run-level uncertainty summaries."""
import csv
import json
from pathlib import Path
import numpy as np
from .data import read_json

METRICS = ("iou", "f1", "precision", "cavity_occupancy_error",
           "velocity_magnitude_mae_m_s", "pressure_demeaned_mae_pa", "alpha_mae")


def aggregate(summary_paths, output):
    records = [read_json(p) for p in summary_paths]
    if len(records) < 2:
        raise ValueError("At least two independent runs required for sample standard deviation")
    if len({r["seed"] for r in records}) != len(records):
        raise ValueError("Duplicate seeds are not independent runs")
    for key in ("mode", "snapshots", "manifest_sha256", "config_sha256", "data_kind"):
        if len({r[key] for r in records}) != 1:
            raise ValueError(f"Cannot pool runs with different {key}")
    rows = []
    for metric in METRICS:
        a = np.array([r[metric] for r in records], dtype=float)
        if not np.isfinite(a).all():
            raise ValueError(f"Nonfinite values for {metric}")
        rows.append({"metric": metric, "mean": float(a.mean()), "sample_std": float(a.std(ddof=1)),
                     "n_runs": len(a)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/"aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = {"data_kind": records[0]["data_kind"], "mode": records[0]["mode"],
              "seeds": [r["seed"] for r in records], "metrics": rows,
              "definition": "mean and sample SD across per-run temporal averages"}
    (output/"aggregate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Independent-run summary", "", f"Data: **{result['data_kind']}**; mode: **{result['mode']}**.", "",
             "| Metric | Mean | Sample SD | Runs |", "|---|---:|---:|---:|"]
    lines += [f"| {r['metric']} | {r['mean']:.6g} | {r['sample_std']:.6g} | {r['n_runs']} |" for r in rows]
    (output/"SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def run_experiment(cfg, manifest, output, seeds=(1337, 2024, 3407), mode="final", device="cpu"):
    from .engine import train, evaluate
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("Provide at least two distinct seeds (paper protocol: three)")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    paths = []
    for seed in seeds:
        checkpoint = train(cfg, manifest, root/f"seed_{seed}", mode, seed, device)
        evaluate(checkpoint, manifest, root/f"evaluation_{seed}", device)
        paths.append(root/f"evaluation_{seed}/summary.json")
    return aggregate(paths, root/"aggregate")
