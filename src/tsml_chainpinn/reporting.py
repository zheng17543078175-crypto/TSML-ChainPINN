"""Figures made from actual saved logs and metrics, without fabricated scores."""
import csv
import json
from pathlib import Path
import numpy as np


def plot_run(run_dir, evaluation_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir, evaluation_dir, output = map(Path, (run_dir, evaluation_dir, output))
    with (run_dir/"history.csv").open(encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    with (evaluation_dir/"metrics.csv").open(encoding="utf-8") as f:
        metrics = list(csv.DictReader(f))
    summary = json.loads((evaluation_dir/"summary.json").read_text())
    synthetic = summary.get("data_kind", "").startswith("SYNTHETIC")
    output.mkdir(parents=True, exist_ok=False)
    colors = ["#167c80", "#cc762c", "#5963b7"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        for stage, color in zip((1, 2, 3), colors):
            ids = [i for i, r in enumerate(history) if int(r["stage"]) == stage]
            axes[0].plot(np.array(ids)+1, [float(history[i]["loss"]) for i in ids],
                         "o-", ms=3, label=f"Stage {stage}", color=color)
        axes[0].set(xlabel="Training epoch", ylabel="Objective (stage-dependent)", title="Training history")
        if all(float(r["loss"]) > 0 for r in history):
            axes[0].set_yscale("log")
        axes[0].legend(frameon=False, fontsize=8)
        time = [float(r["time_s"]) for r in metrics]
        for key, color in zip(("iou", "f1", "precision"), colors):
            axes[1].plot(time, [float(r[key]) for r in metrics], label=key.upper(), color=color)
        axes[1].set(xlabel="Physical time (s)", ylabel="Score", ylim=(-.03, 1.03), title="Held-out cavity overlap")
        axes[1].legend(frameon=False, fontsize=8)
        axes[2].plot(time, [float(r["alpha_mae"]) for r in metrics], color=colors[0])
        axes[2].set(xlabel="Physical time (s)", ylabel="Liquid-fraction MAE", title="Held-out phase error")
        for ax in axes:
            ax.grid(alpha=.16)
        title = "SYNTHETIC WORKFLOW DEMO — not CFD or paper results" if synthetic else "TSML-ChainPINN · recorded run"
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.savefig(output/"run_summary.png", dpi=180)
        fig.savefig(output/"run_summary.svg")
        plt.close(fig)
    return output/"run_summary.png"
