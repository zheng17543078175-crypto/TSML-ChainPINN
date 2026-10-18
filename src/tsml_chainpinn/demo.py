"""Self-contained synthetic workflow example. NOT a CFD/paper benchmark."""
import hashlib
import json
from importlib.resources import files
from pathlib import Path
import numpy as np


def make_demo(output, points_per_axis=5):
    if points_per_axis < 3:
        raise ValueError("points_per_axis must be at least three")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    grid = np.linspace(0, .0155, points_per_axis)
    xyz = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    x, y, z = (xyz/.0155).T
    records = []
    for sid in range(1, 201):
        time = (sid-1)*.005
        phase = 2*np.pi*(x - time)
        alpha = .65 + .30*np.sin(phase)
        spatial = np.zeros((len(x), 3))
        spatial[:, 0] = .30*np.cos(phase)*2*np.pi/.0155
        velocity = np.column_stack((18+np.sin(phase), .2*np.cos(2*np.pi*y), .2*np.sin(2*np.pi*z)))
        pressure = 3169 + 100*np.cos(phase)
        name = f"snapshot_{sid:03d}.npz"
        np.savez_compressed(output/name, xyz=xyz, flow=np.column_stack((pressure, velocity)),
                            alpha_liquid=alpha[:, None], alpha_gradient=spatial,
                            volume_weights=np.full(len(x), .0155**3/len(x)))
        records.append({"id": sid, "time_s": time, "file": name,
                        "sha256": hashlib.sha256((output/name).read_bytes()).hexdigest()})
    manifest = {"data_kind": "SYNTHETIC_WORKFLOW_DEMO_NOT_CFD", "condition": "synthetic",
                "units": {"coordinates": "m", "time": "s", "velocity": "m/s", "pressure": "Pa_absolute"},
                "snapshots": records}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cfg = json.loads(files("tsml_chainpinn").joinpath("defaults/paper.json").read_text(encoding="utf-8"))
    cfg["model"].update(width=16, depth=2, bands=4, heads=2)
    cfg["training"].update(batch_size=16, microbatch_size=8, steps_per_epoch=2,
                           epochs=[2, 2, 3], warmup_epochs=1)
    cfg_path = output/"demo.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (output/"README.txt").write_text("Synthetic workflow example only. Fields do not solve the CFD equations.\n"
                                     "Reduced model/schedule, uniform illustrative quadrature. Not paper performance.\n")
    return output/"manifest.json", cfg_path


def run_demo(output, device="cpu", plot=False):
    import torch
    from .config import load_config
    from .engine import train, evaluate
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    if device == "cpu":
        torch.set_num_threads(min(2, old_threads))
    try:
        manifest, config = make_demo(output/"data")
        checkpoint = train(load_config(config), manifest, output/"run", "final", 1337, device)
        summary = evaluate(checkpoint, manifest, output/"evaluation", device, save_predictions=True)
        if plot:
            # Isolate plotting from PyTorch's OpenMP runtime (notably on conda
            # Windows). Do not suppress duplicate-runtime safety checks.
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "tsml_chainpinn", "plot",
                            "--run", str(output/"run"), "--evaluation", str(output/"evaluation"),
                            "--output", str(output/"figures")], check=True)
        return summary
    finally:
        torch.set_num_threads(old_threads)
