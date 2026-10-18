"""Three-stage training and full-original-grid inference."""
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from .model import ChainPINN
from .data import Sequence, split_ids, fit_normalization, sample_indices
from .losses import hybrid_loss, flow_loss
from .physics import physics_losses
from .checkpoint import capture_rng, restore_rng, atomic_save, load_checkpoint


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def objective(model, coords, target_flow, alpha, alpha_grad, norm, cfg, stage, warmup=False,
              phase_denominators=None):
    pred = model(coords)
    t = cfg["training"]
    if stage == 1:
        r = physics_losses(coords, pred, norm, cfg["physics"], alpha_reference=alpha,
                           alpha_reference_gradient=alpha_grad, include_vapor=False)
        return flow_loss(pred["flow"], target_flow) + sum(
            w * r[k] for w, k in zip(t["stage1_physics_weights"], ("divergence", "momentum")))
    phase = hybrid_loss(pred["alpha_logits"], alpha, t["cavitation_threshold"],
                        t["focal_gamma"], t["hybrid_weights"])
    if phase_denominators is not None:
        # Restore whole-batch masked means across memory microbatches. This is
        # necessary because masks may have different occupancy in each chunk.
        nc, nn, total = phase_denominators
        mask = alpha < t["cavitation_threshold"]
        n = len(alpha)
        wc, wr, wf = t["hybrid_weights"]
        phase["total"] = (wc * phase["focal"] +
                          wr * phase["reg"] * mask.sum() * total / (max(nc, 1) * n) +
                          wf * phase["fp"] * (~mask).sum() * total / (max(nn, 1) * n))
    if stage == 2 or warmup:
        return phase["total"] * (t["stage3_phase_weight"] if warmup else 1.)
    r = physics_losses(coords, pred, norm, cfg["physics"])
    return (flow_loss(pred["flow"], target_flow) + t["stage3_phase_weight"] * phase["total"]
            + sum(w * r[k] for w, k in zip(t["stage3_physics_weights"],
                                          ("divergence", "momentum", "vapor"))))


def train(cfg, manifest, output, mode="final", seed=1337, device="cpu", resume=False):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    sequence = Sequence(manifest)
    fitting, evaluation = split_ids(mode)
    out = Path(output)
    previous = load_checkpoint(out / "latest.pt") if resume else None
    manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    if previous is not None:
        provenance = previous["provenance"]
        if (previous["config"] != cfg or provenance["mode"] != mode or provenance["seed"] != seed
                or provenance["manifest_sha256"] != manifest_hash):
            raise ValueError("Resume requires identical config, seed, mode and data manifest")
        if "optimizer" not in previous or "rng" not in previous:
            raise ValueError("Resume requires latest.pt with optimizer and RNG states")
        norm = previous["normalization"]
    else:
        norm = fit_normalization(sequence, fitting)
    # Preflight every fitting snapshot, rather than fail hours into a run.
    for sid in fitting:
        if not np.any(sequence.load(sid)["alpha_liquid"] < cfg["training"]["cavitation_threshold"]):
            raise ValueError(f"Snapshot {sid} has no cavitation pool.")
    if not resume:
        out.mkdir(parents=True, exist_ok=False)  # Never overwrite another seed/run.
    (out / "normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    provenance = {"format": "tsml-paper-v2", "config": cfg, "mode": mode, "seed": seed,
                  "fitting_ids": fitting, "evaluation_ids": evaluation,
                  "manifest_sha256": manifest_hash,
                  "torch_version": str(torch.__version__), "device": device,
                  "data_kind": sequence.manifest.get("data_kind", "user_supplied_CFD"),
                  "status": "NEW_TRAINING_NOT_LEGACY_PAPER_RESULTS"}
    (out / "run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    model = ChainPINN(**cfg["model"]).to(device)
    t = cfg["training"]
    history = previous.get("history", []) if previous is not None else []
    if previous is not None:
        model.load_state_dict(previous["model"])
        restore_rng(previous["rng"], rng)
    history_file = (out / "history.csv").open("w", newline="", encoding="utf-8")
    try:
        writer = csv.writer(history_file)
        writer.writerow(["stage", "epoch", "warmup", "loss"])
        writer.writerows(history)
        for stage in (1, 2, 3):
            if previous is not None and stage < previous["stage"]:
                continue
            # Include frozen parameters so the warm-up transition can unfreeze
            # them without discarding Adam state for the phase branch.
            optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rates"][stage - 1])
            start_epoch = 0
            if previous is not None and stage == previous["stage"]:
                start_epoch = previous["epoch"]
                optimizer.load_state_dict(previous["optimizer"])
            for epoch in range(start_epoch, t["epochs"][stage - 1]):
                warmup = stage == 3 and epoch < t["warmup_epochs"]
                model.train()
                model.set_stage(stage, warmup)
                epoch_loss = 0.
                for _ in range(t["steps_per_epoch"]):
                    sid = int(rng.choice(fitting))
                    d = sequence.load(sid)
                    idx = sample_indices(d["alpha_liquid"], t["batch_size"], rng,
                                         t["cavitation_fraction"], t["cavitation_threshold"])
                    nc = int((d["alpha_liquid"][idx] < t["cavitation_threshold"]).sum())
                    denominators = (nc, len(idx) - nc, len(idx))
                    alpha_gradient = sequence.fitting_alpha_gradient(sid, fitting) if stage == 1 else None
                    optimizer.zero_grad(set_to_none=True)
                    for start in range(0, len(idx), t["microbatch_size"]):
                        ix = idx[start:start + t["microbatch_size"]]
                        tensor = lambda a: torch.as_tensor(a, device=device, dtype=torch.float32)
                        x = tensor((d["coords"][ix] - norm["coord_mean"]) / norm["coord_scale"]).requires_grad_(True)
                        y = tensor((d["flow"][ix] - norm["flow_mean"]) / norm["flow_scale"])
                        loss = objective(model, x, y, tensor(d["alpha_liquid"][ix]),
                                         tensor(alpha_gradient[ix]) if alpha_gradient is not None else None, norm, cfg, stage,
                                         warmup, denominators) * len(ix) / len(idx)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite loss: stage={stage}, epoch={epoch + 1}")
                        loss.backward()
                        epoch_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                mean = epoch_loss / t["steps_per_epoch"]
                row = [stage, epoch + 1, warmup, mean]
                history.append(row)
                atomic_save({"format": "tsml-paper-v2", "model": model.state_dict(),
                             "optimizer": optimizer.state_dict(), "rng": capture_rng(rng),
                             "config": cfg, "normalization": norm, "provenance": provenance,
                             "stage": stage, "epoch": epoch + 1, "history": history}, out / "latest.pt")
                writer.writerow(row)
                history_file.flush()
                print(f"Stage {stage}, epoch {epoch + 1}: {mean:.6g}", flush=True)
            atomic_save({"format": "tsml-paper-v2", "model": model.state_dict(),
                        "config": cfg, "normalization": norm, "provenance": provenance,
                        "stage": stage, "epoch": t["epochs"][stage - 1], "completed_stage": True}, out / f"stage{stage}.pt")
    finally:
        history_file.close()
    return out / "stage3.pt"


def evaluate(checkpoint, manifest, output, device="cpu", batch_size=4096, save_predictions=False):
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    state = load_checkpoint(checkpoint)
    if state.get("stage") != 3 or state.get("epoch") != state["config"]["training"]["epochs"][2]:
        raise ValueError("Use a completed paper-v2 Stage-3 checkpoint, not legacy TSML weights.")
    norm, cfg = state["normalization"], state["config"]
    sequence = Sequence(manifest)
    digest = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    if digest != state["provenance"]["manifest_sha256"]:
        raise ValueError("Manifest differs from the training record")
    model = ChainPINN(**cfg["model"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for sid in state["provenance"]["evaluation_ids"]:
        d = sequence.load(sid)
        flows, alphas = [], []
        with torch.no_grad():
            for start in range(0, len(d["coords"]), batch_size):
                x = (d["coords"][start:start + batch_size] - norm["coord_mean"]) / norm["coord_scale"]
                pred = model(torch.as_tensor(x, dtype=torch.float32, device=device))
                flows.append(pred["flow"].cpu().numpy() * norm["flow_scale"] + norm["flow_mean"])
                alphas.append(pred["alpha"].cpu().numpy())
        f, a = np.concatenate(flows), np.concatenate(alphas)
        threshold = cfg["training"]["cavitation_threshold"]
        pc, tc = a.ravel() < threshold, d["alpha_liquid"].ravel() < threshold
        tp, fp, fn = np.sum(pc & tc), np.sum(pc & ~tc), np.sum(~pc & tc)
        weights = d["volume_weights"] / d["volume_weights"].sum()
        # Empty/empty sets: IoU=F1=1, precision=1; otherwise undefined
        # precision with no predictions is reported as 0.
        iou = float(tp / (tp + fp + fn)) if tp + fp + fn else 1.
        f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.
        precision = float(tp / (tp + fp)) if tp + fp else (1. if not tc.any() else 0.)
        pressure_error = (f[:, 0] - f[:, 0].mean()) - (d["flow"][:, 0] - d["flow"][:, 0].mean())
        rows.append({"snapshot": sid, "time_s": sequence.records[sid - 1]["time_s"],
                     "iou": iou, "f1": f1, "precision": precision,
                     "cavity_occupancy_error": float(abs(weights @ pc - weights @ tc)),
                     "velocity_magnitude_mae_m_s": float(np.abs(np.linalg.norm(f[:, 1:], axis=1) - np.linalg.norm(d["flow"][:, 1:], axis=1)).mean()),
                     "pressure_demeaned_mae_pa": float(np.abs(pressure_error).mean()),
                     "alpha_mae": float(np.abs(a - d["alpha_liquid"]).mean())})
        if save_predictions:
            np.savez_compressed(out / f"prediction_{sid:03d}.npz", xyz=d["xyz"], flow=f,
                                alpha_liquid=a, time_s=sequence.records[sid - 1]["time_s"],
                                volume_weights=d["volume_weights"])
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {k: float(np.mean([row[k] for row in rows])) for k in rows[0] if k not in ("snapshot", "time_s")}
    summary.update({"mode": state["provenance"]["mode"], "seed": state["provenance"]["seed"],
                    "snapshots": len(rows), "evaluation": "full_original_mesh_no_resampling",
                    "manifest_sha256": digest,
                    "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                    "data_kind": state["provenance"].get("data_kind", "user_supplied_CFD")})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
