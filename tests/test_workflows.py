"""Public workflow checks, including interrupted training continuation."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from test_core import configuration, make_sequence
from tsml_chainpinn.engine import train, evaluate
from tsml_chainpinn.inference import Surrogate
from tsml_chainpinn.experiments import aggregate, METRICS
from tsml_chainpinn.demo import make_demo
from tsml_chainpinn.diagnostics import inspect_data


def test_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    import tsml_chainpinn.engine as engine
    cfg = configuration()
    cfg["training"]["epochs"] = [2, 1, 2]
    manifest = make_sequence(tmp_path/"data")
    full = train(cfg, manifest, tmp_path/"full", seed=23)
    save = engine.atomic_save

    def interrupt_after_first_epoch(state, path):
        save(state, path)
        if Path(path).name == "latest.pt":
            raise KeyboardInterrupt("simulated interruption after durable checkpoint")

    monkeypatch.setattr(engine, "atomic_save", interrupt_after_first_epoch)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, manifest, tmp_path/"interrupted", seed=23)
    monkeypatch.setattr(engine, "atomic_save", save)
    resumed = train(cfg, manifest, tmp_path/"interrupted", seed=23, resume=True)
    f = torch.load(full, weights_only=True)
    r = torch.load(resumed, weights_only=True)
    assert all(torch.equal(f["model"][k], r["model"][k]) for k in f["model"])
    assert (tmp_path/"full/history.csv").read_bytes() == (tmp_path/"interrupted/history.csv").read_bytes()
    changed = copy.deepcopy(cfg)
    changed["training"]["learning_rates"][0] *= 2
    with pytest.raises(ValueError, match="Resume requires identical"):
        train(changed, manifest, tmp_path/"interrupted", seed=23, resume=True)


def test_python_inference_matches_saved_evaluation(tmp_path):
    manifest = make_sequence(tmp_path/"data")
    cp = train(configuration(), manifest, tmp_path/"run", mode="selection")
    evaluate(cp, manifest, tmp_path/"evaluation", batch_size=2, save_predictions=True)
    api = Surrogate(cp)
    with np.load(tmp_path/"evaluation/prediction_129.npz") as saved:
        coordinates = np.column_stack((saved["xyz"], np.full(4, saved["time_s"])))
        result = api.predict(coordinates, batch_size=2)
        assert np.array_equal(result["flow"], saved["flow"])
        assert np.array_equal(result["alpha_liquid"], saved["alpha_liquid"])
        assert np.allclose(result["alpha_liquid"] + result["alpha_vapor"], 1.)
    with pytest.raises(ValueError, match="finite"):
        api.predict(np.zeros((5, 3)))


def test_aggregate_run_statistics_and_mismatch_rejection(tmp_path):
    paths = []
    for i in range(3):
        data = dict(seed=i, mode="final", snapshots=40, manifest_sha256="same", config_sha256="same",
                    data_kind="SYNTHETIC_TEST", **{k: float(i+1) for k in METRICS})
        path = tmp_path/f"{i}.json"
        path.write_text(json.dumps(data))
        paths.append(path)
    result = aggregate(paths, tmp_path/"summary")
    assert result["metrics"][0]["mean"] == 2.
    assert result["metrics"][0]["sample_std"] == 1.
    with pytest.raises(ValueError, match="Duplicate seeds"):
        aggregate([paths[0], paths[0]], tmp_path/"invalid")
    data["manifest_sha256"] = "different"
    paths[2].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different manifest"):
        aggregate(paths, tmp_path/"mismatch")


def test_demo_data_contract_hashes_and_separate_configuration(tmp_path):
    manifest, config = make_demo(tmp_path/"synthetic", points_per_axis=3)
    report = inspect_data(manifest, verify_hashes=True)
    assert report["snapshots"] == 200 and report["points_min"] == 27
    assert report["data_kind"] == "SYNTHETIC_WORKFLOW_DEMO_NOT_CFD"
    cfg = json.loads(config.read_text())
    assert cfg["model"]["width"] == 16
    official = json.loads((Path(__file__).parents[1]/"configs/paper.json").read_text())
    assert official["model"]["width"] == 512 and official["training"]["batch_size"] == 4096
    with (tmp_path/"synthetic/snapshot_001.npz").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        inspect_data(manifest, verify_hashes=True)


def test_packaged_defaults_match_paper_config():
    from importlib.resources import files
    packaged = json.loads(files("tsml_chainpinn").joinpath("defaults/paper.json").read_text())
    repo = json.loads((Path(__file__).parents[1]/"configs/paper.json").read_text())
    assert packaged == repo
