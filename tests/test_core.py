"""Analytic and synthetic tests; these do not reproduce manuscript results."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tsml_chainpinn.config import load_config
from tsml_chainpinn.data import Sequence, split_ids, fit_normalization, sample_indices
from tsml_chainpinn.model import ChainPINN, FourierEmbedding
from tsml_chainpinn.losses import hybrid_loss
from tsml_chainpinn.physics import residuals, zgb_sources
from tsml_chainpinn.engine import objective, train, evaluate

torch.set_num_threads(1)


def configuration():
    c = json.loads((Path(__file__).parents[1] / "configs/paper.json").read_text())
    # Synthetic-test choices, never presented as recovered CFD constants.
    c["physics"].update(rho_liquid=2., rho_vapor=1., mu_liquid=.1, mu_vapor=.05,
                          reference_velocity_m_s=2., reference_length_m=3.)
    c["model"] = dict(width=8, depth=1, bands=2, heads=2, softplus_beta=2.)
    c["training"].update(batch_size=10, microbatch_size=3, steps_per_epoch=1,
                           epochs=[1, 1, 2], warmup_epochs=1)
    return c


def normalization():
    return dict(coord_mean=[0]*4, coord_scale=[2., 3., 4., 5.],
                flow_mean=[0]*4, flow_scale=[1]*4)


def test_fourier_equation_11():
    layer = FourierEmbedding(2).double()
    z = layer(torch.zeros(2, 4, dtype=torch.float64))
    assert z.shape == (2, 16)
    assert torch.equal(z[:, :8], torch.ones_like(z[:, :8]))
    x = torch.zeros(1, 4, dtype=torch.float64)
    x[:, 0] = .25
    assert abs(layer(x)[0, 8].item() - 1) < 1e-12


def test_focal_has_no_class_weight_and_masks():
    logits = torch.zeros(2, 1, requires_grad=True)
    loss = hybrid_loss(logits, torch.tensor([[.2], [1.]]))
    assert loss["focal"].item() == pytest.approx(np.log(2) / 4)
    assert loss["reg"].item() == pytest.approx(.3)
    assert loss["fp"].item() == pytest.approx(.4)
    for alpha in (torch.zeros(2, 1), torch.ones(2, 1)):
        empty = hybrid_loss(logits, alpha)["total"]
        assert torch.isfinite(empty)
        empty.backward()
    assert torch.isfinite(logits.grad).all()


def test_physical_derivatives_and_conservative_momentum():
    c = configuration()["physics"]
    q = torch.tensor([[.2, .3, .4, .1], [.5, .2, .1, .3]], dtype=torch.float64, requires_grad=True)
    norm = normalization()
    physical = q * q.new_tensor(norm["coord_scale"])
    x, y, z, t = physical.split(1, dim=1)
    # rho=.1*x+.2*t+1.3, velocity=(x,0,0), pressure=x^2.
    a = .3 + .1*x + .2*t
    flow = torch.cat((x*x, x, x*0, x*0), 1)
    r = residuals(q, {"flow": flow, "alpha": a}, norm, c, include_vapor=False)
    rho = 1 + a
    # d(rho*x)/dt + d(rho*x*x)/dx + dp/dx - d(2*mu)/dx
    expected = .2*x + .1*x*x + 2*rho*x + 2*x - 2*.05*.1
    assert torch.allclose(r["divergence"], torch.full_like(x, 1.5))
    assert torch.allclose(r["momentum"][:, :1], expected * 3 / 8)
    assert torch.allclose(r["momentum"][:, 1:], torch.zeros_like(r["momentum"][:, 1:]))
    r["momentum"].square().sum().backward()
    assert torch.isfinite(q.grad).all()


def test_stage1_cfd_property_gradients_match_analytic_field():
    c = configuration()["physics"]
    q = torch.rand(3, 4, dtype=torch.float64, requires_grad=True)
    norm = normalization()
    x = q[:, :1] * 2
    a = .2 + .1*x
    pred = {"flow": torch.cat((x*x, x, x*0, x*0), 1), "alpha": a}
    direct = residuals(q, pred, norm, c, include_vapor=False)
    lifted = residuals(q, pred, norm, c, alpha_reference=a.detach(),
                       alpha_reference_gradient=q.new_tensor([[.1, 0, 0, 0]]).expand(3, 4),
                       include_vapor=False)
    assert torch.allclose(direct["momentum"], lifted["momentum"])


def test_vapor_equation_and_source_sign():
    c = configuration()["physics"]
    q = torch.rand(4, 4, dtype=torch.float64, requires_grad=True)
    p = q[:, :1]*0 + c["vapor_pressure_pa"]
    a = .3 + .02*q[:, 3:4]*5
    flow = torch.cat((p, p*0, p*0, p*0), 1)
    r = residuals(q, {"flow": flow, "alpha": a}, normalization(), c, smooth=False)
    assert torch.allclose(r["vapor"], torch.full_like(a, -.02*3/2))
    pressure = torch.tensor([[3168.], [3170.]])
    ev, co = zgb_sources(pressure, torch.full_like(pressure, .5), c, smooth=False)
    assert ev[0] > 0 and ev[1] == 0 and co[0] == 0 and co[1] > 0
    pressure = torch.tensor([[3169.]], requires_grad=True)
    ev, co = zgb_sources(pressure, torch.tensor([[.5]]), c, smooth=True)
    (ev+co).sum().backward()
    assert torch.isfinite(pressure.grad).all()


def test_stages_freeze_exact_modules_and_inference_is_pointwise():
    model = ChainPINN(**configuration()["model"])
    for stage, warm, prefix in [(1, False, "flow."), (2, False, "phase."), (3, True, "phase.")]:
        model.set_stage(stage, warm)
        assert all(p.requires_grad == name.startswith(prefix) for name, p in model.named_parameters())
    model.set_stage(3)
    assert all(p.requires_grad for p in model.parameters())
    model.eval()
    x = torch.rand(9, 4)
    whole = model(x)["flow"]
    chunked = torch.cat([model(chunk)["flow"] for chunk in x.split(2)])
    assert torch.allclose(whole, chunked, atol=2e-6)


@pytest.mark.parametrize("stage,warmup", [(1, False), (2, False), (3, True), (3, False)])
def test_microbatch_gradients_preserve_full_batch_objective(stage, warmup):
    torch.manual_seed(42)
    cfg = configuration()
    a = ChainPINN(**cfg["model"]).double()
    b = copy.deepcopy(a)
    a.set_stage(stage, warmup)
    b.set_stage(stage, warmup)
    x = torch.rand(5, 4, dtype=torch.float64)
    y = torch.rand(5, 4, dtype=torch.float64)
    alpha = torch.tensor([[.1], [.2], [.95], [1.], [.96]], dtype=torch.float64)
    ag = torch.zeros(5, 4, dtype=torch.float64)
    whole = objective(a, x.clone().requires_grad_(True), y, alpha, ag, normalization(), cfg, stage, warmup)
    whole.backward()
    chunk_loss = 0.
    for indices in (slice(0, 2), slice(2, 5)):
        n = len(x[indices])
        loss = objective(b, x[indices].clone().requires_grad_(True), y[indices], alpha[indices],
                         ag[indices], normalization(), cfg, stage, warmup, (2, 3, 5)) * n/5
        loss.backward()
        chunk_loss += loss.item()
    assert chunk_loss == pytest.approx(whole.item(), rel=1e-7, abs=1e-7)
    for p, q in zip(a.parameters(), b.parameters()):
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad, rtol=1e-6, atol=1e-6)


def make_sequence(path):
    path.mkdir()
    snapshots = []
    xyz = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
    for sid in range(1, 201):
        # Deliberately different held-out targets to expose normalization leakage.
        shift = 10000 if sid > 128 else 0
        flow = np.column_stack((np.full(4, 3169.+shift), xyz + sid*.001))
        name = f"{sid:03d}.npz"
        np.savez(path/name, xyz=xyz, flow=flow, alpha_liquid=np.array([[.2], [.3], [.95], [1.]]),
                 alpha_gradient=np.zeros((4, 3)), volume_weights=np.ones(4))
        snapshots.append(dict(id=sid, time_s=(sid-1)*.005, file=name))
    manifest = dict(units=dict(coordinates="m", time="s", velocity="m/s", pressure="Pa_absolute"),
                    snapshots=snapshots)
    target = path / "manifest.json"
    target.write_text(json.dumps(manifest))
    return target


def test_split_normalization_and_sampling(tmp_path):
    manifest = make_sequence(tmp_path/"data")
    sequence = Sequence(manifest)
    fitting, validation = split_ids("selection")
    assert fitting[-1] == 128 and validation == list(range(129, 161))
    assert split_ids("final") == (list(range(1, 161)), list(range(161, 201)))
    norm = fit_normalization(sequence, fitting)
    assert norm["flow_mean"][0] == pytest.approx(3169.)
    assert sequence.fitting_alpha_gradient(128, fitting).shape == (4, 4)
    alpha = np.array([.1, 1., 1., 1., 1.])
    indices = sample_indices(alpha, 100000, np.random.default_rng(2))
    # Enrichment from full domain gives .2 + .8*.2 = .36, not exactly .20.
    assert np.mean(alpha[indices] < .9) == pytest.approx(.36, abs=.005)
    with pytest.raises(ValueError, match="no cavitating"):
        sample_indices(np.ones(5), 10, np.random.default_rng(2))


def test_end_to_end_new_training_only(tmp_path):
    manifest = make_sequence(tmp_path/"data")
    cfg = configuration()
    checkpoint = train(cfg, manifest, tmp_path/"run", mode="selection")
    summary = evaluate(checkpoint, manifest, tmp_path/"eval", batch_size=2, save_predictions=True)
    assert summary["snapshots"] == 32
    assert np.isfinite(summary["alpha_mae"])
    assert len(list((tmp_path/"eval").glob("prediction_*.npz"))) == 32
    with np.load(tmp_path/"eval/prediction_129.npz") as saved:
        assert saved["xyz"].shape == (4, 3)
    with pytest.raises(FileExistsError):
        train(cfg, manifest, tmp_path/"run", mode="selection")


def test_incomplete_config_refuses_to_train(tmp_path):
    cfg = configuration()
    cfg["physics"]["rho_liquid"] = None
    path = tmp_path/"incomplete.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="unresolved"):
        load_config(path)


def test_vtk_adapter_preserves_original_grid_and_units(tmp_path):
    pv = pytest.importorskip("pyvista")
    from tsml_chainpinn.prepare import prepare
    source = tmp_path/"vtk"
    source.mkdir()
    xyz = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
    records = []
    for sid in range(1, 201):
        mesh = pv.UnstructuredGrid(np.array([4, 0, 1, 2, 3]), np.array([10], dtype=np.uint8), xyz)
        time = (sid-1)*.005
        mesh["v"] = xyz
        mesh["p"] = np.full(4, -98000.)
        mesh["a"] = .2 + .1*xyz[:, 0] + .01*time
        name = f"{sid:03d}.vtu"
        mesh.save(source/name)
        records.append(dict(id=sid, time_s=time, file=name))
    spec = dict(condition="SYNTHETIC_TEST", coordinate_scale_to_m=1., velocity_scale_to_m_s=1.,
                pressure_scale_to_pa=1., pressure_offset_to_absolute_pa=101325.,
                volume_block_path=[], velocity_field="v", pressure_field="p", phase_field="a",
                phase_kind="liquid", snapshots=records)
    manifest = source/"source.json"
    manifest.write_text(json.dumps(spec))
    result = Sequence(prepare(manifest, tmp_path/"prepared"))
    d = result.load(128)
    assert np.array_equal(d["xyz"], xyz)
    assert np.all(d["flow"][:, 0] == 3325.)
    assert d["volume_weights"].sum() == pytest.approx(1/6)
    grad = result.fitting_alpha_gradient(128, list(range(1, 129)))
    assert np.allclose(grad, np.tile([.1, 0, 0, .01], (4, 1)), atol=1e-7)
