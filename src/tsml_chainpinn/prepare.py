"""Optional VTK adapter. No spatial interpolation or uniform-grid resampling."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import read_json, Sequence


def prepare(source_manifest, output):
    import pyvista as pv
    import vtk
    source_path = Path(source_manifest).resolve()
    spec = read_json(source_path)
    records = spec["snapshots"]
    if len(records) != 200 or [r["id"] for r in records] != list(range(1, 201)):
        raise ValueError("Provide the actual chronological 200-state sequence")
    if not np.allclose(np.diff([r["time_s"] for r in records]), .005, rtol=0, atol=1e-9):
        raise ValueError("Expected 0.005 s spacing; do not fabricate times for old exports")
    if spec["phase_kind"] not in ("liquid", "vapor"):
        raise ValueError("phase_kind must be liquid or vapor; field names are not sufficient evidence")
    for k in ("coordinate_scale_to_m", "velocity_scale_to_m_s", "pressure_scale_to_pa"):
        if not np.isfinite(spec[k]) or spec[k] <= 0:
            raise ValueError(f"Invalid {k}")
    if not np.isfinite(spec["pressure_offset_to_absolute_pa"]):
        raise ValueError("Explicit finite pressure offset required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    exported, first_xyz = [], None
    for record in records:
        source = source_path.parent / record["file"]
        if source.suffix.lower() == ".vtkhdf":
            reader = vtk.vtkHDFReader()
            reader.SetFileName(str(source))
            reader.Update()
            mesh = pv.wrap(reader.GetOutputDataObject(0))
        else:
            mesh = pv.read(source)
        # Explicit block path, e.g. ["s-fluent"]. Never guess a similarly named block.
        for key in spec.get("volume_block_path", []):
            mesh = mesh[key]
        if isinstance(mesh, pv.MultiBlock):
            raise ValueError("Specify volume_block_path down to one volume dataset")
        if mesh.cell_data:
            mesh = mesh.cell_data_to_point_data(pass_cell_data=True)
        xyz = np.asarray(mesh.points, dtype=np.float64) * spec["coordinate_scale_to_m"]
        if first_xyz is None:
            first_xyz = xyz.copy()
        elif not np.array_equal(first_xyz, xyz):
            raise ValueError("The minimal adapter requires a fixed mesh and stable point ordering")
        mesh.points = xyz
        alpha = np.asarray(mesh.point_data[spec["phase_field"]], dtype=np.float64).reshape(-1)
        if spec["phase_kind"] == "vapor":
            alpha = 1 - alpha
        if not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
            raise ValueError("Invalid phase fractions; no automatic clipping")
        mesh.point_data["_tsml_alpha_liquid"] = alpha
        gradient = np.asarray(mesh.compute_derivative(scalars="_tsml_alpha_liquid",
                                                       gradient="_tsml_gradient").point_data["_tsml_gradient"])
        pressure = np.asarray(mesh.point_data[spec["pressure_field"]], dtype=float).reshape(-1)
        pressure = pressure * spec["pressure_scale_to_pa"] + spec["pressure_offset_to_absolute_pa"]
        velocity = np.asarray(mesh.point_data[spec["velocity_field"]], dtype=float) * spec["velocity_scale_to_m_s"]
        volumes = np.asarray(mesh.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"])
        if np.any(volumes < 0) or volumes.sum() <= 0:
            raise ValueError("Expected positive-volume 3D cells")
        # Lump each cell's volume equally to its vertices; preserves total volume.
        weights = np.zeros(len(xyz))
        for i, volume in enumerate(volumes):
            ids = np.asarray(mesh.get_cell(i).point_ids)
            weights[ids] += volume / len(ids)
        name = f"snapshot_{record['id']:03d}.npz"
        np.savez_compressed(output / name, xyz=xyz, flow=np.column_stack((pressure, velocity)),
                            alpha_liquid=alpha[:, None], alpha_gradient=gradient, volume_weights=weights)
        exported.append({"id": record["id"], "time_s": record["time_s"], "file": name,
                         "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                         "source_file": source.name})
    manifest = {"units": {"coordinates": "m", "time": "s", "velocity": "m/s", "pressure": "Pa_absolute"},
                "condition": spec["condition"], "snapshots": exported,
                "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "phase_spatial_derivative": "VTK nodal gradient on original fixed mesh",
                "volume_quadrature": "cell-volume lumping to original vertices"}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Sequence(path)
    return path
