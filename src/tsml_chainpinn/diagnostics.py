"""Local environment and prepared-data diagnostics; no external services."""
import hashlib
import importlib.util
import platform
import sys
import torch
from . import __version__
from .data import Sequence


def doctor():
    return {"package": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
            "pytorch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "optional_dependencies": {name: importlib.util.find_spec(name) is not None
                                      for name in ("pyvista", "vtk", "matplotlib", "pytest")}}


def inspect_data(manifest, verify_hashes=False):
    seq = Sequence(manifest)
    counts, fractions = [], []
    for r in seq.records:
        d = seq.load(r["id"])
        counts.append(len(d["xyz"]))
        fractions.append(float((d["alpha_liquid"] < .9).mean()))
        if verify_hashes:
            if "sha256" not in r:
                raise ValueError(f"Snapshot {r['id']} has no expected sha256")
            sha = hashlib.sha256()
            with (seq.path.parent/r["file"]).open("rb") as f:
                for block in iter(lambda: f.read(1024*1024), b""):
                    sha.update(block)
            if sha.hexdigest() != r["sha256"]:
                raise ValueError(f"Snapshot {r['id']} SHA256 mismatch")
    return {"snapshots": len(counts), "points_min": min(counts), "points_max": max(counts),
            "cavitation_point_fraction_min": min(fractions), "cavitation_point_fraction_max": max(fractions),
            "data_kind": seq.manifest.get("data_kind", "user_supplied_CFD"),
            "hashes_verified": verify_hashes,
            "empty_cavitation_snapshot_ids": [r["id"] for r, f in zip(seq.records, fractions) if f == 0]}
