"""Explicit chronological data contract; no replacement by legacy snapshots."""
import json
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Sequence:
    def __init__(self, manifest_path, expected_count=200):
        self.path = Path(manifest_path).resolve()
        self.manifest = read_json(self.path)
        self.records = self.manifest["snapshots"]
        if len(self.records) != expected_count:
            raise ValueError(f"Expected {expected_count} snapshots, found {len(self.records)}; no relabeling is allowed.")
        if [r["id"] for r in self.records] != list(range(1, expected_count + 1)):
            raise ValueError("Snapshot IDs must be chronological 1..200.")
        times = np.array([r["time_s"] for r in self.records], dtype=float)
        if not np.all(np.isfinite(times)) or not np.allclose(np.diff(times), .005, rtol=0, atol=1e-9):
            raise ValueError("Paper requires physical snapshot spacing 0.005 s.")
        if self.manifest.get("units") != {"coordinates": "m", "time": "s", "velocity": "m/s", "pressure": "Pa_absolute"}:
            raise ValueError("Manifest units must explicitly specify SI and absolute pressure.")
        paths = [self.path.parent / r["file"] for r in self.records]
        if len(set(p.resolve() for p in paths)) != expected_count:
            raise ValueError("Each snapshot must have a distinct file.")
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing snapshot: {missing[0]}")
        self._cache = {}

    def load(self, snapshot_id):
        if snapshot_id in self._cache:
            return self._cache[snapshot_id]
        record = self.records[snapshot_id - 1]
        with np.load(self.path.parent / record["file"], allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        n = len(data["xyz"])
        shapes = {"xyz": (n, 3), "flow": (n, 4), "alpha_liquid": (n, 1),
                  "alpha_gradient": (n, 3), "volume_weights": (n,)}
        for name, shape in shapes.items():
            if name not in data or data[name].shape != shape or not np.isfinite(data[name]).all():
                raise ValueError(f"Snapshot {snapshot_id}: invalid {name}, expected finite {shape}.")
        if n == 0 or np.any(data["alpha_liquid"] < 0) or np.any(data["alpha_liquid"] > 1):
            raise ValueError("Empty mesh or alpha_L outside [0,1].")
        weights = data["volume_weights"]
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Volume quadrature weights must be nonnegative with positive sum.")
        data["coords"] = np.column_stack((data["xyz"], np.full(n, record["time_s"])))
        # Bound memory; do not load the complete 200-state sequence into RAM.
        if len(self._cache) >= 2:
            self._cache.pop(next(iter(self._cache)))
        self._cache[snapshot_id] = data
        return data

    def fitting_alpha_gradient(self, snapshot_id, fitting_ids):
        """Spatial CFD gradient plus time difference using ONLY fitting states."""
        pos = fitting_ids.index(snapshot_id)
        left = fitting_ids[max(0, pos - 1)]
        right = fitting_ids[min(len(fitting_ids) - 1, pos + 1)]
        if left == right:
            raise ValueError("At least two fitting snapshots required for temporal derivatives")
        d = self.load(snapshot_id)
        dl, dr = self.load(left), self.load(right)
        if not np.array_equal(d["xyz"], dl["xyz"]) or not np.array_equal(d["xyz"], dr["xyz"]):
            raise ValueError("CFD phase temporal differences require a fixed mesh with stable point ordering")
        dt = self.records[right - 1]["time_s"] - self.records[left - 1]["time_s"]
        return np.column_stack((d["alpha_gradient"], (dr["alpha_liquid"] - dl["alpha_liquid"]) / dt))


def split_ids(mode):
    if mode == "selection":
        return list(range(1, 129)), list(range(129, 161))
    if mode == "final":
        return list(range(1, 161)), list(range(161, 201))
    raise ValueError("mode must be selection or final")


def fit_normalization(sequence, ids):
    # Stable merge of population moments, using fitting data only.
    count, mean, m2 = 0, np.zeros(8), np.zeros(8)
    for sid in ids:
        d = sequence.load(sid)
        a = np.column_stack((d["coords"], d["flow"])).astype(np.float64)
        n, batch_mean = len(a), a.mean(0)
        delta = batch_mean - mean
        m2 += ((a - batch_mean) ** 2).sum(0) + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    scale = np.sqrt(m2 / count)
    scale = np.where(scale > 1e-12, scale, 1.)
    return {"coord_mean": mean[:4].tolist(), "coord_scale": scale[:4].tolist(),
            "flow_mean": mean[4:].tolist(), "flow_scale": scale[4:].tolist(),
            "fitted_snapshot_ids": list(ids), "flow_order": ["p", "u", "v", "w"]}


def sample_indices(alpha, size, rng, cav_fraction=.2, threshold=.9):
    # Figure 2: rho*N from cavitating region, (1-rho)*N from FULL domain.
    cav = np.flatnonzero(alpha.reshape(-1) < threshold)
    count = int(round(size * cav_fraction))
    if count and not len(cav):
        raise ValueError("Chosen snapshot has no cavitating points; fixed enrichment is impossible.")
    indices = np.concatenate((rng.choice(cav, count, replace=True),
                              rng.integers(0, len(alpha), size - count)))
    rng.shuffle(indices)
    return indices
