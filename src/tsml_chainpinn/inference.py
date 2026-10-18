"""Python API for dimensional coordinate-to-field prediction."""
import numpy as np
import torch
from .checkpoint import load_checkpoint
from .model import ChainPINN


class Surrogate:
    """Load a completed model and predict at arbitrary SI coordinates.

    Input columns: x, y, z [m], t [s]. Output flow: p [Pa absolute], u,v,w [m/s].
    This API does not imply validity outside the trained geometry/condition.
    """
    def __init__(self, checkpoint, device="cpu"):
        state = load_checkpoint(checkpoint)
        if state["stage"] != 3 or state["epoch"] != state["config"]["training"]["epochs"][2]:
            raise ValueError("Inference requires completed Stage 3")
        self.device = device
        self.normalization = state["normalization"]
        self.config = state["config"]
        self.provenance = state["provenance"]
        self.model = ChainPINN(**self.config["model"]).to(device)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()

    def predict(self, coordinates, batch_size=4096):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 4 or not np.isfinite(coordinates).all():
            raise ValueError("coordinates must be a finite (N,4) SI array")
        if batch_size < 1 or not len(coordinates):
            raise ValueError("Nonempty coordinates and positive batch_size required")
        norm = self.normalization
        flow, alpha = [], []
        with torch.no_grad():
            for begin in range(0, len(coordinates), batch_size):
                x = (coordinates[begin:begin+batch_size] - norm["coord_mean"]) / norm["coord_scale"]
                p = self.model(torch.as_tensor(x, dtype=torch.float32, device=self.device))
                flow.append(p["flow"].cpu().numpy()*norm["flow_scale"] + norm["flow_mean"])
                alpha.append(p["alpha"].cpu().numpy())
        a = np.concatenate(alpha)
        return {"coordinates": coordinates.copy(), "flow": np.concatenate(flow),
                "alpha_liquid": a, "alpha_vapor": 1-a,
                "cavitation_mask": a < self.config["training"]["cavitation_threshold"]}
