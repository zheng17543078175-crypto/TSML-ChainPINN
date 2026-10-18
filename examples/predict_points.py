"""Predict dimensional fields at saved (N,4) coordinates, with no VTK dependency.

python examples/predict_points.py --checkpoint runs/c1/stage3.pt \
    --coordinates query_xyzt.npy --output prediction.npz
"""
import argparse
from pathlib import Path
import numpy as np
from tsml_chainpinn.inference import Surrogate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--coordinates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    coords = np.load(args.coordinates, allow_pickle=False)
    result = Surrogate(args.checkpoint, args.device).predict(coords)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    print(f"Saved {len(coords)} predictions to {output}")


if __name__ == "__main__":
    main()
