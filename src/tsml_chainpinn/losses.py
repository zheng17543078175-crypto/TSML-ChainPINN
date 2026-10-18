"""Equations (15)-(20); alpha always means liquid volume fraction."""
import torch
from torch.nn import functional as F


def hybrid_loss(logits, alpha, threshold=0.9, gamma=2.0, weights=(1., .5, 2.)):
    logits, alpha = logits.reshape(-1), alpha.reshape(-1)
    mask = alpha < threshold
    target = mask.to(logits.dtype)
    cav_logits = -logits
    probability = cav_logits.sigmoid()
    pt = target * probability + (1 - target) * (1 - probability)
    focal = ((1 - pt).pow(gamma) * F.binary_cross_entropy_with_logits(
        cav_logits, target, reduction="none")).mean()
    pred = logits.sigmoid()
    # Mask sums avoid NaN for empty cavitating/noncavitating subsets.
    reg = ((pred - alpha).abs() * mask).sum() / mask.sum().clamp_min(1)
    fp = (F.relu(threshold - pred) * ~mask).sum() / (~mask).sum().clamp_min(1)
    return {"total": weights[0] * focal + weights[1] * reg + weights[2] * fp,
            "focal": focal, "reg": reg, "fp": fp}


def flow_loss(pred, truth):
    # Standardized outputs make this identical to dimensional channel-scaled L1.
    # Eq. (22) averages the L1 vector norm over N (not over 4*N).
    return (pred - truth).abs().sum(dim=1).mean()
