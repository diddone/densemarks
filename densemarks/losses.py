"""Training losses for the original DenseMarks configuration.

Three free functions, no classes, no state. Ported from dense_lmks/loss_head.py
(corr / segm) and dense_lmks/heads/reg_heads.py (lmks L1).
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F


def corr_loss(
    f_src: Tensor,
    f_tgt: Tensor,
    close_mask: Tensor,
    logits_scale: Tensor,
    weight: float,
) -> Tensor:
    """Batched CLIP-style contrastive correspondence loss.

    Args:
        f_src, f_tgt: [P, N, D] descriptors at corresponding points.
        close_mask: [P, N, N] bool, True where two correspondences are the same pixel.
        P = number of image pairs; N = points per pair; D = descriptor dim.
        logits_scale: scalar Parameter, the *log* of the temperature.
        weight: float loss weight.
    """
    f_src = F.normalize(f_src, dim=-1)
    f_tgt = F.normalize(f_tgt, dim=-1)

    # [P, N, D] x [P, D, N] -> [P, N, N]
    s = logits_scale.exp() * f_src @ rearrange(f_tgt, "b n d -> b d n")
    t = logits_scale.exp() * f_tgt @ rearrange(f_src, "b n d -> b d n")

    mask = torch.where(close_mask, -torch.inf, 0.0)
    s = s + mask.detach()
    t = t + mask.detach()

    p, n, _ = f_src.shape
    labels = torch.arange(n, device=f_src.device, dtype=torch.long).unsqueeze(0).expand(p, -1)
    return weight * (F.cross_entropy(s, labels) + F.cross_entropy(t, labels)) / 2.0


def segm_loss(logits: Tensor, gt: Tensor, weight: float) -> Tensor:
    """Segmentation cross-entropy.

    Args:
        logits: [K, n_segm_channels, H, W] float; n_segm_channels = segmentation classes (incl. background).
        gt: [K, H, W] int64 class indices.
        weight: float loss weight.
    """
    return weight * F.cross_entropy(logits, gt)


def lmks_loss(pred_uvw: Tensor, target_uvw: Tensor, weight: float) -> Tensor:
    """Landmark UVW L1. pred/target: [M, 3] in [0, 1]. weight=50.0 published."""
    return weight * (pred_uvw - target_uvw).abs().mean()
