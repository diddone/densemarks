"""Learned UVW cube with isotropic 3D Gaussian smoothing (uvw_gauss).

Every pixel is mapped to a position in a canonical cube [0,1]^3; the same
anatomical point should land on the same position across images.

The UVW output comes from `to_uvw` alone and never touches `_E`. `_E` is a
side channel: a learned voxel grid mapping cube position -> descriptor, so
the contrastive loss has something to compare between two pixels. Hence
`dense_marks_model.py` can drop it and still reproduce the released output.

State-dict names here are frozen (`strict=True` load of the checkpoint).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from densemarks.config import Config


class RegHead(nn.Module):
    """The learned cube. Owns `_E`, `gauss_kernel`, `fixed_vertices_uv`."""

    def __init__(self, cfg: Config):
        super().__init__()
        uvw_path = Path(cfg.fixed_lmks_uvw_file)
        if not uvw_path.is_file():
            raise FileNotFoundError(
                f"Missing landmark UVW file {cfg.fixed_lmks_uvw_file!r}. "
                "Place assets/fixed_positions_of_landmarks_in_cube.npy as described in the README."
            )

        # Live and load-bearing: act() is on the inference path (see to_uvw).
        self.act_tanh_scale = cfg.act_tanh_scale
        self.embed_size = cfg.embed_size
        self.gauss_padding = cfg.gauss_kernel_size // 2

        # One descriptor per voxel of [0,1]^3. Trained by the contrastive
        # loss only -- UVW prediction never reads it.
        self._E = nn.Parameter(
            torch.empty(cfg.embed_size, cfg.e_size, cfg.e_size, cfg.e_size)
        )
        nn.init.normal_(self._E)

        # add smoothing with gaussian kernel
        ax = torch.arange(cfg.gauss_kernel_size) - (cfg.gauss_kernel_size - 1) / 2.0
        zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * cfg.gauss_sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None].repeat(cfg.embed_size, 1, 1, 1, 1)
        self.register_buffer("gauss_kernel", kernel)

        # [L,3] canonical UVW per fixed landmark, from a FLAME fit.
        # Used as regularisation term for better interpretabiity of the canonical space
        # Holds UVW despite the "_uv" name; frozen for checkpoint compat.
        self.register_buffer(
            "fixed_vertices_uv",
            torch.from_numpy(np.load(uvw_path).astype(np.float32)),
        )

    def act(self, x: Tensor) -> Tensor:
        """Squash to [-1, 1]. scale=3 lets the network reach the cube's
        extremes (UVW near 0/1) without huge raw activations."""
        return torch.tanh(self.act_tanh_scale * x)

    def get_E(self) -> Tensor:
        """Gaussian-smoothed cube [1, D, e, e, e].

        Recomputed every call, so the smoothing sits in the graph and
        gradients reach `_E` through it. Depthwise: each channel separately.
        """
        return F.conv3d(
            self._E[None],
            self.gauss_kernel,
            padding=self.gauss_padding,
            groups=self.embed_size,
        )

    def forward(self, raw_uvw: Tensor) -> Tensor:
        """raw_uvw: [B,3,H,W] pre-activation -> [B,D,H,W] descriptors.

        Samples the cube at each pixel's predicted position. act() already
        lands in [-1,1], grid_sample's own coordinate convention. The grid is
        a degenerate depth-1 volume of H*W points, hence the squeeze(2). Its
        last dim is (x,y,z) -- reversed w.r.t. the cube's (D,H,W) axes.
        align_corners=False matches the checkpoint
        """
        grid = self.act(raw_uvw).permute(0, 2, 3, 1).unsqueeze(1)  # [B,1,H,W,3]
        E = self.get_E().expand(raw_uvw.shape[0], -1, -1, -1, -1)
        return F.grid_sample(E, grid, mode="bilinear", align_corners=False).squeeze(2)

    def to_uvw(self, raw_uvw: Tensor) -> Tensor:
        """raw_uvw: [B,3,H,W] pre-activation -> [B,3,H,W] UVW in [0,1].

        The model's actual output; never touches the cube.
        """
        return self.act(raw_uvw) * 0.5 + 0.5

    def lmks_target(self, fixed_l_idx: Tensor) -> Tensor:
        """Lookup canonical UVW targets: [M] indices -> [M, 3] UVW."""
        return self.fixed_vertices_uv[fixed_l_idx]
