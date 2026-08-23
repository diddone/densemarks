"""DenseMarks training model: embedder -> predictor -> reg/segm heads + losses.

Pipeline: RGB image -> Embedder (DINOv3+DPT) -> Predictor (raw UVW logits) ->
RegHead cube lookup (UVW -> 8-dim descriptor) -> {contrastive correspondence,
segmentation, landmark} losses.

Input is a `data.Batch`: `batch.frames` is flat, one `Frame` per *image*, but
the frames are interleaved source/target pairs of the same head
(`batch.sources` / `batch.targets` split them). Row `k` of `corrs_yx` in a
pair's two frames is the same physical point on the head tracked across the
two frames; that row-correspondence is the entire correspondence-loss
training signal.

This is the training path; `dense_marks_model.DenseMarksModel` is the
inference-only counterpart that consumes the checkpoints written by train.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from densemarks import losses
from densemarks.config import Config
from densemarks.data import Batch
from densemarks.embedder import Embedder
from densemarks.reg_head import RegHead

# The cube is parametrised by 3 coordinates
UVW_CHANNELS = 3


class Predictor(nn.Module):
    """ConvTranspose upsample of DPT features to raw UVW (pre-tanh; see RegHead.to_uvw)."""

    def __init__(self, cfg: Config, in_channels: int):
        super().__init__()
        k = cfg.predictor_deconv_kernel
        # Plain PyTorch default init (not densepose initialize_module_params).
        # `embed_lowres` is a legacy name, kept because it's a frozen checkpoint key
        self.embed_lowres = nn.ConvTranspose2d(
            in_channels,  # Embedder.out_channels (256, the DPT neck's fusion width)
            UVW_CHANNELS,
            kernel_size=k,
            stride=2,
            padding=int(k / 2 - 1),
        )

    def forward(self, features: Tensor, output_size: tuple[int, int]) -> Tensor:
        # features: [B,256,H/2,W/2] (256 = DPT neck); deconv upsamples 2x, then
        # bilinear resize corrects rounding to hit the exact input resolution.
        embed_lowres = self.embed_lowres(features)
        return F.interpolate(
            embed_lowres, size=output_size, mode="bilinear", align_corners=False
        )


class OutputHead(nn.Module):
    """Learned UVW cube (`RegHead`) + segmentation head.

    `embed_to_segm` predicts from the cube's 8-dim descriptor, not raw
    image features, tying segmentation classes to canonical cube regions.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.reg_head = RegHead(cfg)
        self.embed_to_segm = nn.Conv2d(cfg.embed_size, cfg.n_segm_channels, 1)

    def forward(self, raw_uvw: Tensor) -> tuple[Tensor, Tensor]:
        desc = self.reg_head(raw_uvw)  # [B, D, H, W]; D = embed_size (8)
        segm = self.embed_to_segm(desc)  # [B, 22, H, W]; 22 = face-parsing classes
        return desc, segm


class LossHead(nn.Module):
    """Runs `OutputHead` and computes the correspondence/segm/landmark losses."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.output_head = OutputHead(cfg)
        self.logits_scale = nn.Parameter(torch.ones([]) * cfg.logits_scale_init)
        # Checkpoint-compatibility ballast; no loss reads this and it never
        # gets a gradient. Exists only because it's in the released
        # model.safetensors and strict=True loading requires it. Do not delete.
        self.eyes_logits_scale = nn.Parameter(torch.ones([]) * cfg.logits_scale_init)
        self.corr_loss_weight = cfg.corr_loss_weight
        self.segm_loss_weight = cfg.segm_loss_weight
        self.lmks_l1_loss_weight = cfg.lmks_l1_loss_weight

    def forward(self, raw_uvw: Tensor, batch: Batch) -> dict[str, Tensor]:
        desc, segm = self.output_head(raw_uvw)
        B, _, H, W = desc.shape
        size = (H, W)
        loss_dict: dict[str, Tensor] = {}

        # Correspondence loss over interleaved source/target pairs.
        # collate_fn resamples every frame's corrs_yx (with replacement) up to
        # the batch-wide max N, so reading N off the first frame is safe here.
        n = batch.frames[0].corrs_yx.size(0)
        h1 = torch.cat([f.corrs_yx[:, 0] for f in batch.sources], dim=0)
        w1 = torch.cat([f.corrs_yx[:, 1] for f in batch.sources], dim=0)
        h2 = torch.cat([f.corrs_yx[:, 0] for f in batch.targets], dim=0)
        w2 = torch.cat([f.corrs_yx[:, 1] for f in batch.targets], dim=0)
        # b1/b2: which batch (image) index each of the P*N points belongs to.
        b1 = torch.arange(0, B, 2, device=desc.device).repeat_interleave(n)
        b2 = torch.arange(1, B, 2, device=desc.device).repeat_interleave(n)
        p = B // 2
        # gathers one descriptor per point -> [P*N, D], reshaped to [P, N, D] (P pairs, N points/pair).
        f1 = desc[b1, :, h1, w1].reshape(p, n, -1)
        f2 = desc[b2, :, h2, w2].reshape(p, n, -1)
        # batch.close_mask [P,N,N]: True where two rows land on the identical
        # pixel in either frame, so corr_loss excludes them from the negatives.
        loss_dict["loss_corr"] = losses.corr_loss(
            f1, f2, batch.close_mask, self.logits_scale, self.corr_loss_weight
        )

        # Optional: not every image has a face-parsing mask. `gt` must
        # enumerate the frames in the same order as batch.seg_frame_inds,
        # since that's what indexes into segm.
        inds = batch.seg_frame_inds
        if inds is not None and len(inds):
            gt = torch.cat(
                [
                    F.interpolate(
                        f.seg_mask[None, None].float(), size=size, mode="nearest"
                    ).squeeze(1)
                    for f in batch.frames
                    if f.seg_mask is not None
                ]
            ).long()
            loss_dict["loss_segm"] = losses.segm_loss(
                segm[inds], gt, self.segm_loss_weight
            )

        # Optional: not every frame has MediaPipe landmarks. idx/hs/ws/l_idx
        # and the three comprehensions below must all stay in lockstep or
        # supervision silently misaligns. This is the only loss that anchors
        # the cube to a fixed, reproducible canonical frame
        # (assets/fixed_positions_of_landmarks_in_cube.npy) -- without it the cube's
        # parametrisation is otherwise free to drift.
        idx = batch.lmks_frame_inds
        if idx is not None:
            hs = torch.cat(
                [f.fixed_lmks_yx[:, 0] for f in batch.frames if f.fixed_lmks_yx is not None]
            )
            ws = torch.cat(
                [f.fixed_lmks_yx[:, 1] for f in batch.frames if f.fixed_lmks_yx is not None]
            )
            l_idx = torch.cat(
                [f.fixed_l_idx for f in batch.frames if f.fixed_l_idx is not None]
            )
            reg_head = self.output_head.reg_head
            # raw_uvw (pre-tanh) at landmark pixels -> squashed UVW, compared
            # against fixed canonical targets looked up by landmark index.
            pred = reg_head.to_uvw(raw_uvw[idx, :, hs, ws])
            target = reg_head.lmks_target(l_idx)
            loss_dict["loss_lmks"] = losses.lmks_loss(
                pred, target, self.lmks_l1_loss_weight
            )

        return loss_dict


class DenseMarksTrainModel(nn.Module):
    """Full training model: Embedder -> Predictor -> LossHead.

    Frozen state-dict key paths that must not be renamed/deleted (breaks
    strict=True loading of the released checkpoint): embedder.backbone.*,
    embedder.dpt_neck.*, predictor.embed_lowres.*,
    loss_head.output_head.reg_head.{_E,fixed_vertices_uv,gauss_kernel},
    loss_head.output_head.embed_to_segm.*, loss_head.logits_scale,
    loss_head.eyes_logits_scale.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.embedder = Embedder(cfg)
        self.predictor = Predictor(cfg, in_channels=self.embedder.out_channels)
        self.loss_head = LossHead(cfg)

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        images = batch.images
        _, _, h, w = images.shape
        raw_uvw = self.predictor(self.embedder(images), output_size=(h, w))
        return self.loss_head(raw_uvw, batch)

    def predict_uvw(self, images: Tensor) -> Tensor:
        """Inference. [B,3,H,W] RGB -> [B,3,H,W] UVW in [0,1].

        Must stay numerically identical to dense_marks_model.py's inference
        path, which drops the `_E` cube entirely: to_uvw() never reads `_E`,
        it's a pure tanh squash of the raw predictor output, so the cube
        (used only for the contrastive/segm losses at train time) can be
        dropped for inference without changing this output.
        """
        raw = self.predictor(self.embedder(images), output_size=images.shape[-2:])
        return self.loss_head.output_head.reg_head.to_uvw(raw)
