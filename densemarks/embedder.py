"""Dense feature extractor: DINOv3 ViT backbone + HuggingFace DPT neck."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation

from densemarks.config import Config


class Embedder(nn.Module):
    """Turns RGB images into dense feature maps for the regression / seg heads.

    Pipeline: ImageNet-normalise → DINOv3 ViT (last 4 blocks) → DPT neck →
    float32 features at half resolution ``[B, 256, H/2, W/2]``.

    Any DINOv3 ViT backbone works: the neck is sized from the backbone's
    ``embed_dim``, and ``out_channels`` (DPT's fusion width, 256) is the same
    either way. Only ``dinov3_vitb16`` matches the released checkpoint.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        repo = Path(cfg.dinov3_repo)
        if not repo.is_dir():
            raise FileNotFoundError(
                f"DINOv3 repo not found at {cfg.dinov3_repo!r}. "
                "Clone it as described in the README (third_party_dinov3)."
            )

        self.backbone = self._load_backbone(cfg)
        for attr in ("embed_dim", "patch_size", "n_blocks"):
            if not hasattr(self.backbone, attr):
                raise ValueError(
                    f"{cfg.dinov3_model!r} does not expose `{attr}`; only DINOv3 "
                    "ViT backbones are supported (not the ConvNeXt variants)."
                )
        embed_dim = self.backbone.embed_dim
        if embed_dim % 64:
            raise ValueError(
                f"{cfg.dinov3_model!r} has embed_dim={embed_dim}, which is not a "
                "multiple of 64; the DPT neck's backbone config cannot be derived."
            )
        self.patch_size = self.backbone.patch_size
        n_blocks = self.backbone.n_blocks
        # DPT expects four multi-scale stages; take the last four ViT blocks.
        self.backbone_out_indices = list(range(n_blocks - 4, n_blocks))
        self.use_bf16 = cfg.embedder_use_bf16

        # Architecture-only DPT neck (no pretrained depth weights). The DINOv2
        # config exists purely to tell DPT the token width to reassemble from,
        # so it is built here rather than downloaded; at embed_dim=768 (ViT-B)
        # the resulting neck is identical to the one built from the
        # `facebook/dinov2-base` config the released checkpoint was trained with.
        # num_attention_heads is unused by the neck but must divide hidden_size
        # or Dinov2Config rejects it.
        backbone_config = Dinov2Config(
            hidden_size=embed_dim,
            num_attention_heads=embed_dim // 64,
            out_features=["stage1", "stage2", "stage3", "stage4"],
            reshape_hidden_states=False,
        )
        dpt_config = DPTConfig(backbone_config=backbone_config)
        self.dpt_neck = DPTForDepthEstimation(dpt_config).neck
        # What this Embedder emits, and what Predictor consumes. DPT's fusion
        # width: 256, and independent of which backbone feeds the neck.
        self.out_channels = dpt_config.fusion_hidden_size

        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    @staticmethod
    def _load_backbone(cfg: Config):
        """Load the local DINOv3 hub model (pretrained weights or random init)."""
        kwargs = {"source": "local"}
        if cfg.dinov3_weights == "random":
            # Escape hatch for smoke tests. Does NOT reproduce the checkpoint.
            kwargs["pretrained"] = False
        elif cfg.dinov3_weights:
            kwargs["weights"] = cfg.dinov3_weights
        else:
            raise ValueError(
                "--dinov3_weights is required: the published run starts from the "
                "LVD-1689M DINOv3 ViT-B/16 checkpoint "
                "(dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth), which we cannot "
                "redistribute. Get it from https://github.com/facebookresearch/dinov3 "
                "and see the README. Pass --dinov3_weights random for a random-init "
                "smoke test."
            )
        return torch.hub.load(cfg.dinov3_repo, cfg.dinov3_model, **kwargs)

    @staticmethod
    def _bf16_enabled(use_bf16: bool) -> bool:
        """Use bf16 only on CUDA Ampere+ (compute capability >= 8)."""
        return (
            use_bf16
            and torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).major >= 8
        )

    def _preprocess(self, images: Tensor) -> Tensor:
        """Scale uint8/float images to [0, 1] then apply ImageNet mean/std."""
        images = images.to(device=self.pixel_mean.device, dtype=torch.float32)
        if images.max() > 1.0:
            images = images / 255.0
        return (images - self.pixel_mean) / self.pixel_std

    def forward(self, images: Tensor) -> Tensor:
        """RGB ``[B, 3, H, W]`` → dense features ``[B, 256, H/2, W/2]`` float32."""
        images = self._preprocess(images)
        use_bf16 = self._bf16_enabled(self.use_bf16)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            feats = self.backbone.get_intermediate_layers(
                images,
                n=self.backbone_out_indices,
                reshape=False,
                return_class_token=True,
                norm=True,
            )
            _, _, height, width = images.shape
            # DPT neck wants the CLS token prepended to each stage's tokens.
            feats = [torch.cat([cls[:, None, :], f], dim=1) for f, cls in feats]
            out = self.dpt_neck(feats, height // self.patch_size, width // self.patch_size)[-1]
        return out.float()
