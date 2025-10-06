"""
Standalone DenseMarks Model

A self-contained model class that loads a trained model from weights and performs
inference to produce UVW coordinates (B x 3 x H x W) from input images.

Dependencies: torch, numpy, transformers
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import Dinov2Config, DPTConfig, DPTForDepthEstimation
from PIL import Image

def read_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"))

class DINOv3DPTEmbedder(nn.Module):
    """
    DINOv3 backbone with DPT head for feature extraction.
    Based on embedders/vit_dinov3.py but simplified for standalone use.
    """

    def __init__(self):
        super().__init__()

        # Environment detection for dinov3 repo path

        DINOV3_REPO_PATH = Path(__file__).parent / "third_party_dinov3" # Path to dinov3 repo

        self.dinov3_repo_path = str(DINOV3_REPO_PATH)
        self.backbone = self._load_dinov3_backbone()

        # Get backbone properties
        self.embedding_dim = self.backbone.embed_dim  # 768
        self.patch_size = self.backbone.patch_size    # 16

        # Create DPT head architecture
        backbone_config = Dinov2Config.from_pretrained(
            "facebook/dinov2-base",
            out_features=["stage1", "stage2", "stage3", "stage4"],
            reshape_hidden_states=False
        )
        config = DPTConfig(backbone_config=backbone_config)
        self.dpt_neck = DPTForDepthEstimation(config).neck

        # Image preprocessing parameters
        self.register_buffer('pixel_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Get output layers (last 4 layers)
        n_blocks = self.backbone.n_blocks
        self.backbone_out_indices = [i for i in range(n_blocks - 4, n_blocks)]

        print(f"Initialized DINOv3+DPT model:")
        print(f"  - Embedding dimension: {self.embedding_dim}")
        print(f"  - Patch size: {self.patch_size}x{self.patch_size}")

    def _load_dinov3_backbone(self):
        """Load DINOv3 backbone using torch.hub.load"""
        # checkpoint_mapping = {
        #     "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        # }

        #backbone_weights = str(Path("/mnt/hdd/checkpoints/") / checkpoint_mapping["dinov3_vitb16"])

        #print(f"Loading DINOv3 model from: {backbone_weights}")
        backbone = torch.hub.load(
            self.dinov3_repo_path,
            "dinov3_vitb16",
            source="local",
            pretrained=False,
            #weights=backbone_weights
        )
        print(f"✓ Successfully loaded DINOv3 backbone")

        return backbone

    @property
    def device(self):
        return next(self.parameters()).device

    def process_images(self, images):
        """Preprocess images for DINOv3"""
        images = images.to(self.device)

        # Convert BGR to RGB
        #images = images[:, [2, 1, 0], :, :]

        # Normalize to [0, 1] if needed
        if images.max() > 1.0:
            images = images / 255.0

        # Apply normalization
        images = (images - self.pixel_mean) / self.pixel_std
        return images

    @staticmethod
    def is_bfloat_16_available():
        """Check if bfloat16 is available"""
        return torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8

    def forward(self, images):
        """
        Forward pass through DINOv3 backbone and DPT head

        Args:
            images: Input images tensor (B, 3, H, W)

        Returns:
            tensor of size [B, 256, H_i, W_i] where H_i, W_i are the input image dimensions
        """
        images = self.process_images(images)

        # Use bfloat16 if available
        use_bfloat16 = self.is_bfloat_16_available()

        # Forward through DINOv3 backbone
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
            backbone_features = self.backbone.get_intermediate_layers(
                images,
                n=self.backbone_out_indices,
                reshape=False,
                return_class_token=True,
                norm=True,  # important, apply layernorm
            )

            B, _, height, width = images.shape
            patch_size = self.patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size

            # Concatenate class token with features
            backbone_features_cls = [torch.cat([cls_token[:, None, :], f], dim=1)
                                    for f, cls_token in backbone_features]

            # Apply DPT neck
            output = self.dpt_neck(backbone_features_cls, patch_height, patch_width)[-1]

        return output.float()


class Predictor(nn.Module):
    """
    Predictor that interpolates backbone features to the original image size.
    Based on predictors.py but simplified for standalone use.
    """

    def __init__(self, input_channels=256, output_dim=3, kernel_size=4):
        super().__init__()

        # ConvTranspose2d layer: input_channels -> output_dim
        self.embed_lowres = nn.ConvTranspose2d(
            input_channels, output_dim, kernel_size,
            stride=2, padding=int(kernel_size / 2 - 1)
        )

        # Initialize parameters
        nn.init.kaiming_normal_(self.embed_lowres.weight, mode='fan_out', nonlinearity='relu')
        if self.embed_lowres.bias is not None:
            nn.init.constant_(self.embed_lowres.bias, 0)

    def forward(self, head_outputs, output_size=None):
        """
        Forward step on backbone outputs

        Args:
            head_outputs: Backbone outputs, tensor of shape [N, D, H, W]
            output_size: Output size for interpolation, usually the same as input size

        Returns:
            embeddings: tensor of shape [N, output_dim, output_size[0], output_size[1]]
        """
        assert output_size is not None, "output_size is required to interpolate to the same input size"

        # Apply ConvTranspose2d
        embed_lowres = self.embed_lowres(head_outputs)

        # Interpolate to target size
        embeddings = F.interpolate(
            embed_lowres,
            size=output_size,
            mode='bilinear',
            align_corners=False
        )

        return embeddings


class DenseMarksModel(nn.Module):
    """
    Standalone DenseMarks model for UVW coordinate prediction.

    This model loads a trained checkpoint and performs inference to produce
    UVW coordinates (B x 3 x H x W) from input images.
    """

    def __init__(self, weights_path):
        """
        Initialize the model and load weights from checkpoint.

        Args:
            weights_path: Path to the model checkpoint file
        """
        super().__init__()

        # Initialize components
        self.embedder = DINOv3DPTEmbedder()
        self.predictor = Predictor(
            input_channels=256,  # DPT neck output
            output_dim=3,        # UVW coordinates
            kernel_size=4
        )

        # UV activation parameters
        self.uv_act_tanh_scale = 3.0

        # Load weights
        self.load_weights(weights_path)

        # Set to eval mode
        self.eval()

        print(f"✓ DenseMarksModel initialized successfully")

    def load_weights(self, weights_path):
        """
        Load model weights from checkpoint.

        Args:
            weights_path: Path to checkpoint file
        """
        print(f"Loading weights from: {weights_path}")

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')
        model_state_dict = checkpoint['model']

        # Create new state dict with proper key mapping
        new_state_dict = {}

        for key, value in model_state_dict.items():
            if key.startswith('embedder.backbone.'):
                # Map embedder.backbone.* -> backbone.*
                new_key = key.replace('embedder.backbone.', 'embedder.backbone.')
                new_state_dict[new_key] = value
            elif key.startswith('embedder.dpt_neck.'):
                # Map embedder.dpt_neck.* -> dpt_neck.*
                new_key = key.replace('embedder.dpt_neck.', 'embedder.dpt_neck.')
                new_state_dict[new_key] = value
            elif key.startswith('predictor.embed_lowres.'):
                # Map predictor.embed_lowres.* -> embed_lowres.*
                new_key = key.replace('predictor.embed_lowres.', 'predictor.embed_lowres.')
                new_state_dict[new_key] = value

        # Load the mapped weights with strict=False to handle shape mismatches
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print("✓ Model weights loaded successfully")

    def act(self, x):
        """
        Activation function for UVW conversion.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        return torch.tanh(self.uv_act_tanh_scale * x)

    def forward(self, images):
        """
        Forward pass to predict UVW coordinates.

        Args:
            images: Input images as numpy array or torch tensor
                   Shape: (B, H, W, 3) or (B, 3, H, W)

        Returns:
            uvw: UVW coordinates tensor (B, 3, H, W) with values in [0, 1]
        """
        # Convert to torch tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()

        # Ensure proper shape (B, 3, H, W)
        if images.dim() == 4:
            if images.shape[1] == 3:  # Already (B, 3, H, W)
                pass
            elif images.shape[-1] == 3:  # (B, H, W, 3) -> (B, 3, H, W)
                images = images.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unexpected image shape: {images.shape}")
        elif images.dim() == 3:  # Single image (H, W, 3) or (3, H, W)
            if images.shape[0] == 3:  # (3, H, W)
                images = images.unsqueeze(0)  # -> (1, 3, H, W)
            elif images.shape[-1] == 3:  # (H, W, 3)
                images = images.permute(2, 0, 1).unsqueeze(0)  # -> (1, 3, H, W)
            else:
                raise ValueError(f"Unexpected image shape: {images.shape}")
        else:
            raise ValueError(f"Images must be 3D or 4D tensor, got {images.dim()}D")

        # Get original image size for interpolation
        B, C, H, W = images.shape
        original_size = (H, W)

        # Forward pass
        with torch.no_grad():
            # Extract features using DINOv3+DPT
            backbone_outputs = self.embedder(images)  # (B, 256, H', W')

            # Apply predictor to get UVW coordinates
            predictor_outputs = self.predictor(backbone_outputs, output_size=original_size)  # (B, 3, H, W)

            # Apply activation function and convert to [0, 1] range
            uvw = self.act(predictor_outputs) * 0.5 + 0.5  # [-1, 1] -> [0, 1]

        return uvw

    @property
    def device(self):
        return next(self.parameters()).device


# Example usage
if __name__ == "__main__":
    # Example of how to use the model
    from huggingface_hub import hf_hub_download
    #model = DenseMarksModel(hf_hub_download("diddone/densemarks", "pytorch_model.bin"))
    model = DenseMarksModel(hf_hub_download("diddone/densemarks", "model.safetensors"))
    images = read_image("assets/00000.png")
    print(f"Input shape: {images.shape}")
    print(f"Output UVW shape: {uvw.shape}")
    print(f"UVW range: [{uvw.min():.3f}, {uvw.max():.3f}]")
