"""Configuration for training the original DenseMarks model.

Every default in this file is the value used for the released checkpoint.
Running `python train.py --data_dir <...> --dinov3_weights <...>` with no other
flags reproduces it (those two have no usable default; see README).
"""

import math
from dataclasses import dataclass


@dataclass
class Config:
    # ---- data ----------------------------------------------------------
    data_dir: str = ""  # root of tracked CelebV-HQ; see README for layout
    tracks_file_name: str = "pred_tracks.pt"
    image_size: int = 512  # data is not resized; the loader asserts frames match this
    points_min_lim: int = 80  # resample pairs with fewer visible tracked points
    cotracker_conf_threshold: float = 0.6  # drop CoTracker points below this confidence
    pairs_per_batch: int = 8
    num_workers: int = 4

    # ---- embedder ------------------------------------------------------
    dinov3_repo: str = "third_party_dinov3"  # local torch.hub repo path for DINOv3
    # Any DINOv3 ViT variant works (vits16/vitb16/vitl16/...): the DPT neck is
    # sized from the backbone's embed_dim. The released checkpoint is vitb16;
    # anything else means training from scratch. ConvNeXt variants are not supported.
    dinov3_model: str = "dinov3_vitb16"
    dinov3_weights: str = ""  # REQUIRED: path to the LVD-1689M .pth (see README); "random" = smoke test only
    embedder_use_bf16: bool = True  # bf16 autocast inside the embedder only

    # ---- predictor / reg head ------------------------------------------
    predictor_deconv_kernel: int = 4
    embed_size: int = 8  # D: descriptor dim sampled from the cube
    e_size: int = 128  # cube side length; the cube is e_size^3
    gauss_kernel_size: int = 7  # fixed 3D Gaussian smoother on cube lookups
    gauss_sigma: float = 1.5
    act_tanh_scale: float = 3.0  # UVW = tanh(scale * raw); steeper -> sharper cube coords
    n_segm_channels: int = 22  # face-parsing class count (incl. background)

    # ---- losses --------------------------------------------------------
    logits_scale_init: float = math.log(5.0)  # init for learnable CLIP temperature
    corr_loss_weight: float = 1.0
    segm_loss_weight: float = 1.0
    lmks_l1_loss_weight: float = 50.0
    # Row-aligned pair: row i of the UVW file is the target for the landmark
    # whose MediaPipe index is row i of the idx file.
    fixed_lmks_uvw_file: str = "assets/fixed_positions_of_landmarks_in_cube.npy"  # [L,3] canonical UVW targets
    mediapipe_lmks_idx_file: str = "assets/mediapipe_indices_of_fixed_landmarks.npy"  # [L] indices into MediaPipe's 468

    # ---- optimisation --------------------------------------------------
    max_iter: int = 140_000
    base_lr: float = 1e-4
    backbone_lr: float = 1e-5  # DINOv3 backbone (pretrained, smaller LR)
    e_lr: float = 1e-3  # initialised from 0, learning rate for the embedding cube `_E`
    weight_decay: float = 1e-4
    weight_decay_norm: float = 0.0  # WD for 0/1-D params (biases, norms, logits_scale)
    warmup_frac: float = 0.02  # warmup length as a fraction of max_iter
    warmup_factor: float = 0.001  # start LR = base_lr * warmup_factor

    # ---- run -----------------------------------------------------------
    output_dir: str = "output"
    seed: int = 0
    device: str = "cuda"
    log_period: int = 20  # tensorboard / stdout every N iters
    val_period: int = 7000  # run validation every N iters
    val_iters: int = 50  # batches per validation pass
    ckpt_period: int = 5000  # checkpoint every N iters

    def __post_init__(self):
        if self.image_size % 16 != 0:
            raise ValueError("image_size must be divisible by the ViT patch size (16)")

    @property
    def warmup_iters(self) -> int:
        return int(self.warmup_frac * self.max_iter)
