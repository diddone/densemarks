<div align="center">
<h1>DenseMarks: Learning Canonical Embeddings for Human Heads Images via Point Tracks</h1>

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge)](https://arxiv.org/pdf/2511.02830#page=5.27)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge)](https://diddone.github.io/densemarks/)
[![YouTube](https://img.shields.io/badge/YouTube-Video-red?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/o8DOOYFW0gI)

Accepted to ICLR 2026.

A PyTorch implementation for dense UVW coordinate prediction from human head images using a DINOv3 backbone and a DPT-style head architecture.
</div>

## Overview

We propose a replacement of standard face landmarks with a dense per-pixel embeddings of a full human head that live in 3D canonical space, consistent across subjects and poses. Predicted embeddings enable a variety of applications, ranging from head tracking to MVS. DenseMarks achieves 2× higher correspondence accuracy than prior state of the art for keypoint matching across human heads. Additionally, DenseMarks greatly improves the robustness of in-the-wild monocular head tracking and enables CPU-based multi-view stereo reconstruction of human heads from 2–5 posed images in seconds.

- **Input:** RGB image of size 512×512
- **Output:** UVW coordinate tensor `(B, 3, 512, 512)` with values in `[0, 1]`

---

## Quick start

```python
from dense_marks_model import DenseMarksModel, read_image
from huggingface_hub import hf_hub_download

model = DenseMarksModel(hf_hub_download("diddone/densemarks", "model.safetensors"))
images = read_image("assets/00000.png")  # RGB, 512x512
uvw = model(images)  # predict UVW coordinates
```

See [Installation](#installation) for setup (DINOv3 submodule, dependencies).

---

## Prerequisites

- Python 3.10+ (DINOv3 requires it)
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/diddone/densemarks.git
   cd densemarks
   ```

2. **Install DINOv3 submodule:**
   ```bash
   git clone https://github.com/facebookresearch/dinov3 third_party_dinov3
   ```

3. **Modify DINOv3 for compatibility:**
   ```bash
   # For Linux (GNU sed):
   sed -i '/dinov3\.hub\.segmentors/s/^/#/; /dinov3\.hub\.classifiers/s/^/#/; /dinov3\.hub\.detectors/s/^/#/; /dinov3\.hub\.dinotxt/s/^/#/; /dinov3\.hub\.depthers/s/^/#/' third_party_dinov3/hubconf.py

   # For macOS (BSD sed):
   sed -i '' '/dinov3\.hub\.segmentors/s/^/#/; /dinov3\.hub\.classifiers/s/^/#/; /dinov3\.hub\.detectors/s/^/#/; /dinov3\.hub\.dinotxt/s/^/#/; /dinov3\.hub\.depthers/s/^/#/' third_party_dinov3/hubconf.py
   ```

4. **Install dependencies:**
   ```bash
   pip install torch transformers numpy pillow huggingface_hub safetensors
   ```

Model weights are downloaded automatically from Hugging Face on first use — see [Quick start](#quick-start).

## Training

### Data layout

```
data_dir/
  train_tracked_points_paths.json    # JSON array of paths to per-video pred_tracks.pt
  val_tracked_points_paths.json
  <video_id>/
    pred_tracks.pt        # [1, T, P, 2] float, (x, y) pixel coords per tracked point
    pred_visibility.pt    # [1, T, P] bool  (or [1, T, P, 2] for AllTracker: vis * conf)
    images/{frame}.png    # or .jpg; 512x512 RGB
    comb_seg/{frame}.png  # segmentation, single-channel class ids 0..21
    mediapipe/{frame}.npy            # [468, 2] normalised (x, y) in [0,1]
    mediapipe/{frame}_vis_mask.npy   # [468] bool
```

Data requires separate preprocessing (not supported in this repo). In the paper we used CelebV-HQ, with CoTracker3 for point tracks, MediaPipe for landmarks, and face parsing for segmentation masks.

`train_tracked_points_paths.json` and `val_tracked_points_paths.json` are not auto-generated — hand-write them after preprocessing. Each is a JSON array of strings, one per video, pointing to that video's `pred_tracks.pt`. Only the parent folder name is used, so a path relative to `data_dir` like `"<video_id>/pred_tracks.pt"` is simplest:

```json
[
  "video001/pred_tracks.pt",
  "video002/pred_tracks.pt",
  "video003/pred_tracks.pt"
]
```

### Running training

```bash
pip install -r requirements.txt
python train.py \
    --data_dir /path/to/CelebV-HQ_tracked \
    --dinov3_weights /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

`--dinov3_weights` is required. Training starts from the LVD-1689M DINOv3 ViT-B/16
checkpoint, which we cannot redistribute — obtain it from
[facebookresearch/dinov3](https://github.com/facebookresearch/dinov3). Passing
`--dinov3_weights random` initialises the backbone randomly; that is a smoke-test
escape hatch and will not reproduce the released model.

Apart from that path, the command uses the published defaults: DINOv3 ViT-B/16 + DPT neck, 512×512, 8 pairs/batch, 140k steps, AdamW at 1e-4 (backbone 1e-5, cube `_E` at 1e-3), cosine decay with 2% linear warmup.

### Required assets

Both ship with the repo in `assets/` (the paths are `Config` defaults) and define the 79 fixed facial landmarks that anchor the canonical cube:

- `assets/mediapipe_indices_of_fixed_landmarks.npy` — `[79]` int, which of MediaPipe's 468
  face landmarks we supervise
- `assets/fixed_positions_of_landmarks_in_cube.npy` — `[79, 3]` float, the canonical UVW target for each,
  from a FLAME head-model fit

The two are row-aligned: row `i` of the UVW file is the target for the landmark named by row `i` of the index file.

### Memory and resume

~24GB VRAM is enough at 8 pairs/batch (with bf16 autocast). On smaller cards, one can e.g. lower `--pairs_per_batch`.

Resume a crashed run with `--resume` (loads `{output_dir}/checkpoint.pt`). Checkpoints are written every `--ckpt_period` steps (default 5000); the final weights are `{output_dir}/model_final.pt`.

TensorBoard logs go to `--output_dir` (default `output`).

---

## Tracking

We use [VHAP](https://github.com/ShenhanQian/VHAP) for monocular head tracking. UVW is treated as an additional RGB image: we define an extra RGB texture in the FLAME UV space and supervise it with a photometric loss.

---

## Citation

```bibtex
@inproceedings{pozdeev2026densemarks,
  title     = {DenseMarks: Learning Canonical Embeddings for Human Heads Images via Point Tracks},
  author    = {Pozdeev, Dmitrii and Artemov, Alexey and Bhattarai, Ananta R. and Sevastopolsky, Artem},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  eprint    = {2511.02830},
  archivePrefix = {arXiv},
  url       = {https://openreview.net/forum?id=KOvRxAMBzV},
}
```

## License

Code is released under the [MIT License](LICENSE). The released model checkpoint is released under a
[Creative Commons (CC BY-NC 4.0)](LICENSE-CHECKPOINT) license, due to restrictions on the training data (in particular CelebV-HQ).

## Acknowledgements

Built on [DINOv3](https://github.com/facebookresearch/dinov3) and DPT (via 🤗 Transformers). Training data
was annotated using [CoTracker3](https://github.com/facebookresearch/co-tracker) for point tracks,
[GroundedSAM2](https://github.com/IDEA-Research/Grounded-SAM-2) for foreground segmentation,
[MediaPipe](https://github.com/google-ai-edge/mediapipe) for landmarks, and
[FaRL](https://github.com/FacePerceiver/FaRL) refined with
[face-parsing](https://github.com/zllrunning/face-parsing.PyTorch) for segmentation masks. Monocular tracking
uses [VHAP](https://github.com/ShenhanQian/VHAP).
