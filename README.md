# DenseMarks

A PyTorch implementation for dense UVW coordinate prediction from human head images using a DINOv3 backbone and a DPT-style head architecture.

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge)](https://diddone.github.io/densemarks/)
[![YouTube](https://img.shields.io/badge/YouTube-Video-red?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/vVuSXFeZNL8)

## Overview

DenseMarks predicts per-pixel positions in the canonical space (cube $[0, 1] ^ 3$) from human head images.

- **Input:** RGB image of size 512×512  
- **Output:** UVW coordinate tensor `(B, 3, 512, 512)` with values in `[0, 1]`

---

## 🚀 Current Status

DenseMarks currently supports **inference only** — you can run the model to generate dense UVW predictions from input images.  
🧠 **Training support is coming soon!** Stay tuned :)

---

## Prerequisites

- Python 3.8+
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
   pip install torch transformers numpy
   ```

5. **Download model weights from Hugging Face:**
   ```python3
   from dense_marks_model import DenseMarksModel, read_image
   from huggingface_hub import hf_hub_download
   model = DenseMarksModel(hf_hub_download("diddone/densemarks", "model.safetensors"))
   images = read_image("assets/00000.png") # rgb, 512x512
   uvw = model(images) # Predict UVW coordinates
   ```
