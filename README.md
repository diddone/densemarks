# DenseMarks

A PyTorch implementation for dense UVW coordinate prediction from human head images using DINOv3 backbone with DPT head architecture.

## Overview

DenseMarks predicts per-pixel positions in canonical space (cube [0, 1]³) from human head images.

**Input**: RGB images of size 512×512 pixels

**Output**: UVW coordinates tensor (B, 3, 512, 512) with values in [0, 1]

## Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/densemarks.git
   cd densemarks
   ```

2. **Install DINOv3 submodule:**
   ```bash
   git clone https://github.com/facebookresearch/dinov3 third_party_dinov3
   ```

3. **Modify DINOv3 for compatibility:**
   ```bash
   sed -i '/dinov3\.hub\.segmentors/s/^/#/' third_party_dinov3/hubconf.py
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