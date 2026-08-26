"""Segment the "person" in each video frame with Grounded-SAM-2.

For every `<data_dir>/<video_id>/images/{frame}.jpg` (written by
track_cotracker.py), runs Grounding DINO (HF `grounding-dino-tiny`) + SAM 2
with text prompt "person." and writes a binary mask to
`<video_id>/gsam/{frame}.png` (uint8, 0/255, single channel, same H x W as
input). Frames with no detection are skipped with a warning, not a crash.

Ported from dense_lmks/scripts/utils_sam.py + run_sam_on_video_by_frames.py.

Needs a Grounded-SAM-2 env (see requirements-gsam2.txt) - a separate,
heavier env from track_cotracker.py and landmarks_mediapipe.py.

Usage: python segment_gsam2.py --data_dir /path/to/data_dir \
    --gsam2_repo /path/to/Grounded-SAM-2
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.io import imsave
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class GroundedSam:
    """Grounding DINO (HF) + SAM 2 person segmenter for a single image."""

    def __init__(
        self,
        sam2_checkpoint: str,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        text_prompt: str = "person.",
        grounding_model: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
    ):
        self.text_prompt = text_prompt
        self.device = device if torch.cuda.is_available() else "cpu"

        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        self.processor = AutoProcessor.from_pretrained(grounding_model)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_model
        ).to(self.device)

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def run(self, img_path: str, target_size: tuple[int, int] = (800, 800)) -> np.ndarray:
        """Return a uint8 0/255 mask at the original image resolution."""
        orig_img = Image.open(img_path)
        orig_size = orig_img.size
        image = orig_img.resize(target_size)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            self.sam2_predictor.set_image(np.array(image.convert("RGB")))

            inputs = self.processor(
                images=image, text=self.text_prompt, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.grounding_model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, text_threshold=0.3, target_sizes=[image.size[::-1]]
            )

        input_boxes = results[0]["boxes"].cpu().numpy()
        masks, _, _ = self.sam2_predictor.predict(
            point_coords=None, point_labels=None, box=input_boxes, multimask_output=False
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = results[0]["scores"].cpu().numpy()
        final_mask = (masks[np.argmax(confidences)] * 255).astype(np.uint8)
        return cv2.resize(final_mask, orig_size, interpolation=cv2.INTER_NEAREST)


def _sort_by_frame_number(paths: list[str]) -> list[str]:
    return sorted(paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def run_on_video(grounded_sam: GroundedSam, video_dir: str) -> None:
    """Write gsam/{frame}.png for every images/{frame}.png missing one."""
    image_paths = _sort_by_frame_number(glob.glob(os.path.join(video_dir, "images", "*.jpg")))
    if not image_paths:
        return

    out_dir = os.path.join(video_dir, "gsam")
    os.makedirs(out_dir, exist_ok=True)
    for image_path in image_paths:
        out_path = os.path.join(out_dir, Path(image_path).with_suffix(".png").name)
        if os.path.isfile(out_path):
            continue
        try:
            mask = grounded_sam.run(image_path)
            assert np.any(mask >= 128), "no object found"
        except Exception as e:
            print(f"Warning: skipping {image_path}: {e}")
            continue
        imsave(out_path, mask)


def main(
    data_dir: str,
    gsam2_repo: str,
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    text_prompt: str = "person.", # dot is apparantly important
    device: str = "cuda",
    part: int = 0,
    total_part: int = 1,
) -> None:
    """Run GSAM2 person segmentation over all videos in data_dir (sharded).

    sam2_checkpoint is resolved relative to gsam2_repo if not absolute.
    """
    if not os.path.isabs(sam2_checkpoint):
        sam2_checkpoint = os.path.join(gsam2_repo, sam2_checkpoint)

    videos = sorted(
        name
        for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name, "images"))
    )[part::total_part]
    if not videos:
        print("No videos to process for this shard.")
        return

    grounded_sam = GroundedSam(
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        text_prompt=text_prompt,
        device=device,
    )
    for video_id in videos:
        run_on_video(grounded_sam, os.path.join(data_dir, video_id))


if __name__ == "__main__":
    tyro.cli(main)
