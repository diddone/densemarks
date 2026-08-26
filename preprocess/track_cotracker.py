"""Extract CoTracker point tracks for a directory of raw videos.

For every input video, writes under `<data_dir>/<video_id>/`:
- `images/{frame}.jpg`  resized RGB frames, frame = bare int
- `pred_tracks.pt`      [1, T, P, 2] float32 xy pixel coords
- `pred_visibility.pt`  [1, T, P] bool

Matches the contract in densemarks/data.py.

Ported from track_video.py (see cotracker_data_preprocessing.zip), not
launch_tracking.py (needs external "gsam" masks we're not using). Default
path tracks a plain dense grid over the whole frame via the online
predictor (cotracker3_online), chunked by model.step.

Needs a CoTracker-capable env (torch, co-tracker, Pillow, tyro) - likely
different from landmarks_mediapipe.py's env (needs pytorch3d).

GSAM2 workflow (--use_gsam_mask): first run writes images/ then skips
tracking until `<video_dir>/gsam/0.png` exists; run segment_gsam2.py (its
own env) in between; rerun with --use_gsam_mask to pick up masks and track.
The masked path uses the offline predictor (cotracker3_offline), whose
forward() natively accepts segm_mask and filters the seed grid by it -
the online predictor's forward() has no segm_mask parameter.

After processing, hand-write `{data_dir}/{train,val}_tracked_points_paths.json`
yourself (see preprocess/README.md) - this script no longer generates a split.

Usage: python track_cotracker.py --video_dir /path/to/raw_videos --data_dir /path/to/data_dir
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import tyro
from cotracker.utils.visualizer import read_video_from_path
from PIL import Image
from tqdm import tqdm

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _load_cotracker(device: str, offline: bool) -> torch.nn.Module:
    variant = "cotracker3_offline" if offline else "cotracker3_online"
    model = torch.hub.load("facebookresearch/co-tracker", variant)
    return model.to(device)


@torch.no_grad()
def _run_cotracker_online(
    model: torch.nn.Module, video: torch.Tensor, grid_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feed a video through online CoTracker, chunked by model.step."""
    model(video_chunk=video, is_first_step=True, grid_size=grid_size)
    pred_tracks, pred_visibility = None, None
    for ind in range(0, video.shape[1] - model.step, model.step):
        pred_tracks, pred_visibility = model(
            video_chunk=video[:, ind : ind + model.step * 2]
        )
    if pred_tracks is None:
        raise ValueError(
            f"Video too short to track (needs > model.step={model.step} frames)"
        )
    return pred_tracks, pred_visibility


@torch.no_grad()
def _run_cotracker_masked(
    model: torch.nn.Module, video: torch.Tensor, grid_size: int, segm_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feed a whole video through offline CoTracker, seeding the grid from segm_mask."""
    return model(video=video, grid_size=grid_size, segm_mask=segm_mask.float())


def _resize_frames(frames: np.ndarray, image_size: int) -> np.ndarray:
    """Resize ``[T, H, W, 3]`` uint8 RGB frames to ``[T, image_size, image_size, 3]``."""
    resized = np.empty((frames.shape[0], image_size, image_size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        resized_frame = Image.fromarray(frame).resize(
            (image_size, image_size), Image.BILINEAR
        )
        resized[i] = np.array(resized_frame)
    return resized


def _discover_videos(video_dir: str) -> list[Path]:
    paths = [
        p
        for p in Path(video_dir).iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(paths)


def _load_gsam_mask(out_dir: str) -> torch.Tensor | None:
    """Load, threshold, and area-check `<out_dir>/gsam/0.png`. Returns None if it should be skipped."""
    mask_path = os.path.join(out_dir, "gsam", "0.png")
    if not os.path.isfile(mask_path):
        print(f"{out_dir}: no gsam mask yet, skipping tracking for now")
        return None
    mask = np.array(Image.open(mask_path).convert("L")) >= 128
    if mask.mean() < 0.2:
        print(f"{out_dir}: gsam mask is too small, skipping tracking")
        return None
    return torch.from_numpy(mask)[None, None]


def process_video(
    video_path: Path,
    out_dir: str,
    model: torch.nn.Module,
    grid_size: int,
    image_size: int,
    device: str,
    use_gsam_mask: bool = False,
) -> None:
    """Track one video and write its images/, pred_tracks.pt, pred_visibility.pt."""
    frames = read_video_from_path(str(video_path))  # [T, H, W, 3] uint8 RGB
    frames = _resize_frames(frames, image_size)

    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(images_dir, f"{i}.jpg"), quality=95)

    segm_mask = None
    if use_gsam_mask:
        segm_mask = _load_gsam_mask(out_dir)
        if segm_mask is None:
            return
        segm_mask = segm_mask.to(device)

    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    if segm_mask is not None:
        pred_tracks, pred_visibility = _run_cotracker_masked(
            model, video, grid_size, segm_mask
        )
    else:
        pred_tracks, pred_visibility = _run_cotracker_online(model, video, grid_size)

    torch.save(pred_tracks.float().cpu(), os.path.join(out_dir, "pred_tracks.pt"))
    torch.save(pred_visibility.bool().cpu(), os.path.join(out_dir, "pred_visibility.pt"))


def main(
    video_dir: str,
    data_dir: str,
    video_paths: tuple[str, ...] | None = None,
    grid_size: int = 40,
    image_size: int = 512,
    device: str = "cuda",
    part: int = 0,
    total_part: int = 1,
    use_gsam_mask: bool = False,
) -> None:
    """Track CoTracker points for all videos in video_dir into data_dir."""
    if video_paths is not None:
        videos = sorted(Path(p) for p in video_paths)
    else:
        videos = _discover_videos(video_dir)
    videos = videos[part::total_part]

    if not videos:
        print("No videos to process for this shard.")
    else:
        os.makedirs(data_dir, exist_ok=True)
        model = _load_cotracker(device, offline=use_gsam_mask)
        for video_path in tqdm(videos, desc="Tracking videos"):
            video_id = video_path.stem
            out_dir = os.path.join(data_dir, video_id)
            if os.path.isfile(os.path.join(out_dir, "pred_visibility.pt")):
                continue
            os.makedirs(out_dir, exist_ok=True)
            process_video(
                video_path, out_dir, model, grid_size, image_size, device, use_gsam_mask
            )


if __name__ == "__main__":
    tyro.cli(main)
