"""Training data pipeline for the original DenseMarks configuration.

On-disk layout under ``cfg.data_dir``::

    data_dir/
      train_tracked_points_paths.pkl   # pickled list of paths to per-video pred_tracks.pt
      val_tracked_points_paths.pkl
      <video_id>/
        pred_tracks.pt                 # [1, T, P, 2] float, (x, y) pixel coords
        pred_visibility.pt             # [1, T, P] bool, or [1, T, P, 2] (vis * conf)
        images/{frame}.png|jpg         # RGB, cfg.image_size x cfg.image_size
        comb_seg/{frame}.png           # segmentation, class ids 0..n_segm_channels-1
        mediapipe/{frame}.npy          # [468, 2] normalised (x, y) in [0, 1]
        mediapipe/{frame}_vis_mask.npy # [468] bool, True if the point is visible

Visibility path is derived as ``tracks_file_name.replace("_tracks", "_visibility")``.

The segmentation mask is taken as-is: it must already cover everything you want
supervised (body *and* face), since nonzero also selects which tracked points
become correspondences. The released checkpoint was trained on masks whose body
region is class 3 with the face classes painted on top.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

from densemarks.config import Config

logger = logging.getLogger(__name__)

# Arrays are numpy while the dataset builds a frame, torch from `Frame.to_torch()` on.
Array = np.ndarray | torch.Tensor


@dataclass
class Frame:
    """One image and the supervision attached to it."""

    tag: str
    image: Array  # [3, H, W] uint8 RGB
    corrs_yx: Array  # [N, 2] tracked points; row k is the same physical point in both
    # frames of the pair -- that row-alignment is the correspondence training signal
    seg_mask: Array | None = None  # [H, W] class ids; None when unavailable/too small
    fixed_lmks_yx: Array | None = None  # [L, 2] visible MediaPipe subset, or None
    fixed_l_idx: Array | None = None  # [L] row in the canonical landmark table

    def to_torch(self) -> Frame:
        """Convert the numpy arrays to the torch dtypes the rest of training expects."""

        def as_int64(arr: np.ndarray | None) -> torch.Tensor | None:
            if arr is None:
                return None
            return torch.from_numpy(np.ascontiguousarray(arr).astype(np.int64))

        return Frame(
            tag=self.tag,
            image=torch.from_numpy(np.ascontiguousarray(self.image)).to(torch.uint8),
            corrs_yx=as_int64(self.corrs_yx),
            seg_mask=as_int64(self.seg_mask),
            fixed_lmks_yx=as_int64(self.fixed_lmks_yx),
            fixed_l_idx=as_int64(self.fixed_l_idx),
        )

    def to(self, device: torch.device) -> Frame:
        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device, non_blocking=True)

        return Frame(
            tag=self.tag,
            image=move(self.image),
            corrs_yx=move(self.corrs_yx),
            seg_mask=move(self.seg_mask),
            fixed_lmks_yx=move(self.fixed_lmks_yx),
            fixed_l_idx=move(self.fixed_l_idx),
        )


@dataclass
class Batch:
    """A collated batch of P frame pairs, flattened to [src0, tgt0, src1, tgt1, ...]."""

    frames: list[Frame]
    close_mask: torch.Tensor  # [P, N, N] bool: two rows of a pair on the same pixel
    seg_frame_inds: torch.Tensor | None  # [S] frames that carry a seg_mask
    lmks_frame_inds: torch.Tensor | None  # [sum L] source frame of each landmark, in
    # the order the frames' landmarks concatenate

    @property
    def sources(self) -> list[Frame]:
        return self.frames[0::2]

    @property
    def targets(self) -> list[Frame]:
        return self.frames[1::2]

    @property
    def images(self) -> torch.Tensor:
        return torch.stack([f.image for f in self.frames], dim=0)

    def to(self, device: torch.device) -> Batch:
        """The collate_fn emits CPU tensors; the loss head indexes ``desc`` with them."""

        def move(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device, non_blocking=True)

        return Batch(
            frames=[f.to(device) for f in self.frames],
            close_mask=move(self.close_mask),
            seg_frame_inds=move(self.seg_frame_inds),
            lmks_frame_inds=move(self.lmks_frame_inds),
        )


def _require_mediapipe_asset(path: str) -> np.ndarray:
    """Load the Mediapipe subset index (which of the 468 landmarks we use)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required asset '{path}' is missing "
            f"(expected assets/mediapipe_indices_of_fixed_landmarks.npy). "
            f"See README for how to obtain the training assets."
        )
    return np.load(path)


def check_borders_yx(corrs_yx: np.ndarray, height: int, width: int) -> np.ndarray:
    """True for (y, x) points that lie inside the image."""
    return (
        (corrs_yx[:, 0] >= 0)
        & (corrs_yx[:, 1] >= 0)
        & (corrs_yx[:, 0] < height)
        & (corrs_yx[:, 1] < width)
    )


def check_borders_xy(corrs_xy: np.ndarray, height: int, width: int) -> np.ndarray:
    """True for (x, y) points that lie inside the image."""
    return (
        (corrs_xy[:, 1] >= 0)
        & (corrs_xy[:, 0] >= 0)
        & (corrs_xy[:, 0] < width)
        & (corrs_xy[:, 1] < height)
    )


def load_image_rgb(image_path: str) -> np.ndarray:
    """Load an image as uint8 RGB in CHW layout."""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image)
    return np.transpose(arr, (2, 0, 1)).copy()


def load_seg_mask(seg_mask_path: str, n_classes: int) -> np.ndarray:
    """Load a segmentation mask of class ids; reject anything that isn't one."""
    mask = np.array(Image.open(seg_mask_path))
    if mask.ndim != 2 or mask.max() >= n_classes:
        raise ValueError(
            f"'{seg_mask_path}' is not a single-channel class-id mask with values in "
            f"[0, {n_classes - 1}] (got shape {mask.shape}, max {mask.max()}). "
            f"Masks are used as-is; see README for the expected format."
        )
    return mask


def unnormalize_kpts_and_get_border_mask(
    lms: np.ndarray, image_size: tuple[int, int], yx: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalised landmarks to pixel ints; also return an in-bounds mask."""
    assert lms.min() >= -10.0 and lms.max() <= 10.0, (
        f"Landmarks should be normalized, min: {lms.min()}, max: {lms.max()}"
    )
    lms = lms.copy()
    if yx:
        lms[:, 0] *= image_size[0]
        lms[:, 1] *= image_size[1]
    else:
        # (x, y) normalised; image_size is (H, W).
        lms[:, 1] *= image_size[0]
        lms[:, 0] *= image_size[1]
    lms = np.floor(lms).astype(np.int32)
    bord_mask = (
        check_borders_yx(lms, height=image_size[0], width=image_size[1])
        if yx
        else check_borders_xy(lms, height=image_size[0], width=image_size[1])
    )
    return lms, bord_mask


def load_mediapipe_vis_lmks(
    landmarks_path: str,
    vis_mask_path: str,
    fixed_lmks_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Take the fixed Mediapipe subset and drop landmarks marked invisible."""
    lmks = np.load(landmarks_path)[fixed_lmks_idx]
    vis_mask = np.load(vis_mask_path)[fixed_lmks_idx]
    l_idx = np.arange(lmks.shape[0])
    return lmks[vis_mask], l_idx[vis_mask]


def remove_invisible_lmks(frame: Frame) -> Frame:
    """Drop fixed landmarks that landed outside the image (e.g. after aug)."""
    h, w = frame.image.shape[-2], frame.image.shape[-1]
    vis_mask = check_borders_yx(frame.fixed_lmks_yx, height=h, width=w)
    frame.fixed_lmks_yx = frame.fixed_lmks_yx[vis_mask]
    frame.fixed_l_idx = frame.fixed_l_idx[vis_mask]
    if len(frame.fixed_lmks_yx) == 0:
        frame.fixed_lmks_yx = None
        frame.fixed_l_idx = None
    return frame


def remove_invisible_corrs(pair: list[Frame]) -> list[Frame]:
    """Keep only tracked points that are still in-bounds in both frames."""
    assert len(pair) == 2
    h1, w1 = pair[0].image.shape[-2], pair[0].image.shape[-1]
    h2, w2 = pair[1].image.shape[-2], pair[1].image.shape[-1]
    valid = check_borders_yx(pair[0].corrs_yx, height=h1, width=w1) & check_borders_yx(
        pair[1].corrs_yx, height=h2, width=w2
    )
    for frame in pair:
        frame.corrs_yx = frame.corrs_yx[valid]
    return pair


def _compute_close_mask(yx: torch.Tensor) -> torch.Tensor:
    """Mark pairs of distinct points that share the same pixel (collisions)."""
    same = (yx[:, None, :] == yx[None, :, :]).all(dim=-1)
    return same & ~torch.eye(yx.shape[0], dtype=torch.bool)


def collate_fn(batch: list[list[Frame]]) -> Batch:
    """Turn a list of pairs into a flat batch: pad corrs, add close_mask + frame indices."""
    frames = [frame for pair in batch for frame in pair]
    assert len(frames) % 2 == 0, "Batch size should be even, as we use pairs"

    # Which frames have a seg_mask.
    seg_inds = [i for i, f in enumerate(frames) if f.seg_mask is not None]
    seg_frame_inds = torch.tensor(seg_inds, dtype=torch.int64) if seg_inds else None

    # For concatenated landmarks: which frame each landmark came from.
    lmks_sizes = [
        0 if f.fixed_lmks_yx is None else f.fixed_lmks_yx.shape[0] for f in frames
    ]
    lmks_frame_inds = None
    if any(lmks_sizes):
        lmks_frame_inds = torch.repeat_interleave(
            torch.arange(len(frames), dtype=torch.int64),
            torch.tensor(lmks_sizes, dtype=torch.int64),
        )

    # Make every pair have the same number of corrs by resampling extras.
    max_corr_sizes = max(len(f.corrs_yx) for f in frames)
    close_masks = []
    for src, tgt in zip(frames[0::2], frames[1::2]):
        assert src.corrs_yx.shape[0] == tgt.corrs_yx.shape[0]
        n = src.corrs_yx.shape[0]
        pad_size = max_corr_sizes - n
        if pad_size > 0:
            if n == 0:
                raise ValueError(
                    f"Cannot pad empty correspondences for tags {src.tag}, {tgt.tag}"
                )
            random_idx = torch.from_numpy(np.random.choice(n, pad_size, replace=True))
            # Same rows on both sides, so the row-alignment survives padding.
            src.corrs_yx = torch.cat([src.corrs_yx, src.corrs_yx[random_idx]], dim=0)
            tgt.corrs_yx = torch.cat([tgt.corrs_yx, tgt.corrs_yx[random_idx]], dim=0)

        # Collision if either frame has two points on the same pixel.
        close_masks.append(
            _compute_close_mask(src.corrs_yx) | _compute_close_mask(tgt.corrs_yx)
        )

    return Batch(
        frames=frames,
        close_mask=torch.stack(close_masks, dim=0),
        seg_frame_inds=seg_frame_inds,
        lmks_frame_inds=lmks_frame_inds,
    )


def _worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own numpy RNG seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


class DenseMarksDataset:
    """Loads CelebV-HQ videos and samples frame pairs with tracked correspondences.

    Training draws random pairs forever via ``sample_pair``; validation uses
    ``get_eval_pair`` with a fixed seed so the same frames are reused.
    """

    def __init__(self, cfg: Config, is_train: bool):
        assert cfg.image_size % 16 == 0
        self.cfg = cfg
        self.is_train = is_train
        self.data_dir = cfg.data_dir
        self.tracks_file_name = cfg.tracks_file_name
        self.image_size = (cfg.image_size, cfg.image_size)
        self.points_min_lim = cfg.points_min_lim
        self.cotracker_conf_threshold = cfg.cotracker_conf_threshold
        self.n_segm_channels = cfg.n_segm_channels

        self.fixed_lmks_idx = _require_mediapipe_asset(cfg.mediapipe_lmks_idx_file)

        # Intersect the split pickle with folders that actually have tracks on disk.
        split = "train" if is_train else "val"
        path_to_tracked_points_paths = os.path.join(
            cfg.data_dir, f"{split}_tracked_points_paths.pkl"
        )
        if not os.path.isfile(path_to_tracked_points_paths):
            raise FileNotFoundError(
                f"Split file '{path_to_tracked_points_paths}' is missing. "
                f"Expected '{split}_tracked_points_paths.pkl' under data_dir. "
                f"See README for the dataset layout."
            )

        with open(path_to_tracked_points_paths, "rb") as f:
            points_paths = pickle.load(f)

        path_video_folders = [Path(path).parts[-2] for path in points_paths]
        video_folders_with_tracks = [
            name
            for name in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, name))
            and os.path.isfile(os.path.join(self.data_dir, name, self.tracks_file_name))
        ]
        video_folders = set(video_folders_with_tracks) & set(path_video_folders)
        self.tracked_points_paths = [
            os.path.join(self.data_dir, video_folder, self.tracks_file_name)
            for video_folder in sorted(video_folders)
        ]
        if not self.tracked_points_paths:
            raise FileNotFoundError(
                f"No videos with '{self.tracks_file_name}' found matching "
                f"{path_to_tracked_points_paths}"
            )

        if is_train:
            # Geometric aug; keypoints stay in (y, x). Val has no transform.
            self.transform = A.Compose(
                [
                    A.ShiftScaleRotate(
                        rotate_limit=(-18, 18),
                        shift_limit=(-0.1, 0.1),
                        scale_limit=(-0.1, 0.1),
                        p=0.5,
                        border_mode=0,
                    )
                ],
                keypoint_params=A.KeypointParams(format="yx", remove_invisible=False),
                additional_targets={
                    "fixed_lmks_yx": "keypoints",
                },
            )
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.tracked_points_paths)

    def sample_frames_pair(self, total_frames: int, seed=None) -> np.ndarray:
        """Pick two distinct frame indices; ``seed`` makes the choice deterministic."""
        if seed is not None:
            rng = np.random.default_rng(seed)
            return rng.choice(total_frames, size=2, replace=False)
        return np.random.choice(total_frames, size=2, replace=False)

    def _load_landmarks_for_frame(
        self, mediapipe_dir: str, frame_idx: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load visible fixed landmarks for one frame as (y, x) pixels, or None."""
        mp_lmks_path = os.path.join(mediapipe_dir, f"{frame_idx}.npy")
        vis_lmks_mask_path = mp_lmks_path.replace(".npy", "_vis_mask.npy")
        if not (os.path.exists(mp_lmks_path) and os.path.exists(vis_lmks_mask_path)):
            return None, None
        fixed_lmks_norm_xy, fixed_l_idx = load_mediapipe_vis_lmks(
            mp_lmks_path, vis_lmks_mask_path, self.fixed_lmks_idx
        )
        if len(fixed_lmks_norm_xy) == 0:
            return None, None
        fixed_lmks_xy, lmks_border_mask = unnormalize_kpts_and_get_border_mask(
            fixed_lmks_norm_xy, self.image_size, yx=False
        )
        # Convert (x, y) → (y, x) after dropping out-of-bounds.
        fixed_lmks_yx = fixed_lmks_xy[lmks_border_mask][:, [1, 0]]
        fixed_l_idx = fixed_l_idx[lmks_border_mask]
        if len(fixed_lmks_yx) == 0:
            return None, None
        return fixed_lmks_yx, fixed_l_idx

    def get_pair(self, idx: int, seed=None) -> list[Frame] | None:
        """Build one frame pair from video ``idx``, or None if the pair is unusable."""
        tracked_points_path = self.tracked_points_paths[idx]
        video_dir = os.path.dirname(tracked_points_path)
        images_dir = os.path.join(video_dir, "images")
        seg_masks_dir = os.path.join(video_dir, "comb_seg")
        mediapipe_dir = os.path.join(video_dir, "mediapipe")

        pred_tracks = torch.load(
            os.path.join(video_dir, self.tracks_file_name),
            weights_only=True,
            map_location="cpu",
        ).numpy()
        vis_name = self.tracks_file_name.replace("_tracks", "_visibility")
        pred_visibility = torch.load(
            os.path.join(video_dir, vis_name),
            weights_only=True,
            map_location="cpu",
        ).numpy()

        # If last dim is 2 (vis, conf), combine them then threshold.
        # (Older code always took this path due to a buggy string check.)
        if pred_visibility.shape[-1] == 2:
            pred_visibility = pred_visibility[..., 0] * pred_visibility[..., 1]
        pred_visibility = pred_visibility > self.cotracker_conf_threshold

        total_frames = pred_tracks.shape[1]
        selected_frames = self.sample_frames_pair(total_frames, seed=seed)

        # Prefer png; fall back to jpg.
        rgb_image_path1 = f"{images_dir}/{selected_frames[0]}.png"
        rgb_image_path2 = f"{images_dir}/{selected_frames[1]}.png"
        if not os.path.exists(rgb_image_path1) or not os.path.exists(rgb_image_path2):
            rgb_image_path1 = f"{images_dir}/{selected_frames[0]}.jpg"
            rgb_image_path2 = f"{images_dir}/{selected_frames[1]}.jpg"
        if not os.path.exists(rgb_image_path1) or not os.path.exists(rgb_image_path2):
            raise ValueError(
                f"Image {rgb_image_path1} or {rgb_image_path2} does not exist, "
                f"tried both png and jpg"
            )

        frame1 = load_image_rgb(rgb_image_path1)
        frame2 = load_image_rgb(rgb_image_path2)
        h, w = frame1.shape[1], frame1.shape[2]
        assert (h, w) == self.image_size, (
            f"Expected image size {self.image_size}, got {(h, w)}. "
            f"Resize is not supported; preprocess data to cfg.image_size."
        )
        assert frame2.shape[1:] == self.image_size

        # Missing mask → skip this pair (do not crash training). A malformed one
        # still raises: silently training on it would corrupt the segm loss.
        try:
            seg_mask1 = load_seg_mask(
                f"{seg_masks_dir}/{selected_frames[0]}.png", self.n_segm_channels
            )
            seg_mask2 = load_seg_mask(
                f"{seg_masks_dir}/{selected_frames[1]}.png", self.n_segm_channels
            )
        except FileNotFoundError:
            return None

        # Keep points visible in both frames, in-bounds, and inside the mask.
        pred_visibility_both = np.logical_and(
            pred_visibility[0, selected_frames[0]],
            pred_visibility[0, selected_frames[1]],
        )
        points1_xy = pred_tracks[0, selected_frames[0], pred_visibility_both]
        points2_xy = pred_tracks[0, selected_frames[1], pred_visibility_both]

        border_mask = check_borders_xy(points1_xy, h, w) & check_borders_xy(points2_xy, h, w)
        points1_xy, points2_xy = points1_xy[border_mask], points2_xy[border_mask]

        points1_mask = (
            seg_mask1[points1_xy[:, 1].astype(np.int32), points1_xy[:, 0].astype(np.int32)]
            != 0
        )
        points2_mask = (
            seg_mask2[points2_xy[:, 1].astype(np.int32), points2_xy[:, 0].astype(np.int32)]
            != 0
        )
        combined_mask = np.logical_and(points1_mask, points2_mask)
        points1_yx = points1_xy[combined_mask][:, [1, 0]].astype(np.int32)
        points2_yx = points2_xy[combined_mask][:, [1, 0]].astype(np.int32)

        if points1_yx.shape[0] < self.points_min_lim:
            return None

        seq_folder = os.path.basename(os.path.normpath(video_dir))
        fixed_lmks_yx1, fixed_l_idx1 = self._load_landmarks_for_frame(
            mediapipe_dir, int(selected_frames[0])
        )
        fixed_lmks_yx2, fixed_l_idx2 = self._load_landmarks_for_frame(
            mediapipe_dir, int(selected_frames[1])
        )

        return [
            Frame(
                tag=f"{seq_folder}_{selected_frames[0]}",
                image=frame1,
                corrs_yx=points1_yx,
                seg_mask=seg_mask1,
                fixed_lmks_yx=fixed_lmks_yx1,
                fixed_l_idx=fixed_l_idx1,
            ),
            Frame(
                tag=f"{seq_folder}_{selected_frames[1]}",
                image=frame2,
                corrs_yx=points2_yx,
                seg_mask=seg_mask2,
                fixed_lmks_yx=fixed_lmks_yx2,
                fixed_l_idx=fixed_l_idx2,
            ),
        ]

    def process_pair(self, pair: list[Frame]) -> list[Frame]:
        """Apply aug (train), drop out-of-bounds points, convert to torch."""
        for frame in pair:
            h, w = frame.image.shape[-2], frame.image.shape[-1]
            seg_mask = frame.seg_mask
            if seg_mask is not None:
                seg_mask = seg_mask.copy()
                # Class 20 is ignored / treated as background.
                seg_mask[seg_mask == 20] = 0

            transform_args: dict = {}
            # Tiny masks are unreliable — drop them rather than supervise on noise.
            if seg_mask is not None and np.sum(seg_mask != 0) < h * w * 0.1:
                logger.debug("Segmentation mask too small for %s; set to None", frame.tag)
                seg_mask = None

            frame.seg_mask = seg_mask
            if seg_mask is not None:
                transform_args["mask"] = seg_mask

            if frame.fixed_lmks_yx is not None:
                transform_args["fixed_lmks_yx"] = frame.fixed_lmks_yx

            if self.transform is not None:
                transformed = self.transform(
                    image=frame.image.transpose(1, 2, 0),
                    keypoints=frame.corrs_yx,
                    **transform_args,
                )
                frame.image = transformed["image"].transpose(2, 0, 1)
                frame.corrs_yx = np.array(transformed["keypoints"]).astype(np.int32)
                if "mask" in transformed:
                    frame.seg_mask = transformed["mask"]
                if "fixed_lmks_yx" in transformed:
                    frame.fixed_lmks_yx = np.array(transformed["fixed_lmks_yx"]).astype(
                        np.int32
                    )
                    remove_invisible_lmks(frame)

        pair = remove_invisible_corrs(pair)
        return [frame.to_torch() for frame in pair]

    def sample_pair(self) -> list[Frame]:
        """Keep resampling videos until a usable train pair is found."""
        for _ in range(5):
            pair = None
            for _attempt in range(50):
                idx = np.random.randint(0, len(self.tracked_points_paths))
                pair = self.get_pair(idx)
                if pair is not None:
                    break
            if pair is None:
                continue
            pair = self.process_pair(pair)
            if pair is not None:
                return pair
        raise ValueError(
            "Could not sample a usable pair after retries; check the dataset layout"
        )

    def get_eval_pair(self, idx: int) -> list[Frame]:
        """Deterministic val pair for video ``idx``; retries a few fixed seeds."""
        pair = None
        for i in range(10):
            pair = self.get_pair(idx, seed=42 + i)
            if pair is None:
                continue
            pair = self.process_pair(pair)
            if pair is not None:
                break
        if pair is None:
            raise ValueError(f"Pair {idx} is None; check val dataset / seg masks")
        return pair


class _TrainIterable(IterableDataset):
    """Infinite stream of random train pairs for the DataLoader."""

    def __init__(self, inner: DenseMarksDataset):
        super().__init__()
        self.inner = inner

    def __iter__(self):
        while True:
            yield self.inner.sample_pair()


class _ValMap(Dataset):
    """Map-style wrapper: one eval pair per video index."""

    def __init__(self, inner: DenseMarksDataset):
        super().__init__()
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> list[Frame]:
        return self.inner.get_eval_pair(idx)


def build_dataloader(cfg: Config, is_train: bool) -> DataLoader:
    """Build a train (infinite) or val (finite, seeded) DataLoader."""
    inner = DenseMarksDataset(cfg, is_train=is_train)
    if is_train:
        dataset: Dataset | IterableDataset = _TrainIterable(inner)
        shuffle = False
    else:
        dataset = _ValMap(inner)
        shuffle = False

    generator = None
    if not is_train:
        generator = torch.Generator()
        generator.manual_seed(cfg.seed)

    return DataLoader(
        dataset,
        batch_size=cfg.pairs_per_batch,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        shuffle=shuffle,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )
