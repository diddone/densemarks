"""Extract MediaPipe FaceMesh landmarks + visibility masks for tracked videos.

For every video folder under `data_dir` (must already have an `images/`
subfolder, e.g. from track_cotracker.py), writes under
`<data_dir>/<video_id>/mediapipe/`:

- `{frame}.npy`          [468, 2] float32, normalized (x, y) in [0, 1]
- `{frame}_vis_mask.npy` [468] bool

Matches the contract in densemarks/data.py. Visibility is a real pytorch3d
mesh-occlusion rasterization (ported from dense_lmks/scripts/mediapipe_utils.py),
not a heuristic.

Needs mediapipe + pytorch3d (see preprocess/requirements.txt).

Usage: python landmarks_mediapipe.py --data_dir /path/to/data_dir
"""

from __future__ import annotations

import glob
import os
import urllib.request

import mediapipe as mp
import numpy as np
import torch
import tyro
from pytorch3d.renderer import FoVOrthographicCameras, MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from skimage.io import imread
from tqdm import tqdm

MEDIAPIPE_FACES_PATH = os.path.join(os.path.dirname(__file__), "assets", "mediapipe_faces.npy")
CANONICAL_FACE_MODEL_URL = (
    "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/"
    "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
)

# pytorch3d camera space: +x left, +y up, +z forward. MediaPipe landmarks are
# image-space (x right, y down, origin top-left) with z in ~[-1, 1].
_ZNEAR, _ZFAR = 0.1, 5.0
_MIN_XY, _MAX_XY = -0.5, 0.5


def generate_mediapipe_faces(out_path: str = MEDIAPIPE_FACES_PATH, obj_path: str | None = None) -> np.ndarray:
    """Download/parse MediaPipe's canonical face model into the [898, 3] triangle topology."""
    if obj_path is not None:
        with open(obj_path) as f:
            obj_text = f.read()
    else:
        with urllib.request.urlopen(CANONICAL_FACE_MODEL_URL) as response:
            obj_text = response.read().decode("utf-8")

    faces = []
    for line in obj_text.splitlines():
        if not line.startswith("f "):
            continue
        idx = [int(tok.split("/")[0]) - 1 for tok in line.strip().split()[1:]]
        faces.append(idx)
    faces = np.array(faces, dtype=np.int64)

    n_verts = int(faces.max()) + 1
    if n_verts != 468:
        raise ValueError(f"Expected a 468-vertex face model, got {n_verts} vertices")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, faces)
    return faces


def get_visible_vertices_mask(landmarks, mediapipe_faces: np.ndarray, image_size: int, device: str) -> np.ndarray:
    """[468] bool mask via pytorch3d mesh rasterization: True where a vertex sits on a visible face."""
    points = np.array([(1 - p.x - 0.5, 1 - p.y - 0.5, 1.5 + p.z) for p in landmarks], dtype=np.float32)
    meshes = Meshes(verts=[torch.from_numpy(points)], faces=[torch.from_numpy(mediapipe_faces)])

    cameras = FoVOrthographicCameras(
        device=device, znear=_ZNEAR, zfar=_ZFAR,
        min_x=_MIN_XY, max_x=_MAX_XY, min_y=_MIN_XY, max_y=_MAX_XY,
    )
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    pix_to_face = rasterizer(meshes.to(device)).pix_to_face.detach().cpu().numpy()

    packed_faces = meshes.faces_packed()
    vertex_visibility = torch.zeros(meshes.verts_packed().shape[0], dtype=torch.bool)
    visible_faces = np.unique(pix_to_face)
    visible_faces = visible_faces[visible_faces != -1]
    if visible_faces.size > 0:
        vertex_visibility[torch.unique(packed_faces[visible_faces])] = True
    return vertex_visibility.numpy()


def _discover_video_dirs(data_dir: str, images_subdir: str) -> list[str]:
    video_dirs = []
    for name in sorted(os.listdir(data_dir)):
        images_dir = os.path.join(data_dir, name, images_subdir)
        if glob.glob(os.path.join(images_dir, "*.png")) or glob.glob(os.path.join(images_dir, "*.jpg")):
            video_dirs.append(os.path.join(data_dir, name))
    return video_dirs


def process_video(video_dir: str, face_mesh, mediapipe_faces: np.ndarray, images_subdir: str, device: str) -> list[str]:
    """Detect landmarks + visibility for un-processed frames. Returns failed image paths."""
    images_dir = os.path.join(video_dir, images_subdir)
    out_dir = os.path.join(video_dir, "mediapipe")
    os.makedirs(out_dir, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png")))

    failed = []
    for frame_file in frame_files:
        tag = os.path.splitext(frame_file)[0]
        out_lmks_path = os.path.join(out_dir, f"{tag}.npy")
        out_vis_mask_path = os.path.join(out_dir, f"{tag}_vis_mask.npy")
        if os.path.exists(out_lmks_path) and os.path.exists(out_vis_mask_path):
            continue

        img_path = os.path.join(images_dir, frame_file)
        try:
            img = imread(img_path)
            multi_face_landmarks = face_mesh.process(img).multi_face_landmarks
            if not multi_face_landmarks:
                raise ValueError("No face detected")
            # refine_landmarks=True adds 10 iris points (468:478) beyond the base 468-vertex mesh; drop them.
            landmarks = multi_face_landmarks[0].landmark[:468]

            visibility_mask = get_visible_vertices_mask(landmarks, mediapipe_faces, img.shape[1], device)
            mp_lmks = np.array([[p.x, p.y] for p in landmarks], dtype=np.float32)

            np.save(out_lmks_path, mp_lmks)
            np.save(out_vis_mask_path, visibility_mask)
        except Exception:
            failed.append(img_path)
            continue

    return failed


def main(
    data_dir: str,
    images_subdir: str = "images",
    device: str = "cuda:0",
    part: int = 0,
    total_parts: int = 1,
) -> None:
    """Extract MediaPipe landmarks + visibility masks for every video in data_dir."""
    if not os.path.isfile(MEDIAPIPE_FACES_PATH):
        print(f"{MEDIAPIPE_FACES_PATH} missing, generating from MediaPipe's canonical face model...")
        generate_mediapipe_faces()
    mediapipe_faces = np.load(MEDIAPIPE_FACES_PATH)

    video_dirs = _discover_video_dirs(data_dir, images_subdir)[part::total_parts]
    if not video_dirs:
        print("No video folders to process for this shard.")
        return

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    failed_images: list[str] = []
    for video_dir in tqdm(video_dirs, desc="Videos"):
        failed_images.extend(process_video(video_dir, face_mesh, mediapipe_faces, images_subdir, device))

    if failed_images:
        print(f"{len(failed_images)} frame(s) failed (no face detected / unreadable), e.g.: {failed_images[:5]}")


if __name__ == "__main__":
    tyro.cli(main)
