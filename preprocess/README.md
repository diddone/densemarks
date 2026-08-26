# Preprocessing scripts

Everything needed to go from raw videos to the `densemarks/data.py` data
layout lives in this folder (scripts, generated assets, requirements).
Commands below assume the repo root as the working directory.

- `track_cotracker.py` — raw videos -> `images/{frame}.jpg`, `pred_tracks.pt`,
  `pred_visibility.pt`. Optional `--use_gsam_mask` seeds tracking from a
  GSAM2 person mask instead of a plain grid (see two-pass workflow below).
- `segment_gsam2.py` — `images/` -> `gsam/{frame}.png` person masks
  (Grounding DINO + SAM 2, prompt `"person."`). Feeds `track_cotracker.py
  --use_gsam_mask`.
- `landmarks_mediapipe.py` — `images/` -> `mediapipe/{frame}.npy` +
  `_vis_mask.npy`. Auto-generates `assets/mediapipe_faces.npy` on first run.
- `visualize_processed_data.ipynb` (source: `visualize_processed_data.py`,
  jupytext py:percent — edit the `.py`, regenerate the notebook with
  `jupytext --to notebook visualize_processed_data.py`) — QA/visualization
  of already-produced outputs: tracks + visibility overlay, landmark
  overlay, gsam mask overlay, shape/dtype sanity checks. Doesn't reprocess
  anything.
- `requirements.txt` — single env for all three scripts.

## Setup

All three scripts run from one venv, `preprocess/.venv` (python 3.10, torch
2.3.1+cu121 — needed for the prebuilt pytorch3d wheel, and satisfies
GSAM2's `transformers<5` constraint).

```bash
uv venv preprocess/.venv --python 3.10
uv pip install --python preprocess/.venv/bin/python \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python preprocess/.venv/bin/python \
    "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt231/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl"
uv pip install --python preprocess/.venv/bin/python -r preprocess/requirements.txt
```

SAM 2 is not a pip package — clone Grounded-SAM-2, grab a checkpoint, then
install it editable into the same venv:

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git ~/tmp/gsam2_work/Grounded-SAM-2
cd ~/tmp/gsam2_work/Grounded-SAM-2/checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
uv pip install --python preprocess/.venv/bin/python -e ~/tmp/gsam2_work/Grounded-SAM-2
```

(`checkpoints/download_ckpts.sh` in that repo fetches all four SAM2.1 sizes —
tiny through large — but only the tiny one is verified working here, via the
`--sam2_checkpoint`/`--sam2_config` overrides below.)

(Grounding DINO itself loads via HF transformers — `grounding-dino-tiny`
auto-downloads on first run, no local config/checkpoint needed.)

## Running

```bash
preprocess/.venv/bin/python preprocess/track_cotracker.py \
    --video_dir local_dev/tmp/raw3 --data_dir local_dev/tmp/dataset --device cuda

preprocess/.venv/bin/python preprocess/segment_gsam2.py \
    --data_dir local_dev/tmp/dataset --gsam2_repo ~/tmp/gsam2_work/Grounded-SAM-2 \
    --sam2_checkpoint checkpoints/sam2.1_hiera_tiny.pt \
    --sam2_config configs/sam2.1/sam2.1_hiera_t.yaml --device cuda

preprocess/.venv/bin/python preprocess/landmarks_mediapipe.py \
    --data-dir local_dev/tmp/dataset --device cuda:0
```

(paths above put working data under `local_dev/` — gitignored — since raw
videos and derived tracks/masks shouldn't be committed; swap in whatever
`--data_dir` you actually want.)

(`landmarks_mediapipe.py` uses a hyphenated `--data-dir` flag; the other two
use underscores — tyro derives flags from each script's own arg names.)

## GSAM2 two-pass workflow

`--use_gsam_mask` needs `gsam/` to already exist, so masking is a two-pass
loop over the same `data_dir`:

1. `track_cotracker.py --use_gsam_mask ...` — writes `images/`, skips
   tracking for any video without a mask yet.
2. `segment_gsam2.py --data_dir ... --gsam2_repo ...` — fills in
   `gsam/{frame}.png` for videos that have `images/`.
3. Rerun step 1 — videos with a mask now get tracked; those still missing
   one (no "person" detected) are skipped again.

## Train/val split

None of the scripts generate a split. Once preprocessing is done, please create
`<data_dir>/train_tracked_points_paths.json` and
`val_tracked_points_paths.json` yourself — see the top-level
[README](../README.md#data-layout) for the format.

## Visualization notebook

```bash
jupytext --to notebook preprocess/visualize_processed_data.py   # after editing the .py
jupyter nbconvert --to notebook --execute preprocess/visualize_processed_data.ipynb
```

## Code style

Write concise comments, only where the code isn't self-explanatory. No
restating what a line obviously does, no multi-paragraph docstrings.
