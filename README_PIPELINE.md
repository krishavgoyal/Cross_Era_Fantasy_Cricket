# Cross-Era Fantasy Cricket - VACE generation pipeline

> This is the deadline pipeline (rtmlib DWPose on CPU -> Wan2.1-VACE-1.3B on a free Kaggle T4 -> era filters -> metrics -> final edit). The teammates' Student 1/2/3 engine is documented in `README.md`; this file covers only the new pipeline.

Animate a historical cricketer from a single reference photo, using motion extracted from a modern cricket clip. Then style the result like footage from the player's era.

> All generated videos are **AI-generated, for academic use only**. They are not real footage.

## Pipeline

```
 modern clip (.mp4)                        reference photo (.png/.jpg)
        │                                            │
 [1] extract_pose.py   (laptop CPU, rtmlib DWPose)   │
        │  raw.npz + overlay.mp4                     │
        ▼                                            │
 [2] make_control.py   (laptop CPU) ◄────────────────┤  --ref-photo: per-bone retarget
        │  control_33.mp4 (480x832, 16 fps, 33 frames, OpenPose skeleton on black)
        │  + control_33.npz (keypoints for metrics)
        ▼                                            ▼
 [3] kaggle/vace_generate.ipynb   (Kaggle T4, ComfyUI, Wan2.1-VACE-1.3B)
        │  gen_<pair>.mp4 + kaggle_runs.csv (time, peak VRAM)
        ▼
 [4] era_filters.py    (laptop CPU)  1930s_film | 1970s_tv | 1990s_broadcast
        ▼
 [5] eval_metrics.py   (laptop CPU)  PCK@0.2, DINOv2 identity, flicker → results/metrics.csv
        ▼
 [6] make_sequence.py  (ffmpeg)      bowler clip + batter clip (+ title cards) → final.mp4
```

The bowler and the batter are generated as **separate clips** and cut together, the way a TV broadcast cuts between them.

## Setup (laptop, once)

```powershell
cd "C:\Users\ASUS V16\OneDrive\Desktop\Cross_Era_Fantasy_Cricket"
py -3.12 -m venv C:\cefc\venv            # keep the venv outside OneDrive
C:\cefc\venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-pipeline.txt
```

`ffmpeg` is optional for stages 1–2. If it is on PATH, videos are re-encoded to H.264. It is required for stage 6 (`winget install Gyan.FFmpeg`).

## Run each stage

Put your inputs in `data/` (git-ignored), for example `data/clips/bowler.mp4`, `data/clips/batter.mp4`, `data/refs/holding.png`.

**1. Pose extraction.** First list the people detected on frame 0 so you can pick the target player:

```powershell
python scripts\extract_pose.py --video data\clips\batter.mp4 --out data\pose\batter --list-boxes
# open data\pose\batter\frame0_boxes.jpg, choose the player's box, then:
python scripts\extract_pose.py --video data\clips\batter.mp4 --out data\pose\batter --init-box 410,120,180,460
```

Output: `data\pose\batter\raw.npz`, plus `overlay.mp4`, which you should check by eye.

**2. Control video.**

```powershell
python scripts\make_control.py --pose data\pose\batter\raw.npz --out data\pose\batter --start 0.5 --ref-photo data\refs\tendulkar.png
```

Output: `control_33.mp4`, `control_33.npz` and `control_33_preview.jpg`. Options:
- `--frames` (default 33)
- `--duration` (seconds of source squeezed into the clip)
- `--no-face`, `--no-hands`

**3. Generation (Kaggle).** See `kaggle/` (task 3).

**4–6.** `era_filters.py`, `eval_metrics.py` and `make_sequence.py` (tasks 4–6).

## Background
- `README.md`: the teammates' original engine (AnimateDiff-based)
- `docs/PROJECT_PROPOSAL.md`: the original proposal.
- `docs/research/`: Phase 0 and 0.1 research summaries.
- `CLAUDE.md`: the final technical decisions.
