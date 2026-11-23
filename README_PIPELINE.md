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

## Relation to team engine
- **Student 2's AnimateDiff engine on `main`** (`generation_engine.py`, `generate_cli.py`) is our **baseline**.
- **VACE (this pipeline)** is the **new generator**.
- Both are compared on the **same cover-drive clip**, with the same reference photo and the same metrics, so the report can show the difference directly.
- We do not modify the team's files; this pipeline lives in `scripts/`, `kaggle/` and the `*_PIPELINE`/`-pipeline` docs.

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

**3. Generation (Kaggle, `kaggle/vace_generate.ipynb`).**
1. Make a folder `cefc-inputs/` with `refs/` (photos), `controls/` (the `control_33.mp4` files) and `pairs.csv` (`name,ref,control[,prompt]`).
2. Kaggle -> Datasets -> New Dataset -> upload that folder (Private).
3. New Notebook -> File -> Import Notebook -> `kaggle/vace_generate.ipynb` -> Add Input -> your dataset.
4. Settings: Accelerator **GPU T4 x2**, Internet **On** (phone verification needed), keep it Private.
5. Run cells top to bottom until the 17-frame smoke test passes, then **Save Version -> Save & Run All**.

Outputs: `outputs/*.mp4` (labelled "AI-generated - academic project"), `outputs_raw/*.mp4` (no label, for metrics), `outputs.zip`, `kaggle_runs.csv` (time and peak VRAM per run). Download them into `data/gen/`.

**4. Era styling** (use the UNLABELLED `outputs_raw/` videos; the label is added by this script):

```powershell
python scripts\era_filters.py --input data\gen\outputs_raw\batter_33f.mp4 --preset 1930s_film
python scripts\era_filters.py --input data\gen\outputs_raw --preset all --out-dir data\styled
```

Presets: `1930s_film` (grayscale, grain, blur, gate jitter, flicker, vignette, 18 fps; `--sepia` for a warm tint), `1970s_tv` (desaturation, colour fringing, scanlines, tape noise, 25 fps), `1990s_broadcast` (mild colour, chroma bleed, softness, 25 fps). Seeded, so the same input and `--seed` always give the same output. Output: `data\styled\<name>_<preset>.mp4`.

**5. Metrics** (CPU; identity needs CPU torch + transformers + pillow, see `requirements-pipeline.txt`):

```powershell
python scripts\eval_metrics.py --name batter_vace --video data\gen\outputs_raw\batter_33f.mp4 `
  --styled data\styled\batter_33f_1930s_film.mp4 data\styled\batter_33f_1970s_tv.mp4 `
  --control data\pose\batter\control_33.npz --ref-photo data\refs\bradman.png
```

Appends one row per video to `results/metrics.csv` (commit it): `pck@0.2` (17 body keypoints vs the control), `dino_mean` / `dino_min` (identity vs the photo, grayscale), `brightness_std` and `frame_diff_mean` (flicker). Run the same command on the AnimateDiff baseline output with a different `--name` to compare.

**6. Final sequence** (needs ffmpeg):

```powershell
python scripts\make_sequence.py --clips data\styled\bowler_33f_1970s_tv.mp4 data\styled\batter_33f_1990s_broadcast.mp4 `
  --cards "Holding bowls" "Tendulkar drives" --intro "Michael Holding vs Sachin Tendulkar" --out data\final\final.mp4
```

All parts are letterboxed to the first clip's size and 25 fps (`--size`, `--fps` to change); title cards carry the AI-generated label.

## Background
- `README.md`: the teammates' original engine (AnimateDiff-based)
- `docs/PROJECT_PROPOSAL.md`: the original proposal.
- `docs/research/`: Phase 0 and 0.1 research summaries.
- `CLAUDE.md`: the final technical decisions.
