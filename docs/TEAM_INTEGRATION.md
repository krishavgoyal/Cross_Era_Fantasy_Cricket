# Team integration: scoring the AnimateDiff baseline against VACE

Student 2's AnimateDiff engine (`generation_engine.py` / `generate_cli.py` on `main`) is the **baseline**. VACE (this pipeline) is the **new generator**. Both are scored with the same script, `scripts/eval_metrics.py`, and the same reference photo.

## What can be compared directly

| Metric | AnimateDiff | VACE | Comparable? |
|---|---|---|---|
| Identity (DINOv2 mean/min vs the reference photo) | yes | yes | **Yes**: same photo, same model, same crop rule |
| Flicker (brightness std, frame diff) | yes | yes | **Mostly**: AnimateDiff output repeats held frames (16 keyframes in 26 frames), which lowers its frame-difference number |
| PCK@0.2 | different canvas and timeline | vs `control_33.npz` | **Not with one control**: the AnimateDiff clip is 384x512, 26 frames at 7 fps, cropped by Student 2's geometry; our control is 480x832, 33 frames at 16 fps. Run `eval_metrics.py` **without `--control`** for it (PCK skipped) and take its PCK from Student 3's evaluator (same definition, see step 5) |

## Steps (PowerShell, repo root, `C:\cefc\venv` active)

**1. Get the AnimateDiff output from the teammates' branch** (it is not on `main`). Use `git restore`, not `git show > file`: PowerShell 5.1 corrupts binary output when redirecting.

```powershell
git fetch origin
git restore --source=origin/Ahaan-main --worktree -- data/generated_clips/test_run/m3_magicanimate.mp4
New-Item -ItemType Directory -Force data\gen\baseline | Out-Null
Move-Item data\generated_clips\test_run\m3_magicanimate.mp4 data\gen\baseline\animatediff_cover_drive.mp4
```

**2. Use the same reference photo and the same clip for both.** Ask Student 2 which reference image `generate_cli.py` used and save it as `data\refs\bradman.png` (or use that file for both). For the fairest comparison, drive VACE with the **same** cover-drive clip the baseline used (tracked on `main`):

```powershell
python scripts\extract_pose.py --video data\processed_clips\good_cover_drive_final\source_512.mp4 --out data\pose\cover_drive --list-boxes
python scripts\extract_pose.py --video data\processed_clips\good_cover_drive_final\source_512.mp4 --out data\pose\cover_drive --init-box X,Y,W,H
python scripts\make_control.py --pose data\pose\cover_drive\raw.npz --out data\pose\cover_drive --ref-photo data\refs\bradman.png
# upload data\pose\cover_drive\control_33.mp4 to the Kaggle dataset, run the notebook, download outputs_raw\<name>_33f.mp4
```

**3. Style both with the same preset:**

```powershell
python scripts\era_filters.py --input data\gen\baseline\animatediff_cover_drive.mp4 --preset 1930s_film --out-dir data\styled
python scripts\era_filters.py --input data\gen\outputs_raw\bradman_batting_33f.mp4 --preset 1930s_film --out-dir data\styled
```

**4. Score both into the same CSV:**

```powershell
# AnimateDiff baseline: identity + flicker (no --control -> PCK skipped)
python scripts\eval_metrics.py --name animatediff_baseline --video data\gen\baseline\animatediff_cover_drive.mp4 `
  --styled data\styled\animatediff_cover_drive_1930s_film.mp4 --ref-photo data\refs\bradman.png

# VACE: identity + flicker + PCK against its control
python scripts\eval_metrics.py --name vace --video data\gen\outputs_raw\bradman_batting_33f.mp4 `
  --styled data\styled\bradman_batting_33f_1930s_film.mp4 `
  --control data\pose\batter\control_33.npz --ref-photo data\refs\bradman.png
```

Both append to `results/metrics.csv`. Expected console output: one table row per video. For the baseline, `PCK@0.2` shows `nan` (by design) and `frames_sampled` = 26.

**5. Baseline PCK (optional, Student 3's tool).** Student 3's evaluator (`student3_v3/evaluate.py`, in the earlier Student 3 package) maps Student 1's reference keypoints through Student 2's crop geometry and held-frame layout, which our script cannot know. It reports PCK@0.2 normalised by torso length (neck to mid-hip), the same definition as ours. Its README gives the command: `pose_engine.py` on the generated clip, then `python -m student3_v3.evaluate --processing ... --reference-pose ... --generated-pose ... --report pck.json`. Report its `pck@0.2` next to the VACE PCK, with a note that each model is scored against its own control.

## Checklist for a fair table

- Same reference photo for both (`--ref-photo`).
- Same source clip (`good_cover_drive_final/source_512.mp4`) driving both.
- Same era preset and seed when comparing styled versions.
- State in the report: resolution (384x512 vs 480x832), frames/fps (26 @ 7 fps with held frames vs 33 @ 16 fps), hardware (AnimateDiff on the team's GPU vs VACE on a Kaggle T4).
