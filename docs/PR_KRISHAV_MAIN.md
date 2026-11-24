# PR: Krishav-main -> main - VACE pipeline, ablation, era styling, Larwood vs Bradman

## Summary
Adds a second, free generation pipeline next to the team's AnimateDiff/MagicAnimate engine:
- pose extraction: DWPose via rtmlib, on CPU;
- skeleton or depth controls;
- Wan2.1-VACE-1.3B in headless ComfyUI on a free Kaggle T4;
- retime to real speed;
- 1930s / 1970s / 1990s era filters;
- metrics and final videos.

It also covers the likeness/bat ablation, the batter final (F1 `final_cap`), the Larwood bowler pipeline and the report materials.

**No existing file on `main` is modified except `.gitignore`**, which gets appended entries scoped to this pipeline's work folders (`data/clips`, `data/pose`, `data/gen`, ...). Teammates' tracked `data/` files are unaffected. The branch is based on `main` and merges without conflicts as of this writing.

## What's added
| Area | Files |
|---|---|
| Pose + controls | `scripts/extract_pose.py`, `make_control.py`, `make_depth_control.py` (Depth Anything V2 Small, `--soften-head`, `--target-only`), `pose_utils.py` |
| Reference photos | `scripts/prepare_ref.py` (rembg, tight crop on white) |
| Generation | `kaggle/vace_generate.ipynb` (Kaggle T4 / Colab, headless ComfyUI, `RUN_MODE` = `pairs` / `finals`), `scripts/build_kaggle_inputs.py` (presets `batter-v2`, `bowler`) |
| Post | `scripts/retime.py` (default `blend`; `hold` for slow motion), `era_filters.py`, `make_sequence.py`, `make_final.py`, `make_match.py` (`--bowler-era` / `--batter-era`), `make_catalog.py`, `make_presentation.py` |
| Evaluation | `scripts/eval_metrics.py` (PCK@0.2, DINOv2 identity, flicker), `score_ablation.py`, `era_eval.py` (warping error), `results/ablation.csv`, `results/ablation_runs.csv`, `results/era_eval.csv` |
| Docs | `docs/CATALOG.md`, `docs/VIDEO_NARRATION.md`, `README_PIPELINE.md`, `CLAUDE.md`, `requirements-pipeline.txt`, `docs/REPORT_NOTES.md`, `docs/TEAM_INTEGRATION.md`, `docs/research/*`, `docs/report/` (`main.tex`, `refs.bib`, `figs/`), `docs/VIDEO_SCRIPT.md`, `scripts/make_figures.py` |

## Key results
- **Depth control fixes the bat.** One bat for the whole clip in 5/5 depth runs; the bat was lost in 2/2 skeleton runs. Depth also lowers brightness flicker. PCK@0.2 is 0.90-1.00 for all runs.
- **A tighter reference crop did not clearly raise identity.** Seed variation (D1-D4) is larger than the reference effect.
- **Batter final:** F1 `final_cap` (seed 42), chosen visually.
- **Bowler (Larwood, L1-L3):**
  - All three are bare-headed, with no fedora, cigarette or suit, no extra people (target-only depth) and no bat.
  - Chosen: **L3** (seed 2024), for the most 1930s kit and the lowest flicker. It is shown at **2x slow motion** (frame holds; blending ghosted).
  - PCK@0.2 0.79-0.85. Each 33-frame clip took ~520 s on the Kaggle T4, with a 12.8-14.9 GB peak.
- **Milestone 4:**
  - Era filters keep the motion: batter PCK -0.007 to -0.08, bowler up to -0.12. The 1930s look deliberately adds flicker.
  - Cross-era catalog: one match per era, `match_crossera.mp4` (1930s Larwood vs 1990s Bradman) and a 46 s reel (`docs/CATALOG.md`).

## How to test
```powershell
py -3.12 -m venv C:\cefc\venv; C:\cefc\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-pipeline.txt        # + winget install Gyan.FFmpeg
python scripts\make_final.py --input data\gen\v3_raw\final_cap_s42_33f.mp4 --tag _cap
python scripts\score_ablation.py
```
Generated videos, model weights and `.npz` files are **not** in git (shared via Drive).

## Before making the repo public: please review (files already on `main`, not added by this PR)
- **Broadcast-derived media.** `data/processed_clips/*/source_512.mp4` and `control_pose*.mp4` (crops of broadcast footage), plus `data/student3/bradman_1930s_graded_master.mp4` and `data/student3_improved/bradman_1930s_improved_sam2.mp4` (generated videos of a real person). Consider removing them, or keeping them in Drive.
- **Large JSON files.** `data/processed_clips/good_cover_drive*/pose_data.json` (6.4 MB each).
- **Redundant zip.** `student3.zip` duplicates the `student3/` code and contains `__pycache__/*.pyc` files.
- **Secret scan.** No `kaggle.json`, API tokens, keys or passwords were found in any tracked file.
- Removing a file only from the latest commit leaves it in the git history. Rewriting history needs the team's agreement and **force-pushing `main`, which this PR does not do**.

## Checklist
- [ ] Team agrees on merging into `main`
- [ ] Individual contributions in `docs/report/main.tex` confirmed
- [ ] Media on `main` reviewed before the repo goes public

🤖 Generated with [Claude Code](https://claude.com/claude-code)
