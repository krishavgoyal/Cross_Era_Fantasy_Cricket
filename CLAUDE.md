# CLAUDE.md – Cross-Era Fantasy Cricket

## Deadline mode
Final submission: **Wed 25 Nov 2026, 11 PM IST**. Build a simple working pipeline and get results.
- No more web research, long plans or benchmark studies. The decisions below are FINAL.
- Look something up only when a real error forces it.
- Keep code simple, commented and runnable on CPU on the laptop.

## Machine (do not ask to change it)
- Windows 11, RTX 4050 Laptop **6 GB VRAM**, **16 GB RAM**, NVIDIA driver **561** (do NOT ask to update), Python **3.12**.
- Nothing heavy runs on the laptop GPU. The laptop handles pose, filters, metrics and editing on the **CPU**.

## Final decisions
| Stage | Choice |
|---|---|
| Pose | rtmlib `Wholebody` with the **DWPose-l 384x288** model, `device='cpu'`, CPU `onnxruntime` only. Do not use onnxruntime-gpu, and do not use torch for this step. |
| Player tracking | `--init-box x,y,w,h` on frame 0, then best-IoU tracking (we do not use rtmlib's PoseTracker) |
| Control video | OpenPose-style skeleton on black, **480x832 portrait, 16 fps, 33 frames** |
| Generator | **Wan2.1-VACE-1.3B** in **ComfyUI on a free Kaggle T4**: fp16, sdpa attention, core nodes only, seed 42, 20 steps, cfg 5, uni_pc/simple, shift 8 |
| Weights (Kaggle only) | `wan2.1_vace_1.3B_fp16.safetensors`, `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `wan_2.1_vae.safetensors` from `Comfy-Org/Wan_2.1_ComfyUI_repackaged` |
| Era filters | OpenCV + ffmpeg: `1930s_film`, `1970s_tv`, `1990s_broadcast`, seeded, with an "AI-generated – academic project" label |
| Metrics | CPU: PCK@0.2 (17 body keypoints vs the control), DINOv2-small identity cosine, flicker. Computed on raw and styled videos |
| Final video | Bowler clip and batter clip generated separately, then cut together with ffmpeg (title cards optional) |

## Reuse, don't rewrite
Earlier Student 3 code lives in `C:\Users\ASUS V16\Downloads\student3_v3_bowler\student3_v3\`:
- `era_filters.py`: the `newsreel_1930s` and `broadcast_1970s` presets and `retime`. Port these into `scripts/era_filters.py`.
- `evaluate.py`: `pck()`, the torso normaliser and Farneback `warping_error()`. Reuse them in `scripts/eval_metrics.py`.
- `colorize.py`: DensePose-based colourisation (not in the main pipeline).

## Repo layout
- Work happens on branch `Krishav-main` (branched from `origin/main`). Never force-push. The teammates' files at the repo root (`README.md`, `requirements.txt`, engine `.py` files) belong to them: do not edit them. Our docs are `README_PIPELINE.md` and `requirements-pipeline.txt`.
- `scripts/pose_utils.py`: keypoint layout, OpenPose drawing, video I/O, DWPose loader. Shared by all scripts.
- `scripts/extract_pose.py`: clip → `raw.npz` + `overlay.mp4`.
- `scripts/make_control.py`: `raw.npz` → `control_33.mp4` (+ `control_33.npz` for metrics), with optional `--ref-photo` retarget.
- `kaggle/vace_generate.ipynb`: the VACE generation notebook (task 3).
- `scripts/era_filters.py`, `scripts/eval_metrics.py`, `scripts/make_sequence.py` (tasks 4–6).
- `docs/research/`: short Phase 0 / 0.1 summaries. Background only; do not redo the research.
- v2/v3 additions: `prepare_ref.py` (tight ref on white), `make_depth_control.py` (Depth Anything V2 Small, aligned with `make_control.py`), `build_kaggle_inputs.py`, `score_ablation.py` (-> `results/ablation.csv`), `retime.py` (real speed, default `blend`; `mci` tears the bat), `make_final.py` (one command -> `data/final/final*.mp4`, `final_eras*.mp4`, `compare*.mp4`).
- Ablation result: **depth control fixes the bat**; the tight crop did not clearly raise identity; seed matters more than the reference. Finals: depth + `bradman_crop.png`, seed 42, 33 frames (`final_cap`, `final_nocap`).
- ffmpeg 9.1 is installed via winget; `pose_utils.find_ffmpeg()` also finds it when PATH is stale.

## Conventions
- Videos, weights and `.npz` files are git-ignored. Work data goes in `data/` (ignored).
- Keypoint arrays are COCO-WholeBody 133: 0–16 body, 17–22 feet, 23–90 face, 91–111 left hand, 112–132 right hand. Missing points are NaN.
- Label every generated output as AI-generated, for academic use. Real players' likeness is not to be used to mislead.
