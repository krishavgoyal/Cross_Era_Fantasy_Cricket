# Cross-Era Fantasy Cricket: notes for Claude Code

Modern cricket clip + historical player photo → the same motion performed by the historical player,
in 1930s film style. Team: Student 1 motion (pose, DensePose), Student 2 generation (Ahaan),
Student 3 colour and evaluation (Krishav), integration (Pranahv, this branch: `Pranahv-main`).

## Layout
- `master_pipeline.py`: one command, 7 stages (motion, densepose, generate, colour, retime, demo, evaluate). Resumable; writes `runs/<run>/manifest.json`. See `INTEGRATION.md`.
- `generation_engine.py`, `generate_cli.py`: Student 2 (SD 1.5 + AnimateDiff + DensePose ControlNet + IP-Adapter). Needs a CUDA GPU.
- `student3_v3/`: default colour stage. `student3_improved/` is v2.
- `densepose/`: how the DensePose data in `data/processed_clips/*` was made (detectron2, separate env).
- `integration/eval_pck.py`: motion score (needs `easy-dwpose`).
- `colab/run_on_colab.sh`: runs the pipeline on a Google Colab GPU (see below).
- `kaggle_runner/run_on_kaggle.sh`: same pipeline as a private Kaggle kernel on a free T4 (needs `~/.kaggle/kaggle.json`); results in `kaggle_results/`.
- `docs/DISTORTIONS_AND_ALTERNATIVES.md`: why the output distorts and what to try next.
- `animate/`: "photo of a player performs the motion" (MusePose): skeleton + tracked bat from a motion clip,
  retargeted onto the player photo placed on a ground photo. `python -m animate.build_jobs`, then
  `KERNEL=cricket-animate EXTRA_DATA=animate/jobs kaggle_runner/run_on_kaggle.sh animate`; score with
  `python -m animate.evaluate`. See `docs/RUN_LOG.md`.
- `motion3d/`: rigid 3D skeleton per motion clip (DWPose + MediaPipe, fixed bone lengths, planted feet on a
  tracked ground, fixed-length bat, hand-keyframed blurred bowling arm), rendered from 6 angles; drives the
  `*_3d` MusePose jobs. `motion3d/run_all.sh`, then `python -m motion3d.build_jobs`. See `docs/MOTION3D.md`.
  MediaPipe runs in `~/pose3d` (needs `LD_LIBRARY_PATH=~/pose3d/sysroot/usr/lib/x86_64-linux-gnu`).
- `data/README.md`: data layout (reference_players/, backgrounds/, motion_videos/), sources in `data/SOURCES_commons.md`.

## Tests (CPU, no models)
```bash
python -m pytest -q tests --ignore=tests/test_renderer.py
PYTHONPATH=. python -m pytest -q student3_v3/tests
```

## Running on Google Colab (preferred GPU)
The user has Colab Pro. The laptop GPU (RTX 5060, 8 GB) works but limits model size.
1. One-time: `uv tool install google-colab-cli`, then `colab sessions` and complete the browser login
   with the user's Colab Pro Google account (the user does this step; it needs a human).
2. Run: `GPU=A100 colab/run_on_colab.sh both` (or `bowler` / `batsman`). Run it in the background and
   check its output every few minutes; it prints the VM log tail every minute.
3. Results land in `colab_results/runs/colab_bowler/` and `colab_results/runs/colab_batsman/`
   (`final_24fps.mp4`, `demo_side_by_side.mp4`, `student3_v3/era_*.mp4`, `evaluation.json`, `manifest.json`).

Rules for Colab:
- A running VM costs compute units. The script stops the VM when it ends; if you interrupt it or
  use `KEEP=1`, run `colab stop -s cricket` yourself. Check `colab usage` before starting.
- Never run `colab auth`, `colab drivemount`, `colab repl` or bare `colab ssh` yourself; they need a human.
- `colab new` returning 400 for a GPU means no quota for it: retry with `GPU=L4` or `GPU=T4`.
- First run in a VM installs a Python 3.11 env and downloads about 6 GB of models (several minutes).
- If the user ran `colab drivemount -s cricket` in their own terminal, models are cached on Drive.
- Debug a failed run with `colab_results/job.log`, then `colab log -s cricket -n 30` if the VM is still up.

## Running locally (WSL, RTX 5060)
See `RUN_ON_GPU.md`. WSL needs `.wslconfig` with memory=12GB, swap=16GB or generation is killed silently.

## Conventions
- Do not delete other students' files; add new modules instead of rewriting theirs.
- `--generated <video>` runs everything after generation on an existing video (no GPU needed).
- Label any output with a real player as an AI reconstruction.
