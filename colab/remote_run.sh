#!/usr/bin/env bash
# Runs ON the Colab VM (started by colab/run_on_colab.sh). Do not run locally.
#   bash /content/cricket/colab/remote_run.sh [both|bowler|batsman]
# Writes /content/job.log (via the caller), /content/results.zip, and a final "PIPELINE_DONE rc=N" line.
set -uo pipefail
WHICH="${1:-both}"
EXTRA="${EXTRA_ARGS:-}"
log() { echo "[$(date +%H:%M:%S)] $*"; }
cd /content/cricket || { echo "PIPELINE_DONE rc=4 (repo missing)"; exit 4; }
# "animate": MusePose (photo + skeleton -> that player moving), separate env; see animate/remote_musepose.sh
if [ "$WHICH" = animate ]; then exec bash animate/remote_musepose.sh; fi

log "STAGE gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"

# Same isolated env as the laptop (Python 3.11 + torch cu128), so Colab's preinstalled
# transformers/huggingface-hub versions can't clash with diffusers 0.32.
if [ ! -x /content/gen/bin/python ]; then
  log "STAGE install (first run in this VM, about 3 to 5 min)"
  pip install -q uv
  uv venv -q /content/gen --python 3.11
  uv pip install -q --python /content/gen/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python /content/gen/bin/python -r requirements-generation.txt \
      "numpy<3" "opencv-python-headless<5" "Pillow<12" "scipy<2" onnxruntime "huggingface-hub<1.0"
  uv pip install -q --python /content/gen/bin/python --no-deps easy-dwpose==1.0.2
fi
source /content/gen/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
# Models: reuse a Drive cache if the user mounted Drive (colab drivemount), else VM disk.
if [ -d /content/drive/MyDrive ]; then export HF_HOME=/content/drive/MyDrive/cricket_hf_cache; else export HF_HOME=/content/hf_cache; fi
log "HF_HOME=$HF_HOME"
python -c "import importlib.metadata as m; print('versions:', *[p+'='+m.version(p) for p in ('torch','diffusers','transformers','accelerate','huggingface-hub')])" || true
python -c "import torch,sys; ok=torch.cuda.is_available(); print('cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 3)" \
  || { echo "PIPELINE_DONE rc=3 (no GPU)"; exit 3; }

RC=0
FORCE="--force generate colour retime demo evaluate"
# CLEAN=1: generation v2 (skeleton ControlNet + view direction) in clean mode, 32 frames with FreeNoise
# windows, frames repeated (not blended) to 24 fps, Student 3 clean configs. Run names get a _v2 suffix
# so the earlier outputs stay separate.
if [ "${CLEAN:-0}" = 1 ]; then
  SFX=_v2; GEN="--gen-engine v2 --gen-clean --frames 32 --retime-mode hold"
  CB="--colour-config student3_v3/configs/bowler_clean.json"; CT="--colour-config student3_v3/configs/batsman_clean.json"
else
  SFX=""; GEN=""; CB=""; CT=""
fi
want() { [ "$WHICH" = "$1" ] || [ "$WHICH" = all ] || { [ "$WHICH" = both ] && [ "$1" != bowler_new ]; }; }
if want bowler; then
  log "STAGE bowler"
  python master_pipeline.py --clip-id bowler_delivery_001 --image data/historical_images/modern_bowler.jpeg \
      --role bowler --auto-window $FORCE $GEN $CB --run-name colab_bowler$SFX $EXTRA || RC=$?
fi
if want batsman; then
  log "STAGE batsman"
  python master_pipeline.py --clip-id good_cover_drive_final --image data/historical_images/bradman.jpg \
      --role batsman $FORCE $GEN $CT --run-name colab_batsman$SFX $EXTRA || RC=$?
fi
if want bowler_new; then
  # New clip (Pat Cummins slow-motion, cropped to the player, 5.7-8.3 s) mapped onto bowler_new.jpg.
  # Frames 0-44 = gather to release (DensePose and skeleton are reliable there). --ref-box = player in the photo.
  log "STAGE bowler_new"
  python master_pipeline.py --clip-id bowler_new_cummins --image data/historical_images/bowler_new.jpg \
      --role bowler --window 0 44 --ref-box 0.06 0.13 0.96 0.95 $FORCE $GEN $CB --run-name colab_bowler_new$SFX $EXTRA || RC=$?
fi

log "STAGE package"
rm -f /content/results.zip
zip -qr /content/results.zip runs -x '*/_keyframes.mp4' || RC=6
echo "PIPELINE_DONE rc=$RC"
