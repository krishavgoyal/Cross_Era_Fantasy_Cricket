#!/usr/bin/env bash
# Runs ON the Kaggle/Colab GPU VM (via colab/remote_run.sh animate). Do not run locally.
# For every animate/jobs/<name>/ (ref.png + pose.mp4): MusePose makes the player in ref.png perform the
# skeleton motion in pose.mp4. Writes /content/results.zip (animate/<name>.mp4 + grid) and PIPELINE_DONE rc=N.
#
# Env: MP_STEPS (20) MP_CFG (3.5) MP_SLICE (24) MP_OVERLAP (4) MP_SEED (99) MP_W (512) MP_H (672) JOBS ("" = all)
set -uo pipefail
log() { echo "[$(date +%H:%M:%S)] $*"; }
REPO=/content/cricket; MP=/content/MusePose; PY=/content/mp/bin/python; OUT=/content/results/animate
STEPS="${MP_STEPS:-20}"; CFG="${MP_CFG:-3.5}"; S="${MP_SLICE:-24}"; O="${MP_OVERLAP:-4}"; SEED="${MP_SEED:-99}"
W="${MP_W:-512}"; H="${MP_H:-672}"
log "STAGE gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"

if [ ! -x "$PY" ] || ! "$PY" -c "import diffusers, av, huggingface_hub" 2>/dev/null; then
  log "STAGE install MusePose env (python 3.10, torch 2.0.1 cu118, diffusers 0.27.2)"
  # av 11.0.0 (MusePose's pin) has no Python 3.10 wheel and fails to build without FFmpeg headers: use 12.3.0.
  # decord is only used by MusePose's training datasets, not by inference.
  { pip install -q uv \
    && uv venv -q --allow-existing /content/mp --python 3.10 \
    && uv pip install -q --python "$PY" torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 \
    && uv pip install -q --python "$PY" "diffusers==0.27.2" "transformers==4.33.1" "accelerate==0.29.3" \
         "huggingface_hub==0.20.3" "tokenizers<0.14" einops==0.4.1 omegaconf==2.2.3 "av==12.3.0" \
         imageio==2.33.0 imageio-ffmpeg==0.4.9 "opencv-python-headless==4.8.1.78" "numpy<2" safetensors pandas tqdm; } \
    || { echo "PIPELINE_DONE rc=3 (install)"; exit 3; }
fi
"$PY" -c "import torch, diffusers, transformers, av; print('env ok: torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'diffusers', diffusers.__version__)" \
  || { echo "PIPELINE_DONE rc=3 (env check)"; exit 3; }
[ -d "$MP" ] || git clone -q --depth 1 https://github.com/TMElyralab/MusePose "$MP"

log "STAGE weights"
export HF_HUB_DISABLE_XET=1
"$PY" - <<'EOF' || { echo "PIPELINE_DONE rc=3 (weights)"; exit 3; }
import os, shutil
from huggingface_hub import hf_hub_download, snapshot_download
pw = "/content/MusePose/pretrained_weights"
os.makedirs(f"{pw}/MusePose", exist_ok=True)
for f in ("denoising_unet.pth", "reference_unet.pth", "pose_guider.pth", "motion_module.pth"):
    p = hf_hub_download("TMElyralab/MusePose", f"MusePose/{f}")
    if not os.path.exists(f"{pw}/MusePose/{f}"):
        os.symlink(p, f"{pw}/MusePose/{f}")
# lambdalabs/sd-image-variations-diffusers is no longer public. MusePose only takes the UNet *architecture*
# from it (its own .pth weights replace all UNet weights; reference_unet is loaded strictly), and SD 1.5's
# UNet has the same architecture. Its image encoder is OpenAI CLIP ViT-L/14, which is public.
unet = snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
                         allow_patterns=["unet/config.json", "unet/diffusion_pytorch_model.safetensors"])
os.makedirs(f"{pw}/sd-image-variations-diffusers", exist_ok=True)
if not os.path.exists(f"{pw}/sd-image-variations-diffusers/unet"):
    os.symlink(f"{unet}/unet", f"{pw}/sd-image-variations-diffusers/unet")
clip = snapshot_download("openai/clip-vit-large-patch14", allow_patterns=["config.json", "model.safetensors",
                         "preprocessor_config.json"])
if not os.path.exists(f"{pw}/image_encoder"):
    os.symlink(clip, f"{pw}/image_encoder")
vae = snapshot_download("stabilityai/sd-vae-ft-mse", allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
if not os.path.exists(f"{pw}/sd-vae-ft-mse"):
    os.symlink(vae, f"{pw}/sd-vae-ft-mse")
print("weights ready")
EOF

log "STAGE config"
mkdir -p "$OUT"
CFGF="$MP/configs/cricket.yaml"
{
  echo 'pretrained_base_model_path: "./pretrained_weights/sd-image-variations-diffusers"'
  echo 'pretrained_vae_path: "./pretrained_weights/sd-vae-ft-mse"'
  echo 'image_encoder_path: "./pretrained_weights/image_encoder"'
  echo 'denoising_unet_path: "./pretrained_weights/MusePose/denoising_unet.pth"'
  echo 'reference_unet_path: "./pretrained_weights/MusePose/reference_unet.pth"'
  echo 'pose_guider_path: "./pretrained_weights/MusePose/pose_guider.pth"'
  echo 'motion_module_path: "./pretrained_weights/MusePose/motion_module.pth"'
  echo 'inference_config: "./configs/inference_v2.yaml"'
  echo "weight_dtype: 'fp16'"
  echo 'test_cases:'
} > "$CFGF"
N=0
for d in "$REPO"/animate/jobs/*/; do
  name="$(basename "$d")"
  if [ -n "${JOBS:-}" ] && ! grep -qw "$name" <<<"$JOBS"; then continue; fi
  [ -f "$d/ref.png" ] && [ -f "$d/pose.mp4" ] || { log "skip $name (ref.png/pose.mp4 missing)"; continue; }
  mkdir -p "$MP/cricket/$name"
  cp "$d/ref.png" "$MP/cricket/$name/$name.png"; cp "$d/pose.mp4" "$MP/cricket/$name/${name}_pose.mp4"
  echo "  \"./cricket/$name/$name.png\":" >> "$CFGF"
  echo "    - \"./cricket/$name/${name}_pose.mp4\"" >> "$CFGF"
  N=$((N + 1))
done
cat "$CFGF"
[ "$N" -gt 0 ] || { echo "PIPELINE_DONE rc=4 (no jobs)"; exit 4; }

log "STAGE generate ($N jobs, ${W}x${H}, steps $STEPS, cfg $CFG, slice $S overlap $O)"
cd "$MP"
RC=0
"$PY" test_stage_2.py --config ./configs/cricket.yaml -W "$W" -H "$H" -L 400 -S "$S" -O "$O" \
    --skip 0 --steps "$STEPS" --cfg "$CFG" --seed "$SEED" || RC=5
for f in "$MP"/output/*/*/*.mp4; do
  [ -e "$f" ] || continue
  cp "$f" "$OUT/"
done
ls -la "$OUT"
log "STAGE package"
cd /content && rm -f results.zip && (cd /content/results && zip -qr /content/results.zip animate) || RC=6
echo "PIPELINE_DONE rc=$RC"
