#!/usr/bin/env bash
# Real DensePose for one clip folder (needs source_512.mp4 + metadata.json in it).
#   bash densepose/run_densepose.sh data/processed_clips/<clip_id>
# Env overrides (defaults match the WSL setup we used):
#   DP_PROJECT  detectron2 DensePose project dir   (~/detectron2/projects/DensePose)
#   DP_WEIGHTS  model weights .pkl (rename .pkl.1)  (~/densepose_model.pkl)
#   DP_CONFIG   config inside DP_PROJECT            (configs/densepose_rcnn_R_50_FPN_s1x.yaml)
#   DP_DEVICE   cpu or cuda                         (cpu; RTX 50 series needs a cu128 torch build for cuda)
set -euo pipefail
CLIP="$(cd "$1" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DP_PROJECT="${DP_PROJECT:-$HOME/detectron2/projects/DensePose}"
DP_WEIGHTS="${DP_WEIGHTS:-$HOME/densepose_model.pkl}"
DP_CONFIG="${DP_CONFIG:-configs/densepose_rcnn_R_50_FPN_s1x.yaml}"
DP_DEVICE="${DP_DEVICE:-cpu}"
FPS="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['processed'].get('fps',24))" "$CLIP/metadata.json" 2>/dev/null || echo 24)"

[ -f "$CLIP/source_512.mp4" ] || { echo "source_512.mp4 not found in $CLIP" >&2; exit 4; }
[ -f "$CLIP/metadata.json" ] || { echo "metadata.json not found in $CLIP" >&2; exit 4; }
[ -f "$DP_WEIGHTS" ] || { echo "weights not found: $DP_WEIGHTS" >&2; exit 3; }
[ -d "$DP_PROJECT" ] || { echo "DensePose project not found: $DP_PROJECT" >&2; exit 3; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/frames"
ffmpeg -v error -i "$CLIP/source_512.mp4" -start_number 0 "$WORK/frames/%05d.png"
echo "frames: $(ls "$WORK/frames" | wc -l)"

( cd "$DP_PROJECT" && python apply_net.py dump "$DP_CONFIG" "$DP_WEIGHTS" "$WORK/frames/*.png" \
    --output "$WORK/dp_results.pkl" --opts MODEL.DEVICE "$DP_DEVICE" )

python "$HERE/dp_to_control.py" "$WORK/dp_results.pkl" "$CLIP" --metadata "$CLIP/metadata.json" --fps "$FPS"
python "$HERE/patch_weak_frames.py" "$CLIP" --fps "$FPS"
