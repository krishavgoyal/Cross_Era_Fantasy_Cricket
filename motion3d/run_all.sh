#!/usr/bin/env bash
# Rebuild every motion3d skeleton (CPU, ~15 min).   motion3d/run_all.sh [clip ...]
# Needs ~/gen (torch, easy-dwpose, ultralytics, simple-lama) and ~/pose3d (mediapipe, see docs/MOTION3D.md).
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="$HOME/pose3d/sysroot/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"  # libGLESv2 for mediapipe
CLIPS=("$@")
[ ${#CLIPS[@]} -gt 0 ] || mapfile -t CLIPS < <(~/gen/bin/python -c "from motion3d.clips import CLIPS; print('\n'.join(CLIPS))")
for c in "${CLIPS[@]}"; do
  echo "== $c"
  if [[ "$c" == extra_* && ! -f "data/processed_clips/$c/pose_data.json" ]]; then ~/gen/bin/python -m motion3d.extra prep "$c"; fi
  ~/pose3d/bin/python -m motion3d.mediapipe_pass "data/processed_clips/$c/source_512.mp4" "data/motion3d/$c/mediapipe.json" 2>/dev/null | tail -1
  ~/gen/bin/python -m motion3d.camera "data/processed_clips/$c" "data/motion3d/$c/camera.json" 2>/dev/null | tail -1
  ~/gen/bin/python -m motion3d.fit "$c" > "data/motion3d/$c/fit.log" 2>&1 || { tail -5 "data/motion3d/$c/fit.log"; exit 1; }
  ~/gen/bin/python -m motion3d.render "$c" | tail -1
  case "$c" in good_cover_drive_final|bowler_new_cummins) ~/gen/bin/python -m motion3d.export "$c" | tail -1;; esac
done
