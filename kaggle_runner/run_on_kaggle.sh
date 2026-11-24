#!/usr/bin/env bash
# Run the full pipeline on a free Kaggle T4 GPU from your own terminal (WSL/Linux/macOS).
#
#   kaggle_runner/run_on_kaggle.sh [both|bowler|batsman]
#
# env:  BRANCH=Pranahv-main   KERNEL=cricket-pipeline   KAGGLE_USER=<your kaggle username>
#       EXTRA_ARGS="..." extra master_pipeline.py flags for both clips
#       WATCH_ONLY=1 don't push; just wait for the latest kernel version and download its results
#       EXTRA_DATA="data/processed_clips/<id> data/historical_images/<img>" local data not on GitHub,
#                  uploaded as the private dataset <user>/<KERNEL>-extra-data and attached to the kernel
#       EXTRA_DATASET=<user>/<slug> attach a dataset uploaded earlier instead of EXTRA_DATA (no upload)
#       CLEAN=1 clean v2 settings (see colab/remote_run.sh); run names get a _v2 suffix
#
# What it does: builds a private Kaggle script kernel (T4 GPU, internet on) that clones BRANCH from
# GitHub, overlays your local code changes (embedded in the script), runs colab/remote_run.sh
# (same pipeline as Colab), then downloads results into ./kaggle_results/.
# Kaggle runs the kernel as a batch job; nothing keeps running or billing after it ends.
#
# One-time setup:
#   uv tool install kaggle
#   kaggle.com -> Settings -> API -> "Create New Token" -> save as ~/.kaggle/kaggle.json, chmod 600
#   Your Kaggle account must be phone-verified (needed for GPU + internet in kernels).
# Free quota: about 30 GPU hours a week, 12 h max per run.
set -euo pipefail
WHICH="${1:-both}"
case "$WHICH" in both|bowler|batsman|bowler_new|all|animate) ;; *) echo "usage: $0 [both|bowler|batsman|bowler_new|all|animate]" >&2; exit 2;; esac
BRANCH="${BRANCH:-Pranahv-main}"; KERNEL="${KERNEL:-cricket-pipeline}"
case "$BRANCH$KERNEL" in *[!A-Za-z0-9_.-]*) echo "BRANCH/KERNEL: letters, digits, . _ - only" >&2; exit 2;; esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/kaggle_results"; mkdir -p "$OUT"
command -v kaggle >/dev/null || { echo "Kaggle CLI missing: uv tool install kaggle" >&2; exit 3; }

if [ -z "${KAGGLE_USER:-}" ]; then
  KAGGLE_USER="$(python3 -c 'import json,os,sys; p=os.path.expanduser("~/.kaggle/kaggle.json"); print(json.load(open(p))["username"]) if os.path.exists(p) else sys.exit(1)' 2>/dev/null)" \
    || KAGGLE_USER="$(kaggle config view 2>/dev/null </dev/null | sed -n 's/^- username: *//p')"
  [ -n "$KAGGLE_USER" ] && [ "$KAGGLE_USER" != None ] \
    || { echo "Set KAGGLE_USER or log in (~/.kaggle/kaggle.json or ~/.kaggle/access_token)" >&2; exit 3; }
fi
SLUG="$KAGGLE_USER/$KERNEL"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/kernel"
# Code only: model weights (checkpoints/, models/, *.onnx ...) and results must never go up with it
# (a local motion run left 350 MB of DWPose models in checkpoints/ and every push hung).
tar czf "$WORK/code.tgz" -C "$REPO" --exclude=./.git --exclude=./data --exclude=./runs --exclude=./colab_results \
    --exclude=./outputs --exclude='*.mp4' --exclude='*.npz' --exclude='*.zip' --exclude=__pycache__ \
    --exclude=./checkpoints --exclude=./models --exclude=./animate/jobs --exclude='./kaggle_results*' --exclude='./*.jpg' --exclude='./*.jpeg' \
    --exclude='*.onnx' --exclude='*.pt' --exclude='*.pth' --exclude='*.pkl' --exclude='*.bin' --exclude='*.safetensors' .
if [ "$(stat -c %s "$WORK/code.tgz")" -gt 2000000 ]; then
  echo "code overlay is $(du -h "$WORK/code.tgz" | cut -f1); expected ~150 KB. Largest files:" >&2
  tar tzvf "$WORK/code.tgz" | sort -k3 -n | tail -5 >&2; exit 2
fi

DS_SOURCES="[]"
if [ -n "${EXTRA_DATA:-}" ]; then
  # Local data that is not on GitHub goes up as a private Kaggle dataset attached to the kernel.
  # (Embedding it in the script made a 3.4 MB kernel source that `kernels push` never finished.)
  DS="$KAGGLE_USER/${KERNEL}-extra-data"
  mkdir -p "$WORK/ds"
  # shellcheck disable=SC2086
  tar czf "$WORK/ds/extra.tgz" -C "$REPO" --exclude=_before_patch $EXTRA_DATA
  echo "== extra data: $EXTRA_DATA ($(du -h "$WORK/ds/extra.tgz" | cut -f1) compressed) -> dataset $DS"
  printf '{"title": "%s", "id": "%s", "licenses": [{"name": "CC0-1.0"}]}\n' "${KERNEL}-extra-data" "$DS" > "$WORK/ds/dataset-metadata.json"
  if timeout 90 kaggle datasets status "$DS" </dev/null >/dev/null 2>&1; then
    timeout 900 kaggle datasets version -p "$WORK/ds" -m "extra data $(date -u +%FT%TZ)" -r skip </dev/null
  else
    timeout 900 kaggle datasets create -p "$WORK/ds" -r skip </dev/null
  fi
  for _ in $(seq 1 30); do
    DST="$(timeout 90 kaggle datasets status "$DS" </dev/null 2>&1 || true)"
    echo "   dataset status: $DST"; grep -qi ready <<<"$DST" && break; sleep 20
  done
  DS_SOURCES="[\"$DS\"]"
elif [ -n "${EXTRA_DATASET:-}" ]; then
  DS_SOURCES="[\"$EXTRA_DATASET\"]"   # attach an already uploaded dataset, no upload
  echo "== attaching existing dataset $EXTRA_DATASET"
fi

cat > "$WORK/kernel/kernel-metadata.json" <<EOF
{
  "id": "$SLUG",
  "title": "$KERNEL",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": $DS_SOURCES,
  "competition_sources": [],
  "kernel_sources": []
}
EOF

# The kernel script: settings + base64 of the code overlay + the fixed runner body.
{
  echo "WHICH = '$WHICH'"
  echo "BRANCH = '$BRANCH'"
  printf 'EXTRA_ARGS = %s\n' "$(python3 -c 'import sys; print(repr(sys.argv[1]))' "${EXTRA_ARGS:-}")"
  printf "CODE_B64 = '%s'\n" "$(base64 -w0 "$WORK/code.tgz")"
  echo "CLEAN = '${CLEAN:-0}'"
  # MusePose settings and job selection travel to the remote script (MP_*, JOBS)
  printf 'REMOTE_ENV = %s\n' "$(python3 -c 'import json,os;print(repr(json.dumps({k:v for k,v in os.environ.items() if k.startswith("MP_") or k=="JOBS"})))')"
  cat "$REPO/kaggle_runner/kernel_body.py"
} > "$WORK/kernel/run.py"

if [ "${WATCH_ONLY:-0}" = 1 ]; then
  echo "== WATCH_ONLY=1: not pushing; watching the latest version of $SLUG"
else
  echo "== pushing kernel $SLUG ($WHICH, branch $BRANCH, T4)"
  PUSH_OUT="$(timeout 900 kaggle kernels push -p "$WORK/kernel" --accelerator NvidiaTeslaT4 2>&1 </dev/null)" || true
  echo "$PUSH_OUT"
  # Kaggle reports a rejected push (e.g. "Maximum batch GPU session count of 2 reached") on stdout with
  # exit 0; watching would then follow an OLD version. Stop instead.
  if grep -qi "error" <<<"$PUSH_OUT"; then
    echo "== push rejected; not watching. Cancel running/queued versions on kaggle.com (notebook -> Versions)" >&2
    exit 4
  fi
fi

START=$(date +%s); FAILS=0
while true; do
  sleep 60
  # Only a successful call with a terminal KernelWorkerStatus ends the wait; network errors are retried
  # (a traceback containing "Error" once ended the wait early). Queued kernels can wait a long time.
  if ST="$(timeout 90 kaggle kernels status "$SLUG" 2>&1 </dev/null)"; then
    FAILS=0; echo "-- $(( ($(date +%s) - START) / 60 )) min: $ST"
    if grep -qE 'KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL)' <<<"$ST"; then break; fi
  else
    FAILS=$((FAILS + 1)); echo "-- $(( ($(date +%s) - START) / 60 )) min: status call failed ($FAILS in a row): $(tail -n 1 <<<"$ST")"
  fi
  [ "$FAILS" -ge 10 ] && { echo "== kaggle kernels status keeps failing; rerun with WATCH_ONLY=1 or check https://www.kaggle.com/code/$SLUG" >&2; exit 71; }
done

echo "== downloading results to $OUT"
rm -f "$OUT/results.zip" "$OUT/job.log"
timeout 900 kaggle kernels output "$SLUG" -p "$OUT" </dev/null || true
[ -f "$OUT/results.zip" ] && (cd "$OUT" && unzip -oq results.zip)
RC="$(grep -o 'PIPELINE_DONE rc=[0-9]*' "$OUT/job.log" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
echo "== done (rc=${RC:-?}, kernel: $ST)"
echo "   Log: $OUT/job.log   Final videos: $OUT/runs/colab_*/final_24fps.mp4 and demo_side_by_side.mp4"
exit "${RC:-70}"
