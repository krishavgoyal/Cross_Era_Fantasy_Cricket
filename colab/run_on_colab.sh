#!/usr/bin/env bash
# Run the full pipeline on a Google Colab GPU from your own terminal (WSL/Linux/macOS).
#
#   colab/run_on_colab.sh [both|bowler|batsman]
#
# env:  GPU=A100|L4|T4|H100 (default A100)   SESSION=cricket   BRANCH=Pranahv-main
#       KEEP=1 keeps the VM after the run (it keeps using compute units!)
#       EXTRA_ARGS="..." extra master_pipeline.py flags for both clips
#
# What it does: starts (or reuses) a Colab GPU session, clones BRANCH from GitHub (that has the
# clip data), overlays your local code changes, runs colab/remote_run.sh in the background on the
# VM, prints its progress every minute, downloads runs/ into ./colab_results/, stops the VM.
# Needs the Colab CLI: uv tool install google-colab-cli   (one-time login: colab sessions)
set -euo pipefail
WHICH="${1:-both}"
case "$WHICH" in both|bowler|batsman) ;; *) echo "usage: $0 [both|bowler|batsman]" >&2; exit 2;; esac
GPU="${GPU:-A100}"; S="${SESSION:-cricket}"; BRANCH="${BRANCH:-Pranahv-main}"
case "$GPU" in A100|L4|T4|H100|G4) ;; *) echo "GPU must be A100, L4, T4, H100 or G4" >&2; exit 2;; esac
case "$S$BRANCH" in *[!A-Za-z0-9_.-]*) echo "SESSION/BRANCH: letters, digits, . _ - only" >&2; exit 2;; esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/colab_results"; mkdir -p "$OUT"
command -v colab >/dev/null || { echo "Colab CLI missing: uv tool install google-colab-cli" >&2; exit 3; }

# Shell on the VM through IPython '!' (no ssh key needed). --timeout: clone/untar can be silent >30 s.
rx() { printf '%s\n' "$1" | colab exec -s "$S" --timeout "${RX_TIMEOUT:-300}"; }
stop_vm() { [ "${KEEP:-0}" = 1 ] && echo "KEEP=1: session $S left running (colab stop -s $S)" || colab stop -s "$S" || true; }

# Stop the VM on ANY exit (error, Ctrl+C, normal end), not just on Ctrl+C, so a crash never
# leaves a GPU running. INT/TERM exit non-zero, which then runs the EXIT trap once.
trap 'rc=$?; stop_vm; exit $rc' EXIT
trap 'echo "interrupted"; exit 130' INT TERM

echo "== compute units"; colab usage || true
if colab sessions 2>/dev/null | grep -qw -- "$S"; then
  echo "== reusing session $S"
else
  echo "== starting $GPU session $S"; colab new -s "$S" --gpu "$GPU"
fi
echo "== code: $BRANCH from GitHub + your local changes"
rx "!rm -rf /content/cricket && git clone -q --depth 1 -b $BRANCH https://github.com/krishavgoyal/Cross_Era_Fantasy_Cricket.git /content/cricket && echo cloned"
TGZ="$(mktemp -d)/cricket_code.tgz"
tar czf "$TGZ" -C "$REPO" --exclude=./.git --exclude=./data --exclude=./runs --exclude=./colab_results \
    --exclude=./outputs --exclude='*.mp4' --exclude='*.npz' --exclude='*.zip' --exclude=__pycache__ \
    --exclude=./checkpoints --exclude=./models --exclude=./animate/jobs --exclude='./kaggle_results*' --exclude='./*.jpg' --exclude='./*.jpeg' \
    --exclude='*.onnx' --exclude='*.pt' --exclude='*.pth' --exclude='*.pkl' --exclude='*.bin' --exclude='*.safetensors' .
if [ "$(stat -c %s "$TGZ")" -gt 2000000 ]; then
  echo "code overlay is $(du -h "$TGZ" | cut -f1); expected ~150 KB. Largest files:" >&2
  tar tzvf "$TGZ" | sort -k3 -n | tail -5 >&2; exit 2
fi
colab upload -s "$S" "$TGZ" /content/cricket_code.tgz
rx "!tar xzf /content/cricket_code.tgz -C /content/cricket && echo overlaid"

echo "== starting the pipeline on the VM ($WHICH)"
# setsid + </dev/null: fully detach so the IPython '!' call returns at once instead of holding
# the kernel until colab exec times out (that is what broke the first T4 run).
rx "!rm -f /content/job.log /content/results.zip; cd /content/cricket && EXTRA_ARGS='${EXTRA_ARGS:-}' setsid nohup bash colab/remote_run.sh $WHICH </dev/null >/content/job.log 2>&1 & echo launched"

START=$(date +%s); FAILS=0
while true; do
  sleep 60
  if TAIL="$(RX_TIMEOUT=60 rx '!tail -n 3 /content/job.log' 2>&1)"; then FAILS=0; else FAILS=$((FAILS + 1)); fi
  echo "-- $(( ($(date +%s) - START) / 60 )) min"; echo "$TAIL"
  if grep -q "PIPELINE_DONE" <<<"$TAIL"; then break; fi
  if grep -q "appears to be lost" <<<"$TAIL" || [ "$FAILS" -ge 5 ]; then
    echo "== lost contact with session $S (Colab reclaimed the VM?). Giving up." >&2; exit 71
  fi
done

echo "== downloading results to $OUT"
colab download -s "$S" /content/job.log "$OUT/job.log" || true
colab download -s "$S" /content/results.zip "$OUT/results.zip"
(cd "$OUT" && unzip -oq results.zip)
RC="$(grep -o 'PIPELINE_DONE rc=[0-9]*' "$OUT/job.log" | tail -1 | cut -d= -f2)"
echo "== done (rc=${RC:-?}). Final videos: $OUT/runs/colab_*/final_24fps.mp4 and demo_side_by_side.mp4"
exit "${RC:-70}"
