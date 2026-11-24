"""Retime a generated clip to REAL speed.

The VACE controls sample --duration seconds of the source (5.6 s) into 33 frames, and the generator writes them at
16 fps, so the clip plays ~2.7x too fast (~2 s). Frame i shows source time i * duration / (N - 1). This script
stretches the clip back to that real duration and resamples it to --fps (default 24).

Methods (all ffmpeg):
  hold   frame holds (setpts + fps): no new pixels, slightly steppy motion
  blend  framerate filter: cross-fades neighbouring frames (soft ghosting on fast motion, never warps shapes)
  mci    minterpolate mi_mode=mci, mc_mode=aobmc, me_mode=bidir, vsbmc=1: motion-compensated in-betweens
         (smoothest when it works; can warp/tear thin fast objects such as the bat)

Usage
  python scripts/retime.py --input data/gen/v1_raw/bradman_batting_33f.mp4 --out data/final/work/retimed.mp4
  python scripts/retime.py --input clip.mp4 --method mci --source-duration 5.6 --fps 24
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2

from pose_utils import find_ffmpeg

DEFAULT_METHOD = "blend"   # chosen after comparing all three on the fast bat swing (see README_PIPELINE.md)


def retime_filter(method: str, n_frames: int, in_fps: float, source_duration: float, out_fps: float) -> str:
    """ffmpeg -vf chain: stretch timestamps so frame i lands at i*duration/(n-1), then resample to out_fps."""
    k = source_duration / (n_frames - 1) * in_fps          # PTS scale factor (2.8 for 33 frames @16 fps over 5.6 s)
    stretch = f"setpts={k:.6f}*PTS"
    pad = "tpad=stop_mode=clone:stop_duration=1"            # hold the last frame so every method reaches full length
    if method == "hold":
        return f"{stretch},fps={out_fps:g},{pad}"
    if method == "blend":
        return f"{stretch},framerate=fps={out_fps:g},{pad}"
    if method == "mci":
        return f"{stretch},minterpolate=fps={out_fps:g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,{pad}"
    raise ValueError(f"unknown method {method!r}")


def retime(src: str | Path, dst: str | Path, source_duration: float = 5.6, fps: float = 24.0,
           method: str = DEFAULT_METHOD) -> Path:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("ffmpeg not found. Install it (winget install Gyan.FFmpeg).")
    cap = cv2.VideoCapture(str(src))
    n, in_fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 16.0
    cap.release()
    if n < 2:
        raise SystemExit(f"Cannot read {src}")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = retime_filter(method, n, in_fps, source_duration, fps)
    # keep exactly the real duration (+ one output frame so the last source sample is shown)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-t", f"{source_duration + 1 / fps:.4f}",
                    "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-r", f"{fps:g}", str(dst)], check=True)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-duration", type=float, default=5.6, help="seconds of source the clip covers (make_control --duration)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--method", choices=["hold", "blend", "mci"], default=DEFAULT_METHOD)
    args = ap.parse_args()
    out = retime(args.input, args.out, args.source_duration, args.fps, args.method)
    cap = cv2.VideoCapture(str(out))
    n, f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Wrote {out}: {n} frames @ {f:g} fps = {n / f:.2f} s ({args.method})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
