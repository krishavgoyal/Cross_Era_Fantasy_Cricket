#!/usr/bin/env python3
"""Repair frames where real DensePose lost most of the body (motion blur etc).

For each weak frame it keeps whatever real DensePose found and fills the missing legs and feet
(part ids 5 to 14) from the DWPose skeleton shapes in pose_data.json. u,v on filled pixels = 128
(placeholder). Everything else stays real DensePose.

Weak frame = mask box vs tracked target box IoU < --iou (0.5)
             or mask area < --area-frac (0.6) x median of the +-3 neighbouring frames.

usage:  python densepose/patch_weak_frames.py <clip_dir> [--frames 118]
reads:  pose_data.json, metadata.json, densepose_real_parts.npz, densepose_real_iuv.npz
writes: control_densepose_real.mp4, densepose_real_parts.npz, densepose_real_iuv.npz, PATCHED_FRAMES.json
        (originals copied to _before_patch/ first)
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_densepose as rd  # noqa: E402
from dp_common import colorize, expected_boxes, iou  # noqa: E402

LEG_IDS = list(range(5, 15))


def find_weak(P, exp, iou_thr=0.5, area_frac=0.6):
    n = len(P)
    area = (P > 0).reshape(n, -1).mean(1)
    weak = []
    for t in range(n):
        ys, xs = np.where(P[t] > 0)
        ob = [xs.min(), ys.min(), xs.max(), ys.max()] if len(xs) else [0, 0, 1, 1]
        nb = [area[j] for j in range(max(0, t - 3), min(n, t + 4)) if j != t]
        low = area[t] < area_frac * np.median(nb)
        if iou(exp[t], ob) < iou_thr or low:
            weak.append(t)
    return weak, area


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--area-frac", type=float, default=0.6)
    ap.add_argument("--frames", type=int, nargs="*", help="force these frame indexes instead of auto detection")
    ap.add_argument("--fps", type=float, default=24)
    a = ap.parse_args(argv)
    C = Path(a.clip_dir)
    P = np.load(C / "densepose_real_parts.npz")["part_index"].copy()
    iu = np.load(C / "densepose_real_iuv.npz")
    U, V = iu["u"].copy(), iu["v"].copy()
    n, S = P.shape[0], P.shape[1]
    exp = expected_boxes(C / "metadata.json", S)
    if len(exp) != n:
        sys.exit(f"metadata has {len(exp)} frames, DensePose has {n}")

    weak, area = find_weak(P, exp, a.iou, a.area_frac)
    if a.frames is not None:
        weak = a.frames
    print("weak frames:", weak)
    if not weak:
        print("nothing to patch")
        return 0

    _, body, hands, face = rd.load(C / "pose_data.json")
    body = rd.fill_gaps(body)
    state = {}
    approx = [rd.render_frame(t, body, hands, face, state, S) for t in range(n)]

    bk = C / "_before_patch"
    bk.mkdir(exist_ok=True)
    for name in ("control_densepose_real.mp4", "densepose_real_parts.npz", "densepose_real_iuv.npz"):
        if (C / name).exists() and not (bk / name).exists():
            shutil.copy(C / name, bk / name)
    for t in weak:
        base = P[t].copy()
        add = np.isin(approx[t], LEG_IDS) & (base == 0)
        base[add] = approx[t][add]
        P[t] = base
        U[t] = np.where(add, 128, U[t])
        V[t] = np.where(add, 128, V[t])
        print(f"frame {t}: coverage {area[t] * 100:.1f}% -> {(base > 0).mean() * 100:.1f}%")
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{S}x{S}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14",
                           str(C / "control_densepose_real.mp4")], stdin=subprocess.PIPE)
    for t in range(n):
        pr.stdin.write(np.ascontiguousarray(colorize(P[t])).tobytes())
    pr.stdin.close()
    if pr.wait() != 0:
        sys.exit("ffmpeg failed")
    np.savez_compressed(C / "densepose_real_parts.npz", part_index=P)
    np.savez_compressed(C / "densepose_real_iuv.npz", u=U, v=V)
    json.dump({"patched_frames": weak,
               "method": "real DensePose kept; legs and feet (part ids 5 to 14) filled from DWPose skeleton shapes; u,v = 128 placeholder on filled pixels",
               "reason": "DensePose lost most of the body in these frames (motion blur)"},
              open(C / "PATCHED_FRAMES.json", "w"), indent=1)
    print("done. originals are in", bk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
