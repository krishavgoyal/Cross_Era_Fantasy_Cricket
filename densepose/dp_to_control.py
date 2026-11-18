#!/usr/bin/env python3
"""Convert detectron2 DensePose `apply_net.py dump` output (dp_results.pkl) into the clip's DensePose files.

Run inside the DensePose env (needs torch, detectron2, densepose, opencv, numpy, ffmpeg):
    python densepose/dp_to_control.py dp_results.pkl <clip_dir> --metadata <clip_dir>/metadata.json

Outputs in out_dir:
    control_densepose_real.mp4   black background, PARULA colormap on part ids
    densepose_real_parts.npz     part_index (N,H,W) uint8, 0 = background, 1..24
    densepose_real_iuv.npz       u, v (N,H,W) uint8 (0..255) for the same pixels

One person per frame is kept. With --metadata the detection that overlaps the tracked target box
most (IoU) is chosen, so umpires or batters inside the crop are ignored. Without it: score * area.
Frames with no detection are held from the previous frame (reported).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dp_common import colorize, expected_boxes, iou  # noqa: E402


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def best_instance(entry, expected=None):
    dp = entry["pred_densepose"]
    boxes = to_np(entry["pred_boxes_XYXY"]).reshape(-1, 4)
    scores = to_np(entry["scores"]).reshape(-1)
    if len(boxes) == 0:
        return None
    if expected is not None:
        ious = np.array([iou(b, expected) for b in boxes])
        if ious.max() >= 0.10:
            i = int(np.argmax(ious))
            return i, boxes[i], dp[i]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(np.argmax(scores * area))
    return i, boxes[i], dp[i]


def paste(canvas_i, canvas_u, canvas_v, box, res, size):
    labels = to_np(res.labels).astype(np.uint8)
    uv = to_np(res.uv)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    if labels.shape != (bh, bw):
        labels = cv2.resize(labels, (bw, bh), interpolation=cv2.INTER_NEAREST)
        uv = np.stack([cv2.resize(c, (bw, bh), interpolation=cv2.INTER_LINEAR) for c in uv])
    u8 = np.clip(uv * 255, 0, 255).astype(np.uint8)
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + bw, size), min(y0 + bh, size)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sx0, sy0 = cx0 - x0, cy0 - y0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
    canvas_i[cy0:cy1, cx0:cx1] = labels[sy0:sy1, sx0:sx1]
    canvas_u[cy0:cy1, cx0:cx1] = u8[0][sy0:sy1, sx0:sx1]
    canvas_v[cy0:cy1, cx0:cx1] = u8[1][sy0:sy1, sx0:sx1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("out_dir")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=float, default=24)
    ap.add_argument("--metadata", default=None, help="clip metadata.json: pick the tracked target, not the biggest person")
    a = ap.parse_args()

    import torch
    try:
        import densepose  # noqa: F401  needed so torch can unpickle DensePose result classes
    except Exception as e:
        sys.exit(f"cannot import densepose ({e}). Activate the DensePose env first.")
    results = torch.load(a.pkl, map_location="cpu", weights_only=False)
    results = sorted(results, key=lambda e: str(e["file_name"]))
    n, S = len(results), a.size
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    I = np.zeros((n, S, S), np.uint8)
    U = np.zeros_like(I)
    V = np.zeros_like(I)
    missing = []
    exp = expected_boxes(a.metadata, S) if a.metadata else None
    if exp is not None and len(exp) != n:
        sys.exit(f"metadata has {len(exp)} frames but dump has {n}")
    for t, e in enumerate(results):
        got = best_instance(e, exp[t] if exp else None) if len(e.get("pred_densepose", [])) else None
        if got is None:
            missing.append(t)
            if t > 0:
                I[t], U[t], V[t] = I[t - 1], U[t - 1], V[t - 1]
            continue
        _, box, res = got
        paste(I[t], U[t], V[t], box, res, S)

    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{S}x{S}",
           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14",
           str(out / "control_densepose_real.mp4")]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in range(n):
        pr.stdin.write(np.ascontiguousarray(colorize(I[t])).tobytes())
    pr.stdin.close()
    if pr.wait() != 0:
        sys.exit("ffmpeg failed writing control_densepose_real.mp4")
    np.savez_compressed(out / "densepose_real_parts.npz", part_index=I)
    np.savez_compressed(out / "densepose_real_iuv.npz", u=U, v=V)
    print(f"wrote {n} frames to {out}")
    print(f"frames with no detection (held from previous): {len(missing)} {missing[:20]}")
    print(f"mean foreground coverage: {(I > 0).mean() * 100:.1f}% of pixels")


if __name__ == "__main__":
    main()
