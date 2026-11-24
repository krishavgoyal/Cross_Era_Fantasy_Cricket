"""Era-filter evaluation (Milestone 4): raw generated clip vs each era style -> results/era_eval.csv.

For the chosen clips (batter F1, bowler L3) each era preset is applied to the RAW 33-frame clip at its native timing
(no retime), so only the filter's effect is measured. Metrics (CPU):
  pck@0.2          motion kept? (DWPose on the styled video vs the control keypoints)
  dino_crop_*      identity vs the tight reference photo (grayscale crops)
  brightness_std   frame-to-frame brightness flicker
  frame_diff_mean  mean absolute frame-to-frame difference
  warp_err         temporal smoothness: Farneback optical flow warps frame t onto t+1 and the luminance error is
                   averaged where forward/backward flow agree (lower = smoother). Ported from student3_v3/evaluate.py
                   (which compared chroma; we compare luminance because the 1930s style is grayscale).
All styled videos are sampled on the control timeline (33 frames over the clip), so every version is compared on the
same moments. Styled eval clips (with the AI label) go to data/final/eval/.

Usage:  python scripts/era_eval.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from era_filters import PRESETS, apply_preset
from eval_metrics import Dino, evaluate_video, ref_embedding
from pose_utils import load_dwpose, read_video, write_video

CLIPS = [  # name, raw video, control npz (canvas keypoints), reference photo
    ("batter_F1", "data/gen/v3_raw/final_cap_s42_33f.mp4", "data/pose/batter/depth_33.npz", "data/refs/bradman_crop.png"),
    ("bowler_L3", "data/gen/bowler_raw/L3_s2024_33f.mp4", "data/pose/bowler/depth_33.npz", "data/refs/larwood_crop.png"),
]
FIELDS = ["timestamp", "clip", "version", "video", "fps", "frames_sampled", "pck@0.2", "dino_crop_mean", "dino_crop_min",
          "brightness_std", "frame_diff_mean", "warp_err"]


def warping_error(prev_bgr: np.ndarray, cur_bgr: np.ndarray) -> float:
    """Mean |L(cur) - warp(L(prev))| where forward/backward Farneback flow agree (lower = smoother motion)."""
    p = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    back = cv2.calcOpticalFlowFarneback(c, p, None, 0.5, 3, 19, 3, 5, 1.2, 0)
    fwd = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 19, 3, 5, 1.2, 0)
    h, w = c.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + back[..., 0], gy + back[..., 1]
    warped = cv2.remap(p.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb = np.linalg.norm(back + cv2.remap(fwd, mx, my, cv2.INTER_LINEAR), axis=2)
    valid = fb < 1.5
    return float(np.abs(c.astype(np.float32) - warped)[valid].mean()) if valid.any() else float("nan")


def timeline_warp_error(path: Path, ctrl_fps: float, n: int) -> float:
    """Warping error between consecutive frames sampled on the control timeline (same moments for every version)."""
    frames, fps = read_video(path)
    idx = np.clip(np.round(np.arange(n) / ctrl_fps * fps).astype(int), 0, len(frames) - 1)
    errs = [warping_error(frames[a], frames[b]) for a, b in zip(idx[:-1], idx[1:]) if a != b]
    return float(np.nanmean(errs)) if errs else float("nan")


def main() -> int:
    out_dir = Path("data/final/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, dino = load_dwpose(), Dino()
    opts = SimpleNamespace(alpha=0.2, thr=0.3, rgb_identity=False)
    rows = []
    for name, raw, control, ref in CLIPS:
        c = np.load(control)
        ctrl, cfps, csize = c["kpts"][:, :17].astype(np.float32), float(c["fps"]), tuple(int(v) for v in c["size"])
        refs = {"crop": ref_embedding(model, dino, ref)}
        frames, fps = read_video(raw)
        versions = [("raw", Path(raw))]
        for preset in PRESETS:
            styled, sfps = apply_preset(preset, frames, fps)
            versions.append((preset, write_video(out_dir / f"{Path(raw).stem}_{preset}.mp4", styled, sfps)))
        for version, path in versions:
            m = evaluate_video(Path(path), "raw" if version == "raw" else "styled", name, model, ctrl, cfps, csize, refs, dino, opts)
            row = {k: m.get(k, "") for k in FIELDS}
            row.update(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), clip=name, version=version, video=str(path),
                       warp_err=round(timeline_warp_error(Path(path), cfps, len(ctrl)), 3))
            rows.append(row)
            print(f"{name:10s} {version:16s} PCK {row['pck@0.2']:.3f}  ID {row['dino_crop_mean']:.3f}  "
                  f"b.std {row['brightness_std']:.2f}  f.diff {row['frame_diff_mean']:.2f}  warp {row['warp_err']:.2f}")
    out = Path("results/era_eval.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
