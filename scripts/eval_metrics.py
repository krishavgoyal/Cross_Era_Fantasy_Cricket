"""Stage 5: metrics for a generated clip (CPU is fine). Computed on the RAW video and on any STYLED versions.

  (a) Motion fidelity   DWPose is run on the generated video; PCK@0.2 of the 17 body keypoints against the control
                        skeleton (control_33.npz), normalised by torso length (neck to mid-hip).
                        pck() and the torso normaliser are ported from student3_v3/evaluate.py.
  (b) Identity          DINOv2-small cosine similarity between the reference-photo person crop and each frame's
                        person crop (mean and minimum). Compared in grayscale by default (our references are B&W).
  (c) Flicker           std of the per-frame mean brightness, and the mean absolute frame-to-frame difference.

Styled videos have a different fps and frame count, so each video is sampled at the CONTROL's timeline
(control frame i <-> time i/control_fps) before measuring pose and identity. Flicker uses every frame of the file.
Note: styled videos carry deliberate grain/blur/jitter, so their scores are expected to be lower; that is the point
of reporting both.

Usage
  python scripts/eval_metrics.py --name batter --video data/gen/outputs_raw/batter_33f.mp4 ^
      --styled data/styled/batter_33f_1930s_film.mp4 data/styled/batter_33f_1970s_tv.mp4 ^
      --control data/pose/batter/control_33.npz --ref-photo data/refs/bradman.png
Appends one row per video to results/metrics.csv (commit that file).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from pose_utils import L_HIP, L_SHO, R_HIP, R_SHO, detect_people, load_dwpose, pose_in_box, read_video

CSV_FIELDS = ["timestamp", "name", "kind", "video", "frames_sampled", "fps", "size", "person_found_rate",
              "pck@0.2", "pck_valid_joints", "dino_mean", "dino_min", "brightness_std", "frame_diff_mean"]


# ------------------------------------------------------------------ (a) motion fidelity
def torso_length(ctrl: np.ndarray) -> float:
    """Median neck (mid-shoulders) to mid-hip distance over the control sequence, in control pixels."""
    neck = (ctrl[:, L_SHO] + ctrl[:, R_SHO]) / 2
    hip = (ctrl[:, L_HIP] + ctrl[:, R_HIP]) / 2
    d = np.linalg.norm(neck - hip, axis=1)
    d = d[np.isfinite(d) & (d > 1)]
    if len(d) == 0:
        raise ValueError("Cannot measure torso length in the control (shoulders/hips missing)")
    return float(np.median(d))


def pck(ref: np.ndarray, pred: np.ndarray, scale: float, alpha: float) -> tuple[float, int]:
    """PCK over all valid reference joints of all frames. ref/pred: (T, J, 2). Returns (score, joints counted).

    A reference joint counts if it is present in the control. A prediction that is missing (NaN) counts as a miss.
    """
    valid = np.isfinite(ref).all(-1)
    dist = np.linalg.norm(ref - pred, axis=-1)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    n = int(valid.sum())
    if n == 0:
        return float("nan"), 0
    return float((dist[valid] <= alpha * scale).mean()), n


# ------------------------------------------------------------------ (b) identity
class Dino:
    """DINOv2 embedder (CPU). Loaded lazily so the other metrics work without torch/transformers."""

    def __init__(self, name: str = "facebook/dinov2-small"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    def __call__(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
        with self.torch.no_grad():
            out = self.model(**self.proc(images=rgb, return_tensors="pt")).last_hidden_state[:, 0]   # CLS token
        emb = out.numpy()
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def person_crop(frame: np.ndarray, box: np.ndarray, gray: bool, margin: float = 0.1) -> np.ndarray:
    """Crop the person box (+margin), pad to a square so DINO's centre-crop does not cut off head or feet."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    x1, y1, x2, y2 = int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(w, x2 + mx)), int(min(h, y2 + my))
    crop = frame[y1:y2, x1:x2]
    if gray:
        crop = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    s = max(crop.shape[:2])
    canvas = np.full((s, s, 3), 114, np.uint8)
    oy, ox = (s - crop.shape[0]) // 2, (s - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return canvas


# ------------------------------------------------------------------ (c) flicker
def flicker_stats(frames: list[np.ndarray]) -> tuple[float, float]:
    """(std of per-frame mean brightness, mean absolute frame-to-frame difference), both on 0-255 grayscale."""
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    brightness_std = float(np.std([g.mean() for g in gray]))
    diff = float(np.mean([np.abs(a - b).mean() for a, b in zip(gray[:-1], gray[1:])])) if len(gray) > 1 else float("nan")
    return brightness_std, diff


# ------------------------------------------------------------------ driver
def imread_any(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def biggest_box(model, frame: np.ndarray) -> np.ndarray | None:
    boxes = detect_people(model, frame)
    if len(boxes) == 0:
        return None
    return boxes[int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))]


def evaluate_video(path: Path, kind: str, name: str, model, ctrl: np.ndarray, ctrl_fps: float, ctrl_size: tuple[int, int],
                   ref_emb: np.ndarray | None, dino: Dino | None, args) -> dict:
    frames, fps = read_video(path)
    h, w = frames[0].shape[:2]
    n = len(ctrl)
    # sample the video on the control's timeline (works for raw 16 fps and for retimed/styled 18 or 25 fps)
    idx = np.clip(np.round(np.arange(n) / ctrl_fps * fps).astype(int), 0, len(frames) - 1)
    # if the video is shorter than the control (e.g. a 17-frame smoke test), score only the frames it covers
    covered = np.arange(n) / ctrl_fps < len(frames) / fps - 1e-6      # strictly inside the video's duration
    idx, ctrl_use = idx[covered], ctrl[covered]

    pred = np.full_like(ctrl_use, np.nan)
    crops, found = [], 0
    sx, sy = ctrl_size[0] / w, ctrl_size[1] / h                     # map video pixels -> control canvas pixels
    for t, fi in enumerate(idx):
        frame = frames[fi]
        box = biggest_box(model, frame)
        if box is None:
            continue
        found += 1
        k, s = pose_in_box(model, frame, box)
        body = k[:17].copy()
        body[s[:17] < args.thr] = np.nan
        pred[t] = body * np.array([sx, sy], np.float32)
        if dino is not None:
            crops.append(person_crop(frame, box, gray=not args.rgb_identity))
    score, n_joints = pck(ctrl_use[:, :17], pred, torso_length(ctrl), args.alpha)

    dmean = dmin = float("nan")
    if dino is not None and crops and ref_emb is not None:
        sims = dino(crops) @ ref_emb
        dmean, dmin = float(sims.mean()), float(sims.min())
    bstd, fdiff = flicker_stats(frames)
    return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "kind": kind, "video": str(path),
            "frames_sampled": len(idx), "fps": round(fps, 2), "size": f"{w}x{h}",
            "person_found_rate": round(found / len(idx), 3), "pck@0.2": round(score, 4), "pck_valid_joints": n_joints,
            "dino_mean": round(dmean, 4), "dino_min": round(dmin, 4),
            "brightness_std": round(bstd, 3), "frame_diff_mean": round(fdiff, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="label for this clip in the CSV, e.g. batter_vace")
    ap.add_argument("--video", required=True, help="RAW generated video (outputs_raw/...)")
    ap.add_argument("--styled", nargs="*", default=[], help="styled versions of the same clip (data/styled/...)")
    ap.add_argument("--control", required=True, help="control_33.npz written by make_control.py")
    ap.add_argument("--ref-photo", help="reference photo for the identity metric (omit to skip identity)")
    ap.add_argument("--out", default="results/metrics.csv")
    ap.add_argument("--alpha", type=float, default=0.2, help="PCK threshold as a fraction of torso length")
    ap.add_argument("--thr", type=float, default=0.3, help="keypoint confidence threshold")
    ap.add_argument("--rgb-identity", action="store_true", help="compare identity in colour (default: grayscale)")
    ap.add_argument("--dino-model", default="facebook/dinov2-small")
    ap.add_argument("--skip-identity", action="store_true", help="skip DINOv2 (no torch/transformers needed)")
    args = ap.parse_args()

    c = np.load(args.control)
    ctrl, ctrl_fps, ctrl_size = c["kpts"][:, :17].astype(np.float32), float(c["fps"]), tuple(int(v) for v in c["size"])
    model = load_dwpose()

    dino = ref_emb = None
    if args.ref_photo and not args.skip_identity:
        dino = Dino(args.dino_model)
        photo = imread_any(args.ref_photo)
        box = biggest_box(model, photo)
        if box is None:
            print("WARNING: no person found in the reference photo; using the whole image for identity")
            box = np.array([0, 0, photo.shape[1], photo.shape[0]], np.float32)
        ref_emb = dino([person_crop(photo, box, gray=not args.rgb_identity)])[0]

    rows = [evaluate_video(Path(args.video), "raw", args.name, model, ctrl, ctrl_fps, ctrl_size, ref_emb, dino, args)]
    for s in args.styled:
        rows.append(evaluate_video(Path(s), "styled", args.name, model, ctrl, ctrl_fps, ctrl_size, ref_emb, dino, args))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, CSV_FIELDS)
        if new:
            wr.writeheader()
        wr.writerows(rows)
    print(f"{'video':44s} {'kind':7s} {'PCK@0.2':>8s} {'DINO mean/min':>14s} {'bright.std':>10s} {'frame diff':>10s}")
    for r in rows:
        print(f"{Path(r['video']).name[:44]:44s} {r['kind']:7s} {r['pck@0.2']:8.3f} "
              f"{r['dino_mean']:7.3f}/{r['dino_min']:.3f} {r['brightness_std']:10.2f} {r['frame_diff_mean']:10.2f}")
    print(f"Appended {len(rows)} row(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
