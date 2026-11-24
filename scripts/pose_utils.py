"""Shared helpers: keypoint layout, DWPose loading, OpenPose-style drawing and video I/O.

Keypoints follow COCO-WholeBody (133 points):
    0-16    body (COCO-17)     17-22  feet
    23-90   face (68)          91-111 left hand (21)     112-132 right hand (21)
Missing keypoints are stored as NaN.
"""
from __future__ import annotations

import colorsys
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- keypoint layout
BODY = list(range(0, 17))
FEET = list(range(17, 23))
FACE = list(range(23, 91))
LHAND = list(range(91, 112))
RHAND = list(range(112, 133))

# COCO-17 names, used in a few places to keep the code readable
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

# DWPose-l 384x288 (whole body, 133 keypoints). rtmlib downloads it on first use.
DWPOSE_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
              "rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip")


def load_dwpose():
    """Create the rtmlib Wholebody model on CPU: YOLOX-m person detector + DWPose-l pose model."""
    from rtmlib import Wholebody  # imported here so drawing-only scripts do not need rtmlib
    return Wholebody(mode="performance",          # gives the YOLOX-m 640 detector
                     pose=DWPOSE_URL, pose_input_size=(288, 384),
                     backend="onnxruntime", device="cpu")


def _boxes_from_kpts(kpts: np.ndarray, scores: np.ndarray, thr: float = 0.3) -> np.ndarray:
    """Fallback: person boxes from the confident body keypoints of each person."""
    out = []
    for k, s in zip(kpts, scores):
        pts = k[:17][s[:17] > thr]
        if len(pts) < 3:
            pts = k[:17]
        x1, y1 = pts.min(0)
        x2, y2 = pts.max(0)
        pad = 0.1 * max(x2 - x1, y2 - y1)
        out.append([x1 - pad, y1 - pad, x2 + pad, y2 + pad])
    return np.asarray(out, np.float32).reshape(-1, 4)


def detect_people(model, img: np.ndarray) -> np.ndarray:
    """Return person boxes (N, 4) as x1, y1, x2, y2."""
    if hasattr(model, "det_model"):                       # rtmlib Wholebody exposes its detector
        return np.asarray(model.det_model(img), dtype=np.float32).reshape(-1, 4)
    kpts, scores = model(img)                             # fallback: full model, boxes from keypoints
    return _boxes_from_kpts(np.asarray(kpts), np.asarray(scores))


def pose_in_box(model, img: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run DWPose on one person box. Returns kpts (133, 2) and scores (133,)."""
    if hasattr(model, "pose_model"):
        kpts, scores = model.pose_model(img, bboxes=[list(map(float, box))])
        return np.asarray(kpts, np.float32)[0], np.asarray(scores, np.float32)[0]
    # fallback: run on everyone and keep the person whose keypoint box overlaps `box` most
    kpts, scores = model(img)
    kpts, scores = np.asarray(kpts, np.float32), np.asarray(scores, np.float32)
    best = int(np.argmax([iou(b, box) for b in _boxes_from_kpts(kpts, scores)]))
    return kpts[best], scores[best]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two x1, y1, x2, y2 boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def biggest_person(model, img: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Pose of the largest detected person (used for the reference photo)."""
    boxes = detect_people(model, img)
    if len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return pose_in_box(model, img, boxes[int(np.argmax(areas))])


# ---------------------------------------------------------------- video I/O
def read_video(path: str | Path, start: float = 0.0, end: float | None = None) -> tuple[list[np.ndarray], float]:
    """Read frames (BGR) between start and end seconds. Returns (frames, fps)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if t < start:
            continue
        if end is not None and t > end:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path} (start={start}, end={end})")
    return frames, float(fps)


# Social-media overlays in our driving clips (channel watermark, handle/caption/music bar), as fractions of the frame:
# (y0, y1, x0, x1). Blurred in figures and comparison videos; the clips are used only for motion.
OVERLAY_BANDS = {"C2_coverdrive": [(0.80, 1.00, 0.0, 1.0), (0.11, 0.17, 0.0, 0.40)]}


def parse_bands(text: str | None) -> list[tuple[float, float, float, float]]:
    """'y0,y1,x0,x1;y0,y1,x0,x1' -> list of tuples (fractions of height/width)."""
    if not text:
        return []
    return [tuple(float(v) for v in part.split(",")) for part in text.split(";") if part.strip()]


def blur_bands(img: np.ndarray, bands) -> np.ndarray:
    """Return a copy with each band heavily pixelated + blurred (unreadable text/logos)."""
    out = img.copy()
    h, w = img.shape[:2]
    for y0, y1, x0, x1 in bands:
        Y0, Y1, X0, X1 = int(y0 * h), int(y1 * h), int(x0 * w), int(x1 * w)
        if Y1 - Y0 < 2 or X1 - X0 < 2:
            continue
        roi = out[Y0:Y1, X0:X1]
        small = cv2.resize(roi, (max(1, (X1 - X0) // 16), max(1, (Y1 - Y0) // 16)), interpolation=cv2.INTER_AREA)
        out[Y0:Y1, X0:X1] = cv2.GaussianBlur(cv2.resize(small, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR), (0, 0), 6)
    return out


def find_ffmpeg(tool: str = "ffmpeg") -> str | None:
    """Path of ffmpeg/ffprobe: PATH first, then the winget install folder (a terminal opened before
    `winget install Gyan.FFmpeg` does not see the new PATH entry yet)."""
    exe = shutil.which(tool)
    if exe:
        return exe
    import glob
    import os
    pattern = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*",
                           "*", "bin", f"{tool}.exe")
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def write_video(path: str | Path, frames: list[np.ndarray], fps: float) -> Path:
    """Write BGR frames to .mp4. If ffmpeg is on PATH, re-encode to H.264 (yuv420p) for compatibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    tmp = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        subprocess.run([ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(fps), str(path)], check=True)
        tmp.unlink()
    else:
        tmp.replace(path)
    return path


# ---------------------------------------------------------------- OpenPose-style drawing
# OpenPose-18 order: nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rknee, Rank,
#                    Lhip, Lknee, Lank, Reye, Leye, Rear, Lear.   (neck = mid-shoulders)
COCO_TO_OP18 = [NOSE, None, R_SHO, R_ELB, R_WRI, L_SHO, L_ELB, L_WRI, R_HIP, R_KNE, R_ANK,
                L_HIP, L_KNE, L_ANK, R_EYE, L_EYE, R_EAR, L_EAR]
OP18_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
              (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
OP18_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
               (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
               (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]  # RGB
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
              (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def to_openpose18(k133: np.ndarray) -> np.ndarray:
    """(133, 2) COCO-WholeBody -> (18, 2) OpenPose body; neck = mean of the shoulders (NaN if missing)."""
    out = np.full((18, 2), np.nan, np.float32)
    for i, c in enumerate(COCO_TO_OP18):
        if c is not None:
            out[i] = k133[c]
    out[1] = (k133[L_SHO] + k133[R_SHO]) / 2.0
    return out


def _ok(p) -> bool:
    return bool(np.all(np.isfinite(p)))


def _pt(p) -> tuple[int, int]:
    """Point as a tuple of plain Python ints (what OpenCV drawing functions expect)."""
    return int(round(float(p[0]))), int(round(float(p[1])))


def draw_openpose(canvas_rgb: np.ndarray, k133: np.ndarray, hands: bool = True, face: bool = True,
                  stick: int = 4) -> np.ndarray:
    """Draw a DWPose/OpenPose-style skeleton (RGB colours) on canvas_rgb in place and return it."""
    body = to_openpose18(k133)
    # limbs as filled ellipses, blended like the original OpenPose renderer
    for (a, b), color in zip(OP18_LIMBS, OP18_COLORS):
        pa, pb = body[a], body[b]
        if not (_ok(pa) and _ok(pb)):
            continue
        layer = canvas_rgb.copy()
        mx, my = (pa + pb) / 2
        length = float(np.hypot(*(pa - pb)))
        angle = float(np.degrees(np.arctan2(pa[1] - pb[1], pa[0] - pb[0])))
        poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stick), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(layer, poly, color)
        canvas_rgb[:] = cv2.addWeighted(canvas_rgb, 0.4, layer, 0.6, 0)
    for i, p in enumerate(body):
        if _ok(p):
            cv2.circle(canvas_rgb, _pt(p), 4, OP18_COLORS[i], -1)
    if hands:
        for idx in (LHAND, RHAND):
            pts = k133[idx]
            for e, (a, b) in enumerate(HAND_EDGES):
                if _ok(pts[a]) and _ok(pts[b]):
                    r, g, bl = colorsys.hsv_to_rgb(e / len(HAND_EDGES), 1.0, 1.0)
                    cv2.line(canvas_rgb, _pt(pts[a]), _pt(pts[b]),
                             (int(r * 255), int(g * 255), int(bl * 255)), 2)
            for p in pts:
                if _ok(p):
                    cv2.circle(canvas_rgb, _pt(p), 3, (0, 0, 255), -1)
    if face:
        for p in k133[FACE]:
            if _ok(p):
                cv2.circle(canvas_rgb, _pt(p), 2, (255, 255, 255), -1)
    return canvas_rgb


def masked(kpts: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    """Copy of kpts with low-confidence points set to NaN."""
    k = kpts.astype(np.float32).copy()
    k[scores < thr] = np.nan
    return k
