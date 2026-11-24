"""Track the real bat in a processed clip (source_512.mp4) with YOLO segmentation (COCO "baseball bat").

    python -m animate.bat_track data/processed_clips/<clip_id>      # writes bat_track.json next to it

For every frame: the bat mask's main axis (PCA) gives a line; of the detections, the one whose line passes
closest to the hands wins. The handle end is the end nearer the hands. Missing frames are filled by linear
interpolation between detected frames (marked "interpolated"); frames before the first / after the last
detection stay empty. Output coordinates are source_512 pixels, like pose_data.json x * width.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from animate.skeleton import load_poses

ROOT = Path(__file__).resolve().parents[1]
BAT_CLASS = 34  # COCO "baseball bat"


def _axis(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    proj = (pts - c) @ d
    if s[0] < 2.5 * max(s[1], 1e-3):  # not elongated: not a bat
        return None
    return c + d * proj.min(), c + d * proj.max()


def _dist_point_segment(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-6), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def track(clip_dir, model="yolo26x-seg.pt", conf=0.10):
    from ultralytics import YOLO
    clip_dir = Path(clip_dir)
    poses, _, _ = load_poses(clip_dir / "pose_data.json")
    m = YOLO(str(ROOT / "models" / model) if (ROOT / "models" / model).exists() else model)
    cap = cv2.VideoCapture(str(clip_dir / "source_512.mp4"))
    cands, hands_at, i = {}, {}, 0   # pass 1: every elongated detection per frame
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = m.predict(f, classes=[BAT_CLASS], conf=conf, verbose=False, retina_masks=True)[0]
        body = poses.get(i, {}).get("body", {})
        hands_at[i] = [np.array(body[k]) for k in ("right_wrist", "left_wrist") if k in body]
        cands[i] = []
        if r.masks is not None:
            for mk, cf in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                ax = _axis(mk > 0.5)
                if ax is not None:
                    cands[i].append((np.array(ax[0]), np.array(ax[1]), float(cf)))
        i += 1
    n = i

    # pass 2: drop static lines (stumps, crease markings): same place in >= 8 frames
    def same(c, d):
        return min(np.linalg.norm(c[0] - d[0]) + np.linalg.norm(c[1] - d[1]),
                   np.linalg.norm(c[0] - d[1]) + np.linalg.norm(c[1] - d[0])) < 12
    # Stumps: static, near-vertical and reaching down to the batter's feet. (A bat held still in the stance
    # is static too, but it is diagonal and ends well above the feet, so it is kept.)
    feet = [lowest for lowest in (max([poses.get(k, {}).get("body", {}).get(j, (0, 0))[1]
                                       for j in ("right_ankle", "left_ankle")]) for k in range(n)) if lowest > 0]
    foot_y = float(np.median(feet)) if feet else 1e9

    def stump_like(c):
        dx, dy = abs(c[0][0] - c[1][0]), abs(c[0][1] - c[1][1])
        return dx < 0.2 * dy and max(c[0][1], c[1][1]) > foot_y - 60

    flat = [c for k in cands for c in cands[k]]
    static = [c for c in flat if stump_like(c) and sum(1 for d in flat if same(c, d)) >= 8]

    raw = {}  # pass 3: per frame, the moving line whose handle end is at the hands
    for k in range(n):
        hands, best = hands_at[k], None
        for a, b, cf in cands[k]:
            if any(same((a, b), s_) for s_ in static):
                continue
            if hands:  # handle = end nearer the hands; the handle itself must be at the hands
                hc = np.mean(hands, 0)
                if np.linalg.norm(b - hc) < np.linalg.norm(a - hc):
                    a, b = b, a
                d = min(float(np.linalg.norm(a - h)) for h in hands)
            else:
                d = 0.0
            score = d - 40 * cf
            if best is None or score < best[0]:
                best = (score, a, b, cf, d)
        if best is not None and best[4] < 45:  # handle within ~45 px of a wrist
            raw[k] = {"handle": best[1].round(1).tolist(), "toe": best[2].round(1).tolist(), "conf": round(best[3], 3)}
    det = sorted(raw)
    frames = []
    for k in range(n):
        if k in raw:
            frames.append({"frame_index": k, **raw[k], "interpolated": False})
            continue
        lo = max([j for j in det if j < k], default=None)
        hi = min([j for j in det if j > k], default=None)
        if lo is None or hi is None or hi - lo > 24:
            frames.append({"frame_index": k, "handle": None, "toe": None, "interpolated": False})
            continue
        t = (k - lo) / (hi - lo)
        lerp = lambda p, q: (np.array(raw[lo][p]) * (1 - t) + np.array(raw[hi][q]) * t).round(1).tolist()
        frames.append({"frame_index": k, "handle": lerp("handle", "handle"), "toe": lerp("toe", "toe"),
                       "conf": None, "interpolated": True})
    out = {"clip": clip_dir.name, "model": model, "class": "COCO baseball bat (34)", "frames": frames,
           "static_lines_rejected": len(static),
           "detected": len(det), "total": n,
           "note": "pixels of source_512.mp4; handle = end nearer the hands"}
    (clip_dir / "bat_track.json").write_text(json.dumps(out, indent=1))
    return out


def load_bat(clip_dir):
    p = Path(clip_dir) / "bat_track.json"
    if not p.exists():
        return {}
    return {f["frame_index"]: (tuple(f["handle"]), tuple(f["toe"])) for f in json.load(open(p))["frames"] if f["handle"]}


if __name__ == "__main__":
    o = track(sys.argv[1])
    print(f"{o['clip']}: bat detected in {o['detected']} of {o['total']} frames; "
          f"{sum(1 for f in o['frames'] if f['handle'])} with a bat after interpolation")
