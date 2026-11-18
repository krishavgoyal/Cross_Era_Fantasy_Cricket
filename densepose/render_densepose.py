"""Draw DensePose-style part masks (ids 1..24) from DWPose keypoints in pose_data.json.

This is NOT DensePose. It is the skeleton-shape approximation used to fill body parts that real
DensePose loses on a few motion-blurred frames (see patch_weak_frames.py), and as a fallback
control when detectron2 is not available.

    p, body, hands, face = load("pose_data.json")   # body (N,18,3): x_px, y_px, conf
    body = fill_gaps(body)
    state = {}
    labels = [render_frame(t, body, hands, face, state) for t in range(N)]   # sequential: front/back state
"""
import json

import cv2
import numpy as np

# OpenPose-18 order used by pose_data.json
NOSE, NECK, RSH, REL, RWR, LSH, LEL, LWR, RHIP, RKNE, RANK, LHIP, LKNE, LANK, REYE, LEYE, REAR, LEAR = range(18)
CONF = 0.3


def load(path):
    p = json.load(open(path))
    W, H = p.get("resolution", [512, 512])
    n = len(p["frames"])
    body = np.zeros((n, 18, 3), np.float32)
    hands = np.zeros((n, 2, 21, 3), np.float32)
    face = np.zeros((n, 68, 3), np.float32)
    for t, f in enumerate(p["frames"]):
        for j, k in enumerate(f["body_keypoints"]):
            body[t, j] = (k["x"] * W, k["y"] * H, k.get("confidence", 0.0))
        for h, side in enumerate(("left", "right")):
            for j, k in enumerate((f.get("hands") or {}).get(side) or []):
                hands[t, h, j] = (k["x"] * W, k["y"] * H, k.get("confidence", 0.0))
        for j, k in enumerate(f.get("face_keypoints") or []):
            face[t, j] = (k["x"] * W, k["y"] * H, k.get("confidence", 0.0))
    return p, body, hands, face


def fill_gaps(body, max_gap=6):
    """Linearly interpolate joints missing for <= max_gap frames; longer gaps are held."""
    body = body.copy()
    n = len(body)
    for j in range(18):
        ok = body[:, j, 2] >= CONF
        if ok.sum() == 0:
            continue
        idx = np.where(ok)[0]
        for t in range(n):
            if ok[t]:
                continue
            prev = idx[idx < t]
            nxt = idx[idx > t]
            if len(prev) and len(nxt) and nxt[0] - prev[-1] <= max_gap + 1:
                a, b = prev[-1], nxt[0]
                w = (t - a) / (b - a)
                body[t, j, :2] = body[a, j, :2] * (1 - w) + body[b, j, :2] * w
                body[t, j, 2] = CONF
            elif len(prev):
                body[t, j, :2] = body[prev[-1], j, :2]
                body[t, j, 2] = CONF * 0.99
    return body


def _ok(b, *js):
    return all(b[j, 2] >= CONF * 0.99 for j in js)


def _limb(img, a, b, w, pid):
    cv2.line(img, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), int(pid), max(1, int(round(w))), cv2.LINE_8)
    cv2.circle(img, (int(round(b[0])), int(round(b[1]))), max(1, int(round(w / 2))), int(pid), -1)


def _scale(b):
    """Torso length in pixels, the unit for limb thickness."""
    if _ok(b, NECK, RHIP) and _ok(b, NECK, LHIP):
        mid = (b[RHIP, :2] + b[LHIP, :2]) / 2
        return max(8.0, float(np.linalg.norm(b[NECK, :2] - mid)))
    if _ok(b, NECK, NOSE):
        return max(8.0, 2.5 * float(np.linalg.norm(b[NECK, :2] - b[NOSE, :2])))
    return 40.0


def render_frame(t, body, hands, face, state, size=512):
    b = body[t]
    img = np.zeros((size, size), np.uint8)
    s = _scale(b)

    # facing: subject's right shoulder on image left = facing camera (front). Hysteresis via state.
    front = state.get("front", True)
    if _ok(b, RSH, LSH):
        d = (b[LSH, 0] - b[RSH, 0]) / s
        if d > 0.08:
            front = True
        elif d < -0.08:
            front = False
    state["front"] = front

    # torso
    pts = [j for j in (RSH, LSH, LHIP, RHIP) if _ok(b, j)]
    if len(pts) >= 3:
        poly = np.array([b[j, :2] for j in pts], np.int32)
        cv2.fillConvexPoly(img, cv2.convexHull(poly), 2 if front else 1)
        cv2.polylines(img, [cv2.convexHull(poly)], True, 2 if front else 1, max(1, int(s * 0.18)))

    # legs: (hip, knee, ankle, upper_front, upper_back, lower_front, lower_back, foot)
    legs = [(RHIP, RKNE, RANK, 9, 7, 13, 11, 6), (LHIP, LKNE, LANK, 10, 8, 14, 12, 5)]
    for hip, kne, ank, uf, ub, lf, lb, foot in legs:
        if _ok(b, hip, kne):
            _limb(img, b[hip, :2], b[kne, :2], s * 0.30, uf if front else ub)
        if _ok(b, kne, ank):
            _limb(img, b[kne, :2], b[ank, :2], s * 0.24, lf if front else lb)
        if _ok(b, kne, ank):
            v = b[ank, :2] - b[kne, :2]
            v = v / (np.linalg.norm(v) + 1e-6)
            toe = b[ank, :2] + v * s * 0.10 + np.array([s * 0.12 * (1 if (hip == LHIP) == front else -1), 0])
            _limb(img, b[ank, :2], toe, s * 0.16, foot)

    # arms: (sh, el, wr, upper_in, upper_out, lower_in, lower_out, hand)
    arms = [(RSH, REL, RWR, 16, 18, 20, 22, 3), (LSH, LEL, LWR, 15, 17, 19, 21, 4)]
    for sh, el, wr, ui, uo, li, lo, hand in arms:
        if _ok(b, sh, el):
            _limb(img, b[sh, :2], b[el, :2], s * 0.20, uo if front else ui)
        if _ok(b, el, wr):
            _limb(img, b[el, :2], b[wr, :2], s * 0.16, lo if front else li)
            v = b[wr, :2] - b[el, :2]
            v = v / (np.linalg.norm(v) + 1e-6)
            _limb(img, b[wr, :2], b[wr, :2] + v * s * 0.14, s * 0.15, hand)

    # head
    if _ok(b, NOSE, NECK):
        r = max(3, int(s * 0.22))
        c = (int(b[NOSE, 0]), int(b[NOSE, 1] - r * 0.3))
        cv2.ellipse(img, c, (r, int(r * 1.2)), 0, 90, 270, 23 if front else 24, -1)
        cv2.ellipse(img, c, (r, int(r * 1.2)), 0, -90, 90, 24 if front else 23, -1)
    return img
