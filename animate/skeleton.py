"""Skeleton motion (body + hands + bat) from Student 1's pose_data.json, retargeted onto a reference
player photo and drawn the way DWPose/MusePose draw poses.

Coordinates are pixels. A "pose" is a dict:
    {"body": {joint: (x, y)}, "left_hand": [(x, y) or None] * 21, "right_hand": [...], "bat": ((x1,y1),(x2,y2)) or None}
"""
import json
import math

import cv2
import numpy as np

BODY = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
        "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
        "right_eye", "left_eye", "right_ear", "left_ear"]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
         (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
           (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
           (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
              (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
BAT_COLOUR = (200, 160, 90)  # light wood; not a DWPose colour, so the bat can't be mistaken for a limb


def load_poses(pose_json, body_conf=0.3, hand_conf=0.35):
    """frame_index -> pose (pixels of the pose data's resolution). Low-confidence points are dropped."""
    d = json.load(open(pose_json))
    w, h = d.get("resolution", [512, 512])
    out = {}
    for fr in d["frames"]:
        body = {k["joint"]: (k["x"] * w, k["y"] * h) for k in fr.get("body_keypoints", [])
                if k.get("confidence", 0) >= body_conf or k.get("confidence_interpolated")}
        pose = {"body": body, "bat": None}
        for side in ("left", "right"):
            pts = (fr.get("hands") or {}).get(side) or []
            pose[f"{side}_hand"] = [((p["x"] * w, p["y"] * h) if p.get("confidence", 0) >= hand_conf else None)
                                    for p in pts] if len(pts) == 21 else [None] * 21
        out[int(fr["frame_index"])] = pose
    return out, (w, h), float(d.get("fps", 24.0))


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def torso_len(body):
    hips = [body[k] for k in ("right_hip", "left_hip") if k in body]
    if "neck" not in body or not hips:
        return None
    mh = np.mean(hips, axis=0)
    return math.hypot(body["neck"][0] - mh[0], body["neck"][1] - mh[1])


def hip_centre(body):
    hips = [body[k] for k in ("right_hip", "left_hip") if k in body]
    return tuple(np.mean(hips, axis=0)) if hips else None


def lowest_y(body):
    ys = [body[k][1] for k in ("right_ankle", "left_ankle") if k in body]
    return max(ys) if ys else None


def transform_pose(pose, s, tx, ty):
    f = (lambda p: None if p is None else (p[0] * s + tx, p[1] * s + ty))
    out = {"body": {k: f(v) for k, v in pose["body"].items()},
           "left_hand": [f(p) for p in pose["left_hand"]], "right_hand": [f(p) for p in pose["right_hand"]],
           "bat": None if pose.get("bat") is None else (f(pose["bat"][0]), f(pose["bat"][1]))}
    return out


def retarget(seq, ref_body, canvas, margin=0.03):
    """Uniform scale + translation so the source player has the reference player's size and position.

    seq: list of poses (source pixels). ref_body: body joints of the player in the reference image (canvas
    pixels). Scale = median source torso -> reference torso; the first frame's hip centre is placed on the
    reference hip centre (x) and the feet on the reference feet (y). Motion is kept exactly (no per-bone
    changes), then the whole sequence is shifted/shrunk only if it would leave the canvas.
    Returns (poses, info)."""
    W, H = canvas
    src_t = [t for t in (torso_len(p["body"]) for p in seq) if t]
    ref_t = torso_len(ref_body)
    if not src_t or not ref_t:
        raise ValueError("torso not found in source or reference pose")
    s = ref_t / float(np.median(src_t))
    first = next(p["body"] for p in seq if hip_centre(p["body"]) and lowest_y(p["body"]))
    src_hip, src_low = hip_centre(first), lowest_y(first)
    ref_hip, ref_low = hip_centre(ref_body), lowest_y(ref_body) or (H * (1 - margin))
    tx = ref_hip[0] - src_hip[0] * s
    ty = ref_low - src_low * s
    out = [transform_pose(p, s, tx, ty) for p in seq]
    # keep everything on the canvas
    pts = np.array([q for p in out for q in list(p["body"].values()) + [h for h in p["left_hand"] + p["right_hand"] if h]])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    fit = min(1.0, (W * (1 - 2 * margin)) / max(1.0, x1 - x0), (H * (1 - 2 * margin)) / max(1.0, y1 - y0))
    dx = dy = 0.0
    if fit < 1.0:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        out = [transform_pose(p, fit, cx * (1 - fit), cy * (1 - fit)) for p in out]
        pts = (pts - [cx, cy]) * fit + [cx, cy]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
    if x0 < W * margin:
        dx = W * margin - x0
    elif x1 > W * (1 - margin):
        dx = W * (1 - margin) - x1
    if y0 < H * margin:
        dy = H * margin - y0
    elif y1 > H * (1 - margin):
        dy = H * (1 - margin) - y1
    if dx or dy:
        out = [transform_pose(p, 1.0, dx, dy) for p in out]
    return out, {"scale": round(s * fit, 4), "fit": round(fit, 4), "shift": [round(dx, 1), round(dy, 1)]}


def draw(pose, size, hands=True, bat=True, stick=4):
    """RGB uint8 image, black background, DWPose drawing convention (body limbs at 60%, joints full colour,
    hand edges in HSV rainbow, hand joints blue). Optionally the bat as a light-wood stick."""
    W, H = size
    canvas = np.zeros((H, W, 3), np.uint8)
    xy = [pose["body"].get(j) for j in BODY]
    for i, (a, b) in enumerate(LIMBS):
        if xy[a] is None or xy[b] is None:
            continue
        (xa, ya), (xb, yb) = xy[a], xy[b]
        length = math.hypot(xa - xb, ya - yb)
        angle = math.degrees(math.atan2(ya - yb, xa - xb))
        poly = cv2.ellipse2Poly((int((xa + xb) / 2), int((ya + yb) / 2)), (int(length / 2), stick), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in COLOURS[i]])
    for i, p in enumerate(xy):
        if p is not None:
            cv2.circle(canvas, (int(p[0]), int(p[1])), stick, COLOURS[i], thickness=-1)
    if hands:
        for side in ("left_hand", "right_hand"):
            pk = pose[side]
            for ie, (a, b) in enumerate(HAND_EDGES):
                if pk[a] is None or pk[b] is None:
                    continue
                rgb = cv2.cvtColor(np.uint8([[[int(180 * ie / len(HAND_EDGES)), 255, 255]]]), cv2.COLOR_HSV2RGB)[0, 0]
                cv2.line(canvas, tuple(int(v) for v in pk[a]), tuple(int(v) for v in pk[b]), [int(c) for c in rgb], 2)
            for p in pk:
                if p is not None:
                    cv2.circle(canvas, (int(p[0]), int(p[1])), 3, (0, 0, 255), thickness=-1)
    if bat and pose.get("bat"):
        (x1, y1), (x2, y2) = pose["bat"]
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), BAT_COLOUR, max(3, stick * 2), cv2.LINE_AA)
    return canvas


# Kinematic tree from the hip centre ("root"); each bone = (parent, child). Hands follow their wrist.
TREE = [("root", "neck"), ("neck", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("root", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
        ("root", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("neck", "nose"), ("nose", "right_eye"), ("nose", "left_eye"), ("right_eye", "right_ear"), ("left_eye", "left_ear")]


def retarget_bones(seq, ref_body, clip=(0.8, 1.25)):
    """Per-bone retargeting (as MusePose's pose_align does): every bone keeps the source direction and its
    frame-to-frame length changes (foreshortening), scaled to the reference player's own bone length.
    Ratio per bone = reference length / median source length, clipped to clip x the torso ratio.
    Returns poses in source coordinates rebuilt from each frame's root (hip centre), plus the ratios."""
    def withroot(b):
        b = dict(b)
        hc = hip_centre(b)
        if hc:
            b["root"] = hc
        return b
    rb = withroot(ref_body)
    sb = [withroot(p["body"]) for p in seq]

    def blen(b, a, c):
        return math.hypot(b[c][0] - b[a][0], b[c][1] - b[a][1]) if a in b and c in b else None

    t_src = [blen(b, "root", "neck") for b in sb if blen(b, "root", "neck")]
    torso_ratio = blen(rb, "root", "neck") / max(float(np.median(t_src)), 1e-3)
    # Limb lengths only, each averaged over left and right: DWPose may label the photo's legs the other way
    # round from the clip's (front-on photo vs side-on clip), and widths (shoulders, hips) depend on the view,
    # so those and the head keep the torso ratio.
    LIMBS_LR = {"upper_arm": [("right_shoulder", "right_elbow"), ("left_shoulder", "left_elbow")],
                "forearm": [("right_elbow", "right_wrist"), ("left_elbow", "left_wrist")],
                "thigh": [("right_hip", "right_knee"), ("left_hip", "left_knee")],
                "shin": [("right_knee", "right_ankle"), ("left_knee", "left_ankle")]}
    ratios = {}
    for limb, bones in LIMBS_LR.items():
        ref_l = [blen(rb, a, c) for a, c in bones if blen(rb, a, c)]
        src_l = [l for b in sb for a, c in bones for l in [blen(b, a, c)] if l]
        if ref_l and src_l:
            r = float(np.clip(np.mean(ref_l) / max(float(np.median(src_l)), 1e-3),
                              clip[0] * torso_ratio, clip[1] * torso_ratio))
            for bone in bones:
                ratios[bone] = r
    out = []
    for p, b in zip(seq, sb):
        if "root" not in b:
            out.append(p)
            continue
        nb = {"root": b["root"]}
        for a, c in TREE:  # parents come before children in TREE
            if a in nb and c in b and a in b:
                r = ratios.get((a, c), torso_ratio)
                nb[c] = (nb[a][0] + (b[c][0] - b[a][0]) * r, nb[a][1] + (b[c][1] - b[a][1]) * r)
        q = {"body": {k: v for k, v in nb.items() if k != "root"}, "bat": p.get("bat")}
        for side, wj, ej in (("left_hand", "left_wrist", "left_elbow"), ("right_hand", "right_wrist", "right_elbow")):
            hs = p[side]
            if wj in b and wj in nb and any(h is not None for h in hs):
                r = ratios.get((ej, wj), torso_ratio)
                q[side] = [None if h is None else (nb[wj][0] + (h[0] - b[wj][0]) * r, nb[wj][1] + (h[1] - b[wj][1]) * r)
                           for h in hs]
            else:
                q[side] = [None] * 21
        out.append(q)
    return out, {f"{a}->{c}": round(v / torso_ratio, 3) for (a, c), v in ratios.items() if a.startswith("right")}, torso_ratio
