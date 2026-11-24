"""Score a MusePose output against the skeleton it was asked to follow, and check body permanence.

    python -m animate.evaluate <generated.mp4> animate/jobs/<name>/target_pose.json [--out eval.json]

- PCK@a: share of target joints whose DWPose detection in the generated frame is within a x (longest side
  of the target body box) pixels. Missed joints count as wrong.
- completeness: share of frames where every limb joint (shoulders, elbows, wrists, hips, knees, ankles) that
  the target shows is also detected in the generated frame, i.e. no body part disappeared; plus the share of
  frames with a detected face (nose).
- merged_limb_frames: share of frames where a left/right wrist, elbow, knee or ankle pair collapses onto one
  spot although the target keeps them apart.
- silhouette_*: player outline stability (YOLO person masks, frame-to-frame IoU, drops below 0.7, max area
  change between frames).
"""
import argparse
import json

import cv2
import numpy as np
from PIL import Image

from animate.skeleton import BODY

LIMB_JOINTS = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist",
               "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]


def detect_video(path, size, device="cpu"):
    from easy_dwpose import DWposeDetector
    from pose_engine import _extract_person_from_dwpose, _select_primary_person
    det = DWposeDetector(device=device)
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, size)
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        raw = det(img, draw_pose=False, include_hands=False, include_face=False, detect_resolution=512)
        try:
            ex = _extract_person_from_dwpose(raw, _select_primary_person(raw))
        except Exception:
            out.append({})
            continue
        body = {}
        for j, pt, sc in zip(BODY, ex["body"], ex["body_scores"]):
            x, y = float(pt[0]), float(pt[1])
            if x <= 1.5 and y <= 1.5:
                x, y = x * size[0], y * size[1]
            if float(sc) >= 0.3:
                body[j] = (x, y)
        out.append(body)
    return out


PAIRS = [("right_wrist", "left_wrist"), ("right_elbow", "left_elbow"), ("right_knee", "left_knee"),
         ("right_ankle", "left_ankle")]


def merged_limbs(pred, target, near=0.04, far=0.10):
    """Frames where a left/right limb pair collapses onto one spot in the output (< near x body size) while the
    target keeps them apart (> far x body size): the 'limbs merging' failure."""
    bad = []
    for k in range(min(len(pred), len(target))):
        t = {j: v for j, v in target[k]["body"].items() if v is not None}
        p = pred[k]
        if len(t) < 4:
            continue
        pts = np.array(list(t.values()))
        span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0))
        for a, b in PAIRS:
            if a in t and b in t and a in p and b in p:
                if (np.hypot(*np.subtract(t[a], t[b])) > far * span
                        and np.hypot(*np.subtract(p[a], p[b])) < near * span):
                    bad.append(k)
                    break
    return bad


def silhouettes(path, size):
    """Player masks per frame (YOLO person segmentation, largest person) and frame-to-frame IoU: a sudden
    drop means part of the body vanished or the outline broke."""
    from pathlib import Path as _P
    from ultralytics import YOLO
    root = _P(__file__).resolve().parents[1]
    m = YOLO(str(root / "models" / "yolo26x-seg.pt"))
    cap = cv2.VideoCapture(str(path))
    masks = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, size)
        r = m.predict(f, classes=[0], conf=0.25, verbose=False, retina_masks=True)[0]
        if r.masks is None:
            masks.append(None)
            continue
        ms = r.masks.data.cpu().numpy() > 0.5
        masks.append(ms[int(np.argmax(ms.reshape(len(ms), -1).sum(1)))])
    cap.release()
    ious = []
    for a, b in zip(masks, masks[1:]):
        if a is None or b is None:
            ious.append(0.0)
        else:
            ious.append(float((a & b).sum() / max((a | b).sum(), 1)))
    areas = [0 if m_ is None else int(m_.sum()) for m_ in masks]
    return {"player_found_frames": round(sum(m_ is not None for m_ in masks) / max(len(masks), 1), 4),
            "silhouette_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
            "silhouette_iou_min": round(float(np.min(ious)), 4) if ious else None,
            "silhouette_drops_below_0.7": int(sum(i < 0.7 for i in ious)),
            "area_change_max": round(float(max(abs(a - b) / max(a, b, 1) for a, b in zip(areas, areas[1:]))), 4)
            if len(areas) > 1 else None}


def score(pred, target, alphas=(0.05, 0.1, 0.2)):
    n = min(len(pred), len(target))
    hits = {a: 0 for a in alphas}
    total, per_joint = 0, {j: [0, 0] for j in BODY}
    complete = face = 0
    for k in range(n):
        t = {j: v for j, v in target[k]["body"].items() if v is not None}
        p = pred[k]
        if not t:
            continue
        pts = np.array(list(t.values()))
        span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0))
        for j, tv in t.items():
            total += 1
            per_joint[j][1] += 1
            if j in p:
                d = float(np.hypot(p[j][0] - tv[0], p[j][1] - tv[1]))
                for a in alphas:
                    hits[a] += d <= a * span
                per_joint[j][0] += d <= 0.1 * span
        complete += all(j in p for j in LIMB_JOINTS if j in t)  # every limb joint the target shows is there
        face += "nose" in p
    return {
        "frames": n,
        **{f"pck@{a}": round(hits[a] / max(total, 1), 4) for a in alphas},
        "per_joint_pck@0.1": {j: round(h / c, 3) for j, (h, c) in per_joint.items() if c},
        "complete_body_frames": round(complete / max(n, 1), 4),
        "face_detected_frames": round(face / max(n, 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generated")
    ap.add_argument("target")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tg = json.load(open(a.target))
    pred = detect_video(a.generated, tuple(tg["canvas"]))
    res = {"generated": a.generated, "target": a.target, **score(pred, tg["frames"])}
    mg = merged_limbs(pred, tg["frames"])
    res["merged_limb_frames"] = round(len(mg) / max(len(pred), 1), 4)
    res["merged_limb_frame_list"] = mg
    res.update(silhouettes(a.generated, tuple(tg["canvas"])))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_joint_pck@0.1", "merged_limb_frame_list")}, indent=1))


if __name__ == "__main__":
    main()
