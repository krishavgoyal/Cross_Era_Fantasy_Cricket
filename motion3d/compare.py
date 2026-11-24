"""Compare the skeletons that drive MusePose: previous jobs (animate/, 2D cleaning) vs motion3d jobs.

    ~/gen/bin/python -m motion3d.compare        # -> outputs/motion3d/skeleton_comparison.json + *_old_vs_new.mp4

Measured on target_pose.json (the exact skeleton drawn into pose.mp4, canvas pixels):
  bone_cv            coefficient of variation of each limb's on-screen length (median over limbs); 3D-rigid
                     bones still change on screen with foreshortening, but smoothly
  bone_jump_p99      99th pct of frame-to-frame limb length change / limb length (sudden stretching)
  jerk_p99           99th pct |acceleration| of body joints (px/frame^2): jitter and sudden starts
  speed_spike        max joint speed / median joint speed
  missing_joint_frames  frames with at least one of the 18 body joints missing
  bat_len_change_p99 99th pct frame-to-frame change of the bat's on-screen length / median length
"""
import json
from pathlib import Path

import cv2
import numpy as np

from animate.skeleton import BODY, draw

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "animate" / "jobs"
OUT = ROOT / "outputs" / "motion3d"
PAIRS = {"batting": ("bradman_batting", "bradman_batting_3d"), "bowling": ("mcgrath_bowling", "mcgrath_bowling_3d")}
LIMBS = [("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_elbow"),
         ("left_elbow", "left_wrist"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle")]


def metrics(job):
    tp = json.load(open(JOBS / job / "target_pose.json"))
    fr = tp["frames"]
    T = len(fr)
    P = np.full((T, len(BODY), 2), np.nan)
    for t, f in enumerate(fr):
        for i, n in enumerate(BODY):
            if f["body"].get(n):
                P[t, i] = f["body"][n]
    cv, jump = [], []
    for a, b in LIMBS:
        l = np.linalg.norm(P[:, BODY.index(a)] - P[:, BODY.index(b)], axis=1)
        m = np.nanmedian(l)
        cv.append(np.nanstd(l) / m)
        jump.append(np.nanpercentile(np.abs(np.diff(l)) / m, 99))
    acc = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=2)
    v = np.linalg.norm(P[1:] - P[:-1], axis=2)
    out = {"frames": T, "bone_cv": round(float(np.median(cv)), 3), "bone_jump_p99": round(float(np.max(jump)), 3),
           "jerk_p99_px": round(float(np.nanpercentile(acc, 99)), 1),
           "speed_spike": round(float(np.nanmax(v) / np.nanmedian(v)), 1),
           "missing_joint_frames": int(np.isnan(P[:, :, 0]).any(1).sum())}
    bats = [f["bat"] for f in fr if f.get("bat")]
    if bats:
        bl = np.array([np.hypot(b[1][0] - b[0][0], b[1][1] - b[0][1]) for b in bats])
        out["bat_frames"] = len(bats)
        out["bat_len_change_p99"] = round(float(np.percentile(np.abs(np.diff(bl)), 99) / np.median(bl)), 3)
        out["bat_len_range_px"] = [round(float(bl.min()), 1), round(float(bl.max()), 1)]
    return out, tp


def video(old, new, out):
    a, b = old[1], new[1]
    Wa, Ha = a["canvas"]
    Wb, Hb = b["canvas"]
    n = max(len(a["frames"]), len(b["frames"]))
    vw = None
    for t in range(n):
        tiles = []
        for tp, (W, H), label in ((a, (Wa, Ha), "before (2D cleaning)"), (b, (Wb, Hb), "now (rigid 3D fit)")):
            f = tp["frames"][min(t, len(tp["frames"]) - 1)]
            p = {"body": {k: tuple(v) for k, v in f["body"].items() if v},
                 "left_hand": [None if q is None else tuple(q) for q in f["left_hand"]],
                 "right_hand": [None if q is None else tuple(q) for q in f["right_hand"]],
                 "bat": None if not f.get("bat") else (tuple(f["bat"][0]), tuple(f["bat"][1]))}
            im = cv2.cvtColor(draw(p, (W, H)), cv2.COLOR_RGB2BGR)
            if f.get("ball"):
                cv2.circle(im, (int(f["ball"][0]), int(f["ball"][1])), 6, (255, 255, 255), -1)
            cv2.putText(im, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(im, (int(W * 632 / H), 632)))
        frame = np.hstack(tiles)
        if vw is None:
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 24, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    vw.release()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for k, (old, new) in PAIRS.items():
        mo, mn = metrics(old), metrics(new)
        res[k] = {"before": {"job": old, **mo[0]}, "now": {"job": new, **mn[0]}}
        video(mo, mn, OUT / f"{k}_skeleton_old_vs_new.mp4")
    (OUT / "skeleton_comparison.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
