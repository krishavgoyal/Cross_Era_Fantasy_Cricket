"""Turn a 3D fit into the 2D pose files the scene/MusePose code reads (camera view, 0 deg), with hands.

    ~/gen/bin/python -m motion3d.export <name>

Writes data/motion3d/<name>/clip/ (a processed-clip folder: metadata.json, pose_data.json, bat_track.json,
source_512.mp4), so animate.scene.batting_scene / bowling_scene can use it unchanged:
  - batting: coordinates mapped back into each source_512 frame (the scene code re-applies the same camera
    tracking, so the ground stays where the fit put it); same timeline as the source
  - bowling: coordinates in the fit's world (the scene code gets identity crop windows and no raw clip, so it
    applies no camera motion again); the timeline is the fit's (slow-motion delivery)
Every body joint is present in every frame (confidence 1.0, "fitted": true).

Hands: DWPose hands are only kept as a SHAPE (21 points in the forearm's frame: origin wrist, x along the
forearm, unit = forearm length). Good detections (>= 15 points, plausible size, attached to the wrist) are
kept, gaps are interpolated, the rest use the median shape (template); the shape is smoothed in time and
carried by the fitted forearm. So a hand never vanishes, never jumps, and never detaches from its wrist.
"""
import json
import shutil
import sys

import cv2
import numpy as np
from scipy.signal import savgol_filter

from animate.skeleton import BODY
from motion3d.clips import CLIPS, out_dir
from motion3d.fit import JI, fix_swaps, load_obs


def hand_shapes(F, obs, side):
    """(T, 21, 2) hand shape in forearm coordinates per output frame + stats."""
    X, times = F["X"], F["times"]
    T = len(X)
    el, wr = JI[f"{side}_elbow"], JI[f"{side}_wrist"]
    L = float(F["L"][list(F["groups"]).index("forearm")])
    shapes = np.full((T, 21, 2), np.nan)
    window = np.zeros(T, bool)
    if "arm_window" in F and side == F.get("arm_side", ""):
        window = F["arm_window"]
    good = 0
    for t in range(T):
        s = times[t]
        k = int(round(s))
        if abs(s - k) > 0.2 or window[t]:
            continue
        h = obs[k]["hands"].get(side)
        if not h:
            continue
        pts = np.array([(x, y) for x, y, c in h])
        conf = np.array([c for x, y, c in h])
        if (conf >= 0.35).sum() < 15:
            continue
        w2, e2 = X[t, wr, :2], X[t, el, :2]
        ax = w2 - e2
        n = np.linalg.norm(ax)
        if n < 1e-3:
            continue
        ux = ax / n
        uy = np.array([-ux[1], ux[0]])
        rel = pts - w2
        sh = np.stack([rel @ ux, rel @ uy], 1) / L
        ok = conf >= 0.35
        size = np.ptp(sh[ok], axis=0)
        if np.hypot(*size) > 1.3 or np.linalg.norm(sh[0]) > 0.5:  # huge / detached: failed detection
            continue
        sh[~ok] = np.nan
        shapes[t] = sh
        good += 1
    template = np.nanmedian(shapes, axis=0) if good else None
    if template is None or np.isnan(template).any():
        # generic open hand along the forearm (OpenPose order: wrist, thumb 1-4, index..pinky 5-20)
        gen = [(0, 0)] + [(0.1 * i, 0.12 * i) for i in range(1, 5)]
        for f, off in enumerate((0.08, 0.02, -0.04, -0.09)):
            gen += [(0.25 + 0.1 * i, off) for i in range(4)]
        gen = np.array(gen, float)
        template = gen if template is None else np.where(np.isnan(template), gen, template)
    tt = np.arange(T)
    for i in range(21):  # interpolate gaps, template where there is nothing on either side
        for c in range(2):
            v = shapes[:, i, c]
            ok = ~np.isnan(v)
            if ok.sum() >= 2:
                v2 = np.interp(tt, tt[ok], v[ok])
                # beyond the first/last detection, and in long gaps (> 12 frames), ease to the template
                gap = np.array([min(abs(t - tt[ok]).min(), 12) / 12 for t in tt])
                v = v2 * (1 - gap) + template[i, c] * gap
            else:
                v = np.full(T, template[i, c])
            shapes[:, i, c] = v
    if T >= 7:
        shapes = savgol_filter(shapes, 7, 2, axis=0)
    return shapes, {"frames_with_good_detection": good, "frames_total": T}


def place_hand(X, shape, side, L):
    el, wr = JI[f"{side}_elbow"], JI[f"{side}_wrist"]
    w2, e2 = X[wr, :2], X[el, :2]
    ax = w2 - e2
    n = np.linalg.norm(ax)
    ux = ax / n if n > 1e-3 else np.array([1.0, 0.0])
    uy = np.array([-ux[1], ux[0]])
    return w2 + (shape[:, :1] * ux + shape[:, 1:] * uy) * L


def frame_pose(F, t, hands):
    X = F["X"][t]
    body = {n: X[JI[n], :2] for n in BODY if n in JI}
    body["neck"] = (body["right_shoulder"] + body["left_shoulder"]) / 2
    L = float(F["L"][list(F["groups"]).index("forearm")])
    out = {"body": body}
    for side in ("left", "right"):
        out[f"{side}_hand"] = place_hand(X, hands[side][t], side, L)
    if F["toe"].size:
        grip = (X[JI["right_wrist"], :2] + X[JI["left_wrist"], :2]) / 2
        out["bat"] = (grip, F["toe"][t, :2])
    return out


def run(name, view=0):
    """view: turn the 3D skeleton about the vertical before projecting (180 = seen from the other side,
    running the other way; still the same bowling arm). Written to clip/ (0) or clip_view<deg>/."""
    cfg = dict(CLIPS[name])
    o = out_dir(name)
    cfg["out"] = o
    d = np.load(o / "fit.npz")
    F = {k: d[k] for k in d.files}
    if view:
        from motion3d.render import view_points
        F["X"], toe, _, _ = view_points(F, view)
        if toe is not None:
            F["toe"] = toe
    fitj = json.load(open(o / "fit.json"))
    if "bowling_arm" in fitj and "keyframed_out_frames" in fitj["bowling_arm"]:
        a, b = fitj["bowling_arm"]["keyframed_out_frames"]
        w = np.zeros(len(F["X"]), bool)
        w[a:b + 1] = True
        F["arm_window"], F["arm_side"] = w, cfg["bowling_arm"]
    obs = load_obs(cfg)
    fix_swaps(obs)
    hands, hstats = {}, {}
    F0 = {k: v for k, v in F.items()}
    if view:  # hand shapes are measured in the camera view (detections), then mirrored with the view
        F0["X"] = np.load(o / "fit.npz")["X"]
    for side in ("left", "right"):
        hands[side], hstats[side] = hand_shapes(F0, obs, side)
        if view % 360 == 180:
            hands[side][:, :, 1] *= -1
    T = len(F["X"])
    clip = o / ("clip" if not view else f"clip_view{view}")
    clip.mkdir(exist_ok=True)
    src_meta = json.load(open(cfg["clip"] / "metadata.json"))
    cam = json.load(open(o / "camera.json"))["matrices"]
    batting = cfg["kind"] == "batting"
    if batting and view:
        raise ValueError("batting export is only in the camera view (the scene is the clip's own camera)")
    frames, bat_frames, meta_frames = [], [], []
    for t in range(T):
        p = frame_pose(F, t, hands)
        if batting:  # back into this source frame's pixels (the scene code re-applies the camera tracking)
            Minv = cv2.invertAffineTransform(np.array(cam[int(round(F["times"][t]))], np.float32))
            f = lambda q: Minv[:, :2] @ np.asarray(q, float) + Minv[:, 2]
        else:
            f = lambda q: np.asarray(q, float)
        kp = lambda n, q: {"joint": n, "x": float(f(q)[0] / 512), "y": float(f(q)[1] / 512), "confidence": 1.0,
                           "fitted": True}
        fr = {"frame_index": t, "timestamp_ms": round(t * 1000 / 24, 2),
              "source_frame": round(float(F["times"][t]), 3),
              "body_keypoints": [kp(n, p["body"][n]) for n in BODY],
              "hands": {side: [kp(f"hand_{i}", q) for i, q in enumerate(p[f"{side}_hand"])] for side in ("left", "right")}}
        frames.append(fr)
        if "bat" in p:
            g, toe = f(p["bat"][0]), f(p["bat"][1])
            bat_frames.append({"frame_index": t, "handle": [round(float(g[0]), 2), round(float(g[1]), 2)],
                               "toe": [round(float(toe[0]), 2), round(float(toe[1]), 2)], "conf": 1.0, "interpolated": False})
        if batting:
            meta_frames.append(src_meta["frames"][int(round(F["times"][t]))])
        else:
            meta_frames.append({"frame_index": t, "source_frame_index": int(round(F["times"][t])),
                                "crop_window_source_xyxy_px": [0, 0, 512, 512]})
    pose = {"schema_version": "1.1.0", "clip_id": f"{name}_motion3d", "fps": 24.0, "frame_count": T,
            "resolution": [512, 512], "coordinate_system": "normalized_xy_0_to_1_origin_top_left" + (
                "" if batting else " (of the motion3d world: x/512, y/512; may exceed 1)"),
            "z_coordinate": "see fit.npz (3D)", "note": "motion3d fit: rigid bones, planted feet, every joint every frame",
            "frames": frames}
    (clip / "pose_data.json").write_text(json.dumps(pose))
    if bat_frames:
        (clip / "bat_track.json").write_text(json.dumps({
            "clip": name, "model": "motion3d fit (rigid bat)", "frames": bat_frames, "detected": len(bat_frames),
            "total": T, "note": "pixels of source_512.mp4; handle = grip (between the wrists), toe = fitted rigid bat"}))
    meta = dict(src_meta)
    meta["frames"] = meta_frames
    if not batting:
        meta["source"] = dict(src_meta["source"], path="none (motion3d world coordinates; no camera motion to re-apply)")
    meta["motion3d"] = {"times_source_frames": [round(float(v), 3) for v in F["times"]], "view_deg": view}
    (clip / "metadata.json").write_text(json.dumps(meta))
    shutil.copy(cfg["clip"] / "source_512.mp4", clip / "source_512.mp4")
    info = {"frames": T, "hands": hstats, "bat_frames": len(bat_frames)}
    (o / ("export.json" if not view else f"export_view{view}.json")).write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--view=")]
    view = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--view=")), 0)
    for n in args or list(CLIPS):
        print(n, run(n, view))
