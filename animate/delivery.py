"""Rebuild a bowler's delivery: slow it down and drive the bowling arm through a full over-the-top rotation.

In the slow-motion source the whole arm rotation and release take about 4-6 frames (Cummins clip: wrist low at
frame 34, vertical at 38, down at 40; frame 36 has no joints at all). Detected joints there are blurred and
swap sides, so the arm looked like it never rotated. This module:
  - time-stretches the delivery window (default 2.5x) by interpolating every joint between measured frames;
  - replaces the bowling arm with a straight arm (cricket law) of the measured length, whose angle around the
    shoulder goes from its measured start (arm behind/down) over the top (straight up) to its measured end
    (follow-through), with ease-in/ease-out so it is fastest at release;
  - returns the release index (arm just past vertical) for the ball.
This is reconstructed motion and is labelled as such in the job metadata.
"""
import numpy as np


def _interp_pose(a, b, t):
    """Linear in time between two poses (joints present in both; else the nearer one)."""
    out = {"body": {}, "left_hand": [None] * 21, "right_hand": [None] * 21, "bat": None}
    for j in set(a["body"]) | set(b["body"]):
        if j in a["body"] and j in b["body"]:
            out["body"][j] = tuple(np.array(a["body"][j]) * (1 - t) + np.array(b["body"][j]) * t)
        else:
            out["body"][j] = (a if t < 0.5 else b)["body"].get(j) or (b if t < 0.5 else a)["body"][j]
    for side in ("left_hand", "right_hand"):
        ha, hb = a[side], b[side]
        out[side] = [None if (p is None or q is None) else tuple(np.array(p) * (1 - t) + np.array(q) * t)
                     for p, q in zip(ha, hb)]
    return out


def _ease(t):
    return t * t * (3 - 2 * t)  # smoothstep: slow at both ends, fastest in the middle (release)


def rebuild(seq, src_frames, window, stretch=2.5, side="right", run_dir=-1.0):
    """seq/src_frames: poses and their source frame numbers. window = (first, last) source frame of the delivery.
    Returns (new_seq, new_times, info). new_times are (fractional) source frame numbers per output frame."""
    pos = {f: i for i, f in enumerate(src_frames)}
    i0, i1 = pos[window[0]], pos[window[1]]
    sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
    hand = f"{side}_hand"
    # arm lengths: median over frames where the arm was detected outside the blurred window
    ups = [np.hypot(*np.subtract(p["body"][el], p["body"][sh])) for p in seq if sh in p["body"] and el in p["body"]]
    los = [np.hypot(*np.subtract(p["body"][wr], p["body"][el])) for p in seq if el in p["body"] and wr in p["body"]]
    L_up, L_lo = float(np.median(ups)), float(np.median(los))

    def angle(p):
        b = p["body"]
        if sh in b and wr in b:
            return float(np.arctan2(b[wr][1] - b[sh][1], b[wr][0] - b[sh][0]))
        return None
    th0, th1 = angle(seq[i0]), angle(seq[i1])
    if th0 is None or th1 is None:
        raise ValueError("bowling arm not measured at the window ends")
    up = -np.pi / 2  # straight up in image coordinates
    # Rotation direction from physics, not from the (blurred, side-swapping) labels: the arm comes up from
    # behind and goes over the top towards the batter, i.e. counter-clockwise on screen (angle decreasing) for
    # a bowler running left, clockwise for one running right. Turn from th0 that way until th1, and if that
    # does not pass straight up, go one more full turn (the arm swings down, back, up and over).
    direction = -1.0 if run_dir < 0 else 1.0
    d = ((th1 - th0) * direction) % (2 * np.pi)
    to_up = ((up - th0) * direction) % (2 * np.pi)
    if to_up > d:
        d += 2 * np.pi
    best = direction * d
    # Hand template: the bowling hand's shape in forearm coordinates (origin wrist, x along elbow->wrist, unit =
    # forearm length), median over frames where it was well detected. It rides along the rebuilt arm, so
    # MusePose always gets a hand there (with no hand points it drew a white blob in the hand).
    shapes = []
    for p in seq:
        b = p["body"]
        hs = p[hand]
        if el in b and wr in b and sum(h is not None for h in hs) >= 15:
            e, w = np.array(b[el]), np.array(b[wr])
            ax = w - e
            L = float(np.linalg.norm(ax))
            if L < 1:
                continue
            ux, uy = ax / L, np.array([-ax[1], ax[0]]) / L
            shapes.append([None if h is None else (float(np.dot(np.array(h) - w, ux) / L), float(np.dot(np.array(h) - w, uy) / L))
                           for h in hs])
    template = None
    if shapes:
        template = []
        for i in range(21):
            pts = [sh[i] for sh in shapes if sh[i] is not None]
            template.append(tuple(np.median(pts, axis=0)) if pts else None)
    n_out = int(round((i1 - i0) * stretch)) + 1
    new_mid, times = [], []
    release = None
    for k in range(n_out):
        u = k / (n_out - 1)
        x = i0 + u * (i1 - i0)
        a, b = int(np.floor(x)), min(int(np.floor(x)) + 1, len(seq) - 1)
        p = _interp_pose(seq[a], seq[b], x - a)
        th = th0 + best * _ease(u)
        s = np.array(p["body"][sh])
        d = np.array([np.cos(th), np.sin(th)])
        p["body"][el] = tuple(s + d * L_up)
        p["body"][wr] = tuple(s + d * (L_up + L_lo))  # straight arm
        if template is not None:
            w = np.array(p["body"][wr])
            ux, uy = d, np.array([-d[1], d[0]])
            p[hand] = [None if q is None else tuple(w + (q[0] * ux + q[1] * uy) * L_lo) for q in template]
        else:
            p[hand] = [None] * 21
        if release is None and abs(((th - up + np.pi) % (2 * np.pi)) - np.pi) < 0.25:
            release = len(new_mid) + 1  # the arm is at the top: the ball leaves just past vertical
        new_mid.append(p)
        times.append(float(np.interp(x, range(len(src_frames)), src_frames)))
    new_seq = list(seq[:i0]) + new_mid + list(seq[i1 + 1:])
    new_times = [float(f) for f in src_frames[:i0]] + times + [float(f) for f in src_frames[i1 + 1:]]
    rel_global = i0 + (release if release is not None else n_out // 2)
    info = {"window_source_frames": list(window), "stretch": stretch, "frames_out": n_out, "bowling_arm": side,
            "arm_lengths_px": [round(L_up, 1), round(L_lo, 1)], "arm_rotation_deg": round(float(np.degrees(best)), 1),
            "release_index": int(rel_global), "hand_template_frames": len(shapes),
            "note": "reconstructed delivery: straight bowling arm, eased rotation, measured hand shape carried along"}
    return new_seq, new_times, info
