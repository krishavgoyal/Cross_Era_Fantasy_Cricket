"""Clean a skeleton sequence before it drives MusePose, so limbs stay separate and move like the source action.

Problems seen in the first MusePose run (limbs swapping, stretching, merging, random hands) mostly come from
the input skeleton: DWPose on fast, blurred broadcast frames swaps left/right limbs for a frame, shrinks or
stretches bones, and returns half-detected hands. Steps, in order:
  1. fix_swaps      left/right arm and leg labels swapped between frames -> swap back (temporal consistency)
  2. fill_gaps_polar  elbows/wrists/knees/ankles missing for <= 12 frames -> angle + length around the parent
     fill_gaps      other joints missing for <= 8 frames -> linear from neighbours (animate.build_jobs.fill_gaps)
  3. clamp_bones    each bone kept within [0.75, 1.10] x its median length (foreshortening allowed, glitches not)
  4. smooth         Savitzky-Golay (body window 7, hands 5): removes jitter, keeps the fast peaks of the action
  5. clean_hands    a hand with < 12 of 21 points is dropped for that frame (MusePose then draws a whole hand)
"""
import numpy as np
from scipy.signal import savgol_filter

from animate.skeleton import BODY

PAIRS = [("right_shoulder", "left_shoulder"), ("right_elbow", "left_elbow"), ("right_wrist", "left_wrist"),
         ("right_hip", "left_hip"), ("right_knee", "left_knee"), ("right_ankle", "left_ankle")]
ARM = [("right_shoulder", "left_shoulder"), ("right_elbow", "left_elbow"), ("right_wrist", "left_wrist")]
LEG = [("right_hip", "left_hip"), ("right_knee", "left_knee"), ("right_ankle", "left_ankle")]
BONES = [("neck", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("neck", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("neck", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("neck", "nose")]


def _d(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def fix_swaps(seq):
    """If swapping a whole arm (or leg) chain's left/right labels moves it much closer to the previous frame,
    swap it (and the matching hand for arms). Returns (seq, n_swaps)."""
    n = 0
    for k in range(1, len(seq)):
        prev, cur = seq[k - 1]["body"], seq[k]["body"]
        for chain in (ARM, LEG):
            keep = swap = 0.0
            m = 0
            for r, l in chain:
                if r in prev and l in prev and r in cur and l in cur:
                    keep += _d(cur[r], prev[r]) + _d(cur[l], prev[l])
                    swap += _d(cur[l], prev[r]) + _d(cur[r], prev[l])
                    m += 1
            if m >= 2 and swap < 0.6 * keep:
                for r, l in chain:
                    a, b = cur.get(r), cur.get(l)
                    cur.pop(r, None), cur.pop(l, None)
                    if b is not None:
                        cur[r] = b
                    if a is not None:
                        cur[l] = a
                if chain is ARM:
                    seq[k]["left_hand"], seq[k]["right_hand"] = seq[k]["right_hand"], seq[k]["left_hand"]
                n += 1
    return seq, n


def clamp_bones(seq, lo=0.75, hi=1.10):
    """Walk each bone from the neck outwards; rescale bones far from their median length."""
    med = {}
    for a, b in BONES:
        ls = [_d(p["body"][a], p["body"][b]) for p in seq if a in p["body"] and b in p["body"]]
        if ls:
            med[(a, b)] = float(np.median(ls))
    n = 0
    for p in seq:
        body = p["body"]
        for a, b in BONES:
            if (a, b) not in med or a not in body or b not in body:
                continue
            L = _d(body[a], body[b])
            want = min(max(L, lo * med[(a, b)]), hi * med[(a, b)])
            if L > 1e-3 and abs(want - L) > 0.5:
                s = want / L
                dx, dy = (body[b][0] - body[a][0]) * s, (body[b][1] - body[a][1]) * s
                shift = (body[a][0] + dx - body[b][0], body[a][1] + dy - body[b][1])
                # move the child joint and everything below it on the chain by the same amount
                for j in _below(b):
                    if j in body:
                        body[j] = (body[j][0] + shift[0], body[j][1] + shift[1])
                for side in ("left_hand", "right_hand"):
                    if b.startswith(side.split("_")[0]) and b.endswith(("elbow", "wrist")):
                        p[side] = [None if q is None else (q[0] + shift[0], q[1] + shift[1]) for q in p[side]]
                n += 1
    return seq, n


def _below(j):
    chains = {"right_shoulder": ["right_shoulder", "right_elbow", "right_wrist"],
              "right_elbow": ["right_elbow", "right_wrist"], "right_wrist": ["right_wrist"],
              "left_shoulder": ["left_shoulder", "left_elbow", "left_wrist"],
              "left_elbow": ["left_elbow", "left_wrist"], "left_wrist": ["left_wrist"],
              "right_hip": ["right_hip", "right_knee", "right_ankle"], "right_knee": ["right_knee", "right_ankle"],
              "right_ankle": ["right_ankle"], "left_hip": ["left_hip", "left_knee", "left_ankle"],
              "left_knee": ["left_knee", "left_ankle"], "left_ankle": ["left_ankle"], "nose": ["nose"]}
    return chains.get(j, [j])


def smooth(seq, body_win=7, hand_win=5):
    n = len(seq)
    for j in BODY:
        ks = [k for k in range(n) if j in seq[k]["body"]]
        for run in _runs(ks):
            if len(run) >= body_win:
                arr = np.array([seq[k]["body"][j] for k in run])
                sm = savgol_filter(arr, body_win, 2, axis=0)
                for k, v in zip(run, sm):
                    seq[k]["body"][j] = (float(v[0]), float(v[1]))
    for side in ("left_hand", "right_hand"):
        for i in range(21):
            ks = [k for k in range(n) if seq[k][side][i] is not None]
            for run in _runs(ks):
                if len(run) >= hand_win:
                    arr = np.array([seq[k][side][i] for k in run])
                    sm = savgol_filter(arr, hand_win, 2, axis=0)
                    for k, v in zip(run, sm):
                        seq[k][side][i] = (float(v[0]), float(v[1]))
    return seq


def _runs(ks):
    runs, cur = [], []
    for k in ks:
        if cur and k != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(k)
    if cur:
        runs.append(cur)
    return runs


def clean_hands(seq, min_points=12, max_size=1.3, max_offset=0.5):
    """Drop a hand that is half-detected, or geometrically impossible: bigger than max_size x the forearm, or
    whose wrist point is more than max_offset x the forearm away from the body's wrist (failed detections
    made huge stretched hands, which MusePose turned into blobs)."""
    fore = {}
    for side in ("left", "right"):
        ls = [_d(p["body"][f"{side}_elbow"], p["body"][f"{side}_wrist"]) for p in seq
              if f"{side}_elbow" in p["body"] and f"{side}_wrist" in p["body"]]
        fore[side] = float(np.median(ls)) if ls else None
    n = 0
    for p in seq:
        for side in ("left", "right"):
            key = f"{side}_hand"
            pts = [q for q in p[key] if q is not None]
            bad = len(pts) < min_points
            if not bad and fore[side]:
                arr = np.array(pts)
                size = float(np.hypot(*(arr.max(0) - arr.min(0))))
                wr = p["body"].get(f"{side}_wrist")
                h0 = p[key][0]
                bad = size > max_size * fore[side] or (wr is not None and h0 is not None
                                                       and _d(wr, h0) > max_offset * fore[side])
            if bad:
                if pts:
                    n += 1
                p[key] = [None] * 21
    return seq, n


CHAINS = [("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_elbow"),
          ("left_elbow", "left_wrist"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
          ("left_hip", "left_knee"), ("left_knee", "left_ankle")]


def fill_gaps_polar(seq, max_gap=12):
    """Fill a missing elbow/wrist/knee/ankle by interpolating its ANGLE and length around the joint above it,
    turning the same way the limb was already rotating. Linear filling moved a whirling bowling arm straight
    through the body (short arm, limbs overlapping); this keeps the arm's length and completes the rotation.
    Parents must exist in the frame (shoulders/hips are filled linearly first). Returns (seq, filled)."""
    n, filled = len(seq), 0
    for a, c in CHAINS:
        def polar(k):
            b = seq[k]["body"]
            if a in b and c in b:
                dx, dy = b[c][0] - b[a][0], b[c][1] - b[a][1]
                return float(np.hypot(dx, dy)), float(np.arctan2(dy, dx))
            return None
        have = [k for k in range(n) if polar(k) is not None]
        for k in range(n):
            if c in seq[k]["body"] or a not in seq[k]["body"]:
                continue
            lo = max([h for h in have if h < k], default=None)
            hi = min([h for h in have if h > k], default=None)
            if lo is None or hi is None or hi - lo > max_gap:
                continue
            (r0, t0), (r1, t1) = polar(lo), polar(hi)
            d = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi  # shortest turn
            prev = [h for h in have if h < lo][-1:]  # rotation direction just before the gap
            if prev:
                v = (polar(lo)[1] - polar(prev[0])[1] + np.pi) % (2 * np.pi) - np.pi
                if v * d < 0 and abs(d) > np.pi / 2:  # the limb was turning the other way: go round
                    d = d + 2 * np.pi if d < 0 else d - 2 * np.pi
            t = (k - lo) / (hi - lo)
            r, th = r0 + (r1 - r0) * t, t0 + d * t
            pa = seq[k]["body"][a]
            seq[k]["body"][c] = (pa[0] + r * np.cos(th), pa[1] + r * np.sin(th))
            filled += 1
    return seq, filled


def clean(seq):
    from animate.build_jobs import fill_gaps
    seq, swaps = fix_swaps(seq)
    seq, polar = fill_gaps_polar(seq)   # limbs: angle around the parent joint (needs the parent)
    seq, filled = fill_gaps(seq)        # everything else (shoulders, hips, head, hands): linear
    seq, polar2 = fill_gaps_polar(seq)  # limbs whose parent was only just filled
    polar += polar2
    seq, clamped = clamp_bones(seq)
    seq = smooth(seq)
    seq, hands = clean_hands(seq)
    return seq, {"lr_swaps_fixed": swaps, "limb_joints_filled_by_angle": polar, "joints_filled": filled,
                 "bones_clamped": clamped, "hands_dropped": hands}
