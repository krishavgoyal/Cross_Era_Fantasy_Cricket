"""Step 1: conform a generated clip to the project brief: 512 x 512 pixels, at most 100 frames (3-5 s at 24 fps).

The MusePose canvas is 560 x 632 (portrait). A 560 x 560 square is cut from it and resized to 512 (area
interpolation, no stretching). The square's vertical position keeps every target joint, the bat and the ball
inside with a margin; it is fixed for the whole clip when the action fits, otherwise it follows the player
with a smoothed pan. Clips longer than 100 frames keep the 100-frame window with the most motion (the shot
itself). The target skeleton and the reference image are transformed the same way, for scoring.
"""
import json

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

MAX_FRAMES = 100
SIZE = 512


def _pts(f):
    pts = [v for v in f["body"].values() if v]
    if f.get("bat"):
        pts += [p for p in f["bat"] if p]
    if f.get("ball"):
        pts.append(f["ball"])
    return np.array(pts, float)


def window(tg, max_frames=MAX_FRAMES):
    T = len(tg["frames"])
    if T <= max_frames:
        return 0, T
    P = np.array([[f["body"].get(j) or [np.nan, np.nan] for j in sorted(tg["frames"][0]["body"])] for f in tg["frames"]], float)
    v = np.nansum(np.linalg.norm(np.diff(P, axis=0), axis=2), axis=1)
    e = np.convolve(v, np.ones(max_frames - 1), "valid")
    a = int(np.argmax(e))
    return a, a + max_frames


def crop_rows(tg, a, b, margin=16):
    W, H = tg["canvas"]
    S = min(W, H)
    lo = np.array([_pts(f)[:, 1].min() for f in tg["frames"][a:b]])
    hi = np.array([_pts(f)[:, 1].max() for f in tg["frames"][a:b]])
    if hi.max() - lo.min() + 2 * margin <= S:
        y0 = float(np.clip((lo.min() + hi.max()) / 2 - S / 2, 0, H - S))
        return np.full(b - a, y0), S, "fixed"
    want = np.clip((lo + hi) / 2 - S / 2, 0, H - S)
    want = gaussian_filter1d(want, 6)
    # never cut a joint: push the window where needed, then smooth again lightly
    want = np.clip(want, hi + margin - S, lo - margin)
    return gaussian_filter1d(np.clip(want, 0, H - S), 2), S, "pan"


def run(video, target, ref, out_video, out_target, out_ref):
    tg = json.load(open(target))
    W, H = tg["canvas"]
    a, b = window(tg)
    y0, S, mode = crop_rows(tg, a, b)
    x0 = (W - S) / 2
    sc = SIZE / S
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (W, H)))
    vw = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), tg["fps"], (SIZE, SIZE))
    for i, k in enumerate(range(a, min(b, len(frames)))):
        M = np.float32([[sc, 0, -x0 * sc], [0, sc, -y0[i] * sc]])
        vw.write(cv2.warpAffine(frames[k], M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE))
    vw.release()
    tf = lambda p, i: None if p is None else [round((p[0] - x0) * sc, 1), round((p[1] - y0[i]) * sc, 1)]
    nf = []
    for i, f in enumerate(tg["frames"][a:b]):
        nf.append({"body": {k: tf(v, i) for k, v in f["body"].items()},
                   "left_hand": [tf(q, i) for q in f["left_hand"]], "right_hand": [tf(q, i) for q in f["right_hand"]],
                   "bat": None if not f.get("bat") else [tf(f["bat"][0], i), tf(f["bat"][1], i)],
                   "ball": tf(f.get("ball"), i)})
    json.dump({"canvas": [SIZE, SIZE], "fps": tg["fps"], "source_frames": tg["source_frames"][a:b], "frames": nf,
               "conform": {"frames": [a, b], "crop": mode, "square_px": S, "scale": round(sc, 4)}},
              open(out_target, "w"))
    r = cv2.resize(cv2.imread(str(ref)), (W, H))
    M = np.float32([[sc, 0, -x0 * sc], [0, sc, -y0[0] * sc]])
    cv2.imwrite(str(out_ref), cv2.warpAffine(r, M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE))
    return {"frames_in": len(frames), "frames_out": min(b, len(frames)) - a, "window": [a, b], "crop": mode,
            "duration_s": round((min(b, len(frames)) - a) / tg["fps"], 2), "size": [SIZE, SIZE]}
