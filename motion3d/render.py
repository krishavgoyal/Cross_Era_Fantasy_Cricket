"""Render a fitted 3D skeleton as OpenPose stick figures from several camera angles.

    ~/gen/bin/python -m motion3d.render <name> [--angles 0,45,90,135,180,270]

Writes to data/motion3d/<name>/:
  views/pose_<angle>.mp4         OpenPose-style stick figure (body, feet, bat) seen from that angle, turning
                                 around the vertical (the ground plane's normal); 0 = the source camera
  views/openpose_<angle>.json    the same, as OpenPose BODY_18 keypoints per frame (x, y, 1.0)
  overlay.mp4                    camera view (0 deg) drawn over the source frames: does it match?
  multiview.mp4                  source + overlay | four views, side by side, ground grid and planted feet marked
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from animate.skeleton import BODY, draw
from motion3d.clips import CLIPS, out_dir
from motion3d.fit import FEET, JI

FOOT_LINES = [("right_ankle", "right_heel"), ("right_heel", "right_toe"), ("right_ankle", "right_toe"),
              ("left_ankle", "left_heel"), ("left_heel", "left_toe"), ("left_ankle", "left_toe")]


def load(name):
    d = np.load(out_dir(name) / "fit.npz")
    return {k: d[k] for k in d.files}


def up_axis(plane):
    """Views turn about the camera's vertical (broadcast cameras are level; the fitted ground plane's tilt
    depends on the least certain coordinate, depth, so it is not used for the turn)."""
    return np.array([0.0, -1.0, 0.0])


def rotation(axis, deg):
    """Rodrigues rotation matrix about a unit axis."""
    th = np.radians(deg)
    k = axis
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def view_points(F, deg):
    """All 3D points of the fit, rotated by deg about the vertical through the mean hip centre."""
    X = F["X"]
    up = up_axis(F["plane"])
    root = (X[:, JI["right_hip"]] + X[:, JI["left_hip"]]) / 2
    c = root.mean(0)
    R = rotation(up, deg)
    Xr = (X - c) @ R.T + c
    toe = None
    if F["toe"].size:
        toe = (F["toe"] - c) @ R.T + c
    return Xr, toe, R, c


def pose_dict(Xt, toe_t=None):
    g = lambda n: (float(Xt[JI[n], 0]), float(Xt[JI[n], 1]))
    body = {n: g(n) for n in BODY if n in JI}
    body["neck"] = tuple(np.mean([body["right_shoulder"], body["left_shoulder"]], axis=0))
    p = {"body": body, "left_hand": [None] * 21, "right_hand": [None] * 21, "bat": None}
    if toe_t is not None:
        grip = (Xt[JI["right_wrist"], :2] + Xt[JI["left_wrist"], :2]) / 2
        p["bat"] = (tuple(grip), (float(toe_t[0]), float(toe_t[1])))
    return p


def draw_view(Xt, toe_t, size, fit_tf, ground=None, contact=None):
    s, ox, oy = fit_tf
    tf = lambda q: (q[..., :2] * s + [ox, oy])
    X2 = tf(Xt)
    T2 = None if toe_t is None else tf(toe_t)
    img = np.zeros((size[1], size[0], 3), np.uint8)
    if ground is not None:  # ground grid (projected), for orientation
        for seg in ground:
            a, b = tf(seg[0]), tf(seg[1])
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (60, 60, 60), 1, cv2.LINE_AA)
    img = np.maximum(img, draw(pose_dict(X2, T2), size, hands=False, stick=3))
    for a, b in FOOT_LINES:
        cv2.line(img, tuple(int(v) for v in X2[JI[a]]), tuple(int(v) for v in X2[JI[b]]), (200, 200, 200), 2, cv2.LINE_AA)
    if contact is not None:
        for i, n in enumerate(FEET):
            if contact[i]:
                cv2.circle(img, tuple(int(v) for v in X2[JI[n]]), 6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def ground_grid(F, R, c, n=7, t=None):
    """Squares on the ground (height of the nearest planted steps at frame t) around the player (world
    coords, rotated like the view)."""
    a, b, cc = F["plane"]
    if t is not None and "ground" in F:
        a, b, cc = float(F["ground"][t]), 0.0, 0.0
    X = F["X"]
    root = (X[:, JI["right_hip"]] + X[:, JI["left_hip"]]) / 2
    span = max(np.ptp(root[:, 0]), 1) + 200
    xs = np.linspace(root[:, 0].min() - 100, root[:, 0].min() - 100 + span, n)
    zs = np.linspace(-span / 2, span / 2, n) + root[:, 2].mean()
    gy = lambda x, z: a + b * x + cc * z
    segs = []
    for x in xs:
        segs.append(((x, gy(x, zs[0]), zs[0]), (x, gy(x, zs[-1]), zs[-1])))
    for z in zs:
        segs.append(((xs[0], gy(xs[0], z), z), (xs[-1], gy(xs[-1], z), z)))
    segs = np.array(segs)
    return (segs - c) @ R.T + c


def fit_transform(pts, size, margin=0.08):
    x0, y0 = pts[..., 0].min(), pts[..., 1].min()
    x1, y1 = pts[..., 0].max(), pts[..., 1].max()
    s = min(size[0] * (1 - 2 * margin) / max(x1 - x0, 1), size[1] * (1 - 2 * margin) / max(y1 - y0, 1))
    return s, size[0] / 2 - s * (x0 + x1) / 2, size[1] / 2 - s * (y0 + y1) / 2


def openpose_json(Xr, toe):
    frames = []
    for t in range(len(Xr)):
        p = pose_dict(Xr[t], None if toe is None else toe[t])
        kp = []
        for n in BODY:
            q = p["body"][n]
            kp += [round(q[0], 2), round(q[1], 2), 1.0]
        fr = {"pose_keypoints_2d": kp,
              "foot_keypoints_2d": [v for n in FEET for v in (round(float(Xr[t, JI[n], 0]), 2), round(float(Xr[t, JI[n], 1]), 2), 1.0)]}
        if p["bat"]:
            fr["bat_grip_toe"] = [round(v, 2) for q in p["bat"] for v in q]
        frames.append(fr)
    return {"format": "OpenPose BODY_18 (COCO order as DWPose) + feet [ankle, heel, toe] R/L + bat grip/toe",
            "keypoint_names": BODY, "frames": frames}


def main(name, angles=(0, 45, 90, 135, 180, 270), tile=(360, 400)):
    cfg = CLIPS[name]
    o = out_dir(name)
    (o / "views").mkdir(parents=True, exist_ok=True)
    F = load(name)
    T = len(F["X"])
    fps = 24.0
    views = {}
    for deg in angles:
        Xr, toe, R, c = view_points(F, deg)
        pts = Xr if toe is None else np.concatenate([Xr, toe[:, None]], 1)
        tf = fit_transform(pts.reshape(-1, 3), tile)
        frames = [draw_view(Xr[t], None if toe is None else toe[t], tile, tf, ground_grid(F, R, c, t=t), F["contact"][t])
                  for t in range(T)]
        views[deg] = frames
        vw = cv2.VideoWriter(str(o / "views" / f"pose_{deg:03d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, tile)
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        (o / "views" / f"openpose_{deg:03d}.json").write_text(json.dumps(openpose_json(Xr, toe)))
    # overlay on the source (camera view mapped back into each source frame)
    cam = json.load(open(o / "camera.json"))["matrices"]
    cap = cv2.VideoCapture(str(Path(cfg["clip"]) / "source_512.mp4"))
    src = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        src.append(f)
    ov = []
    for t in range(T):
        k = int(round(F["times"][t]))
        Minv = cv2.invertAffineTransform(np.array(cam[k], np.float32))
        X2 = F["X"][t].copy()
        X2[:, :2] = X2[:, :2] @ Minv[:, :2].T + Minv[:, 2]
        toe = None
        if F["toe"].size:
            toe = F["toe"][t].copy()
            toe[:2] = Minv[:, :2] @ toe[:2] + Minv[:, 2]
        sk = draw(pose_dict(X2, toe), (512, 512), hands=False, stick=3)
        for a, b in FOOT_LINES:
            cv2.line(sk, tuple(int(v) for v in X2[JI[a], :2]), tuple(int(v) for v in X2[JI[b], :2]), (200, 200, 200), 2)
        fr = src[min(k, len(src) - 1)].copy()
        m = sk.max(2) > 0
        fr[m] = cv2.cvtColor(sk, cv2.COLOR_RGB2BGR)[m]
        ov.append(fr)
    vw = cv2.VideoWriter(str(o / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (512, 512))
    for f in ov:
        vw.write(f)
    vw.release()
    # multiview sheet video
    H = tile[1] * 2
    vw = None
    four = [a for a in (0, 90, 180, 270) if a in views][:4] or list(views)[:4]
    for t in range(T):
        left = cv2.resize(ov[t], (H, H))
        tiles = [cv2.cvtColor(views[a][t], cv2.COLOR_RGB2BGR) for a in four]
        for a, im in zip(four, tiles):
            cv2.putText(im, f"{a} deg" + (" (camera)" if a == 0 else ""), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        right = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        frame = np.hstack([left, right])
        cv2.putText(frame, f"{name}  frame {t}  src {F['times'][t]:.1f}", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1, cv2.LINE_AA)
        if vw is None:
            vw = cv2.VideoWriter(str(o / "multiview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
    vw.release()
    # contact sheet for a quick look
    ks = np.linspace(0, T - 1, 6).astype(int)
    rows = []
    for k in ks:
        rows.append(np.hstack([cv2.resize(ov[k], (tile[1], tile[1]))] + [cv2.cvtColor(views[a][k], cv2.COLOR_RGB2BGR) for a in four]))
    cv2.imwrite(str(o / "multiview_sheet.jpg"), cv2.resize(np.vstack(rows), None, fx=0.6, fy=0.6))
    return {"views": list(views), "frames": T}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--angles", default="0,45,90,135,180,270")
    a = ap.parse_args()
    for n in a.names or list(CLIPS):
        print(n, main(n, tuple(int(v) for v in a.angles.split(","))))
