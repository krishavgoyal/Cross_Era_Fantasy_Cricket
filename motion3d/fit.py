"""Fit one rigid 3D skeleton (+ bat) to a whole clip.

    ~/gen/bin/python -m motion3d.fit <name>        # names in motion3d/clips.py

Inputs (all mapped into the camera-stabilised world of motion3d/camera.py, so the ground does not move):
  - DWPose body (Student 1's pose_data*.json; the raw detections, not the 1-euro-smoothed ones)
  - MediaPipe Pose heavy (motion3d/mediapipe_pass.py): heels, toes and a per-joint depth estimate
  - the tracked bat line (animate/bat_track.py), batting only

Unknowns: every joint's 3D position in every output frame (x, y in world pixels, z = depth in the same pixel
units), one length per bone shared by all frames (left = right), the bat's toe per frame and one bat length.

Objective (minimised with L-BFGS in PyTorch, float64):
  detections        robust (Huber) distance between the projected joint and each detector's 2D point,
                    weighted by its confidence: a glitch (wrong person, left/right swap, blur) costs little,
                    so it is ignored instead of followed. Missing detections simply have no term.
  depth             weak pull towards MediaPipe's depth (front/back of each joint)
  rigid bones       |joint_a - joint_b| = bone length in every frame (limbs strict, torso slightly soft)
  smooth as a unit  small acceleration of the hip centre and of every bone vector (child - parent): a joint
                    can only move with the joints it is attached to, never jump on its own
  speed limit       (pass 2) every bone vector's speed is capped at 1.15 x its 95th percentile from pass 1
  ground            (pass 2) feet points that are planted (slow and lowest) are held still and on a ground
                    ground height of the nearest steps (interpolated in time between planted steps: the
                    ground reference); no foot point may go below that ground in any frame
  bowling arm       (bowling, blurred delivery frames only) straight arm of fixed length turning over the top
                    from its measured start to its measured end (see animate/delivery.py), detections ignored
  bat               rigid: |toe - grip| = one length; grip = between the wrists; toe follows the tracked bat
Result: motion3d/out/<name>/fit.npz + fit.json (diagnostics). Every joint exists in every frame.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from motion3d.clips import CLIPS, ROOT, out_dir

torch.set_default_dtype(torch.float64)

J = ["nose", "right_eye", "left_eye", "right_ear", "left_ear", "right_shoulder", "right_elbow", "right_wrist",
     "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee",
     "left_ankle", "right_heel", "right_toe", "left_heel", "left_toe"]
JI = {n: i for i, n in enumerate(J)}
MP_OF = {n: n for n in J}
MP_OF.update({"right_toe": "right_foot_index", "left_toe": "left_foot_index"})
DW_JOINTS = [n for n in J if not n.endswith(("heel", "toe"))]
# bones: (a, b, group, weight). Same group = same length (left = right).
BONES = [("right_shoulder", "left_shoulder", "shoulder_width", 0.7), ("right_hip", "left_hip", "hip_width", 0.7),
         ("right_shoulder", "right_hip", "torso_side", 0.5), ("left_shoulder", "left_hip", "torso_side", 0.5),
         ("right_shoulder", "right_elbow", "upper_arm", 1.0), ("left_shoulder", "left_elbow", "upper_arm", 1.0),
         ("right_elbow", "right_wrist", "forearm", 1.0), ("left_elbow", "left_wrist", "forearm", 1.0),
         ("right_hip", "right_knee", "thigh", 1.0), ("left_hip", "left_knee", "thigh", 1.0),
         ("right_knee", "right_ankle", "shin", 1.0), ("left_knee", "left_ankle", "shin", 1.0),
         ("right_ankle", "right_heel", "heel", 1.0), ("left_ankle", "left_heel", "heel", 1.0),
         ("right_ankle", "right_toe", "foot", 1.0), ("left_ankle", "left_toe", "foot", 1.0),
         ("right_heel", "right_toe", "sole", 1.0), ("left_heel", "left_toe", "sole", 1.0),
         ("nose", "right_eye", "nose_eye", 1.0), ("nose", "left_eye", "nose_eye", 1.0),
         ("right_eye", "left_eye", "eye_width", 1.0), ("right_eye", "right_ear", "eye_ear", 1.0),
         ("left_eye", "left_ear", "eye_ear", 1.0), ("right_ear", "left_ear", "ear_width", 1.0),
         ("nose", "right_ear", "nose_ear", 1.0), ("nose", "left_ear", "nose_ear", 1.0)]
GROUPS = sorted({b[2] for b in BONES} | {"neck_nose"})
# Body proportions (anthropometric tables, e.g. Winter 2009: thigh 0.245 H, shin 0.246 H, upper arm 0.186 H,
# forearm 0.146 H, foot 0.152 H): allowed ratio range of a bone group to a reference group. Without these a
# limb that is rarely seen (the bowler's feet are off-screen for 74 of 100 frames) could shrink or grow freely.
PROPORTIONS = [("shin", "thigh", 0.88, 1.12), ("forearm", "upper_arm", 0.70, 0.92), ("sole", "shin", 0.45, 0.75),
               ("foot", "shin", 0.40, 0.70), ("heel", "shin", 0.12, 0.35), ("upper_arm", "torso_side", 0.55, 0.75),
               ("thigh", "torso_side", 0.80, 1.10)]
# bone vectors that must change smoothly (child - parent); "root" = hip centre, "neck" = shoulder centre
EDGES = [("root", "right_hip"), ("root", "left_hip"), ("root", "neck"), ("neck", "right_shoulder"),
         ("neck", "left_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_hip", "right_knee"),
         ("right_knee", "right_ankle"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_ankle", "right_heel"), ("right_ankle", "right_toe"), ("left_ankle", "left_heel"),
         ("left_ankle", "left_toe"), ("neck", "nose"), ("nose", "right_eye"), ("nose", "left_eye"),
         ("right_eye", "right_ear"), ("left_eye", "left_ear")]
LEG_BONES = [("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("left_hip", "left_knee"),
             ("left_knee", "left_ankle"), ("right_ankle", "right_heel"), ("right_ankle", "right_toe"),
             ("left_ankle", "left_heel"), ("left_ankle", "left_toe"),
             ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_elbow"),
             ("left_elbow", "left_wrist")]  # limb bones whose child may be unseen (named LEG_ for history)
FEET = ["right_ankle", "right_heel", "right_toe", "left_ankle", "left_heel", "left_toe"]
SWAP = [("right_shoulder", "left_shoulder"), ("right_elbow", "left_elbow"), ("right_wrist", "left_wrist"),
        ("right_hip", "left_hip"), ("right_knee", "left_knee"), ("right_ankle", "left_ankle"),
        ("right_heel", "left_heel"), ("right_toe", "left_toe"), ("right_eye", "left_eye"), ("right_ear", "left_ear")]
ARM_SW = SWAP[:3]
LEG_SW = SWAP[3:8]


def _apply(M, x, y):
    return M[0][0] * x + M[0][1] * y + M[0][2], M[1][0] * x + M[1][1] * y + M[1][2]


# ------------------------------------------------------------------ detections -> world observations
def load_obs(cfg):
    """Per source frame: {"dw": {joint: (x, y, w)}, "mp": {joint: (x, y, w)}, "mpz": {joint: z_m},
    "hands": {side: [(x, y, conf)]*21}, "bat": ((hx, hy), (tx, ty), w) or None}, world coordinates."""
    clip = Path(cfg["clip"])
    cam = json.load(open(cfg["out"] / "camera.json"))["matrices"]
    dw = json.load(open(clip / cfg.get("pose_file", "pose_data.json")))
    W, H = dw.get("resolution", [512, 512])
    mp = {f["frame"]: f for f in json.load(open(cfg["out"] / "mediapipe.json"))["frames"]}
    bat = {}
    if (clip / "bat_track.json").exists() and cfg["kind"] == "batting":
        for f in json.load(open(clip / "bat_track.json"))["frames"]:
            if f["handle"]:
                bat[f["frame_index"]] = (f["handle"], f["toe"], 0.4 if f.get("interpolated") else 1.0)
    meta = json.load(open(clip / "metadata.json"))
    sw, sh = meta["source"]["source_resolution"]

    def inside(k, x, y, m=10, joint=""):
        """Is (x, y) (512 crop px) inside the real camera frame? Outside it (black padding) a detector can
        only guess: the Cummins clip cuts the bowler's feet off from frame ~36, and both detectors then pin
        the ankles to the frame edge, which would fold the legs."""
        x0, y0, x1, y1 = meta["frames"][k]["crop_window_source_xyxy_px"]
        u, v = x0 + x * (x1 - x0) / W, y0 + y * (y1 - y0) / H
        if joint.endswith(("ankle", "heel", "toe", "foot_index")):
            m = 40  # a foot at the frame edge is usually cut off: the detector's point there is a guess
        e = m * (x1 - x0) / W
        return e <= u <= sw - e and e <= v <= sh - e
    obs = []
    n_out = 0
    for fr in dw["frames"]:
        k = int(fr["frame_index"])
        M = cam[min(k, len(cam) - 1)]
        o = {"dw": {}, "mp": {}, "mpz": {}, "hands": {}, "bat": None}
        for kp in fr["body_keypoints"]:
            c = float(kp.get("confidence", 0))
            if kp["joint"] in JI and c >= 0.2:
                x, y = kp.get("raw_x", kp["x"]) * W, kp.get("raw_y", kp["y"]) * H
                if not inside(k, x, y, joint=kp["joint"]):
                    n_out += 1
                    continue
                o["dw"][kp["joint"]] = (*_apply(M, x, y), c)
        for side in ("left", "right"):
            pts = (fr.get("hands") or {}).get(side) or []
            if len(pts) == 21:
                o["hands"][side] = [(*_apply(M, p.get("raw_x", p["x"]) * W, p.get("raw_y", p["y"]) * H),
                                     float(p.get("confidence", 0))) for p in pts]
        m = mp.get(k)
        if m and m["landmarks"]:
            for n in J:
                x, y, v = m["landmarks"][MP_OF[n]]
                if v >= 0.3 and inside(k, x, y, joint=n):
                    o["mp"][n] = (*_apply(M, x, y), 0.7 * v)
                elif v >= 0.3:
                    n_out += 1
            o["mpz"] = {n: m["world"][MP_OF[n]][2] for n in J}
        if k in bat:
            (hx, hy), (tx, ty), w = bat[k]
            o["bat"] = (_apply(M, hx, hy), _apply(M, tx, ty), w)
        o["feet_seen"] = any(n in o[d] for d in ("dw", "mp") for n in ("right_ankle", "left_ankle", "right_heel",
                                                                          "left_heel", "right_toe", "left_toe"))
        obs.append(o)
    load_obs.dropped_outside_frame = n_out
    return obs


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def fix_swaps(obs):
    """Left/right label fixes, per arm/leg chain: (1) MediaPipe frame disagreeing with DWPose (swapping MP's
    labels matches DWPose much better) -> swap MP; (2) a detector whose chain jumps to the other side from
    one frame to the next -> swap back. Returns the number of swaps."""
    n = 0

    def cost(a, b, chain, swap):
        c, m = 0.0, 0
        for r, l in chain:
            if r in a and l in a and r in b and l in b:
                c += (_dist(a[l], b[r]) + _dist(a[r], b[l])) if swap else (_dist(a[r], b[r]) + _dist(a[l], b[l]))
                m += 1
        return c, m

    def do_swap(d, chain):
        for r, l in chain:
            a, b = d.pop(r, None), d.pop(l, None)
            if b is not None:
                d[r] = b
            if a is not None:
                d[l] = a
    for o in obs:
        for chain in (ARM_SW, LEG_SW):
            keep, m = cost(o["mp"], o["dw"], chain, False)
            sw, _ = cost(o["mp"], o["dw"], chain, True)
            if m >= 2 and sw < 0.5 * keep:
                do_swap(o["mp"], chain)
                do_swap(o["mpz"], chain)
                n += 1
    for det in ("dw", "mp"):
        for k in range(1, len(obs)):
            for chain in (ARM_SW, LEG_SW):
                keep, m = cost(obs[k][det], obs[k - 1][det], chain, False)
                sw, _ = cost(obs[k][det], obs[k - 1][det], chain, True)
                if m >= 2 and sw < 0.5 * keep:
                    do_swap(obs[k][det], chain)
                    if det == "mp":
                        do_swap(obs[k]["mpz"], chain)
                    if det == "dw" and chain is ARM_SW:
                        h = obs[k]["hands"]
                        h["left"], h["right"] = h.get("right"), h.get("left")
                        obs[k]["hands"] = {s: v for s, v in h.items() if v is not None}
                    n += 1
    return n


# ------------------------------------------------------------------ timeline (bowling delivery is stretched)
def timeline(cfg, n_src):
    """Output frame -> fractional source frame. Bowling: the blurred delivery window is slowed down."""
    f0, f1 = cfg.get("frames", (0, n_src - 1))
    if not cfg.get("delivery"):
        return np.arange(f0, f1 + 1, dtype=float)
    d0, d1 = cfg["delivery"]
    n_mid = int(round((d1 - d0) * cfg["stretch"])) + 1
    return np.concatenate([np.arange(f0, d0, dtype=float), np.linspace(d0, d1, n_mid),
                           np.arange(d1 + 1, f1 + 1, dtype=float)])


# ------------------------------------------------------------------ the fit
class Problem:
    def __init__(self, cfg, obs, times):
        self.cfg, self.obs, self.times = cfg, obs, times
        T = self.T = len(times)
        # observation arrays (frame, joint, x, y, weight) for frames whose time is (near) a measured frame
        ot, oj, oxy, ow = [], [], [], []
        dt_, dj, dz, dw_ = [], [], [], []
        self.window = np.zeros(T, bool)
        # output frames whose source frame has no foot detection inside the camera frame
        self.no_feet = np.array([not obs[min(int(round(s)), len(obs) - 1)]["feet_seen"] for s in times])
        # per frame: which leg/foot joints some detector saw (nearest source frame)
        seen = np.zeros((T, len(J)), bool)
        for t, s in enumerate(times):
            o = obs[min(int(round(s)), len(obs) - 1)]
            for d in ("dw", "mp"):
                for n in o[d]:
                    seen[t, JI[n]] = True
        self.foot_seen = torch.tensor(seen[:, [JI[n] for n in FEET]].astype(float))
        self.leg_unseen = torch.tensor((~seen[:, [JI[c] for _, c in LEG_BONES]]).astype(float))
        self.leg_ref = None
        if cfg.get("leg_reference"):
            self.leg_ref, self.leg_ref_w = load_leg_reference(cfg, times)
        arm = cfg.get("bowling_arm")
        arm_j = {f"{arm}_elbow", f"{arm}_wrist"} if arm else set()
        self.kf = None
        if cfg.get("arm_keyframes"):
            # hand-annotated bowling arm (shoulder, wrist) on the blurred delivery frames, mapped to the world
            cam = json.load(open(cfg["out"] / "camera.json"))["matrices"]
            kj = json.load(open(cfg["out"] / cfg["arm_keyframes"]))
            self.kf = {int(k): {n: _apply(cam[int(k)], *xy) for n, xy in v.items()} for k, v in kj["frames"].items()}
            k0, k1 = min(self.kf), max(self.kf)
        for t, s in enumerate(times):
            k = int(round(s))
            if self.kf is not None:
                self.window[t] = k0 - 0.5 <= s <= k1 + 0.5
            elif cfg.get("delivery") and cfg["delivery"][0] < s < cfg["delivery"][1]:
                self.window[t] = True
            if self.kf is not None and abs(s - k) <= 0.2 and k in self.kf:
                for n, (x, y) in self.kf[k].items():
                    ot.append(t), oj.append(JI[n]), oxy.append((x, y)), ow.append(1.2)
            if abs(s - k) > 0.2 or k >= len(obs):
                continue
            o = obs[k]
            for det in ("dw", "mp"):
                for n, (x, y, w) in o[det].items():
                    if self.window[t] and n in arm_j:
                        continue  # blurred bowling arm: keyframes / rotation model instead
                    ot.append(t), oj.append(JI[n]), oxy.append((x, y)), ow.append(w)
            if o["mpz"]:
                for n, z in o["mpz"].items():
                    if self.window[t] and n in arm_j:
                        continue
                    dt_.append(t), dj.append(JI[n]), dz.append(z), dw_.append(1.0)
        self.ot, self.oj = torch.tensor(ot), torch.tensor(oj)
        self.oxy, self.ow = torch.tensor(oxy), torch.tensor(ow)
        self.dt, self.dj, self.dz = torch.tensor(dt_), torch.tensor(dj), torch.tensor(dz)
        # px per metre for MediaPipe depth: 2D bone length / 3D world length, upper percentile (foreshortening)
        ratios = []
        for o in obs:
            if not o["mpz"]:
                continue
        mpw = {f["frame"]: f["world"] for f in json.load(open(cfg["out"] / "mediapipe.json"))["frames"] if f["world"]}
        for k, o in enumerate(obs):
            w3 = mpw.get(k)
            if not w3:
                continue
            for a, b in [("right_shoulder", "right_hip"), ("left_shoulder", "left_hip"), ("right_hip", "right_knee"),
                         ("left_hip", "left_knee"), ("right_knee", "right_ankle"), ("left_knee", "left_ankle")]:
                if a in o["dw"] and b in o["dw"]:
                    l3 = np.linalg.norm(np.subtract(w3[MP_OF[a]], w3[MP_OF[b]]))
                    if l3 > 0.05:
                        ratios.append(_dist(o["dw"][a], o["dw"][b]) / l3)
        self.px_per_m = float(np.percentile(ratios, 80)) if ratios else 500.0
        self.dz = self.dz * self.px_per_m
        # bat observations
        bt, bh, bto, bw = [], [], [], []
        for t, s in enumerate(times):
            k = int(round(s))
            if abs(s - k) <= 0.2 and k < len(obs) and obs[k]["bat"]:
                h, to, w = obs[k]["bat"]
                bt.append(t), bh.append(h), bto.append(to), bw.append(w)
        self.has_bat = len(bt) > 0
        if self.has_bat:
            self.bt, self.bh, self.bto, self.bw = map(torch.tensor, (bt, bh, bto, bw))
        self.init()

    def rescale_depth(self, L):
        mpw = [f["world"] for f in json.load(open(self.cfg["out"] / "mediapipe.json"))["frames"] if f["world"]]
        r = []
        for a, b, g, w in BONES:
            if g in ("upper_arm", "forearm", "thigh", "shin", "torso_side"):
                l3 = np.median([np.linalg.norm(np.subtract(m[MP_OF[a]], m[MP_OF[b]])) for m in mpw])
                r.append(L[GROUPS.index(g)] / max(l3, 1e-3))
        new = float(np.median(r))
        self.dz = self.dz * (new / self.px_per_m)
        self.px_per_m = new

    def init(self):
        """Initial guess: per joint, the confidence-weighted mean of the detections, interpolated in time over
        gaps; depth from MediaPipe; bowling arm from the rotation model."""
        T = self.T
        X = np.full((T, len(J), 3), np.nan)
        acc = np.zeros((T, len(J), 3))
        for t, j, (x, y), w in zip(self.ot.tolist(), self.oj.tolist(), self.oxy.tolist(), self.ow.tolist()):
            acc[t, j] += (x * w, y * w, w)
        have = acc[:, :, 2] > 0
        X[have, 0] = acc[have, 0] / acc[have, 2]
        X[have, 1] = acc[have, 1] / acc[have, 2]
        Z = np.full((T, len(J)), np.nan)
        for t, j, z in zip(self.dt.tolist(), self.dj.tolist(), self.dz.tolist()):
            Z[t, j] = z
        X[:, :, 2] = Z
        tt = np.arange(T)
        for j in range(len(J)):
            for c in range(3):
                v = X[:, j, c]
                ok = ~np.isnan(v)
                if ok.sum() >= 2:
                    X[:, j, c] = np.interp(tt, tt[ok], v[ok])
                elif ok.sum() == 1:
                    X[:, j, c] = v[ok][0]
        # anything still missing: copy a neighbour joint
        for j in range(len(J)):
            if np.isnan(X[:, j]).any():
                X[:, j] = np.nan_to_num(X[:, j], nan=0.0) + np.nanmean(X[:, JI["right_hip"]], axis=0) * np.isnan(X[:, j])
        if self.kf is not None:  # elbow halfway along the straight arm
            side = self.cfg["bowling_arm"]
            sh, el, wr = JI[f"{side}_shoulder"], JI[f"{side}_elbow"], JI[f"{side}_wrist"]
            X[self.window, el] = (X[self.window, sh] + X[self.window, wr]) / 2
        elif self.cfg.get("delivery"):
            self.arm_model(X)
        self.X0 = X
        L = {}
        for g in GROUPS:
            ls = [np.linalg.norm(X[:, JI[a]] - X[:, JI[b]], axis=1) for a, b, gg, _ in BONES if gg == g]
            if g == "neck_nose":
                ls = [np.linalg.norm(X[:, JI["nose"]] - (X[:, JI["right_shoulder"]] + X[:, JI["left_shoulder"]]) / 2, axis=1)]
            L[g] = float(np.median(np.concatenate(ls)))
        self.L0 = L

    def arm_model(self, X):
        """Rotation model for the blurred delivery: the bowling arm's direction (shoulder -> wrist, image plane)
        turns from its measured angle at the window start to the one at the window end, over the top, eased
        (fastest at release). Stored as target unit vectors per window frame; also used for the initial guess."""
        cfg = self.cfg
        side = cfg["bowling_arm"]
        sh, el, wr = JI[f"{side}_shoulder"], JI[f"{side}_elbow"], JI[f"{side}_wrist"]
        idx = np.where(self.window)[0]
        a, b = idx[0] - 1, idx[-1] + 1
        th0 = np.arctan2(*(X[a, wr, :2] - X[a, sh, :2])[::-1])
        th1 = np.arctan2(*(X[b, wr, :2] - X[b, sh, :2])[::-1])
        run_dir = np.sign(X[-1, JI["right_hip"], 0] - X[0, JI["right_hip"], 0]) or -1.0
        direction = 1.0 if run_dir > 0 else -1.0
        up = -np.pi / 2
        d = ((th1 - th0) * direction) % (2 * np.pi)
        if ((up - th0) * direction) % (2 * np.pi) > d:
            d += 2 * np.pi
        total = direction * d
        L_up = np.median(np.linalg.norm(X[:, el] - X[:, sh], axis=1))
        L_lo = np.median(np.linalg.norm(X[:, wr] - X[:, el], axis=1))
        dirs = []
        for i, t in enumerate(range(a, b + 1)):
            u = i / (b - a)
            th = th0 + total * u * u * (3 - 2 * u)
            dvec = np.array([np.cos(th), np.sin(th)])
            dirs.append(dvec)
            if a < t < b:
                X[t, el, :2] = X[t, sh, :2] + dvec * L_up
                X[t, wr, :2] = X[t, sh, :2] + dvec * (L_up + L_lo)
                X[t, el, 2] = X[t, wr, 2] = X[t, sh, 2]
        self.arm_t = torch.tensor(list(range(a + 1, b)))
        self.arm_dir = torch.tensor(np.array(dirs[1:-1]))
        thetas = [th0 + total * (i / (b - a)) ** 2 * (3 - 2 * i / (b - a)) for i in range(b - a + 1)]
        rel = next((a + i for i, th in enumerate(thetas) if abs(((th - up + np.pi) % (2 * np.pi)) - np.pi) < 0.25), (a + b) // 2)
        self.arm_info = {"rotation_deg": round(float(np.degrees(total)), 1), "window_out_frames": [int(a + 1), int(b - 1)],
                         "release_index": int(rel) + 1}

    # -------------------------------------------------------------- energy
    def energy(self, X, L, toe, Lbat, stage, extra):
        cfg = self.cfg
        T = self.T
        E = {}
        hub = torch.nn.functional.huber_loss
        # detections (sigma 4 px), robust
        P = X[self.ot, self.oj, :2]
        r = (P - self.oxy) / 4.0
        E["detections"] = (self.ow[:, None] * hub(r, torch.zeros_like(r), reduction="none", delta=1.5)).sum()
        root = (X[:, JI["right_hip"]] + X[:, JI["left_hip"]]) / 2
        neck = (X[:, JI["right_shoulder"]] + X[:, JI["left_shoulder"]]) / 2
        # depth relative to the hip centre (sigma 40 px), robust
        rz = ((X[self.dt, self.dj, 2] - root[self.dt, 2]) - self.dz) / 40.0
        E["depth"] = hub(rz, torch.zeros_like(rz), reduction="sum", delta=1.0)
        # rigid bones (sigma 1 px)
        eb = 0
        for a, b, g, w in BONES:
            l = torch.linalg.norm(X[:, JI[a]] - X[:, JI[b]], dim=1)
            eb = eb + w * ((l - L[GROUPS.index(g)]) ** 2).sum()
        l = torch.linalg.norm(X[:, JI["nose"]] - neck, dim=1)
        eb = eb + 0.5 * ((l - L[GROUPS.index("neck_nose")]) ** 2).sum()
        E["bones"] = 4.0 * eb
        ep = 0
        for g, ref, lo, hi in PROPORTIONS:
            a, r = L[GROUPS.index(g)], L[GROUPS.index(ref)]
            ep = ep + torch.relu(lo * r - a) ** 2 + torch.relu(a - hi * r) ** 2
        E["proportions"] = 1000.0 * ep
        # smooth as a unit: acceleration of the root and of every bone vector
        pts = {"root": root, "neck": neck}

        def P_(n):
            return pts[n] if n in pts else X[:, JI[n]]
        V = torch.stack([P_(c) - P_(p) for p, c in EDGES], dim=1)  # T, E, 3
        acc_l = V[2:] - 2 * V[1:-1] + V[:-2]
        acc_r = root[2:] - 2 * root[1:-1] + root[:-2]
        E["smooth"] = (acc_l ** 2).sum() / 2.0 ** 2 + (acc_r ** 2).sum() / 1.0 ** 2
        # limb joints not seen in a frame (feet cut off by the frame edge while the bowler strides towards the
        # camera, the batter's back arm hidden behind his body): their bone stays close to the image plane
        # instead of swinging in depth
        if self.leg_unseen.any():
            dz = torch.stack([X[:, JI[c], 2] - X[:, JI[p], 2] for p, c in LEG_BONES], 1)
            E["unseen_legs"] = ((dz * self.leg_unseen) ** 2).sum() / 25.0 ** 2
            # anatomy for what the camera did not see: a shin hangs below its knee (ankle at least half a shin
            # lower), a heel and toe are below their ankle (the foot is roughly flat)
            Ls, Lh, Lf = (L[GROUPS.index(g)] for g in ("shin", "heel", "foot"))
            ea = 0
            for side in ("right", "left"):
                kn, an = X[:, JI[f"{side}_knee"]], X[:, JI[f"{side}_ankle"]]
                he, to = X[:, JI[f"{side}_heel"]], X[:, JI[f"{side}_toe"]]
                u_an = self.leg_unseen[:, LEG_BONES.index((f"{side}_knee", f"{side}_ankle"))]
                u_he = self.leg_unseen[:, LEG_BONES.index((f"{side}_ankle", f"{side}_heel"))]
                u_to = self.leg_unseen[:, LEG_BONES.index((f"{side}_ankle", f"{side}_toe"))]
                ea = ea + (u_an * torch.relu(kn[:, 1] + 0.5 * Ls - an[:, 1]) ** 2).sum()
                ea = ea + (u_he * torch.relu(an[:, 1] + 0.3 * Lh - he[:, 1]) ** 2).sum()
                ea = ea + (u_to * torch.relu(an[:, 1] + 0.2 * Lf - to[:, 1]) ** 2).sum()
            E["unseen_leg_anatomy"] = ea / 5.0 ** 2
        # unseen legs follow a reference bowler's legs (another video, matched by delivery phase), as
        # directions in the pelvis frame, so camera angle does not matter (see load_leg_reference)
        if self.leg_ref is not None:
            R = pelvis_frame(X)  # T, 3, 3 (columns: right, up, forward)
            er = 0
            for bi, (pa, ch) in enumerate(LEG_BONES[:4]):
                v = X[:, JI[ch]] - X[:, JI[pa]]
                n = torch.linalg.norm(v, dim=1, keepdim=True).clamp(min=1e-6)
                want = torch.einsum("tij,tj->ti", R, self.leg_ref[:, bi])
                w = self.leg_unseen[:, bi] * self.leg_ref_w
                er = er + (w[:, None] * ((v / n - want) * n) ** 2).sum()
            E["reference_legs"] = er / 15.0 ** 2
        # depth also changes smoothly (it is weakly observed)
        E["smooth"] = E["smooth"] + ((V[1:, :, 2] - V[:-1, :, 2]) ** 2).sum() / 6.0 ** 2
        if stage >= 2:
            sp = torch.linalg.norm(V[1:] - V[:-1], dim=2)  # T-1, E
            E["speed_cap"] = 4.0 * (torch.relu(sp - extra["vmax"][None, :]) ** 2).sum()
            # ground: planted feet still, nothing below the ground height of the nearest planted steps
            Fi = torch.tensor([JI[n] for n in FEET])
            F = X[:, Fi]  # T, 6, 3
            gy = extra["ground"][:, None].expand(-1, F.shape[1])
            below = torch.relu(F[..., 1] - gy) * self.foot_seen  # only feet seen in that frame: the ground under
            # an off-screen foot is not known (the bowler comes towards the camera, the ground there is lower)
            E["ground"] = 25.0 * (below ** 2).sum()
            ct = extra["contact"]  # T, 6 bool
            if ct.any():
                cid = extra["contact_id"]
                both = ct[1:] & ct[:-1] & (cid[1:] == cid[:-1])
                dv = (F[1:] - F[:-1])[both]
                E["contact_still"] = 25.0 * (dv ** 2).sum()
        if self.kf is not None:
            side = cfg["bowling_arm"]
            sh, el, wr = JI[f"{side}_shoulder"], JI[f"{side}_elbow"], JI[f"{side}_wrist"]
            w = torch.tensor(np.where(self.window)[0])
            l = torch.linalg.norm(X[w, wr] - X[w, sh], dim=1)
            E["bowling_arm"] = 4.0 * ((l - L[GROUPS.index("upper_arm")] - L[GROUPS.index("forearm")]) ** 2).sum()
        elif cfg.get("delivery"):
            side = cfg["bowling_arm"]
            sh, el, wr = JI[f"{side}_shoulder"], JI[f"{side}_elbow"], JI[f"{side}_wrist"]
            t = self.arm_t
            d = self.arm_dir
            ea = 0
            for j, Lj in ((el, L[GROUPS.index("upper_arm")]), (wr, L[GROUPS.index("upper_arm")] + L[GROUPS.index("forearm")])):
                u = X[t, j, :2] - X[t, sh, :2]
                cross = u[:, 0] * d[:, 1] - u[:, 1] * d[:, 0]
                ea = ea + (cross ** 2).sum() / 3.0 ** 2 + (torch.relu(-(u * d).sum(1)) ** 2).sum()
                l = torch.linalg.norm(X[t, j] - X[t, sh], dim=1)
                ea = ea + 4.0 * ((l - Lj) ** 2).sum()  # straight arm (cricket law)
            E["bowling_arm"] = ea
        if self.has_bat:
            # both hands hold the handle: the wrists stay within a third of a forearm of each other
            gap = torch.linalg.norm(X[:, JI["right_wrist"]] - X[:, JI["left_wrist"]], dim=1)
            E["hands_on_bat"] = 4.0 * (torch.relu(gap - 0.35 * L[GROUPS.index("forearm")]) ** 2).sum()
            grip = (X[:, JI["right_wrist"]] + X[:, JI["left_wrist"]]) / 2
            bv = toe - grip
            E["bat_rigid"] = 4.0 * ((torch.linalg.norm(bv, dim=1) - Lbat) ** 2).sum()
            r = (toe[self.bt, :2] - self.bto) / 6.0
            E["bat_toe"] = (self.bw[:, None] * hub(r, torch.zeros_like(r), reduction="none", delta=1.5)).sum()
            # direction of the tracked bat line (handle -> toe) from the grip
            dvec = self.bto - self.bh
            dvec = dvec / torch.linalg.norm(dvec, dim=1, keepdim=True).clamp(min=1e-6)
            u = bv[self.bt, :2]
            cross = (u[:, 0] * dvec[:, 1] - u[:, 1] * dvec[:, 0]) / 4.0
            E["bat_direction"] = (self.bw * hub(cross, torch.zeros_like(cross), reduction="none", delta=1.5)).sum()
            ab = bv[2:] - 2 * bv[1:-1] + bv[:-2]
            E["bat_smooth"] = (ab ** 2).sum() / 3.0 ** 2
            E["bat_depth"] = ((bv[1:, 2] - bv[:-1, 2]) ** 2).sum() / 8.0 ** 2
        return E

    def solve(self, stage, extra, X=None, L=None, toe=None, Lbat=None, iters=400):
        X = torch.tensor(self.X0 if X is None else X, requires_grad=True)
        L = torch.tensor([self.L0[g] for g in GROUPS] if L is None else L, requires_grad=True)
        params = [X, L]
        if self.has_bat:
            if toe is None:
                grip = (self.X0[:, JI["right_wrist"]] + self.X0[:, JI["left_wrist"]]) / 2
                t_obs = np.full((self.T, 2), np.nan)
                h_obs = np.full((self.T, 2), np.nan)
                for t, h, to in zip(self.bt.tolist(), self.bh.tolist(), self.bto.tolist()):
                    t_obs[t], h_obs[t] = to, h
                tt = np.arange(self.T)
                ok = ~np.isnan(t_obs[:, 0])
                tv = np.stack([np.interp(tt, tt[ok], t_obs[ok, c]) for c in range(2)], 1)
                toe = np.concatenate([tv, grip[:, 2:3]], 1)
                Lbat = float(np.percentile(np.linalg.norm(tv - grip[:, :2], axis=1), 90))
            toe = torch.tensor(toe, requires_grad=True)
            Lbat = torch.tensor(float(Lbat), requires_grad=True)
            params += [toe, Lbat]
        opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=50, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            E = self.energy(X, L, toe, Lbat, stage, extra)
            tot = sum(E.values())
            tot.backward()
            return tot
        for _ in range(3):
            opt.step(closure)
        with torch.no_grad():
            E = self.energy(X, L, toe, Lbat, stage, extra)
        out = {"X": X.detach().numpy(), "L": L.detach().numpy(), "energy": {k: round(float(v), 1) for k, v in E.items()}}
        if self.has_bat:
            out["toe"], out["Lbat"] = toe.detach().numpy(), float(Lbat)
        return out


def contacts_and_ground(X, height, seen=None):
    """Planted foot points: slow (< 1.2 % of body height per frame) and near the lowest foot point of the
    frame (within 4 %), for at least 3 frames. Each planted run is a ground reference point (its median
    height); the ground height of any frame is interpolated in time between them (the bowler runs towards
    the camera and the zoom changes, so one flat line does not fit all steps)."""
    Fi = [JI[n] for n in FEET]
    F = X[:, Fi]
    sp = np.zeros(F.shape[:2])
    sp[1:] = np.linalg.norm(F[1:, :, :2] - F[:-1, :, :2], axis=2)
    sp[0] = sp[1]
    low = F[:, :, 1].max(axis=1, keepdims=True)
    ct = (sp < 0.012 * height) & (F[:, :, 1] > low - 0.04 * height)
    if seen is not None:  # a foot outside the camera frame is not a ground reference
        ct &= seen[:, None]
    refs = []  # (time, height) per planted run
    cid = np.zeros(ct.shape, int)  # planted-run id per (frame, foot point); 0 = not planted
    nid = 0
    for f in range(ct.shape[1]):
        start = None
        for t in range(len(ct) + 1):
            # a run ends when the foot stops being planted or has slid > 2 % of body height from its start
            go = t < len(ct) and ct[t, f] and (start is None or np.linalg.norm(F[t, f, :2] - F[start, f, :2]) < 0.02 * height)
            if go and start is None:
                start = t
            if not go and start is not None:
                run = t - start
                if run < 3:
                    ct[start:t, f] = False
                else:
                    nid += 1
                    cid[start:t, f] = nid
                    refs.append((start + run / 2, float(np.median(F[start:t, f, 1]))))
                start = t if (t < len(ct) and ct[t, f]) else None
    if refs:  # a step far from the others (a foot guessed outside the frame) is not a ground reference
        med = np.median([h for _, h in refs])
        refs = [r for r in refs if abs(r[1] - med) < 0.06 * height]
    T = len(X)
    if refs:
        refs.sort()
        # several feet planted at the same time: the ground is the lowest of them
        tt = np.array([r[0] for r in refs])
        hh = np.array([r[1] for r in refs])
        ground = np.interp(np.arange(T), tt, hh)
        for t in range(T):  # never above a foot that is planted in this frame
            if ct[t].any():
                ground[t] = max(ground[t], F[t, ct[t], 1].max())
    else:
        ground = np.full(T, float(np.percentile(F[:, :, 1], 98)))
    return ct, ground, refs, cid


def run(name):
    cfg = dict(CLIPS[name])
    cfg["out"] = out_dir(name)
    obs = load_obs(cfg)
    n_swaps = fix_swaps(obs)
    times = timeline(cfg, len(obs))
    pb = Problem(cfg, obs, times)
    r1 = pb.solve(1, {})
    X1 = r1["X"]
    # MediaPipe's metric skeleton is smaller than the real player: rescale its depth to the fitted bone lengths
    pb.rescale_depth(r1["L"])
    height = float(np.percentile(X1[:, :, 1].max(1) - X1[:, :, 1].min(1), 90))
    V1 = _edge_vectors(X1)
    sp = np.linalg.norm(V1[1:] - V1[:-1], axis=2)
    vmax = np.percentile(sp, 95, axis=0) * 1.15 + 0.5
    ct, ground, refs, cid = contacts_and_ground(X1, height, seen=~pb.no_feet)
    extra = {"vmax": torch.tensor(vmax), "contact": torch.tensor(ct), "ground": torch.tensor(ground),
             "contact_id": torch.tensor(cid), "no_feet": torch.tensor(pb.no_feet)}
    r2 = pb.solve(2, extra, X=X1, L=r1["L"], toe=r1.get("toe"), Lbat=r1.get("Lbat"))
    X = r2["X"]
    # diagnostics
    raw = _raw_track(pb)
    diag = {"frames_out": pb.T, "detections_dropped_outside_camera_frame": load_obs.dropped_outside_frame,
            "frames_without_visible_feet": int(pb.no_feet.sum()), "source_frames": [float(times[0]), float(times[-1])], "lr_swaps_fixed": n_swaps,
            "px_per_metre_mediapipe": round(pb.px_per_m, 1), "body_height_px": round(height, 1),
            "bone_lengths_px": {g: round(float(l), 1) for g, l in zip(GROUPS, r2["L"])},
            "max_bone_length_error_px": round(float(_bone_err(X, r2["L"])), 2),
            "contact_frames_per_foot_point": {n: int(ct[:, i].sum()) for i, n in enumerate(FEET)},
            "ground_reference_steps": [[round(t, 1), round(h, 1)] for t, h in refs],
            "planted_runs": int(cid.max()),
            "max_planted_foot_drift_px": round(float(_drift(X, cid)), 2),
            "max_below_ground_px_seen_feet": round(float(np.max(_below(X, ground) * pb.foot_seen.numpy())), 2),
            "energy_pass1": r1["energy"], "energy_pass2": r2["energy"],
            "jerk_raw_vs_fit": _jerk_cmp(raw, X), "max_speed_ratio_raw_vs_fit": _speed_cmp(raw, X),
            "frames_with_detections": {n: int(sum(1 for t in range(pb.T) if ((pb.ot == t) & (pb.oj == JI[n])).any())) for n in J}}
    if pb.kf is not None:
        diag["bowling_arm"] = arm_info(X, pb)
    elif cfg.get("delivery"):
        diag["bowling_arm"] = pb.arm_info
    if pb.has_bat:
        toe = r2["toe"]
        grip = (X[:, JI["right_wrist"]] + X[:, JI["left_wrist"]]) / 2
        bl = np.linalg.norm(toe - grip, axis=1)
        bl2 = np.linalg.norm(toe[:, :2] - grip[:, :2], axis=1)
        diag["bat"] = {"length_3d_px": round(r2["Lbat"], 1), "length_3d_range": [round(float(bl.min()), 1), round(float(bl.max()), 1)],
                       "length_on_screen_range": [round(float(bl2.min()), 1), round(float(bl2.max()), 1)],
                       "max_on_screen_length_change_per_frame_px": round(float(np.abs(np.diff(bl2)).max()), 1)}
        raw_bl = [np.hypot(*(np.subtract(o["bat"][1], o["bat"][0]))) for o in obs if o["bat"]]
        diag["bat"]["raw_tracked_length_range"] = [round(float(min(raw_bl)), 1), round(float(max(raw_bl)), 1)]
        diag["bat"]["raw_max_length_change_per_frame_px"] = round(float(np.abs(np.diff(raw_bl)).max()), 1)
    np.savez(cfg["out"] / "fit.npz", X=X, L=r2["L"], groups=np.array(GROUPS), joints=np.array(J), times=times,
             contact=ct, contact_id=cid, ground=ground, plane=np.array([float(np.median(ground)), 0.0, 0.0]), toe=r2.get("toe", np.zeros(0)), Lbat=r2.get("Lbat", 0.0),
             X_pass1=X1)
    (cfg["out"] / "fit.json").write_text(json.dumps(diag, indent=1))
    return diag


def arm_info(X, pb):
    """Rotation of the fitted bowling arm (shoulder -> wrist, screen angle, unwrapped) over the keyframed
    window, and the release frame: first frame the arm is 15 deg past vertical towards the batter."""
    side = pb.cfg["bowling_arm"]
    sh, wr = JI[f"{side}_shoulder"], JI[f"{side}_wrist"]
    w = np.where(pb.window)[0]
    d = X[:, wr, :2] - X[:, sh, :2]
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    run_dir = np.sign(X[-1, JI["right_hip"], 0] - X[0, JI["right_hip"], 0]) or -1.0
    # bowler running left: arm turns counter-clockwise on screen (angle decreasing), vertical = -90 deg
    target = np.radians(-90 - 15) if run_dir < 0 else np.radians(-90 + 15)
    thw = th[w] - 2 * np.pi * np.round((th[w][0] - np.arctan2(np.sin(th[w][0]), np.cos(th[w][0]))) / (2 * np.pi))
    past = [t for t, a in zip(w, thw) if (a < target if run_dir < 0 else a > target)]
    rel = int(past[0]) if past else int(w[len(w) // 2])
    return {"keyframed_out_frames": [int(w[0]), int(w[-1])], "rotation_deg": round(float(np.degrees(th[w[-1]] - th[w[0]])), 1),
            "release_index": rel, "release_source_frame": round(float(pb.times[rel]), 2),
            "note": "bowling arm from hand-annotated keyframes (shoulder, wrist), straight arm of the fitted length"}


def pelvis_frame(X):
    """Per frame rotation (columns: body right, up, forward) from hips and shoulders (torch or numpy)."""
    lib = torch if isinstance(X, torch.Tensor) else np
    norm = (lambda v: v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-6)) if lib is torch else \
        (lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6))
    root = (X[:, JI["right_hip"]] + X[:, JI["left_hip"]]) / 2
    neck = (X[:, JI["right_shoulder"]] + X[:, JI["left_shoulder"]]) / 2
    xr = norm(X[:, JI["right_hip"]] - X[:, JI["left_hip"]])
    up0 = neck - root
    up = norm(up0 - (up0 * xr).sum(-1, keepdims=True) * xr) if lib is np else \
        norm(up0 - (up0 * xr).sum(-1, keepdim=True) * xr)
    fw = lib.cross(xr, up, dim=-1) if lib is torch else np.cross(xr, up)
    return lib.stack([xr, up, fw], -1)


def load_leg_reference(cfg, times):
    """Leg bone directions (thigh, shin; pelvis frame) of a reference bowler per output frame: reference frame
    = anchor_ref + (t - anchor_out) / scale, clamped. Applied from output frame from_out on, only to leg bones
    the detectors did not see in that frame."""
    lr = cfg["leg_reference"]
    d = np.load(out_dir(lr["clip"]) / "fit.npz")
    Xr = d["X"]
    R = pelvis_frame(Xr)
    loc = []
    for pa, ch in LEG_BONES[:4]:
        v = Xr[:, JI[ch]] - Xr[:, JI[pa]]
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        loc.append(np.einsum("tji,tj->ti", R, v))  # R^T v
    loc = np.stack(loc, 1)  # Tr, 4, 3
    T = len(times)
    tr = np.clip(lr["anchor"][1] + (np.arange(T) - lr["anchor"][0]) / lr["scale"], 0, len(Xr) - 1)
    a = np.floor(tr).astype(int)
    b = np.minimum(a + 1, len(Xr) - 1)
    f = (tr - a)[:, None, None]
    out = loc[a] * (1 - f) + loc[b] * f
    out = out / np.linalg.norm(out, axis=2, keepdims=True)
    w = (np.arange(T) >= lr["from_out"]).astype(float)
    return torch.tensor(out), torch.tensor(w)


def _edge_vectors(X):
    root = (X[:, JI["right_hip"]] + X[:, JI["left_hip"]]) / 2
    neck = (X[:, JI["right_shoulder"]] + X[:, JI["left_shoulder"]]) / 2
    pts = {"root": root, "neck": neck}
    g = lambda n: pts[n] if n in pts else X[:, JI[n]]
    return np.stack([g(c) - g(p) for p, c in EDGES], 1)


def _bone_err(X, L):
    e = 0
    for a, b, g, w in BONES:
        if w >= 1.0:
            e = max(e, np.abs(np.linalg.norm(X[:, JI[a]] - X[:, JI[b]], axis=1) - L[GROUPS.index(g)]).max())
    return e


def _drift(X, cid):
    """Largest movement of a foot point within one planted run (should be ~0)."""
    Fi = [JI[n] for n in FEET]
    m = 0.0
    for r in range(1, int(cid.max()) + 1):
        t, i = np.where(cid == r)
        P = X[t, [Fi[k] for k in i], :2]
        m = max(m, float(np.linalg.norm(P - P[0], axis=1).max()))
    return m


def _below(X, ground):
    F = X[:, [JI[n] for n in FEET]]
    return np.maximum(0, F[..., 1] - ground[:, None])


def _raw_track(pb):
    """The DWPose-only 2D track on the output timeline (gaps left as NaN), to compare smoothness."""
    R = np.full((pb.T, len(J), 2), np.nan)
    for t, s in enumerate(pb.times):
        k = int(round(s))
        if abs(s - k) <= 0.2 and k < len(pb.obs):
            for n, (x, y, w) in pb.obs[k]["dw"].items():
                R[t, JI[n]] = (x, y)
    return R


def _jerk_cmp(R, X):
    """Median |acceleration| per frame (px/frame^2), raw DWPose vs fit, over joints DWPose measures."""
    js = [JI[n] for n in DW_JOINTS]
    a_raw = np.linalg.norm(R[2:, js] - 2 * R[1:-1, js] + R[:-2, js], axis=2)
    a_fit = np.linalg.norm(X[2:, js, :2] - 2 * X[1:-1, js, :2] + X[:-2, js, :2], axis=2)
    return {"raw_median": round(float(np.nanmedian(a_raw)), 2), "fit_median": round(float(np.median(a_fit)), 2),
            "raw_p99": round(float(np.nanpercentile(a_raw, 99)), 1), "fit_p99": round(float(np.percentile(a_fit, 99)), 1)}


def _speed_cmp(R, X):
    js = [JI[n] for n in DW_JOINTS]
    v_raw = np.linalg.norm(R[1:, js] - R[:-1, js], axis=2)
    v_fit = np.linalg.norm(X[1:, js, :2] - X[:-1, js, :2], axis=2)
    return {"raw_max_over_median": round(float(np.nanmax(v_raw) / max(np.nanmedian(v_raw), 1e-3)), 1),
            "fit_max_over_median": round(float(v_fit.max() / max(np.median(v_fit), 1e-3)), 1)}


if __name__ == "__main__":
    for n in sys.argv[1:] or list(CLIPS):
        print(n, json.dumps(run(n), indent=1))
