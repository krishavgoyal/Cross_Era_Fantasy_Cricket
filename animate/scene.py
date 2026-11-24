"""Scenes that tie the player to the ground (pitch, crease, stumps) instead of a pasted backdrop.

Batting: the background is the motion clip's own camera view (pitch, crease, stumps behind the batter). The
source batter is removed with LaMa inpainting and Bradman is placed exactly where the batter stood, so every
frame's skeleton (mapped into that same view through Student 1's crop windows) keeps the batter's relation to
the crease and stumps. The bat is taken out of Bradman's photo and added back along the tracked bat line
after generation (animate/composite.py), so it is one rigid bat in every frame.

Bowling: McGrath's own photo already shows him at the bowling crease next to the stumps. It is used as the
scene (cropped below the sporting-heroes.net watermark), and the source bowling motion is scaled/moved so
the delivery-stride frame lands exactly on McGrath's pose in the photo.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from animate.reference import cutout_full, detect_body
from animate.skeleton import hip_centre, load_poses, lowest_y, torso_len

ROOT = Path(__file__).resolve().parents[1]
CANVAS = (560, 632)  # W, H; matches McGrath's photo below the watermark (552 x 623), multiples of 8
_YOLO = None
_LAMA = None


def yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO(str(ROOT / "models" / "yolo26x-seg.pt"))
    return _YOLO


def lama(img, mask):
    """LaMa inpainting; mask 255 = fill. Returns an image the size of img."""
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        _LAMA = SimpleLama(device="cpu")
    out = _LAMA(img.convert("RGB"), Image.fromarray(mask).convert("L"))
    return out.crop((0, 0, img.width, img.height))


def seg_masks(bgr, classes, conf=0.15):
    r = yolo().predict(bgr, classes=classes, conf=conf, verbose=False, retina_masks=True)[0]
    if r.masks is None:
        return []
    return [(int(c), float(cf), m > 0.5) for m, c, cf in
            zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]


# ---------------------------------------------------------------- frame mapping (Student 1 crop windows)
def crop_windows(clip_dir):
    m = json.load(open(Path(clip_dir) / "metadata.json"))
    return [f["crop_window_source_xyxy_px"] for f in m["frames"]]


def to_source(pt, win, size=512):
    x0, y0, x1, y1 = win
    return (x0 + pt[0] * (x1 - x0) / size, y0 + pt[1] * (y1 - y0) / size)


def from_source(pt, win, size=512):
    x0, y0, x1, y1 = win
    return ((pt[0] - x0) * size / (x1 - x0), (pt[1] - y0) * size / (y1 - y0))


def map_pose(pose, f):
    g = (lambda p: None if p is None else f(p))
    return {"body": {k: f(v) for k, v in pose["body"].items()}, "left_hand": [g(p) for p in pose["left_hand"]],
            "right_hand": [g(p) for p in pose["right_hand"]],
            "bat": None if pose.get("bat") is None else (f(pose["bat"][0]), f(pose["bat"][1]))}


# ---------------------------------------------------------------- camera motion (broadcast zoom / pan)
def track_camera(video, people_masks=True, max_frames=None, ignore=(), bottom_bar=True):
    """Similarity transform per frame mapping frame k pixels -> frame 0 pixels, estimated from the
    background only (players, bats and black bars masked out), chained frame to frame.
    Returns (list of 2x3 matrices, list of per-step inlier counts)."""
    cap = cv2.VideoCapture(str(video))
    orb = cv2.ORB_create(3000)
    mats, inl = [np.float32([[1, 0, 0], [0, 1, 0]])], [0]
    prev_kp = prev_des = None
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and k >= max_frames):
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        m = (fr.max(axis=2) > 20).astype(np.uint8) * 255
        if bottom_bar:
            m[int(fr.shape[0] * 0.86):] = 0  # bottom overlay bar (channel name, Subscribe)
        for bx0, by0, bx1, by1 in ignore:  # burned-in overlays (watermarks) move with the screen, not the scene
            m[int(by0 * fr.shape[0]):int(by1 * fr.shape[0]), int(bx0 * fr.shape[1]):int(bx1 * fr.shape[1])] = 0
        if people_masks:
            for c, cf, mk in seg_masks(fr, [0, 34]):
                m[cv2.dilate(mk.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0] = 0
        m = cv2.erode(m, np.ones((9, 9), np.uint8))
        kp, des = orb.detectAndCompute(g, m)
        if k > 0:
            A, n = None, 0
            if des is not None and prev_des is not None and len(kp) > 10:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                mt = sorted(bf.match(des, prev_des), key=lambda x: x.distance)[:400]
                if len(mt) >= 12:
                    src = np.float32([kp[x.queryIdx].pt for x in mt])
                    dst = np.float32([prev_kp[x.trainIdx].pt for x in mt])
                    A, keep = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    n = int(keep.sum()) if keep is not None else 0
            if A is None or n < 10:
                A = np.float32([[1, 0, 0], [0, 1, 0]])  # no reliable estimate: assume no camera change
            prev = np.vstack([mats[-1], [0, 0, 1]])
            mats.append((prev @ np.vstack([A, [0, 0, 1]]))[:2].astype(np.float32))
            inl.append(n)
        prev_kp, prev_des = kp, des
        k += 1
    cap.release()
    return mats, inl


def apply_mat(M, p):
    return (float(M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2]), float(M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]))


# ---------------------------------------------------------------- bat sprite from the player's photo
def bat_sprite(photo, body):
    """RGBA of the bat in the photo + its handle/toe points (sprite pixels). handle = end nearer the wrists."""
    bgr = cv2.imread(str(photo))
    bats = [(cf, m) for c, cf, m in seg_masks(bgr, [34]) if c == 34]
    if not bats:
        raise ValueError(f"no bat found in {photo}")
    mask = max(bats, key=lambda t: t[0])[1].astype(np.uint8)
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]
    proj = (pts - c) @ d
    a, b = c + d * proj.min(), c + d * proj.max()
    wr = [np.array(body[k]) for k in ("right_wrist", "left_wrist") if k in body]
    if wr and np.linalg.norm(b - np.mean(wr, 0)) < np.linalg.norm(a - np.mean(wr, 0)):
        a, b = b, a
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.dstack([cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), mask[y0:y1, x0:x1] * 255])
    width = float(np.ptp((pts - c) @ vt[1]))
    return {"rgba": rgba, "handle": (a - [x0, y0]).tolist(), "toe": (b - [x0, y0]).tolist(),
            "width_px": width, "mask_full": mask}


# ---------------------------------------------------------------- batting scene
def batting_scene(clip_dir, photo, out_dir, k0=0, seg_model="u2net_human_seg", grey=True, pad=0.10):
    from animate.bat_track import load_bat
    clip_dir, out_dir = Path(clip_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses, _, fps = load_poses(clip_dir / "pose_data.json")
    bats = load_bat(clip_dir)
    idx = sorted(poses)
    # Camera: the broadcast zooms in during the shot, so map every frame into frame k0's view with the
    # camera motion measured on the background (Student 1's crop windows don't include the camera zoom).
    mats, inliers = track_camera(clip_dir / "source_512.mp4")
    Minv0 = cv2.invertAffineTransform(mats[k0])
    seq = []
    for k in idx:
        Mk = np.vstack([Minv0, [0, 0, 1]]) @ np.vstack([mats[k], [0, 0, 1]])
        f = (lambda p, Mk=Mk: apply_mat(Mk[:2], p))
        p = dict(poses[k])
        p["bat"] = bats.get(k)
        seq.append(map_pose(p, f))
    zoom = [float(np.sqrt(abs(np.linalg.det(m[:, :2])))) for m in mats]

    cap = cv2.VideoCapture(str(clip_dir / "source_512.mp4"))
    for _ in range(k0 + 1):
        ok, frame = cap.read()
    cap.release()
    H0, W0 = frame.shape[:2]
    # visible content (the portrait source sits between black bars) and the region the action covers
    cols = np.where(frame.max(axis=(0, 2)) > 20)[0]
    rows = np.where(frame.max(axis=(1, 2)) > 20)[0]
    vx0, vx1, vy0, vy1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
    pts = np.array([q for p in seq for q in list(p["body"].values()) + [h for h in p["left_hand"] + p["right_hand"] if h]
                    + ([p["bat"][0], p["bat"][1]] if p["bat"] else [])])
    ax0, ay0 = pts.min(0)
    ax1, ay1 = pts.max(0)
    w, h = (ax1 - ax0) * (1 + 2 * pad), (ay1 - ay0) * (1 + 2 * pad)
    cx, cy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    aspect = CANVAS[0] / CANVAS[1]
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    rx0 = float(np.clip(cx - w / 2, 0, max(0, W0 - w)))
    ry0 = float(np.clip(cy - h / 2, vy0, max(vy0, vy1 - h)))
    region = [rx0, ry0, rx0 + w, ry0 + h]

    # Plate = frame k0 with the batter removed. Stumps behind him stay where they are visible: bat-class
    # detections that are near-vertical and reach the feet are stumps and are kept out of the mask.
    m = np.zeros((H0, W0), np.uint8)
    foot = lowest_y(poses[idx[k0]]["body"]) or H0
    keep = np.zeros((H0, W0), bool)
    for c, cf, mk in seg_masks(frame, [0, 34]):
        ys, xs = np.nonzero(mk)
        if not len(xs):
            continue
        if c == 34 and np.ptp(xs) < 0.25 * np.ptp(ys) and ys.max() > foot - 60:
            keep |= mk
            continue
        m[mk] = 255
    if k0 in bats:
        cv2.line(m, tuple(int(v) for v in bats[k0][0]), tuple(int(v) for v in bats[k0][1]), 255, 14)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    m[keep] = 0
    # the portrait source sits between black letterbox bars: fill them with grass too (they are not content)
    m[cv2.dilate((frame.max(axis=2) <= 20).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 255
    plate_full = lama(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), m)
    plate_stats = {"source_frame": k0, "stump_pixels_kept": int(keep.sum()),
                   "camera_zoom_range": [round(min(zoom), 3), round(max(zoom), 3)],
                   "camera_inliers_min": int(min(inliers[1:])) if len(inliers) > 1 else None}
    plate = plate_full.crop(tuple(int(round(v)) for v in region)).resize(CANVAS, Image.Resampling.LANCZOS)
    if grey:
        g = cv2.cvtColor(np.array(plate), cv2.COLOR_RGB2GRAY).astype(np.float32)
        lo, hi = np.percentile(g, 1), np.percentile(g, 99.5)
        g = ((g - lo) / max(hi - lo, 1) * 200 + 25).clip(0, 255)      # stretch to 25..225 (film range)
        g = 255 * (g / 255) ** 0.85                                      # a little brighter in the mid-tones
        plate = Image.fromarray(cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2RGB))
    sx = CANVAS[0] / (region[2] - region[0])
    to_canvas = (lambda p: ((p[0] - region[0]) * sx, (p[1] - region[1]) * sx))
    seq = [map_pose(p, to_canvas) for p in seq]

    # Bradman without his bat (the real bat is composited later along the tracked line)
    bgr = cv2.imread(str(photo))
    photo_body = detect_body(Image.open(photo).convert("RGB"))
    sprite = bat_sprite(photo, photo_body)
    bm = cv2.dilate(sprite["mask_full"] * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    no_bat = lama(Image.open(photo).convert("RGB"), bm)
    no_bat_path = out_dir / "photo_no_bat.png"
    no_bat.save(no_bat_path)
    player = cutout_full(no_bat_path, model=seg_model)       # full photo size: photo px == player px
    pb = detect_body(no_bat)
    # scale/position: Bradman's torso = the batter's torso in frame k0, hips on the batter's hips, feet on the crease
    tgt = seq[k0]["body"]
    s = torso_len(tgt) / torso_len(pb)
    ox = oy = 0
    phip = np.array(hip_centre(pb))
    pw, ph = int(player.width * s), int(player.height * s)
    pl = player.resize((pw, ph), Image.Resampling.LANCZOS)
    if grey:
        al = pl.getchannel("A")
        pl = pl.convert("L").convert("RGB")
        pl.putalpha(al)
    thip = np.array(hip_centre(tgt))
    px, py = int(thip[0] - phip[0] * s), int(thip[1] - phip[1] * s)
    if lowest_y(tgt) and lowest_y(pb):  # put the feet exactly on the batter's feet line
        py = int(lowest_y(tgt) - (lowest_y(pb) - oy) * s)
    ref = plate.copy()
    sh = np.zeros((CANVAS[1], CANVAS[0]), np.float32)
    fy = int(lowest_y(tgt) or py + ph)
    cv2.ellipse(sh, (int(thip[0]), fy), (int(pw * 0.35), 7), 0, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 5) * 0.35
    ref = Image.fromarray((np.array(ref).astype(np.float32) * (1 - sh[..., None])).clip(0, 255).astype(np.uint8))
    ref.paste(pl, (px, py), pl)
    ref.save(out_dir / "ref.png")
    plate.save(out_dir / "plate.png")
    sprite_scale = s
    np.save(out_dir / "bat_sprite.npy", sprite["rgba"])
    meta = {"kind": "batting", "clip": clip_dir.name, "k0": k0, "plate": plate_stats, "region_in_source512": [round(v, 1) for v in region],
            "canvas": list(CANVAS), "player_scale": round(s, 4), "player_xy": [px, py],
            "bat_sprite": {"handle": sprite["handle"], "toe": sprite["toe"], "width_px": sprite["width_px"],
                           "scale": sprite_scale}, "fps": fps, "source_frames": idx}
    (out_dir / "scene.json").write_text(json.dumps(meta, indent=1))
    return ref, seq, meta


def _alpha_box(path, model):
    """Top-left of the cut-out inside the photo (cutout() crops to the alpha bounding box)."""
    from rembg import new_session, remove
    a = np.array(remove(Image.open(path).convert("RGB"), session=new_session(model)))[:, :, 3]
    fg = (a > 40).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    ys, xs = np.nonzero(fg)
    return int(xs.min()), int(ys.min())


# ---------------------------------------------------------------- bowling scene
def bowling_scene(clip_dir, photo, out_dir, top=0.135, frames=None, search=None, stump_box=None, ball=None,
                  pose_file="pose_data.json", margin=40):
    """McGrath's photo (below the watermark) is the scene. The source motion is mapped into the source camera
    (crop windows), then scaled/moved so the frame that best matches the photo's pose lands on it."""
    clip_dir, out_dir = Path(clip_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(photo).convert("RGB")
    W, H = img.size
    crop = img.crop((0, int(H * top), W, H))
    ref = crop.resize(CANVAS, Image.Resampling.LANCZOS)
    rb = detect_body(ref)
    full = ref.copy()
    remove = np.zeros((CANVAS[1], CANVAS[0]), np.uint8)
    stump_alpha = None
    if stump_box:
        # The stump stays in the scene as a fixed layer in front of the bowler (as in the photo) instead of
        # something MusePose redraws: it duplicated and merged with the legs when the bowler ran past it.
        x0, y0, x1, y1 = stump_box
        m = np.zeros((CANVAS[1], CANVAS[0]), np.uint8)
        cv2.rectangle(m, (x0, y0 + (x1 - x0) // 2), (x1, y1), 255, -1)
        cv2.ellipse(m, ((x0 + x1) // 2, y0 + (x1 - x0) // 2), ((x1 - x0) // 2, (x1 - x0) // 2), 0, 180, 360, 255, -1)
        stump_alpha = cv2.GaussianBlur(m, (3, 3), 0.7)
        remove |= cv2.dilate(m, np.ones((17, 17), np.uint8))  # generous: the stump edge must not survive
    if ball:
        # The ball is taken out of his hand and added back as an object that leaves the hand at release
        # (MusePose smeared it); sprite = the ball from the photo.
        bx, by, br = ball
        bm = np.zeros((CANVAS[1], CANVAS[0]), np.uint8)
        cv2.circle(bm, (bx, by), br, 255, -1)
        sprite = np.dstack([np.array(full)[by - br - 1:by + br + 2, bx - br - 1:bx + br + 2],
                            cv2.GaussianBlur(bm, (3, 3), 0.6)[by - br - 1:by + br + 2, bx - br - 1:bx + br + 2]])
        remove |= cv2.dilate(bm, np.ones((9, 9), np.uint8))
    if remove.any():
        ref = lama(ref, remove)
    wins = crop_windows(clip_dir)
    meta_src = json.load(open(clip_dir / "metadata.json"))
    raw = Path(meta_src["source"]["path"])
    src_idx = [f["source_frame_index"] for f in meta_src["frames"]]
    poses, _, fps = load_poses(clip_dir / (pose_file if (clip_dir / pose_file).exists() else "pose_data.json"))
    idx = sorted(poses)
    if frames:
        idx = [k for k in idx if frames[0] <= k <= frames[1]]
    # processed frame -> raw camera frame (crop windows), then camera pan removed (background tracking on the
    # raw clip; the burned-in "CRICKET MASTERCLASS" band and corner logos are ignored)
    mats = None
    if raw.exists():
        mats, _ = track_camera(raw, ignore=[(0.0, 0.30, 1.0, 0.55), (0.9, 0.0, 1.0, 0.12), (0.9, 0.85, 1.0, 1.0)],
                               bottom_bar=False)
    seq = []
    for k in idx:
        M = mats[src_idx[k]] if mats is not None and src_idx[k] < len(mats) else np.float32([[1, 0, 0], [0, 1, 0]])
        seq.append(map_pose(dict(poses[k], bat=None), lambda p, k=k, M=M: apply_mat(M, to_source(p, wins[k]))))
    pan = None if mats is None else [round(float(mats[src_idx[idx[0]]][0, 2]), 1), round(float(mats[src_idx[idx[-1]]][0, 2]), 1)]
    keys = ["neck", "right_shoulder", "left_shoulder", "right_hip", "left_hip", "right_knee", "left_knee",
            "right_ankle", "left_ankle"]
    # Per-bone retargeting onto McGrath's own proportions (source directions and foreshortening kept), then
    # placement: at the frame that matches the photo best, the planted (lower) foot is on the photo's ground
    # and the hips are over his photo hips. (Uniform scaling could not fit both: the source bowler's legs
    # look shorter relative to his torso, so either the feet floated or the head left the frame.)
    from animate.skeleton import retarget_bones
    seq, bone_ratios, s = retarget_bones(seq, rb)
    ground = lowest_y(rb)
    best = None
    cand = range(len(seq)) if not search else [i for i, k in enumerate(idx) if search[0] <= k <= search[1]]
    for i in cand:
        b = seq[i]["body"]
        if not hip_centre(b) or not lowest_y(b):
            continue
        t = np.array([hip_centre(rb)[0] - hip_centre(b)[0], ground - lowest_y(b)])
        common = [j for j in keys if j in b and j in rb]
        if len(common) < 6:
            continue
        sw = {"right_shoulder": "left_shoulder", "left_shoulder": "right_shoulder", "right_hip": "left_hip",
              "left_hip": "right_hip", "right_knee": "left_knee", "left_knee": "right_knee",
              "right_ankle": "left_ankle", "left_ankle": "right_ankle", "neck": "neck"}
        errs = []
        for m in (lambda j: j, lambda j: sw[j]):  # labels as given, or left/right swapped (photo vs clip view)
            cm = [j for j in keys if j in b and m(j) in rb]
            if len(cm) >= 6:
                errs.append(float(np.mean([np.hypot(b[j][0] + t[0] - rb[m(j)][0], b[j][1] + t[1] - rb[m(j)][1])
                                           for j in cm])))
        err = min(errs) if errs else 1e9
        if best is None or err < best[0]:
            best = (err, i, 1.0, t)
    if best is None:
        raise ValueError("could not align the bowling motion to the photo")
    err, i, s1, t = best
    seq = [map_pose(p, lambda q: (q[0] + t[0], q[1] + t[1])) for p in seq]

    # Zoom out so the whole action (run-in, stride, release, follow-through) fits: the photo, the stump layer,
    # the ball and the skeleton all get the same scale z and offset o; LaMa extends the ground around the photo.
    pts = np.array([q for p in seq for q in list(p["body"].values()) + [h for h in p["left_hand"] + p["right_hand"] if h]])
    (bx0, by0), (bx1, by1) = pts.min(0), pts.max(0)
    Wc, Hc = CANVAS
    z = float(min(1.0, (Wc - 2 * margin) / max(bx1 - bx0, 1), (Hc - 2 * margin) / max(by1 - by0, 1)))
    o = np.array([(Wc - z * (bx0 + bx1)) / 2, (Hc - z * (by0 + by1)) / 2])
    if z < 1.0:
        # don't push the photo's ground line up into the sky: keep at least the photo's bottom on the canvas
        o[1] = float(np.clip(o[1], Hc - z * Hc - 40, Hc - z * Hc + 40))
    T = (lambda q: (q[0] * z + o[0], q[1] * z + o[1]))
    seq = [map_pose(p, T) for p in seq]
    ox, oy = int(round(o[0])), int(round(o[1]))
    zw, zh = int(round(Wc * z)), int(round(Hc * z))
    canvas = Image.new("RGB", CANVAS, (0, 0, 0))
    canvas.paste(ref.resize((zw, zh), Image.Resampling.LANCZOS), (ox, oy))
    outside = np.full((Hc, Wc), 255, np.uint8)
    outside[max(0, oy) + 2:min(Hc, oy + zh) - 2, max(0, ox) + 2:min(Wc, ox + zw) - 2] = 0
    ref = lama(canvas, outside) if outside.any() else canvas
    if stump_alpha is not None:
        lay = np.dstack([np.array(full), stump_alpha])
        layer = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
        layer.paste(Image.fromarray(lay).resize((zw, zh), Image.Resampling.LANCZOS), (ox, oy))
        layer.save(out_dir / "fg_layer.png")
    if ball:
        bs = max(5, int(round(sprite.shape[0] * z)))
        np.save(out_dir / "ball_sprite.npy", np.array(Image.fromarray(sprite).resize((bs, bs), Image.Resampling.LANCZOS)))
    ref.save(out_dir / "ref.png")
    meta = {"kind": "bowling", "zoom": round(z, 4), "offset": [ox, oy], "clip": clip_dir.name, "canvas": list(CANVAS), "photo_crop_top": top,
            "camera_pan_x_first_last": pan,
            "aligned_frame": idx[i], "align_error_px": round(err, 1), "torso_ratio": round(s, 4),
            "bone_ratios_vs_torso": bone_ratios,
            "fps": fps, "source_frames": idx, "ref_body_px": {k: [round(v[0], 1), round(v[1], 1)] for k, v in rb.items()}}
    (out_dir / "scene.json").write_text(json.dumps(meta, indent=1))
    return ref, seq, meta
