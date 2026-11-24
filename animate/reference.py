"""Reference frame for MusePose: the player cut out of their photo and placed on a cricket-ground photo.

MusePose keeps the reference image's look (player and scene) and moves the player along a skeleton, so the
background of the final video is this ground photo, with no pasting afterwards.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from animate.skeleton import BODY

CANVAS = (512, 672)  # W, H (3:4, multiples of 8; MusePose runs at this size)


def cutout_full(photo, model="isnet-general-use"):
    """Like cutout() but keeps the photo's full size (player pixels stay at their photo coordinates)."""
    from rembg import new_session, remove
    img = Image.open(photo).convert("RGB")
    a = np.array(remove(img, session=new_session(model)))
    fg = (a[:, :, 3] > 40).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    solid = np.zeros_like(fg)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = [c for c in cnts if cv2.contourArea(c) > 0.01 * fg.size]
    cv2.drawContours(solid, big or cnts, -1, 255, thickness=cv2.FILLED)
    empty = (a[:, :, 3] < 10).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(empty & (solid > 0).astype(np.uint8), connectivity=4)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] > 0.002 * fg.size:
            solid[lab == k] = 0
    rgba = np.dstack([np.array(img), cv2.GaussianBlur(solid, (3, 3), 0.8)])
    return Image.fromarray(rgba)


def cutout(photo, box=None, model="isnet-general-use"):
    """RGBA cut-out of the player (and what they hold, e.g. the bat). box = normalized (x1, y1, x2, y2) crop
    applied first, to leave out watermarks and other people."""
    from rembg import new_session, remove
    img = Image.open(photo).convert("RGB")
    if box:
        w, h = img.size
        img = img.crop((int(w * box[0]), int(h * box[1]), int(w * box[2]), int(h * box[3])))
    rgba = remove(img, session=new_session(model))
    a = np.array(rgba)
    # Solid player: light kit on a light photo background comes out half-transparent (Bradman's whites),
    # so fill the outline of everything that is clearly foreground and keep only a 1 px soft edge.
    fg = (a[:, :, 3] > 40).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    solid = np.zeros_like(fg)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = [c for c in cnts if cv2.contourArea(c) > 0.01 * fg.size]
    cv2.drawContours(solid, big or cnts, -1, 255, thickness=cv2.FILLED)
    # ...but keep real gaps open (between bat and leg, under an arm): big regions the model left empty
    empty = (a[:, :, 3] < 10).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(empty & (solid > 0).astype(np.uint8), connectivity=4)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] > 0.002 * fg.size:
            solid[lab == k] = 0
    a[:, :, 3] = cv2.GaussianBlur(solid, (3, 3), 0.8)
    ys, xs = np.nonzero(a[:, :, 3] > 128)
    return Image.fromarray(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def background(path, size=CANVAS, centre=(0.5, 0.5), grey=False):
    """Crop the ground photo to the canvas aspect around `centre` (normalized) and resize."""
    img = Image.open(path).convert("RGB")
    W, H = size
    w, h = img.size
    if w / h > W / H:
        cw, ch = int(h * W / H), h
    else:
        cw, ch = w, int(w * H / W)
    x0 = int(np.clip(centre[0] * w - cw / 2, 0, w - cw))
    y0 = int(np.clip(centre[1] * h - ch / 2, 0, h - ch))
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize(size, Image.Resampling.LANCZOS)
    if grey:
        img = img.convert("L").convert("RGB")
    return img


def compose(player_rgba, bg, height=0.80, foot_y=0.94, centre_x=0.5, grey=False, shadow=True):
    """Player on the ground: `height` of the canvas tall, feet at `foot_y`, centred at `centre_x`."""
    W, H = bg.size
    p = player_rgba
    s = height * H / p.height
    p = p.resize((max(1, int(p.width * s)), max(1, int(p.height * s))), Image.Resampling.LANCZOS)
    if grey:
        a = p.getchannel("A")
        p = p.convert("L").convert("RGB")
        p.putalpha(a)
    x = int(centre_x * W - p.width / 2)
    y = int(foot_y * H - p.height)
    out = bg.copy()
    if shadow:  # soft contact shadow so the player stands on the grass instead of floating
        sh = np.zeros((H, W), np.float32)
        cv2.ellipse(sh, (int(centre_x * W), int(foot_y * H) - 2), (int(p.width * 0.42), max(4, int(H * 0.012))),
                    0, 0, 360, 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 6) * 0.45
        arr = np.array(out).astype(np.float32) * (1 - sh[..., None])
        out = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    out.paste(p, (x, y), p)
    return out, {"player_box_px": [x, y, x + p.width, y + p.height]}


def detect_body(img, device="cpu"):
    """DWPose body joints of the main person in a PIL image, in pixels (confidence >= 0.3)."""
    from easy_dwpose import DWposeDetector
    from pose_engine import _extract_person_from_dwpose, _select_primary_person
    det = DWposeDetector(device=device)
    raw = det(img, draw_pose=False, include_hands=True, include_face=False, detect_resolution=512)
    ex = _extract_person_from_dwpose(raw, _select_primary_person(raw))
    W, H = img.size
    body = {}
    for j, (pt, sc) in zip(BODY, zip(ex["body"], ex["body_scores"])):
        x, y = float(pt[0]), float(pt[1])
        if x <= 1.5 and y <= 1.5:  # normalized
            x, y = x * W, y * H
        if float(sc) >= 0.3:
            body[j] = (x, y)
    return body


def build(photo, bg_path, out_dir, box=None, bg_centre=(0.5, 0.5), grey=False, height=0.80, centre_x=0.5,
          model="isnet-general-use"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    player = cutout(photo, box, model=model)
    bg = background(bg_path, centre=bg_centre, grey=grey)
    ref, info = compose(player, bg, height=height, centre_x=centre_x, grey=grey)
    ref.save(out_dir / "ref.png")
    body = detect_body(ref)
    meta = {"photo": str(photo), "crop_box": box, "segmentation": model, "background": str(bg_path), "bg_centre": bg_centre,
            "grey": grey, "canvas": list(ref.size), **info, "ref_body_px": {k: [round(v[0], 1), round(v[1], 1)]
                                                                           for k, v in body.items()}}
    (out_dir / "ref.json").write_text(json.dumps(meta, indent=1))
    return ref, body, meta
