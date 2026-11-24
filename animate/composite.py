"""Add the player's real bat (cut from their photo) along the tracked bat line in every generated frame.

    python -m animate.composite <generated.mp4> animate/jobs/<name> <out.mp4>

MusePose has no notion of a bat (a drawn bat line made it wobble and merge with the gloves), so the bat was
taken out of the reference photo and the skeleton. Here it goes back as one rigid object: the bat sprite from
the photo is placed so its handle end sits on the tracked handle and its toe points along the tracked bat;
its width scales with the player. The gloves are then put back on top of the handle (generated pixels around
the hand keypoints), so the batter holds the bat instead of the bat covering the hands.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _tri(h, t, w):
    h, t = np.float32(h), np.float32(t)
    d = t - h
    n = np.float32([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-6)
    return np.float32([h, t, h + n * w / 2])


def composite(generated, job_dir, out_path):
    job_dir = Path(job_dir)
    tgt = json.load(open(job_dir / "target_pose.json"))
    sc = json.load(open(job_dir / "scene.json"))
    has_bat = (job_dir / "bat_sprite.npy").exists()
    if has_bat:
        sprite = np.load(job_dir / "bat_sprite.npy")  # RGBA
        bs = sc["bat_sprite"]
        src_tri = _tri(bs["handle"], bs["toe"], bs["width_px"])
    fg = None
    if (job_dir / "fg_layer.png").exists():  # fixed scene objects in front of the player (the stump)
        fgl = cv2.imread(str(job_dir / "fg_layer.png"), cv2.IMREAD_UNCHANGED)
        fg = (fgl[:, :, :3].astype(np.float32), fgl[:, :, 3:4].astype(np.float32) / 255.0)
    ball = np.load(job_dir / "ball_sprite.npy") if (job_dir / "ball_sprite.npy").exists() else None
    balls = 0
    W, H = tgt["canvas"]
    lens = [float(np.hypot(f["bat"][1][0] - f["bat"][0][0], f["bat"][1][1] - f["bat"][0][1]))
            for f in tgt["frames"] if f.get("bat")] if has_bat else []
    min_len = 0.55 * float(np.median(lens)) if lens else 0.0  # blurred / occluded bats were tracked as stubs
    cap = cv2.VideoCapture(str(generated))
    fps = cap.get(cv2.CAP_PROP_FPS) or tgt["fps"]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    k = drawn = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (W, H))
        f = tgt["frames"][min(k, len(tgt["frames"]) - 1)]
        if has_bat and f.get("bat"):
            (hx, hy), (tx, ty) = f["bat"]
            L = float(np.hypot(tx - hx, ty - hy))
            d = np.array([tx - hx, ty - hy]) / max(L, 1e-3)
            L = max(L, min_len)  # keep the tracked direction, never draw a stub
            # anchor the bat in the hands: the tracker gives direction and length, the wrists give the grip
            # (interpolated tracks drift off the hands in blurred frames and the bat floated)
            wr = [f["body"][j] for j in ("right_wrist", "left_wrist") if f["body"].get(j)]
            if wr:
                g = np.mean(wr, axis=0)
                hx, hy = g - d * 0.12 * L          # handle top just above the top hand
            tx, ty = hx + d[0] * L, hy + d[1] * L
            w = bs["width_px"] * bs["scale"]
            A = cv2.getAffineTransform(src_tri, _tri((hx, hy), (tx, ty), w))
            warped = cv2.warpAffine(sprite, A, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
            a = warped[:, :, 3:4].astype(np.float32) / 255.0
            bat_bgr = cv2.cvtColor(warped[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32) * 0.82  # aged willow
            # thin dark edge so the pale bat stays visible against white flannels (it vanished on the shirt)
            solid = (a[:, :, 0] > 0.5).astype(np.uint8)
            edge = cv2.dilate(solid, np.ones((3, 3), np.uint8)) - solid
            edge_a = cv2.GaussianBlur(edge.astype(np.float32), (3, 3), 0.6)[..., None] * 0.85
            out = fr.astype(np.float32) * (1 - edge_a) + 35.0 * edge_a
            out = out * (1 - a) + bat_bgr * a
            # gloves over the handle: keep generated pixels in discs around the hand keypoints
            hand = np.zeros((H, W), np.uint8)
            body = f["body"]
            torso = None
            if body.get("neck") and (body.get("right_hip") or body.get("left_hip")):
                hp = body.get("right_hip") or body.get("left_hip")
                torso = float(np.hypot(body["neck"][0] - hp[0], body["neck"][1] - hp[1]))
            r = int(max(6, 0.11 * (torso or 100)))
            pts = [q for q in (f.get("left_hand") or []) + (f.get("right_hand") or []) if q]
            pts += [body[j] for j in ("right_wrist", "left_wrist") if body.get(j)]
            for q in pts:
                cv2.circle(hand, (int(q[0]), int(q[1])), r, 255, -1)
            hand = cv2.GaussianBlur(hand, (0, 0), 2).astype(np.float32)[..., None] / 255.0
            out = out * (1 - hand) + fr.astype(np.float32) * hand
            fr = out.clip(0, 255).astype(np.uint8)
            drawn += 1
        if ball is not None and f.get("ball"):
            bx, by = f["ball"]
            bh, bw = ball.shape[:2]
            x0, y0 = int(round(bx - bw / 2)), int(round(by - bh / 2))
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + bw), min(H, y0 + bh)
            if xb > xa and yb > ya:
                sp = ball[ya - y0:yb - y0, xa - x0:xb - x0]
                al = sp[:, :, 3:4].astype(np.float32) / 255.0
                roi = fr[ya:yb, xa:xb].astype(np.float32)
                fr[ya:yb, xa:xb] = (roi * (1 - al) + cv2.cvtColor(sp[:, :, :3], cv2.COLOR_RGB2BGR) * al).astype(np.uint8)
                balls += 1
        if fg is not None:
            fr = (fr.astype(np.float32) * (1 - fg[1]) + fg[0] * fg[1]).clip(0, 255).astype(np.uint8)
        vw.write(fr)
        k += 1
    vw.release()
    return {"frames": k, "frames_with_bat": drawn, "frames_with_ball": balls, "fixed_layer": fg is not None}


if __name__ == "__main__":
    print(composite(sys.argv[1], sys.argv[2], sys.argv[3]))
