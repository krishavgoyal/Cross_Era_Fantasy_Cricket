"""Extra reference motions from the Commons videos in data/motion_videos/ (more bowlers, other camera angles).

    ~/gen/bin/python -m motion3d.extra prep <name>     # trim, track the player, crop, DWPose -> processed clip
    then: mediapipe_pass (pose3d env), camera, fit, render  (motion3d/run_all.sh does all of it)

Each entry: the video, a time window around one delivery, and a seed (time, x, y in video pixels) on the
player to follow (picked by eye from contact sheets; other players are in the frame). The player is followed
frame to frame with YOLO person boxes (nearest box to the previous one), cropped square around him and
written as a processed clip in Student 1's layout (data/processed_clips/<name>/): source_512.mp4,
metadata.json (crop windows), pose_data.json (DWPose body + hands).
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MV = ROOT / "data" / "motion_videos"

EXTRA = {
    "extra_parry_bowling": dict(video=MV / "bowling" / "parry_bowling_2012.ogv", t=(8.4, 10.0), seed=(8.9, 1140, 460),
                                view="side-on, square of the wicket", who="Stephen Parry (Nev1, CC BY-SA 3.0)"),
    "extra_stokes_bowling": dict(video=MV / "bowling" / "stokes_bowling_2012.ogv", t=(0.3, 2.4), seed=(1.2, 1320, 350),
                                 view="side-on, square of the wicket", who="Ben Stokes (Nev1, CC BY-SA 3.0)"),
    "extra_onions_bowling": dict(video=MV / "bowling" / "onions_bowling_2012.ogv", t=(3.6, 5.2), seed=(3.95, 1400, 440),
                                 view="side-on, panning", who="Graham Onions (Nev1, CC BY-SA 3.0)"),
    "extra_amar_endon_bowling": dict(video=MV / "batting" / "amar_kala_batting_2025.webm", t=(9.6, 11.2),
                                     seed=(10.0, 278, 700), view="end-on, from behind the bowler",
                                     who="bowler in 'Amar Kala Batting Video 2025' (DrRevathi25, CC BY-SA 4.0)"),
}


def trim(cfg, out):
    t0, t1 = cfg["t"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t0), "-t", str(t1 - t0), "-i", str(cfg["video"]),
                    "-vf", "fps=24", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out)], check=True)


def follow(frames, seed_k, seed_xy):
    """Box of the chosen player in every frame: start from the box nearest the seed, then nearest-box
    association forwards and backwards (YOLO person boxes); short gaps keep the previous box."""
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "models" / "yolo26x-seg.pt"))
    boxes = []
    for f in frames:
        r = m.predict(f, classes=[0], conf=0.15, verbose=False)[0]
        boxes.append(r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4)))
    ctr = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    out = [None] * len(frames)
    bs = boxes[seed_k]
    out[seed_k] = bs[np.argmin([np.linalg.norm(ctr(b) - seed_xy) for b in bs])]
    for rng in (range(seed_k + 1, len(frames)), range(seed_k - 1, -1, -1)):
        prev = out[seed_k]
        for k in rng:
            if len(boxes[k]):
                h = prev[3] - prev[1]
                d = [np.linalg.norm(ctr(b) - ctr(prev)) / h + abs((b[3] - b[1]) - h) / h for b in boxes[k]]
                i = int(np.argmin(d))
                if d[i] < 0.8:
                    prev = boxes[k][i]
            out[k] = prev
    return np.array(out)


def prep(name):
    from easy_dwpose import DWposeDetector
    from pose_engine import _extract_person_from_dwpose, _select_primary_person
    from animate.skeleton import BODY
    cfg = EXTRA[name]
    raw = ROOT / "data" / "raw_clips" / f"{name}.mp4"
    raw.parent.mkdir(parents=True, exist_ok=True)
    trim(cfg, raw)
    cap = cv2.VideoCapture(str(raw))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    H, W = frames[0].shape[:2]
    seed_k = int(round((cfg["seed"][0] - cfg["t"][0]) * 24))
    bx = follow(frames, seed_k, np.array(cfg["seed"][1:], float))
    # square crop windows, smoothed (side = 1.6 x box height, centre on the box)
    c = np.stack([(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2], 1)
    side = np.full(len(bx), np.median(bx[:, 3] - bx[:, 1]) * 1.6)
    k = np.ones(5) / 5
    c = np.stack([np.convolve(np.pad(c[:, i], 2, mode="edge"), k, "valid") for i in range(2)], 1)
    clip = ROOT / "data" / "processed_clips" / name
    clip.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(clip / "source_512.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (512, 512))
    det = DWposeDetector(device="cpu")
    mframes, pframes = [], []
    for i, f in enumerate(frames):
        x0, y0 = c[i] - side[i] / 2
        x1, y1 = x0 + side[i], y0 + side[i]
        M = np.float32([[512 / side[i], 0, -x0 * 512 / side[i]], [0, 512 / side[i], -y0 * 512 / side[i]]])
        crop = cv2.warpAffine(f, M, (512, 512), flags=cv2.INTER_LANCZOS4, borderValue=(0, 0, 0))
        vw.write(crop)
        win = [round(float(v), 2) for v in (x0, y0, x1, y1)]
        mframes.append({"frame_index": i, "source_frame_index": i, "timestamp_ms": round(i * 1000 / 24, 2),
                        "bbox_source_xyxy_px": [round(float(v), 1) for v in bx[i]], "crop_window_source_xyxy_px": win})
        r = det(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), draw_pose=False, include_hands=True,
                include_face=False, detect_resolution=1024)
        fr = {"frame_index": i, "body_keypoints": [], "hands": {}}
        try:
            ex = _extract_person_from_dwpose(r, _select_primary_person(r))
            norm = lambda x, y: (x / 512, y / 512) if (x > 1.5 or y > 1.5) else (x, y)
            for j, pt, sc in zip(BODY, ex["body"], ex["body_scores"]):
                x, y = norm(float(pt[0]), float(pt[1]))
                fr["body_keypoints"].append({"joint": j, "x": x, "y": y, "confidence": round(float(sc), 4)})
            for side_, key, skey in (("left", "left_hand", "left_hand_scores"), ("right", "right_hand", "right_hand_scores")):
                pts, scs = ex.get(key), ex.get(skey)
                if pts is not None and scs is not None and len(pts) == 21:
                    fr["hands"][side_] = [{"joint": f"hand_{q}", "x": norm(float(p[0]), float(p[1]))[0],
                                           "y": norm(float(p[0]), float(p[1]))[1], "confidence": round(float(s), 4)}
                                          for q, (p, s) in enumerate(zip(pts, scs))]
        except Exception:
            pass
        pframes.append(fr)
    vw.release()
    meta = {"schema_version": "1.1.0", "clip_id": name, "action_type": "bowling", "role": "bowler",
            "source": {"path": str(raw), "source_fps": 24.0, "source_frame_count": len(frames),
                       "source_resolution": [W, H], "original": str(cfg["video"].relative_to(ROOT)),
                       "window_s": list(cfg["t"]), "credit": cfg["who"], "view": cfg["view"]},
            "processed": {"fps": 24.0, "frame_count": len(frames), "resolution": [512, 512],
                          "made_by": "motion3d/extra.py (YOLO person follow + DWPose 1024)"},
            "frames": mframes}
    (clip / "metadata.json").write_text(json.dumps(meta, indent=1))
    (clip / "pose_data.json").write_text(json.dumps({"schema_version": "1.1.0", "clip_id": name, "fps": 24.0,
                                                     "frame_count": len(frames), "resolution": [512, 512],
                                                     "frames": pframes}))
    return {"frames": len(frames), "with_body": sum(1 for f in pframes if f["body_keypoints"])}


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        for n in sys.argv[2:] or list(EXTRA):
            print(n, prep(n))
