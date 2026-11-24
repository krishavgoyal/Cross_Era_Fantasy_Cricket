"""Second DWPose pass on the raw full-resolution frames, merged with Student 1's pose_data.json.

    python -m animate.repose data/processed_clips/<clip>        # writes pose_data_merged.json next to it

Student 1 ran DWPose on the 512 px crop. On fast, blurred frames (a bowler's release) a pass on the raw
frames, cropped with the same crop window and detected at 1024 px, finds joints the first pass missed (and
sometimes misses ones it found). Per frame and joint the more confident detection is kept; hands are taken
as a whole from whichever pass is more confident for that hand. Coordinates stay in the same normalized
source_512 frame (the crop window is the same).
"""
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from animate.skeleton import BODY


def run(clip_dir, resolution=1024):
    from easy_dwpose import DWposeDetector
    from pose_engine import _extract_person_from_dwpose, _select_primary_person
    clip_dir = Path(clip_dir)
    meta = json.load(open(clip_dir / "metadata.json"))
    pose = json.load(open(clip_dir / "pose_data.json"))
    raw = Path(meta["source"]["path"])
    cap = cv2.VideoCapture(str(raw))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    det = DWposeDetector(device="cpu")
    merged = copy.deepcopy(pose)
    gained = lost_kept = 0
    for k, fr in enumerate(meta["frames"]):
        img = frames[min(fr["source_frame_index"], len(frames) - 1)]
        H, W = img.shape[:2]
        x0, y0, x1, y1 = [int(round(v)) for v in fr["crop_window_source_xyxy_px"]]
        px0, py0, px1, py1 = max(0, -x0), max(0, -y0), max(0, x1 - W), max(0, y1 - H)
        pad = cv2.copyMakeBorder(img, py0, py1, px0, px1, cv2.BORDER_CONSTANT)
        crop = pad[y0 + py0:y1 + py0, x0 + px0:x1 + px0]
        r = det(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), draw_pose=False, include_hands=True,
                include_face=False, detect_resolution=resolution)
        try:
            ex = _extract_person_from_dwpose(r, _select_primary_person(r))
        except Exception:
            continue
        out = merged["frames"][k]
        by = {kp["joint"]: kp for kp in out["body_keypoints"]}
        for j, pt, sc in zip(BODY, ex["body"], ex["body_scores"]):
            x, y, sc = float(pt[0]), float(pt[1]), float(sc)
            if x > 1.5 or y > 1.5:  # pixels of the crop -> normalized
                x, y = x / crop.shape[1], y / crop.shape[0]
            old = by.get(j)
            old_c = old.get("confidence", 0) if old else 0
            if sc >= 0.3 and sc > old_c + 0.05:
                gained += old_c < 0.3
                by[j] = {"joint": j, "x": x, "y": y, "confidence": round(sc, 4), "source": "raw_1024"}
            elif old_c >= 0.3:
                lost_kept += sc < 0.3
        out["body_keypoints"] = [by[j] for j in BODY if j in by]
        for side, key, skey in (("left", "left_hand", "left_hand_scores"), ("right", "right_hand", "right_hand_scores")):
            pts, scs = ex.get(key), ex.get(skey)
            if pts is None or scs is None or len(pts) != 21:
                continue
            old = (out.get("hands") or {}).get(side) or []
            old_mean = float(np.mean([p.get("confidence", 0) for p in old])) if len(old) == 21 else 0.0
            if float(np.mean(scs)) > old_mean + 0.05:
                hand = []
                for i, (pt, sc) in enumerate(zip(pts, scs)):
                    x, y = float(pt[0]), float(pt[1])
                    if x > 1.5 or y > 1.5:
                        x, y = x / crop.shape[1], y / crop.shape[0]
                    hand.append({"joint": old[i]["joint"] if len(old) == 21 else f"hand_{i}", "x": x, "y": y,
                                 "confidence": round(float(sc), 4), "source": "raw_1024"})
                out.setdefault("hands", {})[side] = hand
    merged["merge_note"] = (f"body joints from a raw-frame DWPose pass at {resolution} px merged by confidence "
                            "(animate/repose.py)")
    (clip_dir / "pose_data_merged.json").write_text(json.dumps(merged))
    return {"joints_gained": gained, "joints_kept_from_first_pass": lost_kept}


if __name__ == "__main__":
    print(run(sys.argv[1]))
