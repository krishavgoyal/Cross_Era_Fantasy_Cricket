"""MediaPipe Pose (heavy, video mode) on a clip: 33 landmarks per frame, 2D image + 3D world (metres, hip-centred).

    ~/pose3d/bin/python -m motion3d.mediapipe_pass <video> <out.json> [--model ~/pose3d/models/pose_landmarker_heavy.task]
                                                   [--start S --end E] [--crop x,y,w,h] [--every N]

Runs in its own venv (~/pose3d: mediapipe needs numpy 2, the ~/gen env pins numpy 1.26).
MediaPipe gives what DWPose does not: heels, toes (ground contact) and a depth estimate per joint, which
fixes the left/right and front/back ambiguity when the skeleton is lifted to 3D.
"""
import argparse
import json
import os
from pathlib import Path

import cv2

MP_NAMES = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
            "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
            "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index"]


def run(video, out, model, start=None, end=None, crop=None, every=1):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(model)),
                                        running_mode=vision.RunningMode.VIDEO, num_poses=1,
                                        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
                                        min_tracking_confidence=0.3)
    det = vision.PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames, k, kept = [], -1, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        k += 1
        t = k / fps
        if (start is not None and t < start) or (end is not None and t > end) or k % every:
            continue
        if crop:
            x, y, w, h = crop
            fr = fr[y:y + h, x:x + w]
        H, W = fr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        r = det.detect_for_video(img, int(round(t * 1000)))
        rec = {"frame": k, "t": round(t, 4), "size": [W, H], "landmarks": None, "world": None}
        if r.pose_landmarks:
            lm, wl = r.pose_landmarks[0], r.pose_world_landmarks[0]
            rec["landmarks"] = {n: [round(p.x * W, 2), round(p.y * H, 2), round(float(p.visibility or 0), 3)]
                                for n, p in zip(MP_NAMES, lm)}
            rec["world"] = {n: [round(p.x, 4), round(p.y, 4), round(p.z, 4)] for n, p in zip(MP_NAMES, wl)}
            kept += 1
        frames.append(rec)
    cap.release()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"video": str(video), "fps": fps, "crop": crop, "model": Path(model).name,
                                     "frames": frames}))
    return {"frames": len(frames), "with_pose": kept}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--model", default=os.path.expanduser("~/pose3d/models/pose_landmarker_heavy.task"))
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--crop")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    print(run(a.video, a.out, a.model, a.start, a.end,
              [int(v) for v in a.crop.split(",")] if a.crop else None, a.every))
