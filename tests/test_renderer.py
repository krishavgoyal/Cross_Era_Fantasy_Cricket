from __future__ import annotations

import json
from pathlib import Path
import tempfile

import cv2

from render_control import render_control_video


def test_renderer_writes_512_video() -> None:
    frames = []
    for index in range(3):
        frames.append(
            {
                "frame_index": index,
                "timestamp_ms": index * 1000 / 24,
                "body_keypoints": [
                    {"joint": "nose", "x": 0.5, "y": 0.2, "confidence": 1.0},
                    {"joint": "neck", "x": 0.5, "y": 0.3, "confidence": 1.0},
                ],
                "hands": {"left": [], "right": []},
                "face_keypoints": [],
            }
        )
    payload = {"fps": 24, "frame_count": 3, "resolution": [512, 512], "frames": frames}
    with tempfile.TemporaryDirectory() as directory:
        pose_path = Path(directory) / "pose_data.json"
        video_path = Path(directory) / "control_pose.mp4"
        pose_path.write_text(json.dumps(payload), encoding="utf-8")
        render_control_video(pose_path, video_path)
        capture = cv2.VideoCapture(str(video_path))
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 512
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 512
        capture.release()
