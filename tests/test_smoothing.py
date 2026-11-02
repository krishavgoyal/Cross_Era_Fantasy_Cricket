from __future__ import annotations

from pathlib import Path
import json
import tempfile

import numpy as np

from temporal_smoothing import smooth_pose_file


def _make_payload(frame_count: int = 24) -> dict:
    frames = []
    for index in range(frame_count):
        timestamp = index / 24.0 * 1000.0
        noisy_x = 0.5 + 0.15 * (index / max(frame_count - 1, 1)) + (0.015 if index % 2 else -0.015)
        frames.append(
            {
                "frame_index": index,
                "timestamp_ms": timestamp,
                "body_keypoints": [
                    {"joint": "nose", "x": noisy_x, "y": 0.4, "confidence": 0.95},
                ],
                "hands": {"left": [], "right": []},
                "face_keypoints": [],
                "world_keypoints_3d": None,
            }
        )
    return {"schema_version": "1.0.0", "fps": 24, "frame_count": frame_count, "resolution": [512, 512], "frames": frames}


def test_one_euro_preserves_shape_and_reduces_alternating_noise() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pose_data.json"
        payload = _make_payload()
        path.write_text(json.dumps(payload), encoding="utf-8")
        smooth_pose_file(path)
        result = json.loads(path.read_text(encoding="utf-8"))

        raw = np.array([frame["raw_body_keypoints"][0]["x"] for frame in result["frames"]])
        filtered = np.array([frame["body_keypoints"][0]["x"] for frame in result["frames"]])
        assert filtered.shape == raw.shape
        assert np.std(np.diff(filtered, n=2)) < np.std(np.diff(raw, n=2))
