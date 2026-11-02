from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from student3.evaluate import evaluate
from student3.postprocess import grade_frames, load_config, write_video


def _config() -> dict:
    root = Path(__file__).parents[1]
    return load_config(root / "configs" / "1930s_archival.json")


def test_grade_frames_preserves_geometry_and_introduces_chroma() -> None:
    frames = []
    for offset in (0, 8, 16):
        frame = np.full((512, 384, 3), 105, dtype=np.uint8)
        cv2.rectangle(frame, (100 + offset, 120), (260 + offset, 470), (210, 210, 210), -1)
        frames.append(frame)
    graded = grade_frames(frames, _config())
    assert len(graded) == len(frames)
    assert graded[0].shape == frames[0].shape
    lab = cv2.cvtColor(graded[0], cv2.COLOR_BGR2LAB)
    assert float(np.mean(np.abs(lab[..., 1:].astype(np.float32) - 128.0))) > 1.0


def test_evaluation_passes_for_matching_video_geometry(tmp_path: Path) -> None:
    frames = [np.full((512, 384, 3), 110 + index, dtype=np.uint8) for index in range(3)]
    source = tmp_path / "source.mp4"
    result = tmp_path / "result.mp4"
    write_video(frames, source, 7.0)
    write_video(grade_frames(frames, _config()), result, 7.0)
    report = evaluate(source, result)
    assert report["verdict"] == "pass"
    assert report["output_spec"]["frame_count"] == 3
