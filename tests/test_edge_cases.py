from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from clip_processor import TrackObservation, _crop_edge_risk, _expand_for_pose, _select_observation
from pose_engine import _candidate_persons_from_dwpose, _select_primary_person
from temporal_smoothing import smooth_pose_file


def test_primary_selection_prefers_large_central_actor_on_first_frame() -> None:
    observations = [
        TrackObservation(3, np.array([0, 0, 120, 200], dtype=float), 0.95),
        TrackObservation(7, np.array([300, 100, 900, 850], dtype=float), 0.85),
    ]
    selected = _select_observation(observations, None, 1000, 1000, None)
    assert selected is not None
    assert selected.track_id == 7


def test_primary_selection_can_switch_when_existing_track_becomes_implausible() -> None:
    observations = [
        TrackObservation(3, np.array([20, 20, 120, 150], dtype=float), 0.95),
        TrackObservation(7, np.array([250, 100, 850, 900], dtype=float), 0.85),
    ]
    selected = _select_observation(observations, 3, 1000, 1000, None)
    assert selected is not None
    assert selected.track_id == 7


def test_pose_expansion_adds_hand_area_to_crop() -> None:
    bbox = np.array([300, 250, 500, 550], dtype=float)
    wrists = np.array([[120, 280], [680, 520]], dtype=float)
    scores = np.array([0.95, 0.95], dtype=float)
    expanded = _expand_for_pose(bbox, wrists, scores, 800, 800, 0.30, 0.15)
    assert expanded[0] < bbox[0]
    assert expanded[2] > bbox[2]


def test_crop_edge_risk_detects_joint_near_boundary() -> None:
    crop = (100, 100, 500, 500)
    points = np.array([[108, 300], [300, 300]], dtype=float)
    scores = np.array([0.95, 0.95], dtype=float)
    assert _crop_edge_risk(points, scores, crop)


def test_dwpose_raw_structure_is_parsed_correctly() -> None:
    bodies = np.zeros((36, 2), dtype=np.float32)
    bodies[:18] = 0.2
    bodies[18:] = 0.7
    body_scores = np.ones((2, 18), dtype=np.float32)
    hands = np.zeros((4, 21, 2), dtype=np.float32)
    hand_scores = np.ones((4, 21), dtype=np.float32)
    faces = np.zeros((2, 68, 2), dtype=np.float32)
    face_scores = np.ones((2, 68), dtype=np.float32)
    pose = {
        "bodies": bodies,
        "body_scores": body_scores,
        "hands": hands,
        "hands_scores": hand_scores,
        "faces": faces,
        "faces_scores": face_scores,
    }
    persons = _candidate_persons_from_dwpose(pose)
    assert len(persons) == 2
    assert _select_primary_person(pose) == 1


def test_confidence_gap_is_interpolated_before_filtering() -> None:
    frames = []
    for index in range(5):
        confidence = 0.95 if index in (0, 3, 4) else 0.1
        x = float(index) / 4.0 if confidence >= 0.95 else 0.0
        frames.append(
            {
                "frame_index": index,
                "timestamp_ms": index * 1000 / 24,
                "body_keypoints": [
                    {"joint": "nose", "x": x, "y": 0.5, "confidence": confidence}
                ],
                "hands": {"left": [], "right": []},
                "face_keypoints": [],
            }
        )
    payload = {"schema_version": "1.1.0", "fps": 24, "frame_count": 5, "resolution": [512, 512], "frames": frames}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pose_data.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        smooth_pose_file(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["frames"][1]["body_keypoints"][0]["confidence_interpolated"] is True
        assert result["frames"][2]["body_keypoints"][0]["confidence_interpolated"] is True
        assert 0.0 < result["frames"][1]["body_keypoints"][0]["x"] < 0.75
        assert 0.0 < result["frames"][2]["body_keypoints"][0]["x"] < 0.95
