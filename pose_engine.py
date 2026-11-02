from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from common import FRAME_SIZE, PipelineError, atomic_json_dump, iter_video_frames, probe_video

BODY_JOINTS = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]
HAND_JOINTS = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def _import_dwpose():
    try:
        from easy_dwpose import DWposeDetector
    except ImportError as exc:
        raise PipelineError("easy-dwpose is not installed. Install requirements.txt first.") from exc
    return DWposeDetector


def _as_array(value: Any, dtype=np.float32) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _normalize_xy(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    result = array[..., :2].astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def _candidate_persons_from_dwpose(pose: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    bodies = _as_array(pose.get("bodies"))
    body_scores = _as_array(pose.get("body_scores"))
    hands = _as_array(pose.get("hands"))
    hand_scores = _as_array(pose.get("hands_scores", pose.get("hands_score")))
    faces = _as_array(pose.get("faces"))
    face_scores = _as_array(pose.get("faces_scores", pose.get("faces_score")))

    if bodies.size == 0:
        return []

    if bodies.ndim == 2 and bodies.shape[1] >= 2:
        if bodies.shape[0] % 18 != 0:
            raise PipelineError(f"Unexpected DWPose body shape: {bodies.shape}")
        bodies = bodies[:, :2].reshape(-1, 18, 2)

    person_count = bodies.shape[0]
    persons: list[dict[str, np.ndarray]] = []
    for person_index in range(person_count):
        body = _normalize_xy(bodies[person_index])
        raw_body_mask = body_scores[person_index].reshape(-1) if body_scores.ndim >= 2 and len(body_scores) > person_index else np.full(18, -1.0)
        # easy-dwpose rewrites body scores into OpenPose subset indices (-1 for missing),
        # so these are visibility flags rather than calibrated probabilities.
        body_score = (raw_body_mask[:18] >= 0).astype(np.float32)

        hand_left_index = person_index * 2
        hand_right_index = hand_left_index + 1
        left_hand = (
            _normalize_xy(hands[hand_left_index])
            if hands.ndim >= 3 and len(hands) > hand_left_index
            else np.zeros((21, 2), dtype=np.float32)
        )
        right_hand = (
            _normalize_xy(hands[hand_right_index])
            if hands.ndim >= 3 and len(hands) > hand_right_index
            else np.zeros((21, 2), dtype=np.float32)
        )
        left_score = (
            np.clip(hand_scores[hand_left_index].reshape(-1)[:21], 0.0, 1.0)
            if hand_scores.ndim >= 2 and len(hand_scores) > hand_left_index
            else np.zeros(21, dtype=np.float32)
        )
        right_score = (
            np.clip(hand_scores[hand_right_index].reshape(-1)[:21], 0.0, 1.0)
            if hand_scores.ndim >= 2 and len(hand_scores) > hand_right_index
            else np.zeros(21, dtype=np.float32)
        )
        face = (
            _normalize_xy(faces[person_index])
            if faces.ndim >= 3 and len(faces) > person_index
            else np.zeros((68, 2), dtype=np.float32)
        )
        face_score = (
            np.clip(face_scores[person_index].reshape(-1)[:68], 0.0, 1.0)
            if face_scores.ndim >= 2 and len(face_scores) > person_index
            else np.zeros(68, dtype=np.float32)
        )
        persons.append(
            {
                "body": body,
                "body_scores": body_score,
                "left_hand": left_hand,
                "left_hand_scores": left_score,
                "right_hand": right_hand,
                "right_hand_scores": right_score,
                "face": face,
                "face_scores": face_score,
            }
        )
    return persons


def _select_primary_person(pose: dict[str, Any]) -> int | None:
    persons = _candidate_persons_from_dwpose(pose)
    if not persons:
        return None

    candidates: list[tuple[float, int]] = []
    for index, person in enumerate(persons):
        body = person["body"]
        scores = person["body_scores"]
        visible = scores >= 0.30
        if not np.any(visible):
            candidates.append((0.0, index))
            continue

        visible_points = body[visible]
        x1, y1 = np.min(visible_points, axis=0)
        x2, y2 = np.max(visible_points, axis=0)
        area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        center = np.mean(visible_points, axis=0)
        centrality = 1.0 - min(float(np.linalg.norm(center - 0.5)) / 0.7072, 1.0)
        visibility = float(np.mean(scores[visible]))
        candidates.append((0.60 * area + 0.25 * centrality + 0.15 * visibility, index))

    return max(candidates, key=lambda item: item[0])[1]


def _extract_person_from_dwpose(pose: dict[str, Any], person_index: int | None) -> dict[str, np.ndarray]:
    persons = _candidate_persons_from_dwpose(pose)
    if person_index is None or person_index >= len(persons):
        return {
            "body": np.zeros((18, 2), dtype=np.float32),
            "body_scores": np.zeros(18, dtype=np.float32),
            "left_hand": np.zeros((21, 2), dtype=np.float32),
            "left_hand_scores": np.zeros(21, dtype=np.float32),
            "right_hand": np.zeros((21, 2), dtype=np.float32),
            "right_hand_scores": np.zeros(21, dtype=np.float32),
            "face": np.zeros((68, 2), dtype=np.float32),
            "face_scores": np.zeros(68, dtype=np.float32),
        }
    return persons[person_index]


def _joint_objects(points: np.ndarray, scores: np.ndarray, names: list[str]) -> list[dict]:
    return [
        {"joint": name, "x": round(float(point[0]), 7), "y": round(float(point[1]), 7), "confidence": round(float(score), 6)}
        for name, point, score in zip(names, points, scores)
    ]


def extract_pose(
    input_video: Path,
    output_json: Path,
    device: str = "cpu",
    include_face: bool = True,
    include_hands: bool = True,
) -> Path:
    if not input_video.exists():
        raise PipelineError(f"Input video does not exist: {input_video}")
    info = probe_video(input_video)
    DWposeDetector = _import_dwpose()
    detector = DWposeDetector(device=device)

    frames: list[dict] = []
    for frame_index, timestamp_ms, frame_bgr in iter_video_frames(input_video):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb, mode="RGB")
        raw_pose = detector(
            image,
            draw_pose=False,
            include_hands=include_hands,
            include_face=include_face,
            detect_resolution=FRAME_SIZE,
        )
        if not isinstance(raw_pose, dict):
            raise PipelineError("DWPose did not return its expected raw dictionary output")
        person_index = _select_primary_person(raw_pose)
        extracted = _extract_person_from_dwpose(raw_pose, person_index)
        frame_record = {
            "frame_index": frame_index,
            "timestamp_ms": round(timestamp_ms, 3),
            "body_keypoints": _joint_objects(extracted["body"], extracted["body_scores"], BODY_JOINTS),
            "hands": {
                "left": _joint_objects(extracted["left_hand"], extracted["left_hand_scores"], HAND_JOINTS) if include_hands else [],
                "right": _joint_objects(extracted["right_hand"], extracted["right_hand_scores"], HAND_JOINTS) if include_hands else [],
            },
            "face_keypoints": _joint_objects(
                extracted["face"], extracted["face_scores"], [f"face_{index}" for index in range(68)]
            ) if include_face else [],
            "world_keypoints_3d": None,
            "pose_backend": "DWPose/easy-dwpose",
            "selected_person_index": person_index,
        }
        frames.append(frame_record)

    payload = {
        "schema_version": "1.1.0",
        "clip_id": output_json.parent.name,
        "fps": info.fps,
        "frame_count": len(frames),
        "resolution": [FRAME_SIZE, FRAME_SIZE],
        "coordinate_system": "normalized_xy_0_to_1_origin_top_left",
        "z_coordinate": "not_available_from_DWPose_2D",
        "confidence_semantics": {
            "body": "binary_visibility_from_easy_dwpose_openpose_subset",
            "hands": "detector_score_0_to_1",
            "face": "detector_score_0_to_1",
        },
        "frames": frames,
    }
    atomic_json_dump(payload, output_json)
    return output_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract whole-body DWPose coordinates from a normalized clip.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--no-face", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    extract_pose(args.input, args.output, args.device, not args.no_face, not args.no_hands)
    return 0


if __name__ == "__main__":
    main()
