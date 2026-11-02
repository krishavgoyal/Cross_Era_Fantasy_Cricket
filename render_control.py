from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from easy_dwpose.draw.openpose import draw_pose as draw_openpose

from common import FRAME_SIZE, PipelineError, atomic_json_dump, load_json, write_mp4

BODY_EDGES = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]
BODY_COLORS_RGB = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170),
]
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _to_pixel(point: dict) -> tuple[int, int]:
    return int(round(np.clip(float(point["x"]), 0.0, 1.0) * (FRAME_SIZE - 1))), int(round(np.clip(float(point["y"]), 0.0, 1.0) * (FRAME_SIZE - 1)))


def _draw_body(canvas: np.ndarray, keypoints: list[dict], score_threshold: float) -> None:
    for edge_index, (a, b) in enumerate(BODY_EDGES):
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        if keypoints[a]["confidence"] < score_threshold or keypoints[b]["confidence"] < score_threshold:
            continue
        pa = _to_pixel(keypoints[a])
        pb = _to_pixel(keypoints[b])
        color = BODY_COLORS_RGB[edge_index % len(BODY_COLORS_RGB)]
        cv2.line(canvas, pa, pb, color[::-1], thickness=4, lineType=cv2.LINE_AA)
    for keypoint in keypoints:
        if keypoint["confidence"] < score_threshold:
            continue
        cv2.circle(canvas, _to_pixel(keypoint), 4, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)


def _draw_hand(canvas: np.ndarray, keypoints: list[dict], score_threshold: float) -> None:
    for a, b in HAND_EDGES:
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        if keypoints[a]["confidence"] < score_threshold or keypoints[b]["confidence"] < score_threshold:
            continue
        cv2.line(canvas, _to_pixel(keypoints[a]), _to_pixel(keypoints[b]), (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    for keypoint in keypoints:
        if keypoint["confidence"] >= score_threshold:
            cv2.circle(canvas, _to_pixel(keypoint), 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)


def _draw_face(canvas: np.ndarray, keypoints: list[dict], score_threshold: float) -> None:
    for keypoint in keypoints:
        if keypoint["confidence"] >= score_threshold:
            cv2.circle(canvas, _to_pixel(keypoint), 1, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)


BODY_JOINT_ORDER = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
]


def _render_with_easy_dwpose(frame: dict) -> np.ndarray:
    candidate = np.zeros((18, 2), dtype=np.float32)
    subset = np.full((1, 18), -1.0, dtype=np.float32)

    joint_index = {
        name: index
        for index, name in enumerate(BODY_JOINT_ORDER)
    }

    for keypoint in frame.get("body_keypoints", []):
        joint_name = keypoint.get("joint")
        index = joint_index.get(joint_name)

        if index is None:
            continue

        confidence = float(keypoint.get("confidence", 0.0))

        if confidence <= 0.0:
            continue

        candidate[index] = [
            float(keypoint.get("x", 0.0)),
            float(keypoint.get("y", 0.0)),
        ]
        subset[0, index] = float(index)

    def hand_to_array(hand_keypoints: list[dict]) -> np.ndarray:
        output = np.zeros((21, 2), dtype=np.float32)

        for index, keypoint in enumerate(hand_keypoints[:21]):
            confidence = float(keypoint.get("confidence", 0.0))

            if confidence <= 0.20:
                continue

            output[index] = [
                float(keypoint.get("x", 0.0)),
                float(keypoint.get("y", 0.0)),
            ]

        return output

    hands_payload = frame.get("hands", {})

    hands = np.stack(
        [
            hand_to_array(hands_payload.get("left", [])),
            hand_to_array(hands_payload.get("right", [])),
        ],
        axis=0,
    )

    faces_payload = frame.get("face_keypoints", [])

    faces = np.zeros((1, 68, 2), dtype=np.float32)

    for index, keypoint in enumerate(faces_payload[:68]):
        confidence = float(keypoint.get("confidence", 0.0))

        if confidence <= 0.20:
            continue

        faces[0, index] = [
            float(keypoint.get("x", 0.0)),
            float(keypoint.get("y", 0.0)),
        ]

    pose = {
        "bodies": candidate,
        "body_scores": subset,
        "hands": hands,
        "faces": faces,
    }

    return draw_openpose(
        pose,
        height=FRAME_SIZE,
        width=FRAME_SIZE,
        include_face=bool(faces_payload),
        include_hands=True,
    )
def render_control_video(pose_json: Path, output_video: Path, style: str = "dwpose", score_threshold: float = 0.20) -> Path:
    payload = load_json(pose_json)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise PipelineError(f"No pose frames found in {pose_json}")
    if style not in {"dwpose", "openpose"}:
        raise PipelineError(f"Unsupported control style: {style}")

    rendered_frames: list[np.ndarray] = []
    for frame in frames:
        if style == "dwpose":
            try:
                canvas = _render_with_easy_dwpose(frame)
            except PipelineError:
                # Unit tests and offline tooling can still validate dimensions without
                # installing the heavy DWPose runtime; production uses the canonical renderer.
                canvas = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
                _draw_body(canvas, frame.get("body_keypoints", []), max(score_threshold, 0.30))
                hands = frame.get("hands", {})
                _draw_hand(canvas, hands.get("left", []), score_threshold)
                _draw_hand(canvas, hands.get("right", []), score_threshold)
                _draw_face(canvas, frame.get("face_keypoints", []), score_threshold)
        else:
            canvas = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
            _draw_body(canvas, frame.get("body_keypoints", []), max(score_threshold, 0.30))
            hands = frame.get("hands", {})
            _draw_hand(canvas, hands.get("left", []), score_threshold)
            _draw_hand(canvas, hands.get("right", []), score_threshold)
            _draw_face(canvas, frame.get("face_keypoints", []), score_threshold)
        rendered_frames.append(canvas)

    write_mp4(rendered_frames, output_video, float(payload["fps"]))
    return output_video


def export_numpy_matrix(pose_json: Path, output_npz: Path) -> Path:
    payload = load_json(pose_json)
    frames = payload["frames"]
    body = np.asarray(
        [
            [[joint["x"], joint["y"], joint["confidence"]] for joint in frame["body_keypoints"]]
            for frame in frames
        ],
        dtype=np.float32,
    )
    left_hand = np.asarray(
        [[[joint["x"], joint["y"], joint["confidence"]] for joint in frame["hands"]["left"]] for frame in frames],
        dtype=np.float32,
    )
    right_hand = np.asarray(
        [[[joint["x"], joint["y"], joint["confidence"]] for joint in frame["hands"]["right"]] for frame in frames],
        dtype=np.float32,
    )
    face = np.asarray(
        [[[joint["x"], joint["y"], joint["confidence"]] for joint in frame.get("face_keypoints", [])] for frame in frames],
        dtype=np.float32,
    )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, body=body, left_hand=left_hand, right_hand=right_hand, face=face, fps=float(payload["fps"]))
    return output_npz


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render pose coordinates into a diffusion-control video.")
    parser.add_argument("--pose-json", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--style", choices=("dwpose", "openpose"), default="dwpose")
    parser.add_argument("--score-threshold", type=float, default=0.20)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    render_control_video(args.pose_json, args.output_video, args.style, args.score_threshold)
    if args.output_npz:
        export_numpy_matrix(args.pose_json, args.output_npz)
    return 0


if __name__ == "__main__":
    main()
