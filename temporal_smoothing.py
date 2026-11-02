from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np

from common import PipelineError, atomic_json_dump, load_json


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, derivative_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0 or derivative_cutoff <= 0:
            raise ValueError("Cutoff frequencies must be positive")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def __call__(self, timestamp_s: float, value: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = float(value)
            self._t_prev = float(timestamp_s)
            return self._x_prev
        dt = max(timestamp_s - self._t_prev, 1e-6)
        derivative = (value - self._x_prev) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        derivative_hat = derivative_alpha * derivative + (1.0 - derivative_alpha) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(derivative_hat)
        signal_alpha = self._alpha(cutoff, dt)
        filtered = signal_alpha * value + (1.0 - signal_alpha) * self._x_prev
        self._x_prev = float(filtered)
        self._dx_prev = float(derivative_hat)
        self._t_prev = float(timestamp_s)
        return self._x_prev


def _interpolate_short_confidence_gaps(
    x: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = x.astype(np.float32, copy=True)
    confidences = confidence.astype(np.float32, copy=True)
    valid = np.isfinite(values) & (confidences >= threshold)
    interpolated = np.zeros(len(values), dtype=bool)

    index = 0
    while index < len(values):
        if valid[index]:
            index += 1
            continue
        start = index
        while index < len(values) and not valid[index]:
            index += 1
        end = index - 1
        gap_length = end - start + 1
        left = start - 1
        right = index
        if gap_length <= max_gap and left >= 0 and right < len(values) and valid[left] and valid[right]:
            values[start : end + 1] = np.linspace(values[left], values[right], gap_length + 2, dtype=np.float32)[1:-1]
            interpolated[start : end + 1] = True

    return values, interpolated


def _smooth_joint_trajectory(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    timestamps_s: np.ndarray,
    min_cutoff: float,
    beta: float,
    derivative_cutoff: float,
    confidence_threshold: float,
    interpolation_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    interpolated = np.zeros(len(coordinates), dtype=bool)
    working = coordinates.astype(np.float32, copy=True)

    valid = (
        np.isfinite(working).all(axis=1)
        & (confidence >= confidence_threshold)
    )

    for axis in range(2):
        working[:, axis], axis_interpolated = _interpolate_short_confidence_gaps(
            working[:, axis],
            confidence,
            confidence_threshold,
            interpolation_gap,
        )
        interpolated |= axis_interpolated

    usable = valid | interpolated

    output = working.copy()

    for axis in range(2):
        signal_filter: OneEuroFilter | None = None

        for index, timestamp in enumerate(timestamps_s):
            if not usable[index]:
                # Preserve the detector's coordinate and invalid confidence state.
                # Do not manufacture a trajectory across long occlusions.
                output[index, axis] = working[index, axis]
                continue

            if signal_filter is None:
                # Start the filter at the first trustworthy observation after an
                # initial occlusion instead of propagating an invalid frame-0 value.
                signal_filter = OneEuroFilter(
                    min_cutoff,
                    beta,
                    derivative_cutoff,
                )
                output[index, axis] = signal_filter(
                    float(timestamp),
                    float(working[index, axis]),
                )
                continue

            output[index, axis] = signal_filter(
                float(timestamp),
                float(working[index, axis]),
            )

    return output, interpolated


def _process_joint_group(
    frames: list[dict],
    group_getter: Callable[[dict], list[dict]],
    min_cutoff: float,
    beta: float,
    derivative_cutoff: float,
    confidence_threshold: float,
    interpolation_gap: int,
) -> None:
    timestamps = np.array([float(frame["timestamp_ms"]) / 1000.0 for frame in frames], dtype=np.float32)
    if not frames:
        return
    joint_count = len(group_getter(frames[0]))
    if joint_count == 0:
        return

    coordinates = np.zeros((len(frames), joint_count, 2), dtype=np.float32)
    confidences = np.zeros((len(frames), joint_count), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        joints = group_getter(frame)
        if len(joints) != joint_count:
            return
        for joint_index, joint in enumerate(joints):
            coordinates[frame_index, joint_index] = [float(joint["x"]), float(joint["y"])]
            confidences[frame_index, joint_index] = float(joint["confidence"])

    for joint_index in range(joint_count):
        smoothed, interpolated = _smooth_joint_trajectory(
            coordinates[:, joint_index],
            confidences[:, joint_index],
            timestamps,
            min_cutoff,
            beta,
            derivative_cutoff,
            confidence_threshold,
            interpolation_gap,
        )
        for frame_index, frame in enumerate(frames):
            joint = group_getter(frame)[joint_index]
            joint["raw_x"] = joint["x"]
            joint["raw_y"] = joint["y"]
            joint["x"] = round(float(smoothed[frame_index, 0]), 7)
            joint["y"] = round(float(smoothed[frame_index, 1]), 7)
            if interpolated[frame_index]:
                joint["confidence_interpolated"] = True


def smooth_pose_file(
    path: Path,
    min_cutoff: float = 1.4,
    beta: float = 0.35,
    derivative_cutoff: float = 1.0,
    include_hands: bool = True,
    include_face: bool = True,
    confidence_threshold: float = 0.30,
    interpolation_gap: int = 2,
) -> Path:
    payload = load_json(path)
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise PipelineError(f"pose_data.json contains no frames: {path}")
    if interpolation_gap < 0 or interpolation_gap > 10:
        raise ValueError("interpolation_gap must be between 0 and 10")

    for frame in frames:
        frame.setdefault("raw_body_keypoints", [dict(item) for item in frame["body_keypoints"]])
        frame.setdefault(
            "raw_hands",
            {
                "left": [dict(item) for item in frame["hands"].get("left", [])],
                "right": [dict(item) for item in frame["hands"].get("right", [])],
            },
        )
        frame.setdefault("raw_face_keypoints", [dict(item) for item in frame.get("face_keypoints", [])])

    _process_joint_group(
        frames,
        lambda frame: frame["body_keypoints"],
        min_cutoff,
        beta,
        derivative_cutoff,
        confidence_threshold,
        interpolation_gap,
    )

    if include_hands:
        _process_joint_group(
            frames,
            lambda frame: frame["hands"].get("left", []),
            min_cutoff,
            beta,
            derivative_cutoff,
            confidence_threshold,
            interpolation_gap,
        )
        _process_joint_group(
            frames,
            lambda frame: frame["hands"].get("right", []),
            min_cutoff,
            beta,
            derivative_cutoff,
            confidence_threshold,
            interpolation_gap,
        )

    if include_face:
        _process_joint_group(
            frames,
            lambda frame: frame.get("face_keypoints", []),
            min_cutoff,
            beta,
            derivative_cutoff,
            confidence_threshold,
            interpolation_gap,
        )

    payload["smoothing"] = {
        "algorithm": "1_euro_filter",
        "min_cutoff": min_cutoff,
        "beta": beta,
        "derivative_cutoff": derivative_cutoff,
        "confidence_threshold": confidence_threshold,
        "interpolation_gap_frames": interpolation_gap,
        "interpolation_policy": "confidence_gap_<=_2_frames_with_high_confidence_neighbors",
        "long_occlusion_policy": "hold_last_trustworthy_coordinate",
        "applied_to": ["body", "hands", "face"],
    }
    atomic_json_dump(payload, path)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a confidence-aware 1-Euro filter to pose trajectories.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--min-cutoff", type=float, default=1.4)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--derivative-cutoff", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--interpolation-gap", type=int, default=2)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    smooth_pose_file(
        args.input,
        args.min_cutoff,
        args.beta,
        args.derivative_cutoff,
        confidence_threshold=args.confidence_threshold,
        interpolation_gap=args.interpolation_gap,
    )
    return 0


if __name__ == "__main__":
    main()
