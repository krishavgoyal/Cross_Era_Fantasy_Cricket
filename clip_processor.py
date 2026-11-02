from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from common import (
    DEFAULT_FPS,
    FRAME_SIZE,
    PipelineError,
    atomic_json_dump,
    centered_square_crop,
    iter_video_frames,
    probe_video,
    resample_sequence,
    sha256_file,
    write_mp4,
)


@dataclass
class TrackObservation:
    track_id: int
    bbox: np.ndarray
    confidence: float
    keypoints_xy: np.ndarray | None = None
    keypoint_confidence: np.ndarray | None = None


def _import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise PipelineError("Ultralytics is not installed. Install requirements.txt first.") from exc
    return YOLO


def _extract_observations(result) -> list[TrackObservation]:
    observations: list[TrackObservation] = []
    if result is None:
        return observations

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return observations

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones(len(xyxy))
    ids_tensor = getattr(boxes, "id", None)
    ids = ids_tensor.detach().cpu().numpy().astype(int) if ids_tensor is not None else np.arange(len(xyxy))
    classes = boxes.cls.detach().cpu().numpy().astype(int) if getattr(boxes, "cls", None) is not None else np.zeros(len(xyxy), dtype=int)

    keypoints = getattr(result, "keypoints", None)
    keypoint_xy = keypoints.xy.detach().cpu().numpy() if keypoints is not None else None
    keypoint_conf = (
        keypoints.conf.detach().cpu().numpy()
        if keypoints is not None and getattr(keypoints, "conf", None) is not None
        else None
    )

    for index, (bbox, score, track_id, cls_id) in enumerate(zip(xyxy, conf, ids, classes)):
        if cls_id != 0:
            continue
        observations.append(
            TrackObservation(
                track_id=int(track_id),
                bbox=bbox.astype(float),
                confidence=float(score),
                keypoints_xy=None if keypoint_xy is None else keypoint_xy[index].astype(float),
                keypoint_confidence=None if keypoint_conf is None else keypoint_conf[index].astype(float),
            )
        )
    return observations


def _center_score(bbox: np.ndarray, frame_width: int, frame_height: int) -> float:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) * 0.5) / frame_width
    cy = ((y1 + y2) * 0.5) / frame_height
    distance = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    return 1.0 - min(distance / 0.7072, 1.0)


def _area_score(bbox: np.ndarray, frame_width: int, frame_height: int) -> float:
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return min(area / max(frame_width * frame_height, 1), 1.0)


def _observation_score(obs: TrackObservation, frame_width: int, frame_height: int) -> float:
    # Initial actor selection is deliberately dominated by spatial evidence. This
    # makes the first target choice deterministic even when tracker IDs are arbitrary.
    return (
        0.55 * _area_score(obs.bbox, frame_width, frame_height)
        + 0.35 * _center_score(obs.bbox, frame_width, frame_height)
        + 0.10 * obs.confidence
    )


def _select_observation(
    observations: list[TrackObservation],
    previous_track_id: Optional[int],
    frame_width: int,
    frame_height: int,
    forced_track_id: Optional[int],
) -> Optional[TrackObservation]:
    if not observations:
        return None

    if forced_track_id is not None:
        matches = [obs for obs in observations if obs.track_id == forced_track_id]
        return max(matches, key=lambda item: item.confidence) if matches else None

    global_best = max(observations, key=lambda item: _observation_score(item, frame_width, frame_height))
    if previous_track_id is None:
        return global_best

    same_track = [obs for obs in observations if obs.track_id == previous_track_id]
    if not same_track:
        return global_best

    current = max(same_track, key=lambda item: item.confidence)
    current_score = _observation_score(current, frame_width, frame_height)
    global_score = _observation_score(global_best, frame_width, frame_height)

    # ByteTrack continuity is useful, but not sacred. If another person becomes
    # substantially more plausible as the primary actor, allow an ID switch rather
    # than following an umpire/non-striker forever.
    if global_best.track_id != current.track_id and global_score > current_score * 1.30:
        return global_best
    return current


def _ema_bbox(previous: Optional[np.ndarray], current: np.ndarray, alpha: float) -> np.ndarray:
    return current.copy() if previous is None else alpha * current + (1.0 - alpha) * previous


def _expand_for_pose(
    bbox: np.ndarray,
    keypoints_xy: np.ndarray | None,
    keypoint_confidence: np.ndarray | None,
    frame_width: int,
    frame_height: int,
    pose_confidence: float,
    extra_motion_margin: float,
) -> np.ndarray:
    expanded = bbox.astype(float).copy()
    if keypoints_xy is not None and keypoint_confidence is not None:
        valid = (
            np.isfinite(keypoints_xy).all(axis=1)
            & (keypoint_confidence >= pose_confidence)
            & (keypoints_xy[:, 0] >= 0)
            & (keypoints_xy[:, 1] >= 0)
        )
        if np.any(valid):
            visible = keypoints_xy[valid]
            expanded[0] = min(expanded[0], float(np.min(visible[:, 0])))
            expanded[1] = min(expanded[1], float(np.min(visible[:, 1])))
            expanded[2] = max(expanded[2], float(np.max(visible[:, 0])))
            expanded[3] = max(expanded[3], float(np.max(visible[:, 1])))

    width = max(expanded[2] - expanded[0], 1.0)
    height = max(expanded[3] - expanded[1], 1.0)
    pad_x = width * extra_motion_margin
    pad_y = height * extra_motion_margin
    return np.array(
        [
            max(0.0, expanded[0] - pad_x),
            max(0.0, expanded[1] - pad_y),
            min(float(frame_width), expanded[2] + pad_x),
            min(float(frame_height), expanded[3] + pad_y),
        ],
        dtype=float,
    )


def _crop_edge_risk(
    keypoints_xy: np.ndarray | None,
    keypoint_confidence: np.ndarray | None,
    crop_window: tuple[float, float, float, float],
    threshold_px_fraction: float = 0.04,
    confidence_threshold: float = 0.30,
) -> bool:
    if keypoints_xy is None or keypoint_confidence is None:
        return False
    left, top, right, bottom = crop_window
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    points = keypoints_xy[keypoint_confidence >= confidence_threshold]
    if points.size == 0:
        return False
    rel_x = (points[:, 0] - left) / width
    rel_y = (points[:, 1] - top) / height
    return bool(
        np.any(rel_x < threshold_px_fraction)
        or np.any(rel_x > 1.0 - threshold_px_fraction)
        or np.any(rel_y < threshold_px_fraction)
        or np.any(rel_y > 1.0 - threshold_px_fraction)
    )


def process_clip(
    input_path: Path,
    output_dir: Path,
    clip_id: str,
    action_type: str,
    target_fps: float = DEFAULT_FPS,
    crop_margin: float = 0.40,
    bbox_smoothing: float = 0.35,
    confidence: float = 0.25,
    model_name: str = "yolo26n-pose.pt",
    tracker_name: str = "bytetrack.yaml",
    device: str = "",
    forced_track_id: Optional[int] = None,
    pose_expand_margin: float = 0.15,
    max_missed_frames: int = 8,
) -> Path:
    if target_fps <= 0:
        raise PipelineError("target_fps must be positive")
    if not 0.05 <= crop_margin <= 1.50:
        raise PipelineError("crop_margin must be between 0.05 and 1.50")
    if not 0.0 <= pose_expand_margin <= 1.0:
        raise PipelineError("pose_expand_margin must be between 0 and 1")
    if max_missed_frames < 0:
        raise PipelineError("max_missed_frames must be non-negative")
    if not input_path.exists():
        raise PipelineError(f"Input video does not exist: {input_path}")

    info = probe_video(input_path)
    YOLO = _import_yolo()
    model = YOLO(model_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_frames: list[np.ndarray] = []
    frame_meta: list[dict] = []
    previous_track_id: Optional[int] = None
    previous_bbox: Optional[np.ndarray] = None
    missed_frames = 0
    edge_risk_frames = 0

    for source_index, timestamp_ms, frame in iter_video_frames(input_path):
        track_kwargs = {
            "persist": source_index > 0,
            "tracker": tracker_name,
            "conf": confidence,
            "classes": [0],
            "verbose": False,
        }
        if device:
            track_kwargs["device"] = device
        results = model.track(frame, **track_kwargs)
        observations = _extract_observations(results[0] if results else None)
        selected = _select_observation(
            observations,
            previous_track_id,
            frame.shape[1],
            frame.shape[0],
            forced_track_id,
        )

        if selected is None:
            missed_frames += 1
            if previous_bbox is None or missed_frames > max_missed_frames:
                continue
            selected_bbox = previous_bbox.copy()
            track_id = previous_track_id
            keypoints_xy = None
            keypoint_confidence = None
        else:
            missed_frames = 0
            pose_expanded_bbox = _expand_for_pose(
                selected.bbox,
                selected.keypoints_xy,
                selected.keypoint_confidence,
                frame.shape[1],
                frame.shape[0],
                pose_confidence=0.30,
                extra_motion_margin=pose_expand_margin,
            )
            selected_bbox = _ema_bbox(previous_bbox, pose_expanded_bbox, bbox_smoothing)
            track_id = selected.track_id
            previous_track_id = selected.track_id
            keypoints_xy = selected.keypoints_xy
            keypoint_confidence = selected.keypoint_confidence

        crop, crop_window = centered_square_crop(frame, selected_bbox, crop_margin)
        if _crop_edge_risk(keypoints_xy, keypoint_confidence, crop_window):
            edge_risk_frames += 1

        source_frames.append(crop)
        source_width = frame.shape[1]
        source_height = frame.shape[0]
        bx1, by1, bx2, by2 = selected_bbox
        cx1, cy1, cx2, cy2 = crop_window
        frame_meta.append(
            {
                "source_frame_index": source_index,
                "timestamp_ms": round(timestamp_ms, 3),
                "track_id": int(track_id) if track_id is not None else None,
                "bbox_source_xyxy_px": [round(float(v), 3) for v in selected_bbox],
                "bbox_source_xyxy_norm": [
                    round(float(np.clip(bx1 / source_width, 0.0, 1.0)), 6),
                    round(float(np.clip(by1 / source_height, 0.0, 1.0)), 6),
                    round(float(np.clip(bx2 / source_width, 0.0, 1.0)), 6),
                    round(float(np.clip(by2 / source_height, 0.0, 1.0)), 6),
                ],
                "crop_window_source_xyxy_px": [round(float(v), 3) for v in crop_window],
                "crop_window_source_xyxy_norm": [
                    round(float(np.clip(cx1 / source_width, 0.0, 1.0)), 6),
                    round(float(np.clip(cy1 / source_height, 0.0, 1.0)), 6),
                    round(float(np.clip(cx2 / source_width, 0.0, 1.0)), 6),
                    round(float(np.clip(cy2 / source_height, 0.0, 1.0)), 6),
                ],
                "crop_edge_risk": _crop_edge_risk(keypoints_xy, keypoint_confidence, crop_window),
            }
        )
        previous_bbox = selected_bbox

    if not source_frames:
        raise PipelineError("No trackable person frames were produced")

    sampled_frames, sampled_indices = resample_sequence(source_frames, info.fps, target_fps)
    sampled_meta = []
    for output_index, source_list_index in enumerate(sampled_indices.tolist()):
        source_list_index = min(source_list_index, len(frame_meta) - 1)
        payload = dict(frame_meta[source_list_index])
        payload["frame_index"] = output_index
        payload["timestamp_ms"] = round((output_index / target_fps) * 1000.0, 3)
        sampled_meta.append(payload)

    output_edge_risk = sum(bool(item["crop_edge_risk"]) for item in sampled_meta)

    source_path = output_dir / "source_512.mp4"
    metadata_path = output_dir / "metadata.json"
    write_mp4(sampled_frames, source_path, target_fps)
    metadata = {
        "schema_version": "1.1.0",
        "project": "Cross-Era Fantasy Cricket",
        "clip_id": clip_id,
        "action_type": action_type,
        "role": "unknown",
        "source": {
            "path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            "source_fps": info.fps,
            "source_frame_count": info.frame_count,
            "source_resolution": [info.width, info.height],
            "duration_s": round(info.duration_s, 6),
        },
        "processed": {
            "fps": target_fps,
            "frame_count": len(sampled_frames),
            "resolution": [FRAME_SIZE, FRAME_SIZE],
            "crop_margin": crop_margin,
            "pose_expand_margin": pose_expand_margin,
            "bbox_smoothing_alpha": bbox_smoothing,
            "tracker": tracker_name,
            "detector": model_name,
            "coordinate_system": "normalized_xy_0_to_1_origin_top_left",
            "bat_tracking": "not_detected; crop safety margins are based on person pose and configurable padding",
            "crop_edge_risk_frames": output_edge_risk,
            "crop_edge_risk_rate": round(output_edge_risk / max(len(sampled_meta), 1), 6),
            "unresolved_tracking_frames": sum(1 for item in sampled_meta if item["track_id"] is None),
        },
        "frames": sampled_meta,
    }
    atomic_json_dump(metadata, metadata_path)
    return source_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track and spatially normalize a cricket action clip.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--action-type", required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, choices=(24.0, 30.0))
    parser.add_argument("--crop-margin", type=float, default=0.40)
    parser.add_argument("--pose-expand-margin", type=float, default=0.15)
    parser.add_argument("--bbox-smoothing", type=float, default=0.35)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-missed-frames", type=int, default=8)
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="")
    parser.add_argument("--track-id", type=int, default=None, help="Pin a known tracker ID when auto-selection is ambiguous.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    process_clip(
        args.input,
        args.output_dir,
        args.clip_id,
        args.action_type,
        args.fps,
        args.crop_margin,
        args.bbox_smoothing,
        args.confidence,
        args.model,
        args.tracker,
        args.device,
        args.track_id,
        args.pose_expand_margin,
        args.max_missed_frames,
    )
    return 0


if __name__ == "__main__":
    main()
