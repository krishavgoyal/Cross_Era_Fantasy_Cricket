from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

import cv2
import numpy as np

FRAME_SIZE = 512
DEFAULT_FPS = 24
T = TypeVar("T")


class PipelineError(RuntimeError):
    """Expected runtime failure that should be reported with a stable exit code."""


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineError(f"Unable to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise PipelineError(f"Invalid video stream metadata: {path}")
    duration_s = (frame_count - 1) / fps if frame_count > 1 else 0.0
    return VideoInfo(fps, frame_count, width, height, duration_s)


def iter_video_frames(path: Path) -> Iterator[tuple[int, float, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineError(f"Unable to open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = (frame_index / source_fps) * 1000.0 if source_fps > 0 else 0.0
            yield frame_index, timestamp_ms, frame
            frame_index += 1
    finally:
        capture.release()


def resample_sequence(items: Sequence[T], source_fps: float, target_fps: float) -> tuple[list[T], np.ndarray]:
    if not items:
        return [], np.empty(0, dtype=np.int64)
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if len(items) == 1:
        return [items[0]], np.array([0], dtype=np.int64)
    duration_s = (len(items) - 1) / source_fps
    target_count = max(1, int(round(duration_s * target_fps)) + 1)
    source_positions = np.linspace(0, len(items) - 1, target_count)
    indices = np.rint(source_positions).astype(np.int64)
    return [items[int(i)] for i in indices], indices


def centered_square_crop(
    frame: np.ndarray,
    bbox_xyxy: Sequence[float],
    margin: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected BGR frame with shape HxWx3")

    x1, y1, x2, y2 = map(float, bbox_xyxy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bounding box")

    height, width = frame.shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1
    side = max(box_width, box_height) * (1.0 + 2.0 * margin)

    if side <= 0:
        raise ValueError("Invalid crop side length")

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    # First attempt: keep the square entirely inside the original frame.
    if side <= width and side <= height:
        left = min(max(cx - side * 0.5, 0.0), width - side)
        top = min(max(cy - side * 0.5, 0.0), height - side)

        right = left + side
        bottom = top + side

        left_i = int(round(left))
        top_i = int(round(top))
        right_i = int(round(right))
        bottom_i = int(round(bottom))

        crop = frame[top_i:bottom_i, left_i:right_i]

        if crop.size == 0:
            raise ValueError("Crop collapsed to an empty image")

        crop = cv2.resize(
            crop,
            (FRAME_SIZE, FRAME_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        return crop, (
            float(left_i),
            float(top_i),
            float(right_i),
            float(bottom_i),
        )

    # Extremely unusual case: requested square is larger than the source
    # frame itself. In that case, neutral padding is preferable to reflected
    # image content because reflected players/logos become false visual data.
    pad_left = pad_right = pad_top = pad_bottom = 0

    square_side = int(math.ceil(side))

    left = cx - side * 0.5
    top = cy - side * 0.5
    right = left + side
    bottom = top + side

    if left < 0:
        pad_left = int(math.ceil(-left))
    if top < 0:
        pad_top = int(math.ceil(-top))
    if right > width:
        pad_right = int(math.ceil(right - width))
    if bottom > height:
        pad_bottom = int(math.ceil(bottom - height))

    padded = cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    left += pad_left
    top += pad_top
    right += pad_left
    bottom += pad_top

    left_i = max(0, int(round(left)))
    top_i = max(0, int(round(top)))
    right_i = min(padded.shape[1], int(round(right)))
    bottom_i = min(padded.shape[0], int(round(bottom)))

    crop = padded[top_i:bottom_i, left_i:right_i]

    if crop.size == 0:
        raise ValueError("Crop collapsed to an empty image")

    crop = cv2.resize(
        crop,
        (FRAME_SIZE, FRAME_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return crop, (
        float(left_i - pad_left),
        float(top_i - pad_top),
        float(right_i - pad_left),
        float(bottom_i - pad_top),
    )

def normalize_xy(value: Sequence[float], width: float, height: float) -> tuple[float, float]:
    return float(np.clip(value[0] / max(width, 1.0), 0.0, 1.0)), float(np.clip(value[1] / max(height, 1.0), 0.0, 1.0))


def atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Failed to read JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"JSON root must be an object: {path}")
    return payload


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_mp4(frames: Iterable[np.ndarray], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        _write_mp4_opencv(frames, output_path, fps)
        return

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{FRAME_SIZE}x{FRAME_SIZE}",
            "-r",
            f"{fps:.06f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for frame in frames:
            if frame.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
                raise ValueError(f"Frame must be {FRAME_SIZE}x{FRAME_SIZE}, got {frame.shape[:2]}")
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise PipelineError(f"ffmpeg failed while writing {output_path} (exit code {return_code})")


def _write_mp4_opencv(frames: Iterable[np.ndarray], output_path: Path, fps: float) -> None:
    writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for frame in frames:
            if writer is None:
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (FRAME_SIZE, FRAME_SIZE))
                if not writer.isOpened():
                    raise PipelineError(f"OpenCV cannot open MP4 writer: {output_path}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
