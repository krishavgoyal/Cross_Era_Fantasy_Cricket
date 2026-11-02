from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class PostprocessError(RuntimeError):
    """Raised when a supplied video cannot satisfy the Student 3 contract."""


@dataclass(frozen=True)
class VideoSpec:
    width: int
    height: int
    fps: float
    frame_count: int


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"expected_width", "expected_height", "expected_fps", "palette"}
    missing = required.difference(config)
    if missing:
        raise PostprocessError(f"Invalid style configuration; missing {sorted(missing)}")
    return config


def read_video(path: Path) -> tuple[list[np.ndarray], VideoSpec]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PostprocessError(f"Unable to open input video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames or fps <= 0 or width <= 0 or height <= 0:
        raise PostprocessError(f"Input has no usable video frames: {path}")
    return frames, VideoSpec(width, height, fps, len(frames))


def validate_input(spec: VideoSpec, config: dict[str, Any], strict: bool) -> None:
    expected = (int(config["expected_width"]), int(config["expected_height"]))
    actual = (spec.width, spec.height)
    if actual != expected:
        raise PostprocessError(f"Expected {expected[0]}x{expected[1]}, received {actual[0]}x{actual[1]}")
    if strict and abs(spec.fps - float(config["expected_fps"])) > 0.05:
        raise PostprocessError(f"Expected {config['expected_fps']} FPS, received {spec.fps:.3f} FPS")


def _roi_mask(shape: tuple[int, int], roi: list[float]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = roi
    mask = np.zeros((height, width), dtype=np.float32)
    left, top = int(x1 * width), int(y1 * height)
    right, bottom = int(x2 * width), int(y2 * height)
    mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), max(2.0, width * 0.018))


def _mix_palette(ab: np.ndarray, mask: np.ndarray, target: tuple[int, int], opacity: float) -> None:
    alpha = np.clip(mask * opacity, 0.0, 1.0)[..., None]
    target_array = np.array(target, dtype=np.float32).reshape(1, 1, 2)
    ab[:] = ab * (1.0 - alpha) + target_array * alpha


def _semantic_chroma(gray: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Create a conservative, deterministic historical palette from a monochrome frame.

    This is deliberately palette-guided rather than an unconstrained per-frame AI
    colorizer: it keeps the project reproducible and avoids colour flicker when no
    colour reference sequence is available.
    """
    height, width = gray.shape
    palette = config["palette"]
    ab = np.full((height, width, 2), palette["neutral_lab_ab"], dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    yn, xn = yy / max(height - 1, 1), xx / max(width - 1, 1)
    luma = gray.astype(np.float32)

    # Environment is intentionally desaturated. It provides a stable historical
    # setting without recolouring the high-detail athlete pixel-by-pixel.
    sky = np.clip((0.40 - yn) / 0.28, 0.0, 1.0) * np.clip((luma - 70.0) / 110.0, 0.0, 1.0)
    grass = np.clip((yn - 0.48) / 0.34, 0.0, 1.0) * np.clip((145.0 - luma) / 90.0, 0.0, 1.0)
    _mix_palette(ab, sky.astype(np.float32), tuple(palette["sky_lab_ab"]), 0.50)
    _mix_palette(ab, grass.astype(np.float32), tuple(palette["grass_lab_ab"]), 0.52)

    subject = _roi_mask((height, width), config["subject_roi"])
    cap = _roi_mask((height, width), config["cap_roi"])
    face = _roi_mask((height, width), config["face_roi"])
    bat = _roi_mask((height, width), config["bat_roi"])

    cloth = subject * np.clip((luma - 132.0) / 82.0, 0.0, 1.0)
    skin = face * np.clip((190.0 - luma) / 115.0, 0.0, 1.0)
    dark_cap = cap * np.clip((145.0 - luma) / 100.0, 0.0, 1.0)

    # The bat is brightest along the fast swing arc; restrict this warm tint to
    # pixels that are both bright and not in the cap/face regions.
    willow = bat * np.clip((luma - 145.0) / 95.0, 0.0, 1.0) * (1.0 - cap) * (1.0 - face)
    _mix_palette(ab, cloth, tuple(palette["cloth_lab_ab"]), 0.45)
    _mix_palette(ab, skin, tuple(palette["skin_lab_ab"]), 0.44)
    _mix_palette(ab, willow, tuple(palette["willow_lab_ab"]), 0.34)
    _mix_palette(ab, dark_cap, tuple(palette["cap_lab_ab"]), 0.72)
    return ab


def _warp_previous_chroma(previous_luma: np.ndarray, luma: np.ndarray, previous_ab: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        previous_luma, luma, None, 0.5, 3, 19, 3, 5, 1.2, 0,
    )
    height, width = luma.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    # Farneback gives source-to-current displacement. Sample the predecessor at
    # current minus displacement.
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    return cv2.remap(previous_ab, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def _sharpen_luma(luma: np.ndarray, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(luma, (0, 0), 1.1)
    return np.clip(luma.astype(np.float32) + amount * (luma.astype(np.float32) - blurred), 0, 255).astype(np.uint8)


def grade_frames(frames: list[np.ndarray], config: dict[str, Any]) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    previous_luma: np.ndarray | None = None
    previous_ab: np.ndarray | None = None
    static_rng = np.random.default_rng(1932)
    static_grain: np.ndarray | None = None

    for index, frame in enumerate(frames):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        luma = lab[..., 0]
        current_ab = _semantic_chroma(luma, config)
        if previous_luma is not None and previous_ab is not None:
            warped_ab = _warp_previous_chroma(previous_luma, luma, previous_ab)
            mix = float(config["temporal_chroma_mix"])
            current_ab = mix * current_ab + (1.0 - mix) * warped_ab

        sharpened_luma = _sharpen_luma(luma, float(config["edge_sharpen_amount"]))
        if static_grain is None:
            static_grain = static_rng.normal(0.0, float(config["grain_strength"]), luma.shape).astype(np.float32)
        dynamic_rng = np.random.default_rng(5000 + index)
        dynamic_grain = dynamic_rng.normal(0.0, float(config["dynamic_grain_strength"]), luma.shape)
        finished_luma = np.clip(sharpened_luma.astype(np.float32) + static_grain + dynamic_grain, 0, 255).astype(np.uint8)

        finished_lab = np.empty_like(lab)
        finished_lab[..., 0] = finished_luma
        finished_lab[..., 1:] = np.clip(current_ab, 0, 255).astype(np.uint8)
        output.append(cv2.cvtColor(finished_lab, cv2.COLOR_LAB2BGR))
        previous_luma = luma
        previous_ab = current_ab
    return output


def write_video(frames: list[np.ndarray], output: Path, fps: float, ffmpeg: str | None = None) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    executable = ffmpeg or shutil.which("ffmpeg")
    if executable:
        height, width = frames[0].shape[:2]
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-r", f"{fps:.06f}", "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            assert process.stdin is not None
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise PostprocessError("FFmpeg could not encode the output video")
        except Exception:
            process.kill()
            process.wait()
            raise
        return "ffmpeg_libx264"

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise PostprocessError("Neither FFmpeg nor OpenCV could create the output MP4")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return "opencv_mp4v_fallback"


def run(input_path: Path, output_path: Path, config_path: Path, ffmpeg: str | None, strict: bool = True) -> dict[str, Any]:
    config = load_config(config_path)
    frames, spec = read_video(input_path)
    validate_input(spec, config, strict)
    graded = grade_frames(frames, config)
    encoder = write_video(graded, output_path, spec.fps, ffmpeg)
    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "style": config["name"],
        "input_spec": spec.__dict__,
        "encoder": encoder,
        "operations": ["luminance_preserved", "semantic_palette", "flow_warped_chroma", "luma_edge_refinement", "archival_grain"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Student 3 archival finishing stage for Cross-Era Fantasy Cricket.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "1930s_archival.json")
    parser.add_argument("--ffmpeg", default=None, help="Optional absolute path to ffmpeg.exe for H.264 output.")
    parser.add_argument("--allow-nonstandard-fps", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON record of the processing run.")
    args = parser.parse_args()
    report = run(args.input, args.output, args.config, args.ffmpeg, strict=not args.allow_nonstandard_fps)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
