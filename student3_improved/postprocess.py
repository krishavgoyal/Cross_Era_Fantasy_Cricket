from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .segmentation import SegmentationError, run_grabcut_fallback, run_sam2_video
except ImportError:
    from segmentation import SegmentationError, run_grabcut_fallback, run_sam2_video


class PostprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoSpec:
    width: int
    height: int
    fps: float
    frame_count: int


def read_video(path: Path) -> tuple[list[np.ndarray], VideoSpec]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise PostprocessError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames or fps <= 0 or width <= 0 or height <= 0:
        raise PostprocessError(f"Video has no usable frames: {path}")
    return frames, VideoSpec(width, height, fps, len(frames))


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"expected_width", "expected_height", "expected_fps", "palette"}
    missing = required - config.keys()
    if missing:
        raise PostprocessError(f"Configuration is missing: {sorted(missing)}")
    for label, box in config.get("sam2_prompts", {}).items():
        if len(box.get("box", [])) != 4:
            raise PostprocessError(f"SAM2 prompt '{label}' must have a normalized box.")
        x1, y1, x2, y2 = box["box"]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise PostprocessError(f"Invalid SAM2 prompt box for '{label}': {box['box']}")
    return config


def validate_input(spec: VideoSpec, config: dict[str, Any], strict: bool = True) -> None:
    expected = (int(config["expected_width"]), int(config["expected_height"]))
    if (spec.width, spec.height) != expected:
        raise PostprocessError(f"Expected {expected[0]}x{expected[1]}, received {spec.width}x{spec.height}.")
    if strict and abs(spec.fps - float(config["expected_fps"])) > 0.05:
        raise PostprocessError(f"Expected {config['expected_fps']} FPS, received {spec.fps:.3f} FPS.")


def _soft_mask(mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    value = mask.astype(np.float32)
    if sigma > 0:
        value = cv2.GaussianBlur(value, (0, 0), sigma)
    return np.clip(value, 0.0, 1.0)


def _apply_palette(ab: np.ndarray, mask: np.ndarray, target: list[int] | tuple[int, int], strength: float) -> None:
    alpha = np.clip(mask.astype(np.float32) * float(strength), 0.0, 1.0)[..., None]
    color = np.asarray(target, dtype=np.float32).reshape(1, 1, 2)
    ab[:] = ab * (1.0 - alpha) + color * alpha


def _scene_cut(previous: np.ndarray, current: np.ndarray, threshold: float) -> bool:
    a = cv2.calcHist([previous], [0], None, [32], [0, 256])
    b = cv2.calcHist([current], [0], None, [32], [0, 256])
    cv2.normalize(a, a)
    cv2.normalize(b, b)
    return cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA) > threshold


def _warp_previous_chroma(
    previous_luma: np.ndarray,
    current_luma: np.ndarray,
    previous_ab: np.ndarray,
    consistency_threshold: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-warp previous chroma and return a forward/backward confidence map."""
    backward = cv2.calcOpticalFlowFarneback(current_luma, previous_luma, None, 0.5, 3, 19, 3, 5, 1.2, 0)
    forward = cv2.calcOpticalFlowFarneback(previous_luma, current_luma, None, 0.5, 3, 19, 3, 5, 1.2, 0)
    h, w = current_luma.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = gx + backward[..., 0]
    map_y = gy + backward[..., 1]
    warped = cv2.remap(previous_ab, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128))
    sampled_forward = cv2.remap(forward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    error = np.sqrt(np.sum((backward + sampled_forward) ** 2, axis=2))
    valid = (map_x >= 0) & (map_x < w - 1) & (map_y >= 0) & (map_y < h - 1)
    confidence = np.exp(-0.5 * (error / max(consistency_threshold, 1e-3)) ** 2) * valid.astype(np.float32)
    return warped.astype(np.float32), confidence.astype(np.float32)


def _colorize_frame(frame: np.ndarray, masks: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    luma = lab[..., 0]
    h, w = luma.shape
    palette = config["palette"]
    ab = np.full((h, w, 2), palette.get("neutral_lab_ab", [128, 128]), dtype=np.float32)

    player = _soft_mask(masks.get("player", np.zeros((h, w), np.uint8)), float(config.get("mask_feather_sigma", 1.4)))
    # Environment is segmented as the complement of the player; vertical position
    # only separates distant sky from the lower field after foreground removal.
    background = np.clip(1.0 - player, 0.0, 1.0)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    grass_region = np.broadcast_to(np.clip((yy - 0.38) / 0.35, 0.0, 1.0), (h, w)) * background
    sky_region = np.broadcast_to(np.clip((0.48 - yy) / 0.30, 0.0, 1.0), (h, w)) * background
    # Reduce green tint in very bright pixels, which are more likely clothing/boards.
    grass_region *= np.clip((190.0 - luma.astype(np.float32)) / 100.0, 0.25, 1.0)
    _apply_palette(ab, sky_region, palette["sky_lab_ab"], float(config.get("sky_strength", 0.25)))
    _apply_palette(ab, grass_region, palette["grass_lab_ab"], float(config.get("grass_strength", 0.72)))

    # Base player tint is restrained cream. More specific SAM2 masks override it.
    _apply_palette(ab, player, palette["cloth_lab_ab"], float(config.get("player_strength", 0.34)))
    # Specific masks are expected to be separately prompted on frame zero.
    priorities = ["trousers", "shirt", "pads", "skin", "helmet", "bat", "shoes"]
    for label in priorities:
        if label not in masks or label not in palette:
            continue
        mask = _soft_mask(masks[label], float(config.get("mask_feather_sigma", 1.4)))
        _apply_palette(ab, mask, palette[label + "_lab_ab"], float(config.get("class_strength", {}).get(label, 0.78)))

    # OpenCV bilateralFilter accepts 1- or 3-channel input, not 2-channel Lab chroma.
    # Filter the two chroma planes independently so luminance remains untouched.
    ab = np.stack([
        cv2.bilateralFilter(ab[..., channel].astype(np.float32), 5, 12.0, 3.0)
        for channel in range(2)
    ], axis=2)
    finished = lab.copy()
    finished[..., 1:] = np.clip(ab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(finished, cv2.COLOR_LAB2BGR), luma.copy()


def grade_frames(
    frames: list[np.ndarray],
    config: dict[str, Any],
    masks_by_frame: list[dict[str, np.ndarray]] | None = None,
) -> list[np.ndarray]:
    if masks_by_frame is None:
        masks_by_frame = [{"player": np.zeros(frame.shape[:2], np.uint8)} for frame in frames]
    if len(masks_by_frame) != len(frames):
        raise PostprocessError("Segmentation mask count does not match video frame count.")

    output: list[np.ndarray] = []
    previous_luma: np.ndarray | None = None
    previous_ab: np.ndarray | None = None
    for index, (frame, masks) in enumerate(zip(frames, masks_by_frame)):
        colorized, source_luma = _colorize_frame(frame, masks, config)
        lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
        current_ab = lab[..., 1:].astype(np.float32)
        if previous_luma is not None and previous_ab is not None:
            cut = _scene_cut(previous_luma, source_luma, float(config.get("scene_cut_threshold", 0.55)))
            if not cut:
                warped_ab, confidence = _warp_previous_chroma(
                    previous_luma, source_luma, previous_ab,
                    float(config.get("flow_consistency_threshold", 1.5)),
                )
                temporal_strength = float(config.get("temporal_strength", 0.38))
                alpha = (temporal_strength * confidence)[..., None]
                current_ab = current_ab * (1.0 - alpha) + warped_ab * alpha
        # The default archival preset deliberately avoids sharpening and grain until
        # color/mask quality is approved. Both remain optional and independently tunable.
        luma = source_luma.astype(np.float32)
        sharpen = float(config.get("edge_sharpen_amount", 0.0))
        if sharpen > 0:
            blur = cv2.GaussianBlur(luma, (0, 0), 1.0)
            luma = np.clip(luma + sharpen * (luma - blur), 0, 255)
        grain = float(config.get("grain_strength", 0.0))
        if grain > 0:
            rng = np.random.default_rng(int(config.get("grain_seed", 1932)) + index)
            luma = np.clip(luma + rng.normal(0.0, grain, luma.shape), 0, 255)
        finished_lab = np.empty_like(lab)
        finished_lab[..., 0] = np.clip(luma, 0, 255).astype(np.uint8)
        finished_lab[..., 1:] = np.clip(current_ab, 0, 255).astype(np.uint8)
        output.append(cv2.cvtColor(finished_lab, cv2.COLOR_LAB2BGR))
        previous_luma = source_luma
        previous_ab = current_ab
    return output


def write_video(frames: list[np.ndarray], output: Path, fps: float, ffmpeg: str | None = None) -> str:
    if not frames:
        raise PostprocessError("No frames to write.")
    output.parent.mkdir(parents=True, exist_ok=True)
    executable = ffmpeg or shutil.which("ffmpeg")
    h, w = frames[0].shape[:2]
    if executable:
        command = [executable, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.06f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            assert process.stdin is not None
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise PostprocessError("FFmpeg failed to encode the output.")
        except Exception:
            process.kill()
            process.wait()
            raise
        return "ffmpeg_libx264"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise PostprocessError("FFmpeg and OpenCV could not create the output MP4.")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return "opencv_mp4v_fallback"


def run(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    ffmpeg: str | None = None,
    strict: bool = True,
    segmentation: str = "auto",
    sam2_config: str | None = None,
    sam2_checkpoint: str | None = None,
    device: str | None = None,
    mask_dir: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    frames, spec = read_video(input_path)
    validate_input(spec, config, strict)

    backend = segmentation
    if backend == "auto":
        backend = "sam2" if sam2_config and sam2_checkpoint else "grabcut"
    if backend == "sam2":
        if not sam2_config or not sam2_checkpoint:
            raise PostprocessError("SAM2 selected but --sam2-config and --sam2-checkpoint were not supplied.")
        masks = run_sam2_video(frames, config, sam2_config, sam2_checkpoint, device)
    elif backend == "grabcut":
        masks = run_grabcut_fallback(frames, config)
    elif backend == "none":
        masks = [{"player": np.zeros(frame.shape[:2], np.uint8)} for frame in frames]
    else:
        raise PostprocessError(f"Unknown segmentation backend: {backend}")

    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame_masks in enumerate(masks):
            for label, mask in frame_masks.items():
                label_dir = mask_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(label_dir / f"frame_{frame_index:04d}.png"), (mask.astype(np.uint8) * 255))

    graded = grade_frames(frames, config, masks)
    encoder = write_video(graded, output_path, spec.fps, ffmpeg)
    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "style": config["name"],
        "input_spec": asdict(spec),
        "output_spec": {"width": spec.width, "height": spec.height, "fps": spec.fps, "frame_count": len(graded)},
        "segmentation_backend": backend,
        "sam2_prompts": list(config.get("sam2_prompts", {}).keys()) if backend == "sam2" else [],
        "encoder": encoder,
        "operations": ["semantic_masks", "class_palette_colorization", "backward_flow_warp", "forward_backward_occlusion_confidence", "scene_cut_reset", "luminance_preservation"],
        "limitations": ["GrabCut fallback is not a substitute for SAM2."] if backend == "grabcut" else ["SAM2 mask quality depends on first-frame prompt boxes; inspect masks before accepting output."] if backend == "sam2" else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Improved Student 3 archival colorization.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "1930s_archival.json")
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--allow-nonstandard-fps", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--segmentation", choices=["auto", "sam2", "grabcut", "none"], default="auto")
    parser.add_argument("--sam2-config", default=None, help="SAM2 YAML config path, e.g. configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-checkpoint", default=None, help="Path to downloaded SAM2 checkpoint .pt")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda or cpu")
    parser.add_argument("--mask-dir", type=Path, default=None, help="Optional directory for per-frame binary segmentation masks")
    args = parser.parse_args()
    report = run(args.input, args.output, args.config, args.ffmpeg, not args.allow_nonstandard_fps, args.segmentation, args.sam2_config, args.sam2_checkpoint, args.device, args.mask_dir)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
