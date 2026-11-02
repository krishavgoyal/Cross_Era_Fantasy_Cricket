from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clip_processor import process_clip
from pose_engine import extract_pose
from render_control import export_numpy_matrix, render_control_video
from temporal_smoothing import smooth_pose_file

EXIT_OK = 0
EXIT_ARGS = 2
EXIT_DEPENDENCY = 3
EXIT_INPUT = 4
EXIT_MODEL = 5
EXIT_OUTPUT = 6
EXIT_UNEXPECTED = 70


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-Era Fantasy Cricket motion preprocessing pipeline.")
    parser.add_argument("--reference-video", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--action-type", required=True)
    parser.add_argument("--fps", type=float, default=24.0, choices=(24.0, 30.0))
    parser.add_argument("--detector-model", default="yolo26n-pose.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--track-id", type=int, default=None)
    parser.add_argument("--crop-margin", type=float, default=0.40)
    parser.add_argument("--pose-expand-margin", type=float, default=0.15)
    parser.add_argument("--bbox-smoothing", type=float, default=0.35)
    parser.add_argument("--max-missed-frames", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--dwpose-device", default="cpu")
    parser.add_argument("--min-cutoff", type=float, default=1.4)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--derivative-cutoff", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--interpolation-gap", type=int, default=2)
    parser.add_argument("--control-style", choices=("dwpose", "openpose"), default="dwpose")
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--skip-smoothing", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    clip_dir = args.output_root / args.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    source_path = process_clip(
        args.reference_video,
        clip_dir,
        args.clip_id,
        args.action_type,
        args.fps,
        args.crop_margin,
        args.bbox_smoothing,
        args.confidence,
        args.detector_model,
        args.tracker,
        args.device,
        args.track_id,
        args.pose_expand_margin,
        args.max_missed_frames,
    )
    pose_path = clip_dir / "pose_data.json"
    extract_pose(source_path, pose_path, args.dwpose_device)
    if not args.skip_smoothing:
        smooth_pose_file(
            pose_path,
            args.min_cutoff,
            args.beta,
            args.derivative_cutoff,
            confidence_threshold=args.confidence_threshold,
            interpolation_gap=args.interpolation_gap,
        )
    if not args.skip_render:
        render_control_video(pose_path, clip_dir / "control_pose.mp4", args.control_style, args.score_threshold)
        export_numpy_matrix(pose_path, clip_dir / "pose_vectors.npz")
    return clip_dir


def classify_exit_code(exc: BaseException) -> int:
    message = str(exc).lower()
    if isinstance(exc, ModuleNotFoundError) or "not installed" in message:
        return EXIT_DEPENDENCY
    if "does not exist" in message or "unable to open video" in message or "invalid video" in message:
        return EXIT_INPUT
    if "model" in message or "cuda" in message or "dwpose" in message or "ultralytics" in message:
        return EXIT_MODEL
    if "write" in message or "ffmpeg" in message or "output" in message:
        return EXIT_OUTPUT
    return EXIT_UNEXPECTED


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv)
        output_dir = run(args)
        print(f"Pipeline complete: {output_dir}")
        return EXIT_OK
    except SystemExit as exc:
        return int(exc.code)
    except Exception as exc:
        code = classify_exit_code(exc)
        print(f"ERROR [{code}]: {exc}", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
