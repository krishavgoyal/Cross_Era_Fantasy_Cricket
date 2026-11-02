from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .postprocess import PostprocessError, read_video
except ImportError:
    from postprocess import PostprocessError, read_video


def _local_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Gaussian-window SSIM on one grayscale frame (not a global-statistics proxy)."""
    x, y = a.astype(np.float32), b.astype(np.float32)
    c1, c2 = 6.5025, 58.5225
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8)
    return float(np.mean(score))


def _metrics(frames: list[np.ndarray]) -> dict[str, float]:
    labs = [cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) for frame in frames]
    chroma = [float(np.mean(np.linalg.norm(lab[..., 1:].astype(np.float32) - 128.0, axis=2))) for lab in labs]
    temporal_chroma = [float(np.mean(np.abs(labs[i][..., 1:].astype(np.float32) - labs[i - 1][..., 1:].astype(np.float32)))) for i in range(1, len(labs))]
    temporal_luma = [float(np.mean(np.abs(labs[i][..., 0].astype(np.float32) - labs[i - 1][..., 0].astype(np.float32)))) for i in range(1, len(labs))]
    return {
        "mean_chroma_distance_from_neutral": round(float(np.mean(chroma)), 4),
        "mean_adjacent_chroma_change": round(float(np.mean(temporal_chroma)) if temporal_chroma else 0.0, 4),
        "max_adjacent_chroma_change": round(float(np.max(temporal_chroma)) if temporal_chroma else 0.0, 4),
        "mean_adjacent_luma_change": round(float(np.mean(temporal_luma)) if temporal_luma else 0.0, 4),
    }


def evaluate(input_path: Path, output_path: Path) -> dict[str, Any]:
    before, before_spec = read_video(input_path)
    after, after_spec = read_video(output_path)
    same_geometry = (before_spec.width, before_spec.height) == (after_spec.width, after_spec.height)
    same_frames = before_spec.frame_count == after_spec.frame_count
    same_fps = abs(before_spec.fps - after_spec.fps) < 0.05
    if not same_geometry or not same_frames:
        raise PostprocessError("Output frame geometry/count differs from input; aligned evaluation is invalid.")

    before_luma = [cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0] for frame in before]
    after_luma = [cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0] for frame in after]
    abs_errors = [np.abs(a.astype(np.float32) - b.astype(np.float32)) for a, b in zip(before_luma, after_luma)]
    mae = float(np.mean([e.mean() for e in abs_errors]))
    mse = float(np.mean([np.mean(e * e) for e in abs_errors]))
    psnr = float("inf") if mse == 0 else 10.0 * float(np.log10(255.0 ** 2 / mse))
    ssim = float(np.mean([_local_ssim(a, b) for a, b in zip(before_luma, after_luma)]))
    input_metrics, output_metrics = _metrics(before), _metrics(after)

    checks = {
        "resolution_preserved": same_geometry,
        "frame_count_preserved": same_frames,
        "fps_preserved": same_fps,
        "luminance_mae_within_3": mae <= 3.0,
        "luminance_ssim_at_least_0_98": ssim >= 0.98,
    }
    all_technical_checks = all(checks.values())
    return {
        "input_spec": before_spec.__dict__,
        "output_spec": after_spec.__dict__,
        "checks": checks,
        "luminance": {
            "mean_absolute_error": round(mae, 4),
            "psnr_db": round(psnr, 3) if np.isfinite(psnr) else "infinity",
            "mean_local_ssim": round(ssim, 5),
        },
        "input_metrics": input_metrics,
        "output_metrics": output_metrics,
        "technical_verdict": "pass" if all_technical_checks else "review",
        "visual_quality_verdict": "requires_human_review",
        "motion_fidelity": "not_measured: run Student 1 DWPose and compare aligned keypoints for PCK",
        "important_note": "Technical checks do not prove semantic color accuracy, mask quality, historical authenticity, or absence of flicker. Inspect the rendered video and SAM2 masks.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate improved Student 3 output against its upstream animation.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.input, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
