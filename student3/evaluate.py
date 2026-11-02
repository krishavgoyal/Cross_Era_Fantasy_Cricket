from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:  # Package import for tests and installed use.
    from .postprocess import PostprocessError, read_video
except ImportError:  # Direct `python student3/evaluate.py` invocation.
    from postprocess import PostprocessError, read_video


def _global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = 6.5025, 58.5225
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    return float(((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / ((mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)))


def _frame_metrics(frames: list[np.ndarray]) -> dict[str, float]:
    labs = [cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) for frame in frames]
    chroma_distance = [np.mean(np.abs(lab[..., 1:].astype(np.float32) - 128.0)) for lab in labs]
    luma_diffs = [
        float(np.mean(np.abs(labs[index][..., 0].astype(np.float32) - labs[index - 1][..., 0].astype(np.float32))))
        for index in range(1, len(labs))
    ]
    return {
        "mean_chroma_distance_from_neutral": round(float(np.mean(chroma_distance)), 4),
        "adjacent_luma_difference_mean": round(float(np.mean(luma_diffs)) if luma_diffs else 0.0, 4),
        "adjacent_luma_difference_max": round(float(np.max(luma_diffs)) if luma_diffs else 0.0, 4),
    }


def evaluate(input_path: Path, output_path: Path) -> dict[str, Any]:
    before, before_spec = read_video(input_path)
    after, after_spec = read_video(output_path)
    if before_spec.frame_count != after_spec.frame_count or (before_spec.width, before_spec.height) != (after_spec.width, after_spec.height):
        raise PostprocessError("Output no longer matches the source frame geometry; motion comparison is invalid.")
    input_luma = [cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0] for frame in before]
    output_luma = [cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0] for frame in after]
    return {
        "input_spec": before_spec.__dict__,
        "output_spec": after_spec.__dict__,
        "input_metrics": _frame_metrics(before),
        "output_metrics": _frame_metrics(after),
        "mean_luma_ssim_input_to_output": round(float(np.mean([_global_ssim(a, b) for a, b in zip(input_luma, output_luma)])), 5),
        "verdict": "pass" if abs(before_spec.fps - after_spec.fps) < 0.05 else "review_fps",
        "note": "This validates post-processing preservation. Run Student 1's DWPose backend on the final clip and compare against aligned source pose data for full PCK motion-fidelity evaluation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Student 3 output against its upstream animation.")
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
