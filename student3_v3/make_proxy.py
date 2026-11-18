"""Build a stand-in for Student 2's output from Student 1's real footage, for developing Student 3 early.

Takes ``source_512.mp4`` (frame-aligned with the DensePose video), applies Student 2's exact sampling and
crop, converts to grayscale and pads with held frames like Student 2 (3 + 16 + 7 at 7 fps). The result has
the generated clip's geometry but real pixels, so mask placement can be checked against true body edges.
It is a PROXY: never present it as a generated result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from student3_improved.postprocess import write_video

from .student2_geometry import compute_geometry, crop_like_student2, read_rgb_frames


def build_proxy(source: Path, densepose: Path, role: str, out: Path, lead: int = 3, tail: int = 7, preset: str = "ahaan-main", fps: float = 7.0) -> dict:
    dp = read_rgb_frames(densepose)
    src = read_rgb_frames(source)
    if len(src) != len(dp):
        raise ValueError(f"source has {len(src)} frames but DensePose has {len(dp)}; they must be frame-aligned")
    geom = compute_geometry(dp, role, 16, preset)
    keys = []
    for idx in geom.sampled_source_indices:
        rgb = crop_like_student2(src[idx], geom, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keys.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    order = [0] * lead + list(range(16)) + [15] * tail
    rng = np.random.default_rng(42)
    frames = [np.clip(keys[k].astype(np.float32) + rng.normal(0, 1.3, keys[k].shape), 0, 255).astype(np.uint8) for k in order]
    write_video(frames, out, fps)
    return {"proxy": str(out), "frames": len(frames), "sampled_source_indices": geom.sampled_source_indices}


def main() -> int:
    p = argparse.ArgumentParser(description="Grayscale proxy of a Student 2 output built from real footage.")
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--densepose", required=True, type=Path)
    p.add_argument("--role", choices=["batsman", "bowler"], required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(build_proxy(a.source, a.densepose, a.role, a.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
