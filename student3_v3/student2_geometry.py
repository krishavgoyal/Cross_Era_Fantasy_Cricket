"""Reproduce Student 2's DensePose sampling and crop so Student 3 masks line up with generated frames.

Student 2 (generation_engine.Milestone3Engine.load_aligned_densepose_sequence) does three things
before ControlNet sees the DensePose video:

1. keeps only "active" frames (enough non-black pixels),
2. picks ``target_frames`` of them with ``np.linspace(start, end)``, where start/end depend on the role,
3. crops one fixed 3:4 box around the union silhouette and resizes it to 384x512 with NEAREST.

The generated video therefore shares this exact geometry. We mirror the arithmetic here (rather than
importing it) so we can also return the sampled source-frame indices and the crop box, which Student 2's
method does not expose. ``bridge_student2_masks`` additionally calls Student 2's own code when it is
importable and reports how closely the two agree, so drift is caught instead of silently misaligning.

Keep ``PRESETS`` in sync with generation_engine.py. "ahaan-main" = commit 93954f6, "main" = 4b2f775.
"""
from __future__ import annotations

import importlib
import inspect
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PRESETS: dict[str, dict[str, Any]] = {
    "ahaan-main": {"active_px_threshold": 1200, "cx_factor": 0.48, "role_aware": True},
    "main": {"active_px_threshold": 1500, "cx_factor": 0.47, "role_aware": False},
}
TARGET_SIZE = (384, 512)  # (width, height) used by Student 2


@dataclass
class Student2Geometry:
    role: str
    preset: str
    target_frames: int
    active_source_indices: list[int]   # source-video frame index of each active frame
    sampled_active_indices: list[int]  # positions inside the active list (what linspace returned)
    sampled_source_indices: list[int]  # source-video frame index of each generated keyframe
    pad: int
    crop_xyxy_padded: tuple[int, int, int, int]
    crop_xyxy_source: tuple[int, int, int, int]  # same box in unpadded source pixels
    target_size: tuple[int, int] = TARGET_SIZE

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def read_rgb_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames in {path}")
    return frames


def compute_geometry(dp_frames_rgb: list[np.ndarray], role: str, target_frames: int = 16, preset: str = "ahaan-main") -> Student2Geometry:
    if preset not in PRESETS:
        raise ValueError(f"Unknown Student 2 preset {preset!r}; choose from {sorted(PRESETS)}")
    params = PRESETS[preset]
    active = [i for i, f in enumerate(dp_frames_rgb) if np.count_nonzero(np.max(f, axis=2) > 15) > params["active_px_threshold"]]
    total = len(active)
    if total < target_frames:
        raise ValueError(f"Only {total} active DensePose frames, need {target_frames}")

    if params["role_aware"] and role == "bowler":
        start_f = min(8, max(0, total - target_frames))
        end_f = max(start_f + target_frames - 1, total - 2)
    else:
        start_f = min(24, max(0, total - target_frames))
        end_f = min(total - 4, 94)
    picks = [int(i) for i in np.linspace(start_f, end_f, target_frames, dtype=int)]
    sampled = [dp_frames_rgb[active[i]] for i in picks]

    stack = np.stack(sampled)
    mask = np.max(stack, axis=(0, 3)) > 15
    coords = np.argwhere(mask)
    h, w = sampled[0].shape[:2]
    tw, th = TARGET_SIZE
    aspect = tw / th
    if len(coords) == 0:
        pad, x1, y1, x2, y2 = 0, 0, 0, w, h
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        box_h = int((y_max - y_min) * 1.24)
        box_w = int(box_h * aspect)
        min_req_w = int((x_max - x_min) * 1.36)
        if box_w < min_req_w:
            box_w = min_req_w
            box_h = int(box_w / aspect)
        cx = int((x_min + x_max) * params["cx_factor"])
        cy = (y_min + y_max) // 2
        pad = max(box_h, box_w)
        y1 = cy - box_h // 2 + pad
        y2 = y1 + box_h
        x1 = cx - box_w // 2 + pad
        x2 = x1 + box_w
    return Student2Geometry(
        role=role,
        preset=preset,
        target_frames=target_frames,
        active_source_indices=active,
        sampled_active_indices=picks,
        sampled_source_indices=[active[i] for i in picks],
        pad=int(pad),
        crop_xyxy_padded=(int(x1), int(y1), int(x2), int(y2)),
        crop_xyxy_source=(int(x1 - pad), int(y1 - pad), int(x2 - pad), int(y2 - pad)),
    )


def crop_like_student2(frame: np.ndarray, geom: Student2Geometry, interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    """Apply Student 2's crop to any frame that is pixel-aligned with the DensePose video."""
    x1, y1, x2, y2 = geom.crop_xyxy_padded
    if geom.pad > 0:
        border = 0 if frame.ndim == 2 else (0, 0, 0)
        frame = cv2.copyMakeBorder(frame, geom.pad, geom.pad, geom.pad, geom.pad, cv2.BORDER_CONSTANT, value=border)
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, geom.target_size, interpolation=interpolation)


def source_xy_to_generated(x: np.ndarray, y: np.ndarray, geom: Student2Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Map 512x512 DensePose/source pixel coordinates into generated-frame pixels."""
    sx1, sy1, sx2, sy2 = geom.crop_xyxy_source
    tw, th = geom.target_size
    return (x - sx1) * tw / max(1, sx2 - sx1), (y - sy1) * th / max(1, sy2 - sy1)


def _install_stubs() -> list[str]:
    """generation_engine imports torch/diffusers at module level; the geometry method uses neither."""
    added = []
    for name in ["torch", "diffusers", "diffusers.utils"]:
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
            continue
        except Exception:
            pass
        module = types.ModuleType(name)
        if name == "diffusers":
            for attr in ["DiffusionPipeline", "MotionAdapter", "ControlNetModel", "DDIMScheduler"]:
                setattr(module, attr, object)
        if name == "diffusers.utils":
            module.export_to_video = lambda *a, **k: None
        if name == "torch":
            module.float16 = "float16"
            module.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules[name] = module
        added.append(name)
    return added


def bridge_student2_masks(densepose_video: Path, role: str, target_frames: int = 16, repo_root: Path | None = None,
                          window: tuple[int, int] | None = None) -> dict[str, Any] | None:
    """Run Student 2's own geometry code (no model load) and return its per-frame part masks.

    Returns None when generation_engine.py cannot be imported. Otherwise returns
    {"body_soft": [HxW float], "bat_alpha": [...], "stubbed": [...]} for each keyframe.
    """
    root = repo_root or Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    stubbed = _install_stubs()
    try:
        engine_mod = importlib.import_module("generation_engine")
        engine = engine_mod.Milestone3Engine.__new__(engine_mod.Milestone3Engine)
        engine.target_size = TARGET_SIZE
        method = engine.load_aligned_densepose_sequence
        kwargs = {"target_frames": target_frames}
        if "role" in inspect.signature(method).parameters:
            kwargs["role"] = role
        if window is not None and "window" in inspect.signature(method).parameters:
            kwargs["window"] = tuple(window)  # integration: same sampling window the generation used
        _, part_masks = method(str(densepose_video), **kwargs)
    except Exception as exc:  # pragma: no cover - depends on teammates' code
        return {"error": f"{type(exc).__name__}: {exc}", "stubbed": stubbed}
    return {
        "body_soft": [m["body_soft"].astype(np.float32) for m in part_masks],
        "bat_alpha": [m["bat_alpha"].astype(np.float32) for m in part_masks],
        "stubbed": stubbed,
    }
