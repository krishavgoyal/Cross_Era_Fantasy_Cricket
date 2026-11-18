"""Turn Student 1's DensePose control video into per-class masks for Student 3 colouring.

Student 1 renders DensePose fine segmentation (part ids 1..24) with detectron2's convention:
colour = PARULA(part_id * 255 / 24) on black. We invert that by nearest-colour lookup, clean the
result spatially, then map parts to clothing/skin classes with a role-specific table from the config.

DensePose part ids (detectron2):
 1,2 torso | 3 right hand | 4 left hand | 5 left foot | 6 right foot
 7,9 right upper leg | 8,10 left upper leg | 11,13 right lower leg | 12,14 left lower leg
 15,17 left upper arm | 16,18 right upper arm | 19,21 left lower arm | 20,22 right lower arm | 23,24 head
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

N_PARTS = 24
_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_PARULA).reshape(256, 3)
PART_COLORS_RGB = np.array([_LUT[int(p * 255 / N_PARTS)][::-1] for p in range(1, N_PARTS + 1)], dtype=np.float32)

# Exact colours Student 2 paints onto its control frames (generation_engine.py). Not DensePose parts.
STUDENT2_OVERLAYS_RGB = {
    "shin_patch": (45, 160, 145),
    "bat_line": (175, 190, 65),
    "hand_dot": (35, 90, 230),
}


def decode_parts(frame_rgb: np.ndarray, black_threshold: int = 24, max_color_distance: float = 40.0) -> np.ndarray:
    """Return an HxW uint8 map: 0 = background, 1..24 = DensePose part id."""
    h, w = frame_rgb.shape[:2]
    flat = frame_rgb.reshape(-1, 3).astype(np.float32)
    foreground = flat.max(axis=1) > black_threshold
    parts = np.zeros(h * w, dtype=np.uint8)
    if foreground.any():
        fg = flat[foreground]
        dist = ((fg[:, None, :] - PART_COLORS_RGB[None, :, :]) ** 2).sum(axis=2)
        best = dist.argmin(axis=1)
        ok = np.sqrt(dist[np.arange(len(best)), best]) <= max_color_distance
        values = np.where(ok, best + 1, 0).astype(np.uint8)
        parts[np.flatnonzero(foreground)] = values
    parts = parts.reshape(h, w)
    return majority_filter(parts, size=3)


def majority_filter(parts: np.ndarray, size: int = 3) -> np.ndarray:
    """Replace each labelled pixel by the most common non-zero label in its window (removes codec speckle)."""
    labels = np.unique(parts[parts > 0])
    if labels.size <= 1:
        return parts
    votes = np.stack([cv2.boxFilter((parts == lab).astype(np.float32), -1, (size, size), normalize=False) for lab in labels])
    winner = labels[np.argmax(votes, axis=0)]
    return np.where(parts > 0, winner, 0).astype(np.uint8)


def classes_from_parts(
    parts: np.ndarray,
    class_parts: dict[str, list[int]],
    head_split: dict[str, Any] | None = None,
    face_visible: bool = False,
) -> dict[str, np.ndarray]:
    """Map a part-id map to binary class masks. ``head_split`` divides parts 23/24 into cap/hair and face."""
    masks: dict[str, np.ndarray] = {}
    for label, ids in class_parts.items():
        masks[label] = np.isin(parts, ids).astype(np.uint8)
    head = np.isin(parts, [23, 24])
    if head_split and head.any():
        ys, _ = np.nonzero(head)
        top, bottom = ys.min(), ys.max()
        cut = top + float(head_split.get("top_fraction", 0.45)) * max(1, bottom - top)
        yy = np.arange(parts.shape[0])[:, None]
        upper = head & (yy <= cut)
        lower = head & (yy > cut)
        top_label = head_split.get("top_class", "cap")
        face_label = head_split.get("face_class", "skin")
        back_label = head_split.get("back_of_head_class", top_label)
        masks[top_label] = np.maximum(masks.get(top_label, np.zeros_like(parts)), upper.astype(np.uint8))
        lower_label = face_label if face_visible else back_label
        masks[lower_label] = np.maximum(masks.get(lower_label, np.zeros_like(parts)), lower.astype(np.uint8))
    masks["player"] = (parts > 0).astype(np.uint8)
    return masks


def add_student2_overlays(masks: dict[str, np.ndarray], control_rgb: np.ndarray, overlay_classes: dict[str, str]) -> None:
    """Student 2 bridges missing shins with a flat colour; give those pixels the configured class."""
    for overlay, label in overlay_classes.items():
        colour = STUDENT2_OVERLAYS_RGB.get(overlay)
        if colour is None:
            continue
        hit = np.all(control_rgb == np.array(colour, dtype=control_rgb.dtype), axis=2)
        if hit.any():
            masks[label] = np.maximum(masks.get(label, np.zeros(hit.shape, np.uint8)), hit.astype(np.uint8))
            masks["player"] = np.maximum(masks["player"], hit.astype(np.uint8))
