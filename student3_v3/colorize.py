"""Role-aware semantic colourisation for generated cricket clips (batsman or bowler).

Luminance is never changed: only LAB a/b chroma is written. Colour is computed once per generated
keyframe and then re-used for every held copy of that keyframe, so holds cannot flicker. Temporal
chroma blending between keyframes re-uses Student 3 v2's backward-flow + forward/backward confidence.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from student3_improved.postprocess import _scene_cut, _soft_mask, _warp_previous_chroma


def _blend(ab: np.ndarray, weight: np.ndarray, target: list[float], strength: float) -> None:
    alpha = np.clip(weight * strength, 0.0, 1.0)[..., None]
    ab[:] = ab * (1.0 - alpha) + np.asarray(target, np.float32).reshape(1, 1, 2) * alpha


def colorize_keyframe(frame_bgr: np.ndarray, masks: dict[str, np.ndarray], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (ab float32 HxWx2, luma uint8 HxW) for one keyframe."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    luma = lab[..., 0]
    h, w = luma.shape
    pal = cfg["palette"]
    sat = float(cfg.get("saturation", 1.0))
    feather = float(cfg.get("mask_feather_sigma", 1.2))
    ab = np.full((h, w, 2), pal.get("neutral", [128, 128]), np.float32)

    player = _soft_mask(masks.get("player", np.zeros((h, w), np.uint8)), feather)
    bg_cfg = cfg.get("background", {})
    background = 1.0 - player
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    horizon = float(bg_cfg.get("horizon", 0.40))
    ramp = max(1e-3, float(bg_cfg.get("ramp", 0.20)))
    grass = np.broadcast_to(np.clip((yy - horizon) / ramp + 0.5, 0, 1), (h, w)) * background
    sky = np.broadcast_to(np.clip((horizon - yy) / ramp + 0.5, 0, 1), (h, w)) * background
    lf = luma.astype(np.float32)
    grass = grass * np.clip((float(bg_cfg.get("grass_max_luma", 200)) - lf) / 60.0, 0.2, 1.0)
    if "sky" in pal:
        _blend(ab, sky, pal["sky"], float(bg_cfg.get("sky_strength", 0.3)) * sat)
    if "grass" in pal:
        _blend(ab, grass, pal["grass"], float(bg_cfg.get("grass_strength", 0.75)) * sat)

    base = cfg.get("base_player_class")
    if base and base in pal:
        _blend(ab, player, pal[base], float(cfg.get("base_player_strength", 0.5)) * sat)
    strengths = cfg.get("class_strength", {})
    for label in cfg["class_order"]:
        if label not in masks or label not in pal:
            continue
        weight = _soft_mask(masks[label], feather) * player if label != "bat" else _soft_mask(masks[label], feather)
        target = pal[label]
        bright = cfg.get("palette_if_bright", {}).get(label)
        if bright and masks[label].any() and float(lf[masks[label] > 0].mean()) >= float(bright["min_luma"]):
            target = bright["ab"]  # e.g. Student 2 drew a white sun hat: keep it cream, don't tint it mint
        _blend(ab, weight, target, float(strengths.get(label, 0.8)) * sat)

    # Very dark or very bright pixels cannot hold much chroma; taper so they don't turn muddy or neon.
    taper = np.clip(np.minimum(lf, 255.0 - lf) / float(cfg.get("chroma_taper_luma", 40.0)), 0.35, 1.0)[..., None]
    ab = 128.0 + (ab - 128.0) * taper
    ab = np.stack([cv2.bilateralFilter(ab[..., c].astype(np.float32), 5, 10.0, 3.0) for c in range(2)], axis=2)
    return ab.astype(np.float32), luma.copy()


def temporal_blend(keys_bgr: list[np.ndarray], key_ab: list[np.ndarray], key_luma: list[np.ndarray], player_masks: list[np.ndarray], cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    """Blend each keyframe's chroma with the flow-warped previous result where flow is trustworthy."""
    out: list[np.ndarray] = []
    log: list[dict[str, float]] = []
    strength = float(cfg.get("temporal_strength", 0.35))
    min_conf = float(cfg.get("temporal_min_player_confidence", 0.35))
    prev_ab = prev_luma = None
    for i, (ab, luma, player) in enumerate(zip(key_ab, key_luma, player_masks)):
        entry = {"keyframe": i, "blended": 0.0, "player_flow_confidence": None}
        if prev_ab is not None and not _scene_cut(prev_luma, luma, float(cfg.get("scene_cut_threshold", 0.55))):
            warped, conf = _warp_previous_chroma(prev_luma, luma, prev_ab, float(cfg.get("flow_consistency_threshold", 1.5)))
            sel = player > 0
            pconf = float(conf[sel].mean()) if sel.any() else float(conf.mean())
            entry["player_flow_confidence"] = round(pconf, 4)
            if pconf >= min_conf:  # release guard: fast arm whips break flow, fall back to this frame's masks
                alpha = (strength * conf)[..., None]
                ab = ab * (1.0 - alpha) + warped * alpha
                entry["blended"] = 1.0
        out.append(ab)
        log.append(entry)
        prev_ab, prev_luma = ab, luma
    return out, log


def compose(frame_bgr: np.ndarray, ab: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lab[..., 1:] = np.clip(ab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
