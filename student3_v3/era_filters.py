"""Era styling and frame-rate scaling (the proposal's "Era-Styling Filter Module").

Presets (all parameters live in the JSON config under "era_presets"):
- newsreel_1930s : grayscale/sepia map, contrast curve, lens blur + vignette, gate weave,
                   exposure flicker, dust and scratches, grain, retimed to ~18 fps with uneven holds
- broadcast_1970s: desaturation, warm cast, chromatic aberration, scanlines, tape-noise bands,
                   bloom, retimed to 25 fps
- remaster       : the colourised master with light grain, retimed to 24 fps by frame holds

Every random choice uses a seeded generator so the same input always gives the same output.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def retime(frames: list[np.ndarray], src_fps: float, dst_fps: float, jitter: float = 0.0, seed: int = 0) -> list[np.ndarray]:
    """Change frame rate by holding frames (no invented in-betweens). ``jitter`` adds uneven, hand-cranked holds."""
    if dst_fps <= 0 or src_fps <= 0:
        return frames
    duration = len(frames) / src_fps
    n_out = max(1, int(round(duration * dst_fps)))
    rng = np.random.default_rng(seed)
    times = (np.arange(n_out) + 0.5) / dst_fps
    if jitter > 0:
        times = times + rng.uniform(-jitter, jitter, size=n_out) / dst_fps
        times = np.maximum.accumulate(np.clip(times, 0, duration - 1e-6))
    idx = np.clip((times * src_fps).astype(int), 0, len(frames) - 1)
    return [frames[i] for i in idx]


def _vignette(h: int, w: int, strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    return np.clip(1.0 - strength * np.clip(r - 0.55, 0, None) ** 1.6, 0.0, 1.0)


def _gate_weave(n: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency sub-pixel drift, like film wandering in the projector gate."""
    raw = rng.normal(0, 1, size=(n + 6, 2))
    kernel = np.ones(5) / 5
    smooth = np.stack([np.convolve(raw[:, k], kernel, mode="same") for k in range(2)], axis=1)[3:n + 3]
    return smooth * amplitude


def newsreel_1930s(frames: list[np.ndarray], p: dict[str, Any], seed: int = 1932) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    vig = _vignette(h, w, float(p.get("vignette", 0.9)))
    weave = _gate_weave(len(frames), float(p.get("gate_weave_px", 1.2)), rng)
    tone = np.asarray(p.get("sepia_bgr", [0.93, 0.98, 1.04]), np.float32) if p.get("sepia", True) else np.ones(3, np.float32)
    gamma = float(p.get("contrast_gamma", 1.15))
    out = []
    scratches: list[tuple[int, int]] = []
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g = np.clip(0.5 + (g - 0.5) * float(p.get("contrast", 1.12)), 0, 1) ** gamma  # S-ish curve, crushed blacks
        g = cv2.GaussianBlur(g, (0, 0), float(p.get("lens_blur_sigma", 0.7)))
        g *= vig * (1.0 + rng.normal(0, float(p.get("flicker", 0.035))))
        g = g * float(p.get("max_white", 0.94)) + float(p.get("black_lift", 0.04))
        m = np.float32([[1, 0, weave[i, 0]], [0, 1, weave[i, 1]]])
        g = cv2.warpAffine(g, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        g += rng.normal(0, float(p.get("grain", 0.035)), g.shape).astype(np.float32)
        for _ in range(rng.poisson(float(p.get("dust_per_frame", 2.0)))):
            cx, cy, rad = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(1, 3))
            cv2.circle(g, (cx, cy), rad, float(rng.choice([0.05, 0.95])), -1, lineType=cv2.LINE_AA)
        if rng.random() < float(p.get("scratch_start_prob", 0.15)):
            scratches.append((int(rng.integers(int(w * 0.1), int(w * 0.9))), int(rng.integers(2, 6))))
        alive = []
        for x, life in scratches:
            cv2.line(g, (x, 0), (x + int(rng.integers(-1, 2)), h), 0.9, 1, lineType=cv2.LINE_AA)
            if life > 1:
                alive.append((x, life - 1))
        scratches = alive
        g = np.clip(g, 0, 1)
        out.append(np.clip(g[..., None] * tone[None, None, :] * 255.0, 0, 255).astype(np.uint8))
    return out


def broadcast_1970s(frames: list[np.ndarray], p: dict[str, Any], seed: int = 1975) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    shift = int(p.get("chromatic_shift_px", 2))
    out = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= float(p.get("saturation", 0.62))
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        img *= np.asarray(p.get("cast_bgr", [0.94, 1.0, 1.06]), np.float32)
        b, g, r = cv2.split(img)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        img = cv2.merge([b, g, r])
        bloom = cv2.GaussianBlur(np.clip(img - 190, 0, None), (0, 0), 6)
        img = img + float(p.get("bloom", 0.5)) * bloom
        img = cv2.GaussianBlur(img, (0, 0), float(p.get("softness_sigma", 0.6)))
        img[::2] *= 1.0 - float(p.get("scanline_depth", 0.08))
        if rng.random() < float(p.get("tape_noise_prob", 0.25)):
            y = int(rng.integers(0, h - 6))
            band = int(rng.integers(2, 6))
            img[y:y + band] += rng.normal(0, 40, (band, w, 1)).astype(np.float32)
            img[y:y + band] = np.roll(img[y:y + band], int(rng.integers(-6, 7)), axis=1)
        img += rng.normal(0, float(p.get("noise", 3.0)), img.shape).astype(np.float32)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def remaster(frames: list[np.ndarray], p: dict[str, Any], seed: int = 2026) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    grain = float(p.get("grain", 1.5))
    out = []
    for f in frames:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] += rng.normal(0, grain, lab.shape[:2])
        out.append(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR))
    return out


PRESET_FUNCS = {"newsreel_1930s": newsreel_1930s, "broadcast_1970s": broadcast_1970s, "remaster": remaster}


def apply_preset(name: str, frames: list[np.ndarray], fps: float, presets: dict[str, dict[str, Any]]) -> tuple[list[np.ndarray], float]:
    if name not in PRESET_FUNCS:
        raise ValueError(f"Unknown era preset {name!r}; choose from {sorted(PRESET_FUNCS)}")
    params = presets.get(name, {})
    out_fps = float(params.get("fps", fps))
    # Retime first so grain, dust and flicker change on every output frame, as real film does.
    timed = retime(frames, fps, out_fps, float(params.get("retime_jitter", 0.0)), seed=len(name))
    return PRESET_FUNCS[name](timed, params), out_fps
