"""Stage 4: era styling presets (OpenCV + NumPy, CPU). Turns a generated clip into period-looking footage.

Presets
  1930s_film        grayscale, S-curve contrast, slight lens blur, vignette, gate jitter, flicker, dust/scratches,
                    grain, retimed to 18 fps with uneven holds (hand-cranked feel)
  1970s_tv          desaturation + cool/warm cast, chromatic aberration, bloom, softness, scanlines, tape-noise bands,
                    retimed to 25 fps
  1990s_broadcast   mild colour, chroma bleed, slight softness, faint bloom and noise, retimed to 25 fps

1930s_film and 1970s_tv are ported from the earlier Student 3 code (student3_v3/era_filters.py: newsreel_1930s,
broadcast_1970s, retime). 1990s_broadcast is new. All randomness is seeded, so the same input + seed always gives
the same output. Every output gets a small "AI-generated - academic project" label (ASCII hyphen: OpenCV's
fonts cannot draw an en dash). The label is added LAST so it stays crisp. Feed this script the UNLABELLED
videos (outputs_raw/), otherwise the label appears twice.

Usage
  python scripts/era_filters.py --input data/gen/outputs_raw/batter_33f.mp4 --preset 1930s_film
  python scripts/era_filters.py --input data/gen/outputs_raw --preset all --out-dir data/styled
  python scripts/era_filters.py --input clip.mp4 --preset 1930s_film --sepia --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pose_utils import read_video, write_video

LABEL_TEXT = "AI-generated - academic project"


# ------------------------------------------------------------------ shared helpers
def retime(frames: list[np.ndarray], src_fps: float, dst_fps: float, jitter: float = 0.0, seed: int = 0) -> list[np.ndarray]:
    """Change frame rate by HOLDING frames (no invented in-betweens). Duration is preserved.

    jitter > 0 makes the holds uneven, like a hand-cranked camera (ported from student3_v3).
    """
    if dst_fps <= 0 or src_fps <= 0:
        return frames
    duration = len(frames) / src_fps
    n_out = max(1, int(round(duration * dst_fps)))
    rng = np.random.default_rng(seed)
    times = (np.arange(n_out) + 0.5) / dst_fps
    if jitter > 0:
        times = times + rng.uniform(-jitter, jitter, size=n_out) / dst_fps
        times = np.maximum.accumulate(np.clip(times, 0, duration - 1e-6))   # time never runs backwards
    idx = np.clip((times * src_fps).astype(int), 0, len(frames) - 1)
    return [frames[i] for i in idx]


def _vignette(h: int, w: int, strength: float) -> np.ndarray:
    """Radial darkening mask in [0, 1]; 1 in the centre, lower toward the corners."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    return np.clip(1.0 - strength * np.clip(r - 0.55, 0, None) ** 1.6, 0.0, 1.0)


def _gate_weave(n: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency drift (dx, dy per frame) like film wandering in the projector gate."""
    raw = rng.normal(0, 1, size=(n + 6, 2))
    kernel = np.ones(5) / 5
    smooth = np.stack([np.convolve(raw[:, k], kernel, mode="same") for k in range(2)], axis=1)[3:n + 3]
    return smooth * amplitude


def burn_label(frames: list[np.ndarray], text: str = LABEL_TEXT) -> list[np.ndarray]:
    """Burn a small label (bottom-left, dark outline) onto every frame.

    The outline is 8 offset copies of the SAME thin text. (A thick outline stroke does not work: OpenCV spaces
    glyphs by thickness, so a thick outline drifts out of line with the white text.)
    """
    out = []
    for f in frames:
        f = f.copy()
        h, w = f.shape[:2]
        scale = max(0.4, w / 1100)
        x, y = 8, h - 10
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    cv2.putText(f, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(f, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(f)
    return out


# ------------------------------------------------------------------ presets
def film_1930s(frames: list[np.ndarray], p: dict, rng: np.random.Generator) -> list[np.ndarray]:
    """Black-and-white newsreel: S-curve, blur, vignette, weave, flicker, dust, scratches, grain."""
    h, w = frames[0].shape[:2]
    vig = _vignette(h, w, p["vignette"])
    weave = _gate_weave(len(frames), p["gate_weave_px"], rng)
    tint = np.asarray(p["tint_bgr"], np.float32)               # (1,1,1) = pure grayscale
    out, scratches = [], []
    for i, f in enumerate(frames):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g = np.clip(0.5 + (g - 0.5) * p["contrast"], 0, 1) ** p["gamma"]            # S-ish curve, crushed blacks
        g = cv2.GaussianBlur(g, (0, 0), p["lens_blur_sigma"])
        g *= vig * (1.0 + rng.normal(0, p["flicker"]))                              # exposure flicker
        g = g * p["max_white"] + p["black_lift"]                                    # faded print
        m = np.float32([[1, 0, weave[i, 0]], [0, 1, weave[i, 1]]])                  # gate weave / jitter
        g = cv2.warpAffine(g, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        g += rng.normal(0, p["grain"], g.shape).astype(np.float32)                  # film grain
        for _ in range(rng.poisson(p["dust_per_frame"])):                           # dust specks (black or white)
            cx, cy, rad = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(1, 3))
            cv2.circle(g, (cx, cy), rad, float(rng.choice([0.05, 0.95])), -1, lineType=cv2.LINE_AA)
        if rng.random() < p["scratch_start_prob"]:                                  # vertical scratches last a few frames
            scratches.append((int(rng.integers(int(w * 0.1), int(w * 0.9))), int(rng.integers(2, 6))))
        alive = []
        for x, life in scratches:
            cv2.line(g, (x, 0), (x + int(rng.integers(-1, 2)), h), 0.9, 1, lineType=cv2.LINE_AA)
            if life > 1:
                alive.append((x, life - 1))
        scratches = alive
        g = np.clip(g, 0, 1)
        out.append(np.clip(g[..., None] * tint[None, None, :] * 255.0, 0, 255).astype(np.uint8))
    return out


def tv_1970s(frames: list[np.ndarray], p: dict, rng: np.random.Generator) -> list[np.ndarray]:
    """Early analog colour TV: washed colour, colour fringing, bloom, soft picture, scanlines, tape-noise bands."""
    h, w = frames[0].shape[:2]
    shift = int(p["chromatic_shift_px"])
    out = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= p["saturation"]                                              # desaturate
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        img *= np.asarray(p["cast_bgr"], np.float32)                                # colour cast
        b, g, r = cv2.split(img)
        img = cv2.merge([np.roll(b, -shift, axis=1), g, np.roll(r, shift, axis=1)])  # chromatic aberration
        bloom = cv2.GaussianBlur(np.clip(img - 190, 0, None), (0, 0), 6)
        img = img + p["bloom"] * bloom
        img = cv2.GaussianBlur(img, (0, 0), p["softness_sigma"])
        img[::2] *= 1.0 - p["scanline_depth"]                                       # scanlines
        if rng.random() < p["tape_noise_prob"]:                                     # horizontal tape-noise band
            y = int(rng.integers(0, h - 6)); band = int(rng.integers(2, 6))
            img[y:y + band] += rng.normal(0, 40, (band, w, 1)).astype(np.float32)
            img[y:y + band] = np.roll(img[y:y + band], int(rng.integers(-6, 7)), axis=1)
        img += rng.normal(0, p["noise"], img.shape).astype(np.float32)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def broadcast_1990s(frames: list[np.ndarray], p: dict, rng: np.random.Generator) -> list[np.ndarray]:
    """Early-90s TV broadcast (new preset): mild colour, chroma bleed, slight softness, faint bloom and noise."""
    h, w = frames[0].shape[:2]
    vig = _vignette(h, w, p["vignette"])[..., None]
    out = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= p["saturation"]                                              # slightly tamer colour
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        ycc[..., 1:] = cv2.GaussianBlur(ycc[..., 1:], (0, 0), sigmaX=p["chroma_bleed_px"], sigmaY=0.5)   # analog chroma bleed
        img = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR).astype(np.float32)
        img = cv2.GaussianBlur(img, (0, 0), p["softness_sigma"])                    # slight softness
        img = img + p["bloom"] * cv2.GaussianBlur(np.clip(img - 200, 0, None), (0, 0), 5)
        img = img * (1.0 - p["vignette_mix"] + p["vignette_mix"] * vig)             # very mild vignette
        img += rng.normal(0, p["noise"], img.shape).astype(np.float32)
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


# name -> (function, default seed, output fps, jitter for retiming, parameters)
PRESETS: dict[str, dict] = {
    "1930s_film": dict(fn=film_1930s, seed=1932, fps=18, retime_jitter=0.35, params=dict(
        tint_bgr=(1.0, 1.0, 1.0), contrast=1.12, gamma=1.15, lens_blur_sigma=0.8, vignette=0.9, flicker=0.035,
        gate_weave_px=2.0, grain=0.035, dust_per_frame=2.0, scratch_start_prob=0.15, max_white=0.94, black_lift=0.04)),
    "1970s_tv": dict(fn=tv_1970s, seed=1975, fps=25, retime_jitter=0.0, params=dict(
        saturation=0.62, cast_bgr=(0.94, 1.0, 1.06), chromatic_shift_px=2, bloom=0.5, softness_sigma=0.6,
        scanline_depth=0.08, tape_noise_prob=0.25, noise=3.0)),
    "1990s_broadcast": dict(fn=broadcast_1990s, seed=1996, fps=25, retime_jitter=0.0, params=dict(
        saturation=0.92, chroma_bleed_px=1.6, softness_sigma=0.7, bloom=0.25, vignette=0.5, vignette_mix=0.5, noise=1.5)),
}
SEPIA_BGR = (0.93, 0.98, 1.04)          # warm tint used by --sepia (same values as the old newsreel preset)


def apply_preset(name: str, frames: list[np.ndarray], fps: float, seed: int | None = None, label: bool = True,
                 sepia: bool = False) -> tuple[list[np.ndarray], float]:
    """Style frames with a preset. Returns (styled frames, output fps)."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)} or 'all'")
    cfg = PRESETS[name]
    params = dict(cfg["params"])
    if sepia and name == "1930s_film":
        params["tint_bgr"] = SEPIA_BGR
    seed = cfg["seed"] if seed is None else seed
    # Retime FIRST so grain, dust and flicker change on every output frame, as real film/video does.
    timed = retime(frames, fps, cfg["fps"], cfg["retime_jitter"], seed=seed)
    styled = cfg["fn"](timed, params, np.random.default_rng(seed))
    return (burn_label(styled) if label else styled), float(cfg["fps"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="video file, or a folder of .mp4 files (use the UNLABELLED outputs_raw/)")
    ap.add_argument("--preset", required=True, choices=[*PRESETS, "all"])
    ap.add_argument("--out-dir", default="data/styled")
    ap.add_argument("--seed", type=int, default=None, help="override the preset's default seed")
    ap.add_argument("--sepia", action="store_true", help="1930s_film only: warm sepia tint instead of pure grayscale")
    ap.add_argument("--no-label", action="store_true", help="skip the AI-generated label (not for submission)")
    args = ap.parse_args()

    src = Path(args.input)
    videos = sorted(src.glob("*.mp4")) if src.is_dir() else [src]
    if not videos:
        raise SystemExit(f"No .mp4 files in {src}")
    names = list(PRESETS) if args.preset == "all" else [args.preset]
    out_dir = Path(args.out_dir)
    for v in videos:
        frames, fps = read_video(v)
        for name in names:
            styled, out_fps = apply_preset(name, frames, fps, args.seed, label=not args.no_label, sepia=args.sepia)
            out = write_video(out_dir / f"{v.stem}_{name}.mp4", styled, out_fps)
            print(f"{v.name} [{len(frames)} frames @ {fps:g} fps] -> {name}: {out} ({len(styled)} frames @ {out_fps:g} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
