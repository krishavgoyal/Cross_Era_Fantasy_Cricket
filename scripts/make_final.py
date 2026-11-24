"""ONE command: generated 33-frame clip -> real-speed retime -> era styling -> final videos.

Outputs (in --out-dir, default data/final; <tag> lets several inputs coexist, e.g. _cap / _nocap):
  final<tag>.mp4        intro card ("Sir Don Bradman - a cover drive, reimagined" + small AI line) + 1930s clip
  final_eras<tag>.mp4   the same clip in 1930s / 1970s / 1990s styles back to back, each after a short title card
  compare<tag>.mp4      real source clip | control (depth or pose) | generated 1930s result, on the same timeline
  work/                 intermediates (retimed clip, styled clips)

Timing: the 33 frames cover --source-duration seconds (5.6) of --source-clip starting at --source-start (0), i.e.
frame i shows source time start + i*duration/32. retime.py stretches the clip back to that real duration (24 fps).

Usage
  python scripts/make_final.py --input data/gen/v1_raw/bradman_batting_33f.mp4 --control data/pose/batter/control_33.mp4
  python scripts/make_final.py --input data/gen/v3_raw/final_cap_s42_33f.mp4 --tag _cap
  python scripts/make_final.py --input data/gen/v3_raw/final_nocap_s42_33f.mp4 --tag _nocap
  # bowler: the 2.0 s clip was sampled at real speed (33 frames @ 16 fps), so --source-duration 2.0 = no stretch
  python scripts/make_final.py --input data/gen/bowler_raw/L1_s42_33f.mp4 --tag _bowler --source-duration 2.0 ^
      --source-clip data/clips/B1_bowler.mp4 --control data/pose/bowler/depth_33.mp4 --title "Harold Larwood - bowling, reimagined"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from era_filters import apply_preset, burn_label
from make_sequence import build_sequence
from pose_utils import read_video, write_video
from retime import DEFAULT_METHOD, retime

ERAS = [("1930s_film", "1930s newsreel film"), ("1970s_tv", "1970s colour television"),
        ("1990s_broadcast", "1990s broadcast")]
LABEL = "AI-generated - academic project"


def frame_at(frames: list[np.ndarray], fps: float, t: float) -> np.ndarray:
    """Frame shown at time t (seconds) of a constant-fps video."""
    return frames[min(len(frames) - 1, max(0, int(t * fps + 1e-6)))]


def header(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((46, img.shape[1], 3), 25, np.uint8)
    scale = 0.6 if img.shape[1] >= 420 else 0.5
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(bar, text, (max(4, (img.shape[1] - tw) // 2), 30), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def make_compare(source_clip: str, source_start: float, duration: float, control: str, generated: Path,
                 out: Path, fps: float = 24.0, height: int = 832) -> Path:
    """Side by side on one timeline: real source | control | generated result."""
    src, sfps = read_video(source_clip)
    ctl, _ = read_video(control)
    gen, gfps = read_video(generated)
    ctl_step = duration / (len(ctl) - 1)                    # control frame i = source time start + i*step
    kind = "Depth control" if "depth" in Path(control).name.lower() else "Pose control"
    frames = []
    for i in range(int(round(duration * fps)) + 1):
        t = i / fps
        a = frame_at(src, sfps, source_start + t)
        a = cv2.resize(a, (int(round(a.shape[1] * height / a.shape[0])), height))
        b = ctl[min(len(ctl) - 1, int(round(t / ctl_step)))]
        b = cv2.resize(b, (int(round(b.shape[1] * height / b.shape[0])), height))
        c = frame_at(gen, gfps, t)
        c = cv2.resize(c, (int(round(c.shape[1] * height / c.shape[0])), height))
        frames.append(np.hstack([header(a, "Real source (modern)"), header(b, kind),
                                 header(c, "Generated (AI), 1930s style")]))
    return write_video(out, burn_label(frames), fps)


def retime_and_style(src: str | Path, work: Path, source_duration: float, fps: float = 24.0, method: str = DEFAULT_METHOD,
                     presets=tuple(p for p, _ in ERAS)) -> tuple[list[np.ndarray], float, dict]:
    """Retime src to its real duration, then apply the era presets. Returns (retimed frames, fps, {preset: path})."""
    stem = Path(src).stem
    retimed = retime(src, work / f"{stem}_retimed.mp4", source_duration, fps, method)
    frames, rfps = read_video(retimed)
    print(f"{stem}: retimed ({method}) {len(frames)} frames @ {rfps:g} = {len(frames) / rfps:.2f} s")
    styled = {}
    for preset in presets:
        st, sfps = apply_preset(preset, frames, rfps)
        styled[preset] = write_video(work / f"{stem}_retimed_{preset}.mp4", st, sfps)
        print(f"  {preset}: {len(st)} frames @ {sfps:g} = {len(st) / sfps:.2f} s")
    return frames, rfps, styled


def check(path: Path, clip_part: tuple[float, float] | None = None) -> str:
    """Duration, near-black frames (inside clip_part seconds if given) and whether the label pixels are present."""
    frames, fps = read_video(path)
    lo, hi = clip_part if clip_part else (0.0, len(frames) / fps)
    seg = frames[int(lo * fps):int(hi * fps)]
    black = sum(1 for f in seg if f.mean() < 8 and f.std() < 4)
    h, w = frames[-1].shape[:2]
    label_px = int((frames[-1][h - 24:h - 4, 6:min(w, 260)].min(axis=2) > 200).sum())
    return (f"{path.name}: {len(frames)} frames @ {fps:g} fps = {len(frames) / fps:.2f} s | black frames "
            f"{black}/{len(seg)} | label pixels {label_px} ({'OK' if label_px > 30 else 'MISSING'})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="RAW generated 33-frame clip (no label)")
    ap.add_argument("--tag", default="", help="suffix for the outputs, e.g. _cap")
    ap.add_argument("--out-dir", default="data/final")
    ap.add_argument("--source-duration", type=float, default=5.6,
                    help="seconds of source the 33 frames cover (make_control --duration): 5.6 batter, 2.0 bowler")
    ap.add_argument("--source-clip", default="data/clips/C2_coverdrive.mp4")
    ap.add_argument("--source-start", type=float, default=0.0)
    ap.add_argument("--control", default="data/pose/batter/depth_33.mp4", help="control video that generated --input")
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=["hold", "blend", "mci"])
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--title", "--intro", dest="intro", default="Sir Don Bradman - a cover drive, reimagined",
                    help="intro card title")
    ap.add_argument("--intro-sub", default=LABEL)
    ap.add_argument("--card-seconds", type=float, default=2.5)
    ap.add_argument("--era-card-seconds", type=float, default=1.5)
    args = ap.parse_args()

    out, work = Path(args.out_dir), Path(args.out_dir) / "work"
    work.mkdir(parents=True, exist_ok=True)
    # 1) real speed + 2) era styles (each keeps the duration and burns the label)
    frames, fps, styled = retime_and_style(args.input, work, args.source_duration, args.fps, args.method)
    size = (frames[0].shape[1], frames[0].shape[0])

    # 3) final: intro card + 1930s clip
    final = build_sequence([str(styled["1930s_film"])], [None], args.intro, None, out / f"final{args.tag}.mp4", size,
                           args.fps, args.card_seconds, intro_sub=args.intro_sub)
    # 4) all three eras, each with a short card
    eras = build_sequence([str(styled[p]) for p, _ in ERAS], [t for _, t in ERAS], None, None,
                          out / f"final_eras{args.tag}.mp4", size, 25.0, args.era_card_seconds)
    # 5) side-by-side comparison on the source timeline
    compare = make_compare(args.source_clip, args.source_start, args.source_duration, args.control,
                           styled["1930s_film"], out / f"compare{args.tag}.mp4", args.fps)

    print("\nChecks:")
    print(" ", check(final, (args.card_seconds, 99.0)))      # black-frame count over the clip part (after the card)
    print(" ", check(eras))
    print(" ", check(compare))
    for p in (final, eras, compare):
        print("  ->", Path(p).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
