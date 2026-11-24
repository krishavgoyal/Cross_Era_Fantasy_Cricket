"""The match video: intro card -> bowler -> batter, with short cards before each clip.

Each generated clip is retimed (make_final.retime_and_style) and styled with an era preset. The bowler and the batter
can be in DIFFERENT eras (--bowler-era / --batter-era): a true cross-era match. With --style-cards the title cards are
filmed in the era too (grain, flicker, scanlines...). Every frame carries the "AI-generated - academic project" label.

Usage
  # final (defaults): Larwood L3 in 2x slow motion (4.0 s) + Bradman F1, both 1930s
  python scripts/make_match.py
  # cross-era: Larwood in 1930s film bowls, Bradman in 1990s broadcast drives
  python scripts/make_match.py --bowler data/gen/bowler_raw/L3_s2024_33f.mp4 --bowler-duration 4.0 ^
      --bowler-era 1930s_film --batter-era 1990s_broadcast --style-cards ^
      --bowler-card "Larwood, 1930s film" --batter-card "Bradman, 1990s broadcast" --out data/final/catalog/match_crossera.mp4
Output: data/final/final_match.mp4 (default)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from era_filters import PRESETS, apply_preset
from make_final import LABEL, check, retime_and_style
from make_sequence import build_sequence, card_frames
from pose_utils import read_video, write_video


def card_clip(text: str, subtitle: str | None, era: str | None, size, seconds: float, fps: float, path: Path) -> Path:
    """Title card video; if `era` is given the card is styled in that era (label burned after styling)."""
    frames = card_frames(text, size, seconds, fps, subtitle=subtitle, label=era is None)
    if era:
        frames, fps = apply_preset(era, frames, fps)
    return write_video(path, frames, fps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bowler", default="data/gen/bowler_raw/L3_s2024_33f.mp4", help="RAW bowler clip (no label)")
    ap.add_argument("--batter", default="data/gen/v3_raw/final_cap_s42_33f.mp4",
                    help="RAW batter clip (no label); final_cap was chosen as the batter final")
    ap.add_argument("--bowler-duration", type=float, default=4.0,
                    help="seconds the bowler's 33 frames are stretched over (2.0 = real speed, 4.0 = 2x slow motion)")
    ap.add_argument("--batter-duration", type=float, default=5.6,
                    help="seconds the batter's 33 frames are stretched over (5.6 = real speed)")
    ap.add_argument("--bowler-method", default="hold", choices=["hold", "blend", "mci"],
                    help="retime method for the bowler: hold (default; blend ghosts badly in 2x slow motion)")
    ap.add_argument("--batter-method", default="blend", choices=["hold", "blend", "mci"])
    ap.add_argument("--preset", default="1930s_film", choices=sorted(PRESETS), help="era for both players")
    ap.add_argument("--bowler-era", choices=sorted(PRESETS), help="era for the bowler (default --preset)")
    ap.add_argument("--batter-era", choices=sorted(PRESETS), help="era for the batter (default --preset)")
    ap.add_argument("--style-cards", action="store_true", help="film the title cards in the era as well")
    ap.add_argument("--intro-era", choices=[*sorted(PRESETS), "none"],
                    help="era of the intro card with --style-cards (default: bowler era; 'none' = plain)")
    ap.add_argument("--title", default="Harold Larwood vs Sir Don Bradman - Bodyline, reimagined")
    ap.add_argument("--subtitle", default=LABEL)
    ap.add_argument("--bowler-card", default="Larwood bowls (slow motion)")
    ap.add_argument("--batter-card", default="Bradman drives")
    ap.add_argument("--out", default="data/final/final_match.mp4")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--card-seconds", type=float, default=2.0, help="length of the intro card and each clip card")
    args = ap.parse_args()

    bowler_era, batter_era = args.bowler_era or args.preset, args.batter_era or args.preset
    for p in (args.bowler, args.batter):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p} - download the Kaggle outputs_raw clip first")
    out = Path(args.out)
    work = Path("data/final/work")
    work.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    _, _, bowl = retime_and_style(args.bowler, work, args.bowler_duration, args.fps, args.bowler_method, presets=(bowler_era,))
    _, _, bat = retime_and_style(args.batter, work, args.batter_duration, args.fps, args.batter_method, presets=(batter_era,))
    first, _ = read_video(bowl[bowler_era])
    size = (first[0].shape[1], first[0].shape[0])

    if not args.style_cards:          # plain black cards (as before)
        final = build_sequence([str(bowl[bowler_era]), str(bat[batter_era])], [args.bowler_card, args.batter_card],
                               args.title, None, out, size, args.fps, args.card_seconds, intro_sub=args.subtitle)
    else:                             # cards filmed in the era of what follows
        intro_era = None if args.intro_era == "none" else (args.intro_era or bowler_era)
        stem = out.stem
        parts = [card_clip(args.title, args.subtitle, intro_era, size, args.card_seconds, args.fps, work / f"{stem}_card_intro.mp4"),
                 card_clip(args.bowler_card, None, bowler_era, size, args.card_seconds, args.fps, work / f"{stem}_card_bowler.mp4"),
                 bowl[bowler_era],
                 card_clip(args.batter_card, None, batter_era, size, args.card_seconds, args.fps, work / f"{stem}_card_batter.mp4"),
                 bat[batter_era]]
        final = build_sequence([str(p) for p in parts], [None] * len(parts), None, None, out, size, args.fps,
                               args.card_seconds)
    print(check(final))
    print("->", Path(final).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
