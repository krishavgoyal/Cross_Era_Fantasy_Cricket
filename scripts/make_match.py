"""The match video: intro card -> bowler (1930s) -> batter (1930s), with short cards before each clip.

Both generated clips are retimed to their real duration (bowler: 2.0 s source sampled at real speed, batter: 5.6 s
source squeezed into 33 frames) and styled 1930s_film, exactly as make_final.py does. Every frame carries the
"AI-generated - academic project" label; the intro card has it as a second line too.

Usage
  python scripts/make_match.py
  python scripts/make_match.py --bowler data/gen/bowler_raw/L2_s7_33f.mp4 --batter data/gen/v3_raw/final_cap_s42_33f.mp4
Output: data/final/final_match.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from make_final import LABEL, check, retime_and_style
from make_sequence import build_sequence
from pose_utils import read_video


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bowler", default="data/gen/bowler_raw/L1_s42_33f.mp4", help="RAW bowler clip (no label)")
    ap.add_argument("--batter", default="data/gen/v3_raw/final_nocap_s42_33f.mp4", help="RAW batter clip (no label)")
    ap.add_argument("--bowler-duration", type=float, default=2.0, help="seconds of source the bowler clip covers")
    ap.add_argument("--batter-duration", type=float, default=5.6, help="seconds of source the batter clip covers")
    ap.add_argument("--title", default="Harold Larwood vs Sir Don Bradman - Bodyline, reimagined")
    ap.add_argument("--subtitle", default=LABEL)
    ap.add_argument("--bowler-card", default="Larwood bowls")
    ap.add_argument("--batter-card", default="Bradman drives")
    ap.add_argument("--preset", default="1930s_film")
    ap.add_argument("--out", default="data/final/final_match.mp4")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--card-seconds", type=float, default=2.0, help="length of the intro card and each clip card")
    args = ap.parse_args()

    for p in (args.bowler, args.batter):
        if not Path(p).exists():
            raise SystemExit(f"Missing {p} - download the Kaggle outputs_raw clip first")
    out = Path(args.out)
    work = out.parent / "work"
    work.mkdir(parents=True, exist_ok=True)

    _, _, bowl = retime_and_style(args.bowler, work, args.bowler_duration, args.fps, presets=(args.preset,))
    _, _, bat = retime_and_style(args.batter, work, args.batter_duration, args.fps, presets=(args.preset,))
    first, _ = read_video(bowl[args.preset])
    size = (first[0].shape[1], first[0].shape[0])
    final = build_sequence([str(bowl[args.preset]), str(bat[args.preset])], [args.bowler_card, args.batter_card],
                           args.title, None, out, size, args.fps, args.card_seconds, intro_sub=args.subtitle)
    print(check(final))
    print("->", Path(final).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
