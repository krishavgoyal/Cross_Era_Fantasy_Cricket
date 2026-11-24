"""Stage 6: join the bowler clip and the batter clip (plus optional title cards) into one final.mp4 with ffmpeg.

Every clip is scaled to one size (letterboxed, aspect ratio kept), one frame rate and H.264 yuv420p, then
concatenated in the order given. Title cards are rendered with OpenCV (no ffmpeg font setup needed) and carry the
"AI-generated - academic project" label like every other output.

Usage
  python scripts/make_sequence.py --clips data/styled/bowler_1930s_film.mp4 data/styled/batter_1930s_film.mp4 ^
      --intro "Michael Holding vs Sachin Tendulkar" --out data/final/final.mp4
  # one card before each clip as well:
  python scripts/make_sequence.py --clips bowler.mp4 batter.mp4 --cards "Holding bowls" "Tendulkar drives" ^
      --intro "Michael Holding vs Sachin Tendulkar" --outro "AI-generated for an academic project"

Needs ffmpeg on PATH (winget install Gyan.FFmpeg, then open a NEW terminal).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from era_filters import burn_label
from pose_utils import find_ffmpeg, write_video


def probe_size(path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def wrap_text(text: str, max_chars: int) -> list[str]:
    """Greedy word wrap into lines of at most max_chars characters."""
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + ([cur] if cur else [])


def card_frames(text: str, size: tuple[int, int], seconds: float, fps: float, subtitle: str | None = None,
                label: bool = True) -> list[np.ndarray]:
    """Frames of a black title card (centred wrapped text, optional grey subtitle, optional AI label)."""
    return _card(text, size, seconds, fps, subtitle, label)


def make_card(text: str, size: tuple[int, int], seconds: float, fps: float, path: Path, subtitle: str | None = None) -> Path:
    """Write a title card video (with the AI label)."""
    return write_video(path, _card(text, size, seconds, fps, subtitle, True), fps)


def _card(text: str, size: tuple[int, int], seconds: float, fps: float, subtitle: str | None, label: bool) -> list[np.ndarray]:
    """Black title card with centred, wrapped white text, an optional smaller grey subtitle line and the AI label."""
    w, h = size
    scale = max(0.6, w / 640)
    thick = max(1, int(round(scale * 1.8)))
    (cw, _), _ = cv2.getTextSize("M", cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    lines = wrap_text(text, max(8, int(0.85 * w / cw)))
    line_h = int(scale * 42)
    frame = np.zeros((h, w, 3), np.uint8)
    y0 = h // 2 - (len(lines) - 1) * line_h // 2
    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(frame, line, ((w - tw) // 2, y0 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
    if subtitle:
        sub_scale = scale * 0.6
        for j, line in enumerate(wrap_text(subtitle, max(8, int(0.85 * w / (cw * 0.6))))):
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
            cv2.putText(frame, line, ((w - tw) // 2, y0 + len(lines) * line_h + int(scale * 20) + j * int(scale * 28)),
                        cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (190, 190, 190), 1, cv2.LINE_AA)
    frames = [frame.copy() for _ in range(max(1, int(round(seconds * fps))))]
    return burn_label(frames) if label else frames


def build_sequence(clips: list[str], cards: list[str | None], intro: str | None, outro: str | None, out: Path,
                   size: tuple[int, int], fps: float, card_seconds: float, crf: int = 18,
                   intro_sub: str | None = None) -> Path:
    """Concatenate [intro card] + ([card] + clip)* + [outro card] into out."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH. Install it (winget install Gyan.FFmpeg) and open a NEW terminal.")
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    with tempfile.TemporaryDirectory(prefix="cefc_seq_") as tmp:
        tmp = Path(tmp)
        parts: list[str] = []
        if intro:
            parts.append(str(make_card(intro, size, card_seconds, fps, tmp / "intro.mp4", subtitle=intro_sub)))
        for i, clip in enumerate(clips):
            if cards[i]:
                parts.append(str(make_card(cards[i], size, card_seconds, fps, tmp / f"card_{i}.mp4")))
            parts.append(clip)
        if outro:
            parts.append(str(make_card(outro, size, card_seconds, fps, tmp / "outro.mp4")))

        cmd = [ffmpeg, "-nostdin", "-y", "-loglevel", "error"]
        for p in parts:
            cmd += ["-i", p]
        norm = ";".join(                                     # same size, fps, pixel format for every part
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={fps:g},format=yuv420p[v{i}]" for i in range(len(parts)))
        joined = "".join(f"[v{i}]" for i in range(len(parts)))
        cmd += ["-filter_complex", f"{norm};{joined}concat=n={len(parts)}:v=1:a=0[v]", "-map", "[v]",
                "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True, help="clips in order, e.g. bowler then batter")
    ap.add_argument("--cards", nargs="*", default=[], help="optional card text shown before each clip (same order)")
    ap.add_argument("--intro", help="opening title card, e.g. 'Michael Holding vs Sachin Tendulkar'")
    ap.add_argument("--intro-sub", help="smaller second line on the intro card, e.g. 'AI-generated - academic project'")
    ap.add_argument("--outro", help="closing card")
    ap.add_argument("--out", default="data/final/final.mp4")
    ap.add_argument("--size", help="WxH of the final video (default: size of the first clip)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--card-seconds", type=float, default=2.0)
    args = ap.parse_args()

    if len(args.cards) > len(args.clips):
        raise SystemExit("More --cards than --clips")
    cards = args.cards + [None] * (len(args.clips) - len(args.cards))
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else probe_size(args.clips[0])
    out = build_sequence(args.clips, cards, args.intro, args.outro, Path(args.out), size, args.fps, args.card_seconds,
                         intro_sub=args.intro_sub)
    cap = cv2.VideoCapture(str(out))
    n, f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Wrote {out}: {size[0]}x{size[1]}, {n} frames @ {f:g} fps = {n / f:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
