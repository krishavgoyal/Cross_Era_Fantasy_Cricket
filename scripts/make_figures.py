"""Report figures -> docs/report/figs/ (PNG).

  ablation_strip.png   rows A / C / D1 (17-frame runs), 6 evenly spaced frames each
  final_strip.png      source clip | depth control | final_cap (1930s), same timestamps
  pipeline.png         pipeline diagram
  bowler_strip.png     source | depth (target-only) | Larwood L3 1930s (2x slow motion, sampled on source time)
  era_strip.png        the same moment raw / 1930s / 1970s / 1990s, batter and bowler (needs era_eval.py first)

Usage:  python scripts/make_figures.py            (options: --bowler-run L1_s42_33f)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pose_utils import OVERLAY_BANDS, blur_bands

OUT = Path("docs/report/figs")
TILE = (160, 277)            # w, h of one frame tile (480x832 scaled by 1/3)


def font(size: int, bold: bool = False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def frames_of(path: str | Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    if not out:
        raise FileNotFoundError(path)
    return out, fps


def fit(img_bgr: np.ndarray, size=TILE, bg=(255, 255, 255)) -> Image.Image:
    """Letterbox a BGR frame into a tile (keeps aspect ratio)."""
    w, h = size
    s = min(w / img_bgr.shape[1], h / img_bgr.shape[0])
    small = cv2.resize(img_bgr, (max(1, int(img_bgr.shape[1] * s)), max(1, int(img_bgr.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    tile = Image.new("RGB", size, bg)
    tile.paste(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), ((w - small.shape[1]) // 2, (h - small.shape[0]) // 2))
    return tile


def grid(rows: list[tuple[str, list[Image.Image]]], col_titles: list[str] | None, label_w: int = 190, gap: int = 4) -> Image.Image:
    """Rows of tiles with a label column on the left and optional column titles on top."""
    n = max(len(t) for _, t in rows)
    top = 30 if col_titles else 0
    W = label_w + n * (TILE[0] + gap)
    H = top + len(rows) * (TILE[1] + gap)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_lab, f_col = font(18, bold=True), font(15)
    if col_titles:
        for j, t in enumerate(col_titles):
            x = label_w + j * (TILE[0] + gap)
            tw = d.textlength(t, font=f_col)
            d.text((x + (TILE[0] - tw) / 2, 6), t, fill="black", font=f_col)
    for i, (label, tiles) in enumerate(rows):
        y = top + i * (TILE[1] + gap)
        lines = label.split("\n")
        for k, line in enumerate(lines):
            d.text((8, y + TILE[1] / 2 - 12 * len(lines) + 24 * k), line, fill="black", font=f_lab)
        for j, t in enumerate(tiles):
            img.paste(t, (label_w + j * (TILE[0] + gap), y))
    return img


def ablation_strip(n: int = 6) -> Path:
    runs = [("A: skeleton\n+ padded", "data/gen/v1_raw/bradman_batting_17f_smoke.mp4"),
            ("C: skeleton\n+ crop", "data/gen/v2_raw/C_s42_17f.mp4"),
            ("D1: depth\n+ crop", "data/gen/v2_raw/D1_s42_17f.mp4")]
    rows = []
    for label, path in runs:
        fr, _ = frames_of(path)
        idx = np.linspace(0, len(fr) - 1, n).astype(int)
        rows.append((label, [fit(fr[i]) for i in idx]))
    fr, fps = frames_of(runs[0][1])
    titles = [f"frame {i}" for i in np.linspace(0, len(fr) - 1, n).astype(int)]
    out = OUT / "ablation_strip.png"
    grid(rows, titles).save(out)
    return out


def timeline_strip(source: str, control: str, generated: str, duration: float, times: list[float],
                   labels: tuple[str, str, str], out_name: str, source_start: float = 0.0, gen_slow: float = 1.0) -> Path:
    """Rows: source / control / generated (styled), sampled at the same SOURCE times.

    gen_slow = playback slow-down of the generated clip (2.0 for the bowler shown in 2x slow motion)."""
    src, sfps = frames_of(source)
    bands = OVERLAY_BANDS.get(Path(source).stem, [])
    src = [blur_bands(f, bands) for f in src] if bands else src          # hide social-media overlays
    ctl, _ = frames_of(control)
    gen, gfps = frames_of(generated)
    step = duration / (len(ctl) - 1)
    pick = lambda fr, fps, t: fr[min(len(fr) - 1, int((t) * fps + 1e-6))]
    rows = [(labels[0], [fit(pick(src, sfps, source_start + t)) for t in times]),
            (labels[1], [fit(ctl[min(len(ctl) - 1, int(round(t / step)))]) for t in times]),
            (labels[2], [fit(pick(gen, gfps, t * gen_slow)) for t in times])]
    out = OUT / out_name
    grid(rows, [f"t = {t:.1f} s" for t in times]).save(out)
    return out


def box(d: ImageDraw.ImageDraw, xy, text: str, fill: str, f) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=12, fill=fill, outline="#333333", width=2)
    lines = text.split("\n")
    lh = f.size + 4
    for k, line in enumerate(lines):
        tw = d.textlength(line, font=f)
        d.text(((x0 + x1 - tw) / 2, (y0 + y1) / 2 - lh * len(lines) / 2 + k * lh), line, fill="black", font=f)
    return xy


def arrow(d: ImageDraw.ImageDraw, p0, p1, color="#333333"):
    d.line([p0, p1], fill=color, width=3)
    v = np.subtract(p1, p0).astype(float)
    v /= np.linalg.norm(v) + 1e-9
    n = np.array([-v[1], v[0]])
    tip = np.array(p1, float)
    d.polygon([tuple(tip), tuple(tip - 14 * v + 7 * n), tuple(tip - 14 * v - 7 * n)], fill=color)


def pipeline_diagram() -> Path:
    W, H = 1500, 520
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f, fs = font(19), font(15)
    blue, green, orange, grey = "#dbe9f8", "#dff2df", "#fde7cf", "#eeeeee"
    # row 1: inputs -> pose -> control -> VACE
    b1 = box(d, (20, 60, 230, 160), "Modern clip\n(cover drive /\nbowling)", grey, f)
    b2 = box(d, (300, 60, 530, 160), "DWPose (rtmlib)\nCPU, tracked\nplayer", blue, f)
    b3 = box(d, (600, 40, 860, 180), "Controls 480x832,\n16 fps, 33 frames:\npose skeleton or\nDepth Anything V2", blue, f)
    b4 = box(d, (950, 40, 1250, 180), "Wan2.1-VACE-1.3B\nComfyUI (headless)\non a free Kaggle T4", orange, f)
    b5 = box(d, (950, 250, 1250, 340), "Reference photo\n(public domain, rembg crop)", grey, f)
    arrow(d, (230, 110), (300, 110))
    arrow(d, (530, 110), (600, 110))
    arrow(d, (860, 110), (950, 110))
    arrow(d, (1100, 250), (1100, 180))
    d.line([(160, 160), (160, 220), (730, 220), (730, 180)], fill="#999999", width=2)       # depth from the clip
    d.text((240, 196), "depth map from the same frames", fill="#666666", font=fs)
    # row 2: generated clip -> retime -> era filters -> metrics / final video
    c1 = box(d, (1300, 60, 1480, 160), "Generated clip\n(~2 s @ 16 fps)", orange, f)
    arrow(d, (1250, 110), (1300, 110))
    c2 = box(d, (1300, 400, 1480, 500), "Retime to real\nspeed (24 fps)", green, f)
    arrow(d, (1390, 160), (1390, 400))
    c3 = box(d, (1000, 400, 1230, 500), "Era filters\n1930s / 1970s / 1990s\n+ AI label", green, f)
    arrow(d, (1300, 450), (1230, 450))
    c4 = box(d, (640, 400, 900, 500), "Metrics: PCK@0.2,\nDINOv2 identity,\nflicker", green, f)
    c5 = box(d, (300, 400, 560, 500), "Final videos:\nfinal, eras, compare,\nLarwood vs Bradman", green, f)
    arrow(d, (1000, 450), (900, 450))
    arrow(d, (640, 450), (560, 450))
    d.text((20, 10), "Cross-Era Fantasy Cricket: pipeline (laptop CPU = blue/green, Kaggle GPU = orange)", fill="black",
           font=font(20, bold=True))
    out = OUT / "pipeline.png"
    img.save(out)
    return out


def bowler_strip(run: str, slow: float = 2.0) -> Path:
    """source | target-only depth | Larwood 1930s (the styled clip is in `slow`x slow motion)."""
    video = Path(f"data/gen/bowler_raw/{run}.mp4")
    styled = Path(f"data/final/work/{run}_retimed_1930s_film.mp4")
    out = OUT / "bowler_strip.png"
    if video.exists() and styled.exists():
        return timeline_strip("data/clips/B1_bowler.mp4", "data/pose/bowler/depth_33.mp4", str(styled), 2.0,
                              [0.1, 0.5, 0.9, 1.2, 1.5, 1.9],
                              ("Source\n(modern)", "Depth\n(target only)", f"Larwood\n{run.split('_')[0]}, 1930s"),
                              "bowler_strip.png", gen_slow=slow)
    img = Image.new("RGB", (190 + 6 * 164, 3 * 281 + 30), "#f4f4f4")
    d = ImageDraw.Draw(img)
    d.rectangle((4, 4, img.width - 5, img.height - 5), outline="#999999", width=3)
    msg = (f"PLACEHOLDER - bowler strip: missing {video if not video.exists() else styled}.\n"
           "Run make_final.py for the chosen bowler run, then: python scripts/make_figures.py --bowler-run <run>")
    d.multiline_text((40, img.height // 2 - 40), msg, fill="#444444", font=font(22), spacing=10)
    img.save(out)
    return out


def era_strip() -> Path:
    """The same moment in raw / 1930s / 1970s / 1990s (styled at native timing by era_eval.py): batter + bowler rows."""
    clips = [("Bradman\n(batter F1)", "data/gen/v3_raw/final_cap_s42_33f.mp4", 19),
             ("Larwood\n(bowler L3)", "data/gen/bowler_raw/L3_s2024_33f.mp4", 18)]
    presets = ["1930s_film", "1970s_tv", "1990s_broadcast"]
    rows = []
    for label, raw, i in clips:
        fr, fps = frames_of(raw)
        t = i / fps                                              # the same moment in every version
        tiles = [fit(fr[i])]
        for p in presets:
            sfr, sfps = frames_of(Path("data/final/eval") / f"{Path(raw).stem}_{p}.mp4")
            tiles.append(fit(sfr[min(len(sfr) - 1, int(t * sfps + 1e-6))]))
        rows.append((label, tiles))
    out = OUT / "era_strip.png"
    grid(rows, ["raw (VACE)", "1930s film", "1970s TV", "1990s broadcast"]).save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bowler-run", default="L3_s2024_33f", help="bowler output name (without .mp4)")
    ap.add_argument("--final", default="data/final/work/final_cap_s42_33f_retimed_1930s_film.mp4",
                    help="styled batter final (real speed) for the final strip")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(ablation_strip())
    print(timeline_strip("data/clips/C2_coverdrive.mp4", "data/pose/batter/depth_33.mp4", args.final, 5.6,
                         [0.3, 1.5, 2.5, 3.3, 4.2, 5.3],
                         ("Source\n(modern)", "Depth\ncontrol", "Bradman\nfinal_cap, 1930s"), "final_strip.png"))
    print(pipeline_diagram())
    print(bowler_strip(args.bowler_run))
    print(era_strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
