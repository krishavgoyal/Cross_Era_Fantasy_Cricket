"""DRAFT presentation video (2:30, 1920x1080, 25 fps, no audio) -> data/final/presentation_draft.mp4

Follows docs/VIDEO_SCRIPT.md (the 4 proposal milestones). Real files only. Portrait clips sit side by side on a dark
background with short captions. Every frame carries the "AI-generated - academic project" label, and the source
clip's social-media overlays are blurred. Frames are streamed straight into ffmpeg (nothing large is kept in RAM).

Usage:  python scripts/make_presentation.py            (record the voice-over from docs/VIDEO_NARRATION.md on top)
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_control import canvas_transform, clean_resampled
from pose_utils import OVERLAY_BANDS, blur_bands, find_ffmpeg

W, H, FPS = 1920, 1080, 25
BG = (20, 20, 20)                       # BGR dark background
LABEL = "AI-generated - academic project"
GITHUB = "github.com/krishavgoyal/Cross_Era_Fantasy_Cricket"
OUT = Path("data/final/presentation_draft.mp4")


# ------------------------------------------------------------------ text (PIL, rendered once per segment)
def font(size: int, bold: bool = False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_centered(d: ImageDraw.ImageDraw, y: int, text: str, f, fill=(235, 235, 235), x_center: int = W // 2):
    for k, line in enumerate(text.split("\n")):
        tw = d.textlength(line, font=f)
        d.text((x_center - tw / 2, y + k * int(f.size * 1.25)), line, fill=fill, font=f)


# ------------------------------------------------------------------ media
def fit_size(w: int, h: int, box_w: int, box_h: int) -> tuple[int, int]:
    s = min(box_w / w, box_h / h)
    return max(1, int(w * s)), max(1, int(h * s))


@dataclass
class Clip:
    """A video (or still image) shown in a box, with playback speed / start / looping."""
    path: str
    box: tuple[int, int, int, int]               # x, y, w, h on the 1920x1080 frame
    speed: float = 1.0
    start: float = 0.0
    loop: bool = True
    bands: list = field(default_factory=list)    # blur bands (fractions of the source frame)
    band_region: tuple[float, float] | None = None   # blur only x in [x0, x1] fraction of the frame (preview panels)
    frames: list = field(default_factory=list)
    fps: float = 25.0

    def load(self, seconds: float) -> None:
        """Read only the frames needed for `seconds` of screen time, resized to the box."""
        x, y, bw, bh = self.box
        if Path(self.path).suffix.lower() in (".png", ".jpg", ".jpeg"):
            img = cv2.imdecode(np.fromfile(self.path, np.uint8), cv2.IMREAD_COLOR)
            self.frames, self.fps = [cv2.resize(img, fit_size(img.shape[1], img.shape[0], bw, bh), interpolation=cv2.INTER_AREA)], 1.0
            return
        cap = cv2.VideoCapture(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first = int(self.start * self.fps)
        need = n - first if self.loop else min(n - first, int(np.ceil(seconds * self.speed * self.fps)) + 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames = []
        for _ in range(max(1, need)):
            ok, f = cap.read()
            if not ok:
                break
            if self.bands:
                if self.band_region:
                    x0, x1 = int(self.band_region[0] * f.shape[1]), int(self.band_region[1] * f.shape[1])
                    f[:, x0:x1] = blur_bands(f[:, x0:x1], self.bands)
                else:
                    f = blur_bands(f, self.bands)
            frames.append(cv2.resize(f, fit_size(f.shape[1], f.shape[0], bw, bh), interpolation=cv2.INTER_AREA))
        cap.release()
        if not frames:
            raise FileNotFoundError(self.path)
        self.frames = frames

    def at(self, t: float) -> np.ndarray:
        i = int(t * self.speed * self.fps + 1e-6)
        return self.frames[i % len(self.frames)] if self.loop else self.frames[min(i, len(self.frames) - 1)]

    def paste(self, canvas: np.ndarray, t: float) -> None:
        img = self.at(t)
        x, y, bw, bh = self.box
        ox, oy = x + (bw - img.shape[1]) // 2, y + (bh - img.shape[0]) // 2
        # clip to the canvas (never crash on a layout slip)
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(W, ox + img.shape[1]), min(H, oy + img.shape[0])
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]


@dataclass
class Segment:
    seconds: float
    title: str = ""
    caption: str = ""
    clips: list = field(default_factory=list)
    texts: list = field(default_factory=list)        # (x_center, y, text, size, bold, colour)
    card: tuple[str, str] | None = None              # full-screen text card (big, small)

    def background(self) -> np.ndarray:
        img = Image.new("RGB", (W, H), BG[::-1])
        d = ImageDraw.Draw(img)
        if self.title:
            draw_centered(d, 28, self.title, font(44, bold=True))
        if self.caption:
            draw_centered(d, H - 92, self.caption, font(34), fill=(255, 214, 120))
        for xc, y, text, size, bold, colour in self.texts:
            draw_centered(d, y, text, font(size, bold), fill=colour, x_center=xc)
        if self.card:
            draw_centered(d, H // 2 - 170, self.card[0], font(60, bold=True))
            draw_centered(d, H // 2 + 10, self.card[1], font(36), fill=(210, 210, 210))
        d.text((18, H - 38), LABEL, fill=(200, 200, 200), font=font(22))      # label on every frame
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def canvas_overlay_bands(pose_npz: str, video: str, duration: float, top_margin: float) -> list:
    """Overlay bands of the source clip mapped into the 480x832 control canvas (left panel of depth_preview.mp4)."""
    k, _, _, _ = clean_resampled(pose_npz, 33, 16.0, 0.0, duration)
    s, (cx, cy), (ox, oy) = canvas_transform(k, 480, 832, 0.85, top_margin)
    cap = cv2.VideoCapture(video)
    vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    out = []
    for y0, y1, x0, x1 in OVERLAY_BANDS.get(Path(video).stem, []):
        X0, X1 = (x0 * vw - cx) * s + ox, (x1 * vw - cx) * s + ox
        Y0, Y1 = (y0 * vh - cy) * s + oy, (y1 * vh - cy) * s + oy
        out.append((max(0.0, Y0 / 832), min(1.0, Y1 / 832), max(0.0, X0 / 480), min(1.0, X1 / 480)))
    return [b for b in out if b[1] > b[0] and b[3] > b[2]]


# ------------------------------------------------------------------ timeline (docs/VIDEO_SCRIPT.md)
def build_segments() -> list[Segment]:
    PH = 860                                     # portrait panel height
    top = 120
    c2 = OVERLAY_BANDS["C2_coverdrive"]
    batter_prev_bands = canvas_overlay_bands("data/pose/batter/raw.npz", "data/clips/C2_coverdrive.mp4", 5.6, 0.08)

    def row(paths_speeds, y=top, h=PH, gap=40):
        """Centre a row of portrait boxes (width = h * 480/832)."""
        w = int(h * 480 / 832)
        n = len(paths_speeds)
        x0 = (W - (n * w + (n - 1) * gap)) // 2
        return [(x0 + i * (w + gap), y, w, h) for i in range(n)]

    segs = []
    # 0:00-0:08 hook
    b = row([1])[0]
    segs.append(Segment(8, "", "What if Larwood bowled to Bradman?", [Clip("data/final/final_match.mp4", b, start=4.0, loop=False)]))
    # 0:08-0:25 idea + inputs
    boxes = row([1, 1, 1, 1], h=760, y=150, gap=36)
    labels = ["Modern batter (motion)", "Bradman (archival photo)", "Modern bowler (motion)", "Larwood, 1929 (public domain)"]
    segs.append(Segment(17, "Idea: modern motion + one archival photo", "Cross-Era Fantasy Cricket",
                        [Clip("data/clips/C2_coverdrive.mp4", boxes[0], bands=c2),
                         Clip("data/refs/bradman_crop.png", boxes[1]),
                         Clip("data/clips/B1_bowler.mp4", boxes[2]),
                         Clip("data/refs/larwood_crop.png", boxes[3])],
                        texts=[(bx[0] + bx[2] // 2, bx[1] + bx[3] + 8, lab, 26, False, (200, 200, 200)) for bx, lab in zip(boxes, labels)]))
    # 0:25-0:50 Milestone 1
    ov = (W // 2 - 300, top, 600, PH)
    segs.append(Segment(8, "Milestone 1: motion extraction", "DWPose tracks one player (umpire and fielders ignored)",
                        [Clip("data/pose/batter/overlay.mp4", ov, bands=c2)]))
    wide = (60, top, W - 120, PH)
    segs.append(Segment(9, "Milestone 1: motion extraction", "Pose and depth controls, aligned frame by frame (shown at half speed)",
                        [Clip("data/pose/batter/depth_preview.mp4", wide, speed=0.5, bands=batter_prev_bands, band_region=(0.0, 1 / 3))]))
    segs.append(Segment(8, "Milestone 1: motion extraction", "Bowler: target-only depth removes the fielders (half speed)",
                        [Clip("data/pose/bowler/depth_preview.mp4", wide, speed=0.5)]))
    # 0:50-1:10 Milestone 2
    facts_left = ("Laptop RTX 4050: 6 GB - too small\n"
                  "Kaggle T4 (free): 16 GB, peak 14.9 GB\n"
                  "~9 min per 33-frame clip")
    facts_right = ("Wan2.1-VACE-1.3B: photo + control video\n"
                   "LTX-2: 32 GB+, first frame only\n"
                   "MagicAnimate: SD1.5 + DensePose, outdated")
    segs.append(Segment(12, "Milestone 2: running the model", "Generation runs headless in ComfyUI on a free Kaggle T4",
                        [Clip("docs/report/figs/pipeline.png", (90, 110, 1740, 600))],
                        texts=[(W // 2 - 430, 735, facts_left, 34, False, (235, 235, 235)),
                               (W // 2 + 430, 735, facts_right, 34, False, (235, 235, 235))]))
    bl = row([1], h=760, y=150)[0]
    segs.append(Segment(8, "Milestone 2: team baseline (Student 2)", "AnimateDiff / MagicAnimate + DensePose - our baseline, not the final model",
                        [Clip("data/student3/bradman_1930s_graded_master.mp4", bl)],
                        texts=[(W // 2, bl[1] + bl[3] + 10, "TEAM BASELINE", 30, True, (255, 140, 120))]))
    # 1:10-1:45 Milestone 3
    segs.append(Segment(10, "Milestone 3: generation (ablation)", "Skeleton -> bat lost   |   Depth -> bat kept",
                        [Clip("docs/report/figs/ablation_strip.png", (210, 120, 1500, 860))]))
    pair = row([1, 1], h=800, y=130, gap=160)
    segs.append(Segment(8, "Milestone 3: generation (ablation)", "Same photo, same seed - only the control differs (half speed)",
                        [Clip("data/gen/v2_raw/C_s42_17f.mp4", pair[0], speed=0.5),
                         Clip("data/gen/v2_raw/D1_s42_17f.mp4", pair[1], speed=0.5)],
                        texts=[(pair[0][0] + pair[0][2] // 2, 935, "Skeleton -> bat lost", 34, True, (255, 140, 120)),
                               (pair[1][0] + pair[1][2] // 2, 935, "Depth -> bat kept", 34, True, (140, 230, 140))]))
    segs.append(Segment(5, "Milestone 3: results", "",
                        card=("Depth kept the bat in 5/5 runs\nskeleton lost it in 2/2",
                              "PCK@0.2 0.90-1.00 (batter)  |  tighter crop: no clear likeness gain  |  seed matters more")))
    segs.append(Segment(12, "Milestone 3: raw VACE outputs", "Bradman (F1, real speed)   |   Larwood (L3, slow motion)",
                        [Clip("data/final/work/final_cap_s42_33f_retimed.mp4", pair[0]),
                         Clip("data/final/work/L3_s2024_33f_retimed.mp4", pair[1])],
                        texts=[(pair[1][0] + pair[1][2] // 2, 935, "slow motion", 30, True, (255, 214, 120))]))
    # 1:45-2:15 Milestone 4
    segs.append(Segment(5, "Milestone 4: cross-era styling", "The same moment in 1930s / 1970s / 1990s",
                        [Clip("docs/report/figs/era_strip.png", (300, 120, 1320, 860))]))
    segs.append(Segment(9, "Milestone 4: cross-era styling", "1930s / 1970s / 1990s (2x speed)",
                        [Clip("data/final/final_eras_cap.mp4", b, speed=2.0, loop=False)]))
    segs.append(Segment(16, "Milestone 4: cross-era match", "Larwood in 1930s film  vs  Bradman in 1990s broadcast",
                        [Clip("data/final/catalog/match_crossera.mp4", b, loop=False)]))
    # 2:15-2:30 limitations, ethics, team
    segs.append(Segment(8, "Limitations and ethics", "",
                        card=("Limits: weak likeness - helmet outline becomes a dark cap\nreference clothing is copied - short single-player clips",
                              "AI-generated label on every frame  |  deceased public figures  |  public-domain photos\n"
                              "modern clips used only for motion  |  academic use only")))
    segs.append(Segment(7, "", "", card=("Cross-Era Fantasy Cricket", f"Team: [TODO names]\n{GITHUB}")))
    return segs


def main() -> int:
    segs = build_segments()
    total = sum(s.seconds for s in segs)
    assert total <= 150.0 + 1e-6, total
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise SystemExit("ffmpeg not found")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)], stdin=subprocess.PIPE)
    t_global = 0.0
    for k, seg in enumerate(segs):
        for c in seg.clips:
            c.load(seg.seconds)
        bg = seg.background()
        n = int(round(seg.seconds * FPS))
        for i in range(n):
            frame = bg.copy()
            for c in seg.clips:
                c.paste(frame, i / FPS)
            proc.stdin.write(frame.tobytes())
        print(f"{t_global:6.1f}-{t_global + seg.seconds:6.1f} s  {seg.title or '(card)'}: {seg.caption[:60]}")
        t_global += seg.seconds
        for c in seg.clips:
            c.frames = []                         # free memory before the next segment
    proc.stdin.close()
    proc.wait()
    cap = cv2.VideoCapture(str(OUT))
    nf, f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Wrote {OUT.resolve()}: {nf} frames @ {f:g} fps = {nf / f:.1f} s (target {total:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
