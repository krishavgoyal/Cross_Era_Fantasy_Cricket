"""Prepare a tight VACE reference photo: remove the background (rembg), put the player on pure white and crop so
he fills ~90% of a 480x832 portrait canvas with the head near the top (no big white padding).

Usage
  python scripts/prepare_ref.py --input "C:/Users/ASUS V16/Downloads/bradman_raw.tif" --out data/refs/bradman_crop.png ^
      --compare-with data/refs/bradman.png

Outputs
  data/refs/bradman_crop.png     the new reference (480x832, white background)
  data/refs/bradman_crop_mask.png the foreground mask used (for checking)
  data/refs/ref_compare.png      contact sheet: old reference | new reference

Notes
  * The default rembg model (u2net, salient object) keeps the BAT with the player; the human-only model
    (u2net_human_seg) tends to drop it. Change with --model.
  * A seated / wide subject cannot fill 90% of a portrait canvas without cutting the sides. Up to --max-side-crop
    (default 15%) of the subject box is trimmed per side, centred on the head/shoulders/bat so the elbows or lap are
    trimmed first. Use --max-side-crop 0 to never cut (leaves white space at the bottom instead).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def remove_background(img_rgb: Image.Image, model: str) -> np.ndarray:
    """Return the alpha matte (H, W) in [0, 1] for the main subject."""
    from rembg import new_session, remove
    rgba = remove(img_rgb, session=new_session(model))
    return np.asarray(rgba)[..., 3].astype(np.float32) / 255.0


def main_component(alpha: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Keep only the largest connected foreground blob (drops specks of bench/background)."""
    binary = (alpha > thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return alpha
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = cv2.dilate((labels == keep).astype(np.uint8), np.ones((15, 15), np.uint8))   # keep soft edges near it
    return alpha * blob


def compose_tight(gray: np.ndarray, alpha: np.ndarray, width: int, height: int, fill: float, top_margin: float,
                  max_side_crop: float) -> np.ndarray:
    """Place the subject on white: subject height = fill * canvas height, top of subject at top_margin * height."""
    ys, xs = np.where(alpha > 0.5)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    s = min(fill * height / bh, width / (bw * (1 - 2 * max_side_crop)))
    if s < fill * height / bh:
        print(f"NOTE: subject is wide; using {s * bh / height:.0%} of the height instead of {fill:.0%} "
              f"(raise --max-side-crop to fill more)")
    # Horizontal centre = centre of the UPPER part of the subject (head, shoulders, bat handle), so any side
    # cropping trims the elbow / lap rather than the face or the bat.
    upper = alpha[y0:y0 + int(0.45 * bh)] > 0.5
    cols = np.where(upper.any(axis=0))[0]
    cx = (cols.min() + cols.max() + 1) / 2 if len(cols) else (x0 + x1) / 2
    half = width / (2 * s)                                   # keep the window inside the subject box if possible
    cx = float(np.clip(cx, x0 + half, x1 - half)) if bw > 2 * half else (x0 + x1) / 2
    # affine: source -> canvas. Subject top goes to top_margin*height, horizontally centred.
    M = np.float32([[s, 0, width / 2 - s * cx], [0, s, top_margin * height - s * y0]])
    white = np.full_like(gray, 255.0, dtype=np.float32)
    comp = gray.astype(np.float32) * alpha + white * (1 - alpha)                      # subject on pure white
    out = cv2.warpAffine(comp, M, (width, height), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    a = cv2.warpAffine(alpha, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.clip(out, 0, 255).astype(np.uint8), a


def contact_sheet(old_path: str | None, new_img: np.ndarray, out_path: Path) -> None:
    """old | new side by side (old letterboxed, not stretched)."""
    tiles, titles = [], []
    if old_path and Path(old_path).exists():
        old = cv2.imdecode(np.fromfile(old_path, np.uint8), cv2.IMREAD_COLOR)
        H, W = new_img.shape[:2]                                 # letterbox (keep the old image's aspect ratio)
        s = min(W / old.shape[1], H / old.shape[0])
        small = cv2.resize(old, (int(old.shape[1] * s), int(old.shape[0] * s)), interpolation=cv2.INTER_AREA)
        tile = np.full((H, W, 3), 255, np.uint8)
        y, x = (H - small.shape[0]) // 2, (W - small.shape[1]) // 2
        tile[y:y + small.shape[0], x:x + small.shape[1]] = small
        tiles.append(tile)
        titles.append("OLD: " + Path(old_path).name)
    tiles.append(cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR) if new_img.ndim == 2 else new_img)
    titles.append("NEW (tight crop)")
    sheet = []
    for t, title in zip(tiles, titles):
        bar = np.full((40, t.shape[1], 3), 40, np.uint8)
        cv2.putText(bar, title, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        framed = cv2.copyMakeBorder(np.vstack([bar, t]), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 255))
        sheet.append(framed)
    cv2.imwrite(str(out_path), np.hstack(sheet))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="raw photo (tif/png/jpg)")
    ap.add_argument("--out", default="data/refs/bradman_crop.png")
    ap.add_argument("--compare-with", default="data/refs/bradman.png", help="old reference for the contact sheet")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--fill", type=float, default=0.90, help="subject height as a fraction of the canvas height")
    ap.add_argument("--top-margin", type=float, default=0.03, help="white margin above the head (fraction of height)")
    ap.add_argument("--max-side-crop", type=float, default=0.15,
                    help="allowed crop of the subject box per side (0 = never cut the sides, may leave white space)")
    ap.add_argument("--compare-out", default="ref_compare.png", help="file name of the contact sheet (next to --out)")
    ap.add_argument("--model", default="u2net", help="rembg model (u2net keeps the bat; isnet-general-use also ok)")
    args = ap.parse_args()

    img = Image.open(args.input)
    gray = np.asarray(img.convert("L"))                     # archival photo: keep it black and white
    print(f"Input {args.input}: {img.size[0]}x{img.size[1]} mode {img.mode}")
    alpha = main_component(remove_background(img.convert("RGB"), args.model))
    out_img, out_alpha = compose_tight(gray, alpha, args.width, args.height, args.fill, args.top_margin,
                                       args.max_side_crop)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(out.with_name(out.stem + "_mask.png")), (out_alpha * 255).astype(np.uint8))
    contact_sheet(args.compare_with, out_img, out.with_name(args.compare_out))

    ys = np.where(out_alpha.max(axis=1) > 0.5)[0]
    xs = np.where(out_alpha.max(axis=0) > 0.5)[0]
    print(f"Subject spans rows {ys.min()}-{ys.max()} ({(ys.max() - ys.min() + 1) / args.height:.0%} of height), "
          f"cols {xs.min()}-{xs.max()}; top margin {ys.min()} px")
    print(f"Wrote {out}, {out.with_name(out.stem + '_mask.png')} and {out.with_name(args.compare_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
