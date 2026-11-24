"""Build the Kaggle input dataset zip for the likeness/bat ablation (v2).

Creates C:/cefc/cefc-inputs-v2/ and C:/cefc/cefc-inputs-v2.zip (zip entries use forward slashes):
  cefc-inputs-v2/refs/bradman.png                  old padded reference
  cefc-inputs-v2/refs/bradman_crop.png             tight reference from prepare_ref.py
  cefc-inputs-v2/controls/batter_control_33.mp4    pose control WITH top margin (control_33_margin.mp4)
  cefc-inputs-v2/controls/batter_depth_33.mp4      depth control (depth_33.mp4, soft head, same margin)
  cefc-inputs-v2/pairs.csv                         the runs below (all 17 frames)

Runs (A = the existing 17f smoke video: pose + padded bradman.png, seed 42, generated earlier):
  B  depth + bradman.png       seed 42      | C  pose + bradman_crop.png seed 42
  D1 depth + bradman_crop.png  seed 42      | D2 seed 7 | D3 seed 123 | D4 seed 2024

Usage:  python scripts/build_kaggle_inputs.py
"""
from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

FILES = {  # zip path -> local source
    "refs/bradman.png": "data/refs/bradman.png",
    "refs/bradman_crop.png": "data/refs/bradman_crop.png",
    "controls/batter_control_33.mp4": "data/pose/batter/control_33_margin.mp4",
    "controls/batter_depth_33.mp4": "data/pose/batter/depth_33.mp4",
}
DEPTH, POSE = "controls/batter_depth_33.mp4", "controls/batter_control_33.mp4"
PAD, CROP = "refs/bradman.png", "refs/bradman_crop.png"
RUNS = [  # name, ref, control, seed  (prompt/negative blank = notebook defaults)
    ("B", PAD, DEPTH, 42),
    ("C", CROP, POSE, 42),
    ("D1", CROP, DEPTH, 42),
    ("D2", CROP, DEPTH, 7),
    ("D3", CROP, DEPTH, 123),
    ("D4", CROP, DEPTH, 2024),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="C:/cefc", help="where cefc-inputs-v2/ and the zip are written")
    ap.add_argument("--name", default="cefc-inputs-v2")
    ap.add_argument("--frames", type=int, default=17)
    args = ap.parse_args()

    stage = Path(args.out_dir) / args.name
    if stage.exists():
        shutil.rmtree(stage)
    for dst, src in FILES.items():
        if not Path(src).exists():
            raise SystemExit(f"Missing {src} - run prepare_ref.py / make_control.py / make_depth_control.py first")
        (stage / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, stage / dst)
    with open(stage / "pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "ref", "control", "prompt", "negative", "seed", "frames"])
        for name, ref, ctrl, seed in RUNS:
            w.writerow([name, ref, ctrl, "", "", seed, args.frames])

    zip_base = Path(args.out_dir) / args.name
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=args.out_dir, base_dir=args.name))
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    assert all("\\" not in n for n in names), "zip entries must use forward slashes"
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB):")
    for n in names:
        print("   ", n)
    print((stage / "pairs.csv").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
