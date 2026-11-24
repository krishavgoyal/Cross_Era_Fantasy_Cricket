"""Build a Kaggle input dataset zip (refs/, controls/, pairs.csv) for kaggle/vace_generate.ipynb.

Presets (use --preset NAME), or build any dataset with --file / --row / --prompt / --negative:

  batter-v2  C:/cefc/cefc-inputs-v2.zip       likeness/bat ablation: B, C, D1-D4 (17 frames, default prompt)
  bowler     C:/cefc/cefc-inputs-bowler.zip   Larwood: L1/L2/L3 = depth + larwood_crop.png, seeds 42/7/2024, 33 frames

Zip entries use forward slashes and sit in one top folder (<name>/refs/..., <name>/controls/..., <name>/pairs.csv).

Usage
  python scripts/build_kaggle_inputs.py --preset bowler
  python scripts/build_kaggle_inputs.py --preset batter-v2
  python scripts/build_kaggle_inputs.py --name my-inputs --file refs/x.png=data/refs/x.png ^
      --file controls/c.mp4=data/pose/x/depth_33.mp4 --row X1,refs/x.png,controls/c.mp4,42 --frames 33
"""
from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path

# The notebook's default negative prompt (kaggle/vace_generate.ipynb, Cell 1). Keep in sync if it changes.
NOTEBOOK_NEGATIVE = ("blurry, low quality, worst quality, static, deformed, extra fingers, bad hands, bad face, "
                     "missing bat, extra limbs, fused fingers, watermark, subtitles, text, cluttered background, "
                     "extra balls, multiple balls, two bats, deformed bat, hockey stick, cropped head, helmet, "
                     "modern logos, logo, black pads, black trousers")

BOWLER_PROMPT = ("archival 1930s photograph, fast bowler, bare-headed with short neatly combed hair, face clearly visible, "
                 "white long-sleeved flannel shirt and long white trousers, white boots, delivering a cricket ball with a "
                 "high overarm bowling action, natural motion")
# Bowler: the default negative WITHOUT "missing bat" (that phrase would push the model TOWARD drawing a bat), plus
# the bowler terms. The reference (NLA 1929 photo) shows a fedora, suit and cigarette, which must NOT be copied.
BOWLER_NEGATIVE = (NOTEBOOK_NEGATIVE.replace("missing bat, ", "") +
                   ", cricket bat, cap, hat, fedora, suit, blazer, jacket, dark coat, cigarette, helmet, logo, purple shirt, "
                   "coloured kit, shorts, bare legs, sleeveless, extra people")

PRESETS = {
    "batter-v2": dict(
        name="cefc-inputs-v2", frames=17, prompt="", negative="",
        files={"refs/bradman.png": "data/refs/bradman.png",
               "refs/bradman_crop.png": "data/refs/bradman_crop.png",
               "controls/batter_control_33.mp4": "data/pose/batter/control_33_margin.mp4",
               "controls/batter_depth_33.mp4": "data/pose/batter/depth_33.mp4"},
        rows=[("B", "refs/bradman.png", "controls/batter_depth_33.mp4", 42),
              ("C", "refs/bradman_crop.png", "controls/batter_control_33.mp4", 42),
              ("D1", "refs/bradman_crop.png", "controls/batter_depth_33.mp4", 42),
              ("D2", "refs/bradman_crop.png", "controls/batter_depth_33.mp4", 7),
              ("D3", "refs/bradman_crop.png", "controls/batter_depth_33.mp4", 123),
              ("D4", "refs/bradman_crop.png", "controls/batter_depth_33.mp4", 2024)]),
    "bowler": dict(
        name="cefc-inputs-bowler", frames=33, prompt=BOWLER_PROMPT, negative=BOWLER_NEGATIVE,
        files={"refs/larwood_crop.png": "data/refs/larwood_crop.png",
               "controls/bowler_depth_33.mp4": "data/pose/bowler/depth_33.mp4",
               "controls/bowler_control_33.mp4": "data/pose/bowler/control_33_margin.mp4"},
        rows=[("L1", "refs/larwood_crop.png", "controls/bowler_depth_33.mp4", 42),
              ("L2", "refs/larwood_crop.png", "controls/bowler_depth_33.mp4", 7),
              ("L3", "refs/larwood_crop.png", "controls/bowler_depth_33.mp4", 2024)]),
}


def build(name: str, files: dict[str, str], rows: list[tuple], frames: int, prompt: str, negative: str,
          out_dir: str = "C:/cefc") -> Path:
    stage = Path(out_dir) / name
    if stage.exists():
        shutil.rmtree(stage)
    for dst, src in files.items():
        if not Path(src).exists():
            raise SystemExit(f"Missing {src} - run prepare_ref.py / make_control.py / make_depth_control.py first")
        (stage / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, stage / dst)
    for r in rows:
        for rel in r[1:3]:
            if rel not in files:
                raise SystemExit(f"Row {r[0]} uses {rel}, which is not in the dataset files")
    with open(stage / "pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "ref", "control", "prompt", "negative", "seed", "frames"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], prompt, negative, r[3], r[4] if len(r) > 4 else frames])
    zip_path = Path(shutil.make_archive(str(Path(out_dir) / name), "zip", root_dir=out_dir, base_dir=name))
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    assert all("\\" not in n for n in names), "zip entries must use forward slashes"
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB):")
    for n in names:
        print("   ", n)
    print((stage / "pairs.csv").read_text(encoding="utf-8"))
    return zip_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), help="ready-made dataset (other options override it)")
    ap.add_argument("--name", help="dataset / zip name (C:/cefc/<name>.zip)")
    ap.add_argument("--file", action="append", default=[], help="ZIP_PATH=LOCAL_PATH, e.g. refs/a.png=data/refs/a.png")
    ap.add_argument("--row", action="append", default=[], help="NAME,REF,CONTROL,SEED[,FRAMES] (paths as in the zip)")
    ap.add_argument("--frames", type=int, help="frames for rows without their own (4n+1)")
    ap.add_argument("--prompt", help="prompt for every row (blank = notebook default)")
    ap.add_argument("--negative", help="negative prompt for every row (blank = notebook default)")
    ap.add_argument("--out-dir", default="C:/cefc")
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset]) if args.preset else dict(name=None, files={}, rows=[], frames=33, prompt="", negative="")
    if args.name:
        cfg["name"] = args.name
    for f in args.file:
        dst, src = f.split("=", 1)
        cfg["files"] = {**cfg["files"], dst: src}
    if args.row:
        cfg["rows"] = [tuple(int(v) if i >= 3 else v for i, v in enumerate(r.split(","))) for r in args.row]
    for k in ("frames", "prompt", "negative"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if not cfg["name"] or not cfg["files"] or not cfg["rows"]:
        raise SystemExit("Give --preset, or --name with at least one --file and one --row")
    if (cfg["frames"] - 1) % 4:
        raise SystemExit(f"--frames {cfg['frames']} is not 4n+1")
    build(cfg["name"], cfg["files"], cfg["rows"], cfg["frames"], cfg["prompt"], cfg["negative"], args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
