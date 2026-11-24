"""Score every ablation run into results/ablation.csv (CPU): PCK@0.2, identity vs BOTH reference photos, flicker.

The run list is a small CSV (default results/ablation_runs.csv) with columns
    run, control_type, ref_type, seed, video, control
  * video   : the RAW (unlabelled) generated clip, e.g. data/gen/v2_raw/D1_s42_17f.mp4
  * control : the .npz of the control that generated it (canvas keypoints for PCK):
              pose without margin  -> data/pose/batter/control_33.npz          (run A)
              pose with margin     -> data/pose/batter/control_33_margin.npz   (run C)
              depth (with margin)  -> data/pose/batter/depth_33.npz            (runs B, D1-D4)
Runs whose video file is missing are reported and skipped, so you can score as soon as some outputs arrive.
17-frame videos are scored against the first 17 control frames.

Identity columns: dino_crop_* = vs data/refs/bradman_crop.png, dino_orig_* = vs data/refs/bradman.png (grayscale crops).

Usage
  python scripts/score_ablation.py
  python scripts/score_ablation.py --runs results/ablation_runs.csv --out results/ablation.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eval_metrics import Dino, evaluate_video, ref_embedding
from pose_utils import load_dwpose

FIELDS = ["timestamp", "run", "control_type", "ref_type", "seed", "video", "control", "frames_sampled", "fps",
          "person_found_rate", "pck@0.2", "pck_valid_joints", "dino_crop_mean", "dino_crop_min", "dino_orig_mean",
          "dino_orig_min", "brightness_std", "frame_diff_mean"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/ablation_runs.csv")
    ap.add_argument("--out", default="results/ablation.csv")
    ap.add_argument("--ref-crop", default="data/refs/bradman_crop.png")
    ap.add_argument("--ref-orig", default="data/refs/bradman.png")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--skip-identity", action="store_true", help="no DINOv2 (no torch/transformers needed)")
    args = ap.parse_args()

    runs = list(csv.DictReader(open(args.runs, encoding="utf-8-sig")))
    todo = [r for r in runs if Path(r["video"]).exists()]
    for r in runs:
        if r not in todo:
            print(f"skip {r['run']:4s} (video not found: {r['video']})")
    if not todo:
        raise SystemExit("No run videos found yet - download the Kaggle outputs into the paths in " + args.runs)

    model = load_dwpose()
    dino, refs = None, None
    if not args.skip_identity:
        dino = Dino()
        refs = {"crop": ref_embedding(model, dino, args.ref_crop), "orig": ref_embedding(model, dino, args.ref_orig)}
    opts = SimpleNamespace(alpha=args.alpha, thr=args.thr, rgb_identity=False)

    rows = []
    for r in todo:
        c = np.load(r["control"])
        ctrl, cfps, csize = c["kpts"][:, :17].astype(np.float32), float(c["fps"]), tuple(int(v) for v in c["size"])
        m = evaluate_video(Path(r["video"]), "raw", r["run"], model, ctrl, cfps, csize, refs, dino, opts)
        row = {k: m.get(k, "") for k in FIELDS}
        row.update(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), run=r["run"], control_type=r["control_type"],
                   ref_type=r["ref_type"], seed=r["seed"], video=r["video"], control=r["control"])
        rows.append(row)
        print(f"scored {r['run']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:            # rewritten each time = always current
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)

    f = lambda v: f"{float(v):6.3f}" if v not in ("", None) else "   n/a"
    print(f"\n{'run':4s} {'control':7s} {'ref':6s} {'seed':>5s} {'frames':>6s} {'PCK@0.2':>7s} {'ID crop':>7s} "
          f"{'ID orig':>7s} {'b.std':>6s} {'f.diff':>6s}")
    for r in rows:
        print(f"{r['run']:4s} {r['control_type']:7s} {r['ref_type']:6s} {r['seed']:>5s} {r['frames_sampled']:>6} "
              f"{f(r['pck@0.2']):>7s} {f(r['dino_crop_mean']):>7s} {f(r['dino_orig_mean']):>7s} "
              f"{f(r['brightness_std']):>6s} {f(r['frame_diff_mean']):>6s}")
    print(f"\nWrote {out} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
