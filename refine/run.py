"""Refine a COPY of the MusePose outputs, step by step, re-scoring after every step.

    ~/gen/bin/python -m refine.run [--src outputs/animate_v3] [--out outputs/animate_v3_refined] [job ...]

For each job (default: bradman_batting_3d, mcgrath_bowling_3d):
  step0_original.mp4   exact copy of the Kaggle/MusePose result (after bat/ball compositing, animate.finish)
  step1_512.mp4        brief format: 512 x 512, <= 100 frames                       -> <job>_512.mp4
  step2_colour.mp4     colours from the reference photo
  step3_stable.mp4     colour held steady over time, calm background, dithered
  step4_grain.mp4      fine grain matched to the photo's texture                   -> <job>_512_refined.mp4
Each step is scored (refine.score); a step that makes accuracy, object permanence or distortions clearly worse
is rejected and the previous step is kept (noted in the report). The source folder is never written to.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from refine import colour, conform
from refine.score import read, run as score

ROOT = Path(__file__).resolve().parents[1]
JOBS = ["bradman_batting_3d", "mcgrath_bowling_3d"]
GUARD = {"accuracy": 2.0, "object_permanence": 3.0, "distortions(no=100)": 5.0, "limb_rigidity": 5.0}


def write(frames, path, fps=24):
    tmp = Path(str(path) + ".tmp.mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "14", "-preset", "slow",
                    "-pix_fmt", "yuv444p" if False else "yuv420p", str(path)], check=True)
    tmp.unlink()


def regressed(prev, cur):
    bad = []
    for k, tol in GUARD.items():
        if cur["scores_0_100"][k] < prev["scores_0_100"][k] - tol:
            bad.append(f"{k} {prev['scores_0_100'][k]} -> {cur['scores_0_100'][k]}")
    return bad


def side_by_side(paths, labels, out, size=512):
    vids = [read(p) for p in paths]
    n = min(len(v) for v in vids[1:])  # step 0 is longer for batting: show its frames in the same window
    win = json.load(open(Path(out).parent / "target_512.json"))["conform"]["frames"]
    frames = []
    for t in range(n):
        tiles = []
        for i, (v, lab) in enumerate(zip(vids, labels)):
            f = v[min(t + (win[0] if i == 0 else 0), len(v) - 1)]
            f = cv2.resize(f, (int(f.shape[1] * size / f.shape[0]), size))
            cv2.rectangle(f, (0, 0), (f.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(f, lab, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(f)
        fr = np.hstack(tiles)
        cv2.putText(fr, "AI reconstruction", (fr.shape[1] - 190, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(fr)
    write(frames, out)


def refine_job(job, src, out_root):
    jd = ROOT / "animate" / "jobs" / job
    o = out_root / job
    o.mkdir(parents=True, exist_ok=True)
    rep = {"job": job, "steps": {}}
    s0 = o / "step0_original.mp4"
    shutil.copy(src / f"{job}.mp4", s0)
    sc0 = score(s0, jd / "target_pose.json", jd / "ref.png")
    rep["steps"]["0_original"] = sc0
    print(job, "step0", sc0["scores_0_100"])
    # step 1
    s1 = o / "step1_512.mp4"
    tmp = o / "step1_raw.mp4"
    info1 = conform.run(s0, jd / "target_pose.json", jd / "ref.png", tmp, o / "target_512.json", o / "ref_512.png")
    write(read(tmp), s1)
    tmp.unlink()
    sc1 = score(s1, o / "target_512.json", o / "ref_512.png")
    rep["steps"]["1_512"] = {**sc1, "info": info1, "rejected": regressed(sc0, sc1)}
    print(job, "step1", sc1["scores_0_100"], info1)
    # step 2
    frames = read(s1)
    ref = cv2.imread(str(o / "ref_512.png"))
    masks = colour.masks_for(frames)
    rmask = colour.masks_for([ref])[0]
    labs, info2 = colour.transfer(frames, ref, masks, rmask)
    s2 = o / "step2_colour.mp4"
    write([colour.to_bgr(l) for l in labs], s2)
    sc2 = score(s2, o / "target_512.json", o / "ref_512.png")
    bad2 = regressed(sc1, sc2)
    rep["steps"]["2_colour"] = {**sc2, "info": info2, "rejected": bad2}
    print(job, "step2", sc2["scores_0_100"], info2, bad2)
    # step 3 (on step 2's float frames, or on step 1 if step 2 was rejected)
    base = labs if not bad2 else [colour.to_lab(f) for f in frames]
    tgf = json.load(open(o / "target_512.json"))["frames"]
    out3, info3 = colour.stabilise(base, masks, target_frames=tgf)
    s3 = o / "step3_stable.mp4"
    write(out3, s3)
    sc3 = score(s3, o / "target_512.json", o / "ref_512.png")
    prev = sc2 if not bad2 else sc1
    bad3 = regressed(prev, sc3)
    rep["steps"]["3_stable"] = {**sc3, "info": info3, "rejected": bad3}
    print(job, "step3", sc3["scores_0_100"], bad3)
    best3 = (s3, sc3, out3) if not bad3 else ((s2, sc2, None) if not bad2 else (s1, sc1, None))
    # step 4: grain matched to the photo, on the best result so far
    in4 = best3[2] if best3[2] is not None else read(best3[0])
    out4, info4 = colour.grain(in4, ref)
    s4 = o / "step4_grain.mp4"
    write(out4, s4)
    sc4 = score(s4, o / "target_512.json", o / "ref_512.png")
    bad4 = regressed(best3[1], sc4)
    rep["steps"]["4_grain"] = {**sc4, "info": info4, "rejected": bad4}
    print(job, "step4", sc4["scores_0_100"], info4, bad4)
    final = s4 if not bad4 else best3[0]
    rep["final"] = final.name
    shutil.copy(s1, out_root / f"{job}_512.mp4")
    shutil.copy(final, out_root / f"{job}_512_refined.mp4")
    side_by_side([s0, s1, final], ["Kaggle original (560x632)", "512x512, <=100 frames", "512 + photo colours, steady"],
                 o / "compare.mp4")
    (o / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def table(reps):
    keys = ["accuracy", "accuracy_view_tolerant", "limb_rigidity", "fluidity", "object_permanence", "distortions(no=100)", "colour_realism"]
    lines = []
    for r in reps:
        lines.append(f"\n### {r['job']}\n\n| step | " + " | ".join(keys) + " | rejected |\n|---|" + "---|" * (len(keys) + 1))
        for st, v in r["steps"].items():
            rej = "; ".join(v.get("rejected") or []) or ""
            lines.append(f"| {st} | " + " | ".join(str(v["scores_0_100"][k]) for k in keys) + f" | {rej} |")
        lines.append(f"\nfinal: `{r['final']}`")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*", default=JOBS)
    ap.add_argument("--src", default=str(ROOT / "outputs" / "animate_v3"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "animate_v3_refined"))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reps = [refine_job(j, src, out) for j in a.jobs]
    old = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    old.update({r["job"]: {st: v["scores_0_100"] for st, v in r["steps"].items()} | {"final": r["final"]} for r in reps})
    (out / "summary.json").write_text(json.dumps(old, indent=1))
    print(table(reps))
    (out / "scores.md").write_text("# Scores per step (0-100, higher is better)\n" + table(reps) + "\n")


if __name__ == "__main__":
    main()
