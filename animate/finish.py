"""After a Kaggle MusePose run: bat compositing, scoring, demos, contact sheets.

    python -m animate.finish kaggle_results/animate outputs/animate_v2 bradman_batting mcgrath_bowling

For each job: <out>/<job>_raw.mp4 (MusePose output), <job>.mp4 (final; batting gets the real bat added),
<job>_demo.mp4 (motion source | final, labelled), <job>_grid.mp4 (reference | skeleton | raw), <job>_eval.json,
<job>_sheet.jpg (8 frames).
"""
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WHO = {"bradman": "Don Bradman", "mcgrath": "Glenn McGrath"}


def sheet(video, out, n=8, h=300):
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    tiles = [cv2.resize(frames[i], (int(frames[i].shape[1] * h / frames[i].shape[0]), h)) for i in idx]
    cv2.imwrite(str(out), np.hstack(tiles))


def demo(job_dir, final, out, who):
    """Motion source | final, frame by frame. The left panel shows, for every output frame, the source frame
    it comes from (repeated where the delivery was slowed down), so both sides stay in step."""
    job = json.load(open(job_dir / "job.json"))
    clip = ROOT / "data" / "processed_clips" / job["clip"]
    tgt = json.load(open(job_dir / "target_pose.json"))
    W, H = tgt["canvas"]
    cap = cv2.VideoCapture(str(clip / "source_512.mp4"))
    src = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        src.append(f)
    cap = cv2.VideoCapture(str(final))
    tmp = Path(str(out) + ".tmp.mp4")
    vw = None
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = tgt["source_frames"][min(k, len(tgt["source_frames"]) - 1)]
        s_ = src[int(round(min(max(t, 0), len(src) - 1)))]
        s_ = cv2.resize(s_, (int(s_.shape[1] * H / s_.shape[0]), H))
        x0 = max(0, (s_.shape[1] - W) // 2)
        s_ = s_[:, x0:x0 + W]
        if s_.shape[1] < W:
            s_ = cv2.copyMakeBorder(s_, 0, 0, 0, W - s_.shape[1], cv2.BORDER_CONSTANT)
        f = cv2.resize(f, (W, H))
        frame = np.hstack([s_, f])
        slow = abs(t - round(t)) > 1e-3 or (k > 0 and abs(t - tgt["source_frames"][k - 1]) < 0.999)
        for txt, x in (("motion source" + (" (slowed)" if slow else ""), 10), (f"AI reconstruction - {who}", W + 10)):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x - 4, 6), (x + tw + 4, 14 + th), (0, 0, 0), -1)
            cv2.putText(frame, txt, (x, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if vw is None:
            vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), tgt["fps"], (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        k += 1
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "18", "-pix_fmt",
                    "yuv420p", str(out)], check=True)
    tmp.unlink()


def main(src_dir, out_dir, jobs):
    from animate.composite import composite
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in jobs:
        job_dir = ROOT / "animate" / "jobs" / name
        raw = sorted(p for p in glob.glob(str(src_dir / f"{name}_{name}_pose_*.mp4")) if not p.endswith("__.mp4")
                     and "_pose_guider" not in p)
        grid = sorted(glob.glob(str(src_dir / f"{name}_{name}_pose_*__.mp4")))
        if not raw:
            print(f"{name}: no MusePose output in {src_dir}")
            continue
        shutil.copy(raw[-1], out_dir / f"{name}_raw.mp4")
        if grid:
            shutil.copy(grid[-1], out_dir / f"{name}_grid.mp4")
        final = out_dir / f"{name}.mp4"
        if any((job_dir / f).exists() for f in ("bat_sprite.npy", "fg_layer.png", "ball_sprite.npy")):
            tmp = out_dir / f"{name}_bat_tmp.mp4"
            info = composite(out_dir / f"{name}_raw.mp4", job_dir, tmp)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "16",
                            "-pix_fmt", "yuv420p", str(final)], check=True)
            tmp.unlink()
        else:
            info = {"frames_with_bat": 0}
            shutil.copy(out_dir / f"{name}_raw.mp4", final)
        ev = out_dir / f"{name}_eval.json"
        subprocess.run([sys.executable, "-m", "animate.evaluate", str(final), str(job_dir / "target_pose.json"),
                        "--out", str(ev)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        who = next((v for k, v in WHO.items() if name.startswith(k)), name)
        demo(job_dir, final, out_dir / f"{name}_demo.mp4", who)
        sheet(final, out_dir / f"{name}_sheet.jpg")
        summary[name] = {**json.load(open(ev)), **info}
        summary[name].pop("per_joint_pck@0.1", None)
        print(name, {k: summary[name][k] for k in ("pck@0.1", "complete_body_frames", "merged_limb_frames",
                                                   "silhouette_iou_min", "frames_with_bat")})
    old = json.load(open(out_dir / "summary.json")) if (out_dir / "summary.json").exists() else {}
    old.update(summary)  # keep jobs finished in earlier calls
    (out_dir / "summary.json").write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
