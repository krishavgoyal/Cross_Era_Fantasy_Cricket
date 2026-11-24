"""Build MusePose jobs: reference frame (player on a ground photo) + retargeted skeleton video.

    python -m animate.build_jobs                 # all jobs in JOBS
    python -m animate.build_jobs bradman_cover_drive

Writes animate/jobs/<name>/: ref.png, ref.json, pose.mp4 (24 fps, DWPose drawing), target_pose.json (the
skeleton per frame in canvas pixels, for scoring), job.json, preview.jpg.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from animate import reference as R
from animate.bat_track import load_bat
from animate.skeleton import BODY, draw, load_poses, retarget

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "reference_players"
BG = ROOT / "data" / "backgrounds"
CLIPS = ROOT / "data" / "processed_clips"

JOBS = {
    "bradman_cover_drive": dict(
        photo=P / "don_bradman/bradman_original_repo.jpg", model="u2net_human_seg",
        background=BG / "scg_1930s_pd.jpg", bg_centre=(0.35, 0.75), grey=True,
        clip="good_cover_drive_final", frames=None, bat=True),
    "mcgrath_cummins": dict(
        photo=P / "glenn_mcgrath/mcgrath_delivery_stride_sportingheroes.jpg", box=(0.0, 0.13, 1.0, 1.0),
        model="isnet-general-use", background=BG / "lords_pavilion_2017.jpg", bg_centre=(0.15, 0.55),
        clip="bowler_new_cummins", frames=(0, 44), bat=False),
    "mcgrath_runup": dict(
        photo=P / "glenn_mcgrath/mcgrath_delivery_stride_sportingheroes.jpg", box=(0.0, 0.13, 1.0, 1.0),
        model="isnet-general-use", background=BG / "lords_pavilion_2017.jpg", bg_centre=(0.15, 0.55),
        clip="bowler_delivery_001", frames=(40, 120), bat=False),
}


def fill_gaps(seq, max_gap=8):
    """Fill joints missing in a frame from the nearest frames on both sides (linear), so limbs never drop
    out of the skeleton for a few frames. Gaps longer than max_gap stay empty. Returns (seq, filled_count)."""
    n, filled = len(seq), 0
    for j in BODY:
        have = [k for k in range(n) if j in seq[k]["body"]]
        for k in range(n):
            if j in seq[k]["body"]:
                continue
            lo = max([h for h in have if h < k], default=None)
            hi = min([h for h in have if h > k], default=None)
            if lo is not None and hi is not None and hi - lo <= max_gap:
                t = (k - lo) / (hi - lo)
                a, b = np.array(seq[lo]["body"][j]), np.array(seq[hi]["body"][j])
                seq[k]["body"][j] = tuple(a * (1 - t) + b * t)
                filled += 1
    for side in ("left_hand", "right_hand"):
        for i in range(21):
            have = [k for k in range(n) if seq[k][side][i] is not None]
            for k in range(n):
                if seq[k][side][i] is not None:
                    continue
                lo = max([h for h in have if h < k], default=None)
                hi = min([h for h in have if h > k], default=None)
                if lo is not None and hi is not None and hi - lo <= max_gap:
                    t = (k - lo) / (hi - lo)
                    a, b = np.array(seq[lo][side][i]), np.array(seq[hi][side][i])
                    seq[k][side][i] = tuple(a * (1 - t) + b * t)
    return seq, filled


def smooth(seq, w=(0.25, 0.5, 0.25)):
    """Light 3-tap temporal smoothing of joints present in all three frames (removes DWPose jitter)."""
    out = []
    for k in range(len(seq)):
        p = {"body": dict(seq[k]["body"]), "left_hand": list(seq[k]["left_hand"]),
             "right_hand": list(seq[k]["right_hand"]), "bat": seq[k].get("bat")}
        if 0 < k < len(seq) - 1:
            for j, v in seq[k]["body"].items():
                if j in seq[k - 1]["body"] and j in seq[k + 1]["body"]:
                    p["body"][j] = tuple(w[0] * np.array(seq[k - 1]["body"][j]) + w[1] * np.array(v)
                                         + w[2] * np.array(seq[k + 1]["body"][j]))
        out.append(p)
    return out


def build(name, cfg, out_root=ROOT / "animate" / "jobs"):
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    ref, ref_body, meta = R.build(cfg["photo"], cfg["background"], out, box=cfg.get("box"),
                                  bg_centre=cfg["bg_centre"], grey=cfg.get("grey", False), model=cfg["model"])
    clip = CLIPS / cfg["clip"]
    poses, (sw, sh), fps = load_poses(clip / "pose_data.json")
    bats = load_bat(clip) if cfg.get("bat") else {}
    idx = sorted(poses)
    if cfg.get("frames"):
        idx = [i for i in idx if cfg["frames"][0] <= i <= cfg["frames"][1]]
    seq = []
    for i in idx:
        p = poses[i]
        p["bat"] = bats.get(i)
        seq.append(p)
    seq, filled = fill_gaps(seq)
    seq = smooth(seq)
    # View direction (generation_v2 rules): seen from behind -> no face points, as DWPose sees a back view
    from generation_v2 import FACE_POINTS, clip_view
    view, _ = clip_view([p["body"] for p in seq], side_band=(0.3, 0.7))
    if view == "back":
        for p in seq:
            for j in FACE_POINTS:
                p["body"].pop(j, None)
    tgt, info = retarget(seq, ref_body, ref.size)
    W, H = ref.size
    vw = cv2.VideoWriter(str(out / "pose.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for p in tgt:
        vw.write(cv2.cvtColor(draw(p, (W, H)), cv2.COLOR_RGB2BGR))
    vw.release()
    ser = lambda q: None if q is None else [round(q[0], 1), round(q[1], 1)]
    (out / "target_pose.json").write_text(json.dumps({
        "canvas": [W, H], "fps": fps, "source_frames": idx,
        "frames": [{"body": {k: ser(v) for k, v in p["body"].items()},
                    "left_hand": [ser(q) for q in p["left_hand"]], "right_hand": [ser(q) for q in p["right_hand"]],
                    "bat": None if p["bat"] is None else [ser(p["bat"][0]), ser(p["bat"][1])]} for p in tgt]}))
    job = {"name": name, "clip": cfg["clip"], "view": view, "source_frames": [idx[0], idx[-1]], "n_frames": len(idx), "fps": fps,
           "bat": bool(cfg.get("bat")), "joints_filled_from_neighbours": filled, "retarget": info,
           "reference": meta}
    (out / "job.json").write_text(json.dumps(job, indent=1, default=str))
    # preview: reference + 5 skeleton frames over it
    refb = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2BGR)
    tiles = [refb] + [cv2.addWeighted(refb, 0.45, cv2.cvtColor(draw(tgt[k], (W, H)), cv2.COLOR_RGB2BGR), 1.0, 0)
                      for k in np.linspace(0, len(tgt) - 1, 5).astype(int)]
    cv2.imwrite(str(out / "preview.jpg"), cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5))
    return job


if __name__ == "__main__":
    names = sys.argv[1:] or list(JOBS)
    for n in names:
        j = build(n, JOBS[n])
        print(f"{n}: {j['n_frames']} frames from {j['clip']} {j['source_frames']}, filled {j['joints_filled_from_neighbours']} "
              f"joints, retarget {j['retarget']}")
