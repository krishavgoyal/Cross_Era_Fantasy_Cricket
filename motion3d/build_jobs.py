"""MusePose jobs from the motion3d skeletons (new, separate jobs; the earlier ones are kept).

    ~/gen/bin/python -m motion3d.build_jobs [bradman_batting_3d] [mcgrath_bowling_3d]

Same scenes as animate/build_scene_jobs.py (Bradman on the clip's own pitch with the stumps behind him,
McGrath at the crease in his photo with the stump layer and the ball), but the motion comes from
data/motion3d/<clip>/clip/ (motion3d.export): no 2D cleaning and no 2D delivery rebuild are applied on top,
because the 3D fit already did that work with rigid bones, planted feet and the keyframed bowling arm.
Writes animate/jobs/<name>/ like the other builders (ref.png, pose.mp4, target_pose.json, job.json, ...).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from animate import scene
from animate.build_scene_jobs import OUT, P, ball_path
from animate.skeleton import draw
from motion3d.clips import out_dir

JOBS = {
    "bradman_batting_3d": dict(kind="batting", motion="good_cover_drive_final",
                               photo=P / "don_bradman" / "bradman_original_repo.jpg"),
    "mcgrath_bowling_3d": dict(kind="bowling", motion="bowler_new_cummins",
                               photo=P / "glenn_mcgrath" / "mcgrath_delivery_stride_sportingheroes.jpg",
                               search_source=(24, 40), stump_box=(212, 361, 230, 592), ball=(60, 305, 10),
                               bowling_arm="right"),
}
# same scene, motion refit with the reference-bowler legs (the first mcgrath_bowling_3d video is kept)
JOBS["mcgrath_bowling_3d_legfix"] = dict(JOBS["mcgrath_bowling_3d"])
# McGrath's photo shows him bowling towards the right (front arm right, ball hand left); the Cummins camera sees
# a bowler running left. The 3D skeleton seen from the other side (180 deg) matches the photo's pose best
# (Procrustes 59 px vs 156 px; the 0 deg view only matched with left/right swapped = a left-arm bowler), so
# McGrath follows that view: same right-arm action, running right, as in his photo.
JOBS["mcgrath_bowling_3d_view180"] = dict(JOBS["mcgrath_bowling_3d"], clip_dir="clip_view180")


def _to_list(p):
    return {"body": {k: (float(v[0]), float(v[1])) for k, v in p["body"].items()},
            "left_hand": [None if q is None else (float(q[0]), float(q[1])) for q in p["left_hand"]],
            "right_hand": [None if q is None else (float(q[0]), float(q[1])) for q in p["right_hand"]],
            "bat": None if p.get("bat") is None else ((float(p["bat"][0][0]), float(p["bat"][0][1])),
                                                     (float(p["bat"][1][0]), float(p["bat"][1][1]))),
            **({"ball": p["ball"]} if "ball" in p else {})}


def build(name, cfg):
    out = OUT / name
    mdir = out_dir(cfg["motion"])
    clip = mdir / cfg.get("clip_dir", "clip")
    fitj = json.load(open(mdir / "fit.json"))
    times = json.load(open(clip / "metadata.json"))["motion3d"]["times_source_frames"]
    if cfg["kind"] == "batting":
        ref, seq, meta = scene.batting_scene(clip, cfg["photo"], out)
    else:
        search = [t for t, s in enumerate(times) if cfg["search_source"][0] <= s <= cfg["search_source"][1]]
        ref, seq, meta = scene.bowling_scene(clip, cfg["photo"], out, search=(search[0], search[-1]),
                                             stump_box=cfg.get("stump_box"), ball=cfg.get("ball"))
    W, H = ref.size
    stats = {"motion": "motion3d (rigid 3D fit)", "fit": {k: fitj[k] for k in (
        "max_bone_length_error_px", "max_planted_foot_drift_px", "jerk_raw_vs_fit", "max_speed_ratio_raw_vs_fit")}}
    if cfg.get("ball"):
        rel = fitj["bowling_arm"]["release_index"]
        path, info = ball_path(seq, (W, H), side=cfg.get("bowling_arm"), release=rel)
        for p, b in zip(seq, path):
            p["ball"] = b
        stats["ball"] = info
        stats["bowling_arm"] = fitj["bowling_arm"]

    def inside(p, m=6):
        return all(m <= x <= W - m and m <= y <= H - m for x, y in p["body"].values())
    ok = [inside(p) for p in seq]
    best, cur = (0, 0), None
    for i, v in enumerate(ok + [False]):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    first, last = best if best[1] > best[0] else (0, len(seq))
    stats["frames_dropped_outside_canvas"] = len(seq) - (last - first)
    seq = seq[first:last]
    src = [times[int(k)] for k in meta["source_frames"]][first:last]
    stats["kept_source_frames"] = [src[0], src[-1]]
    vw = cv2.VideoWriter(str(out / "pose.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (W, H))
    for p in seq:
        vw.write(cv2.cvtColor(draw(p, (W, H), bat=False), cv2.COLOR_RGB2BGR))
    vw.release()
    ser = lambda q: None if q is None else [round(float(q[0]), 1), round(float(q[1]), 1)]
    (out / "target_pose.json").write_text(json.dumps({
        "canvas": [W, H], "fps": meta["fps"], "source_frames": src,
        "frames": [{"body": {k: ser(v) for k, v in p["body"].items()},
                    "left_hand": [ser(q) for q in p["left_hand"]], "right_hand": [ser(q) for q in p["right_hand"]],
                    "bat": None if p.get("bat") is None else [ser(p["bat"][0]), ser(p["bat"][1])],
                    "ball": ser(p.get("ball"))} for p in seq]}))
    job = {"name": name, "kind": cfg["kind"], "clip": cfg["motion"], "photo": str(cfg["photo"]),
           "n_frames": len(seq), "motion_cleaning": stats, "scene": {k: v for k, v in meta.items() if k != "source_frames"}}
    job["scene"]["source_frames"] = src
    (out / "job.json").write_text(json.dumps(job, indent=1, default=str))
    refb = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2BGR)
    tiles = [refb] + [cv2.addWeighted(refb, 0.45, cv2.cvtColor(draw(seq[k], (W, H)), cv2.COLOR_RGB2BGR), 1.0, 0)
                      for k in np.linspace(0, len(seq) - 1, 5).astype(int)]
    cv2.imwrite(str(out / "preview.jpg"), cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5))
    return job


if __name__ == "__main__":
    for n in sys.argv[1:] or list(JOBS):
        j = build(n, JOBS[n])
        print(n, j["n_frames"], json.dumps(j["motion_cleaning"])[:600])
