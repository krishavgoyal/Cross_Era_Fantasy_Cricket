"""Build MusePose jobs where the player is tied to a real scene (see animate/scene.py) and the motion is cleaned
(animate/motion_clean.py).

    python -m animate.build_scene_jobs                  # all
    python -m animate.build_scene_jobs bradman_batting

Writes animate/jobs/<name>/: ref.png, pose.mp4 (body + hands, no bat line), target_pose.json (with the bat for
compositing), scene.json, job.json, preview.jpg. Old jobs in animate/jobs/ are not touched.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from animate import scene
from animate.motion_clean import clean
from animate.skeleton import draw

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "reference_players"
CLIPS = ROOT / "data" / "processed_clips"
OUT = ROOT / "animate" / "jobs"

JOBS = {
    "bradman_batting": dict(kind="batting", clip=CLIPS / "good_cover_drive_final",
                            photo=P / "don_bradman" / "bradman_original_repo.jpg"),
    "mcgrath_bowling": dict(kind="bowling", clip=CLIPS / "bowler_new_cummins",
                            photo=P / "glenn_mcgrath" / "mcgrath_delivery_stride_sportingheroes.jpg",
                            frames=(0, 52), search=(24, 40), pose_file="pose_data_merged.json",
                            stump_box=(212, 361, 230, 592),   # the stump in the photo canvas (px), colour + zoom
                            ball=(60, 305, 10),               # the ball in his right hand (cx, cy, r)
                            delivery=(30, 46), stretch=2.5, bowling_arm="right"),  # rebuilt delivery (animate/delivery.py)
}


def ball_path(seq, size, gravity=0.5, min_speed=22.0, side=None, release=None):
    """Ball position per frame: in the bowling hand until release, then a straight throw with a little drop.
    Bowling arm = the wrist that travels furthest vertically. Release = the frame after that wrist is highest
    (top of the arc), when it starts coming over. Speed comes from the hand's own motion at release, at least
    min_speed px/frame, always towards the direction the bowler is running."""
    W, H = size
    def hand(p, side):
        b = p["body"]
        w, e = b.get(f"{side}_wrist"), b.get(f"{side}_elbow")
        if w is None:
            return None
        if e is None:
            return w
        return (w[0] + 0.30 * (w[0] - e[0]), w[1] + 0.30 * (w[1] - e[1]))  # palm, just past the wrist
    if side is None:
        ys = {s_: [p["body"][f"{s_}_wrist"][1] for p in seq if f"{s_}_wrist" in p["body"]] for s_ in ("right", "left")}
        side = max(ys, key=lambda s_: (np.ptp(ys[s_]) if ys[s_] else -1))
    if release is None:
        tops = [(p["body"][f"{side}_wrist"][1], k) for k, p in enumerate(seq) if f"{side}_wrist" in p["body"]]
        release = min(min(tops)[1] + 1, len(seq) - 1) if tops else len(seq) // 2
    rel = int(min(max(release, 1), len(seq) - 1))
    hips = [hip_x for hip_x in ((sum(p["body"][j][0] for j in ("right_hip", "left_hip") if j in p["body"])
                                 / max(1, sum(j in p["body"] for j in ("right_hip", "left_hip")))) for p in seq)]
    run_dir = -1.0 if hips[-1] < hips[0] else 1.0
    path = [hand(p, side) if k <= rel else None for k, p in enumerate(seq)]
    h0 = path[rel] or path[rel - 1]
    prev = path[rel - 1] or h0
    v = np.array([h0[0] - prev[0], h0[1] - prev[1]], float)
    if v[0] * run_dir < min_speed:
        v[0] = run_dir * max(min_speed, abs(v[0]))
    v[1] = float(np.clip(v[1], -0.3 * abs(v[0]), 0.4 * abs(v[0])))
    pos = np.array(h0, float)
    for k in range(rel + 1, len(seq)):
        pos = pos + v
        v[1] += gravity
        path[k] = (float(pos[0]), float(pos[1])) if -20 < pos[0] < W + 20 and -20 < pos[1] < H + 20 else None
    return path, {"bowling_arm": side, "release_index": rel, "release_velocity_px_per_frame": [round(v[0], 1), round(v[1], 1)]}


def build(name, cfg):
    out = OUT / name
    if cfg["kind"] == "batting":
        ref, seq, meta = scene.batting_scene(cfg["clip"], cfg["photo"], out)
    else:
        ref, seq, meta = scene.bowling_scene(cfg["clip"], cfg["photo"], out, frames=cfg.get("frames"),
                                             search=cfg.get("search"), stump_box=cfg.get("stump_box"),
                                             ball=cfg.get("ball"), pose_file=cfg.get("pose_file", "pose_data.json"))
    seq, stats = clean(seq)
    W, H = ref.size
    rel = None
    if cfg.get("delivery"):
        from animate.delivery import rebuild
        hx = [np.mean([p["body"][j][0] for j in ("right_hip", "left_hip") if j in p["body"]]) for p in seq
              if any(j in p["body"] for j in ("right_hip", "left_hip"))]
        seq, times, dinfo = rebuild(seq, meta["source_frames"], cfg["delivery"], stretch=cfg.get("stretch", 2.5),
                                    side=cfg.get("bowling_arm", "right"), run_dir=-1.0 if hx[-1] < hx[0] else 1.0)
        meta["source_frames"] = times
        stats["delivery"] = dinfo
        rel = dinfo["release_index"]
    if cfg.get("ball"):
        path, info = ball_path(seq, (W, H), side=cfg.get("bowling_arm"), release=rel)
        for p, b in zip(seq, path):
            p["ball"] = b
        stats["ball"] = info
    # body permanence: start at the first frame where the whole body is on the canvas (a player half outside
    # the frame makes MusePose draw a partial body)
    def inside(p, m=6):
        return all(m <= x <= W - m and m <= y <= H - m for x, y in p["body"].values())
    # keep the longest unbroken run of frames with the whole body on the canvas
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
    meta["source_frames"] = meta["source_frames"][first:last]
    stats["kept_source_frames"] = [meta["source_frames"][0], meta["source_frames"][-1]]
    stats["frames_partly_outside_canvas"] = sum(1 for p in seq if not inside(p))
    vw = cv2.VideoWriter(str(out / "pose.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (W, H))
    for p in seq:
        vw.write(cv2.cvtColor(draw(p, (W, H), bat=False), cv2.COLOR_RGB2BGR))
    vw.release()
    ser = lambda q: None if q is None else [round(float(q[0]), 1), round(float(q[1]), 1)]
    (out / "target_pose.json").write_text(json.dumps({
        "canvas": [W, H], "fps": meta["fps"], "source_frames": meta["source_frames"],
        "frames": [{"body": {k: ser(v) for k, v in p["body"].items()},
                    "left_hand": [ser(q) for q in p["left_hand"]], "right_hand": [ser(q) for q in p["right_hand"]],
                    "bat": None if p.get("bat") is None else [ser(p["bat"][0]), ser(p["bat"][1])],
                    "ball": ser(p.get("ball"))} for p in seq]}))
    job = {"name": name, "kind": cfg["kind"], "clip": Path(cfg["clip"]).name, "photo": str(cfg["photo"]),
           "n_frames": len(seq), "motion_cleaning": stats, "scene": meta}
    (out / "job.json").write_text(json.dumps(job, indent=1, default=str))
    refb = cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2BGR)
    tiles = [refb] + [cv2.addWeighted(refb, 0.45, cv2.cvtColor(draw(seq[k], (W, H)), cv2.COLOR_RGB2BGR), 1.0, 0)
                      for k in np.linspace(0, len(seq) - 1, 5).astype(int)]
    cv2.imwrite(str(out / "preview.jpg"), cv2.resize(np.hstack(tiles), None, fx=0.5, fy=0.5))
    return job


if __name__ == "__main__":
    for n in sys.argv[1:] or list(JOBS):
        j = build(n, JOBS[n])
        extra = {k: j["scene"][k] for k in ("aligned_frame", "align_error_px", "player_scale") if k in j["scene"]}
        print(f"{n}: {j['n_frames']} frames, cleaning {j['motion_cleaning']}, {extra}")
