#!/usr/bin/env python3
"""Motion fidelity of a generated clip: does the generated player move like the reference?

Runs Student 1's DWPose backend on the generated video, maps the reference keypoints
(Student 1 pose_data.json) into generated-frame pixels with the generation sidecar
(<generated>.mp4.json), and reports:

  PCK@a   share of reference-visible body joints whose generated joint is within
          a * max(reference body box side) pixels. A joint DWPose misses counts as wrong.
  velocity cosine  mean cosine similarity between reference and generated joint
          displacement between consecutive keyframes (direction of motion agrees).

usage:
  python integration/eval_pck.py --generated outputs/run/generated.mp4 \
      --reference data/processed_clips/<clip>/pose_data.json --report outputs/run/pck.json
  (--pred-json reuses a DWPose json of the generated clip instead of running DWPose again)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BODY = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
        "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
        "right_eye", "left_eye", "right_ear", "left_ear"]
VIS = 0.30


def body_array(frame, w, h):
    pts = np.zeros((18, 2), np.float32)
    vis = np.zeros(18, bool)
    for j, k in enumerate(frame.get("body_keypoints", [])[:18]):
        pts[j] = (k["x"] * w, k["y"] * h)
        vis[j] = k.get("confidence", 0) >= VIS and not (k["x"] == 0 and k["y"] == 0)
    return pts, vis


def ref_to_generated(pts, sampling):
    """reference crop pixels (512 space) -> generated frame pixels, using the engine's recorded crop."""
    x1, y1, x2, y2 = sampling["crop_in_padded_px"]
    pad = sampling["pad_px"]
    gw, gh = sampling["generated_size"]
    out = pts.copy()
    out[:, 0] = (pts[:, 0] + pad - x1) * gw / (x2 - x1)
    out[:, 1] = (pts[:, 1] + pad - y1) * gh / (y2 - y1)
    return out


def compute_pck(ref, ref_vis, pred, pred_vis, alphas=(0.05, 0.1, 0.2)):
    """ref/pred: (T,J,2) px; *_vis: (T,J) bool. Returns dict with PCK per alpha, per joint at 0.1, counts."""
    T, J = ref_vis.shape
    scale = np.zeros(T, np.float32)
    for t in range(T):
        p = ref[t][ref_vis[t]]
        scale[t] = max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) if len(p) >= 2 else np.nan
    dist = np.linalg.norm(ref - pred, axis=-1)
    valid = ref_vis & ~np.isnan(scale)[:, None]
    out = {"evaluated_joints": int(valid.sum()), "missed_by_dwpose": int((valid & ~pred_vis).sum())}
    for a in alphas:
        hit = valid & pred_vis & (dist <= a * scale[:, None])
        out[f"pck@{a}"] = round(float(hit.sum() / max(1, valid.sum())), 4)
    hit10 = valid & pred_vis & (dist <= 0.1 * scale[:, None])
    out["per_joint_pck@0.1"] = {BODY[j] if J == 18 else str(j): (round(float(hit10[:, j].sum() / valid[:, j].sum()), 3)
                                if valid[:, j].sum() else None) for j in range(J)}
    out["per_frame_pck@0.1"] = [round(float(hit10[t].sum() / valid[t].sum()), 3) if valid[t].sum() else None
                                for t in range(T)]
    return out


def velocity_cosine(ref, ref_vis, pred, pred_vis, min_motion_px=2.0):
    sims = []
    for t in range(1, len(ref)):
        ok = ref_vis[t] & ref_vis[t - 1] & pred_vis[t] & pred_vis[t - 1]
        vr = ref[t] - ref[t - 1]
        vp = pred[t] - pred[t - 1]
        for j in np.where(ok)[0]:
            nr, np_ = np.linalg.norm(vr[j]), np.linalg.norm(vp[j])
            if nr >= min_motion_px and np_ > 1e-6:
                sims.append(float(vr[j] @ vp[j] / (nr * np_)))
    return (round(float(np.mean(sims)), 4) if sims else None), len(sims)


def evaluate(generated, reference, sidecar=None, pred_json=None, device="cpu", alphas=(0.05, 0.1, 0.2)):
    generated = Path(generated)
    side = json.load(open(sidecar or str(generated) + ".json"))
    s = side["sampling"]
    ref_pose = json.load(open(reference))
    rw, rh = ref_pose.get("resolution", [512, 512])
    gw, gh = s["generated_size"]

    if pred_json is None:
        sys.path.insert(0, str(ROOT))
        from pose_engine import extract_pose  # Student 1 backend, same detector as the reference
        pred_json = generated.with_suffix(".dwpose.json")
        extract_pose(generated, pred_json, device)
    pred_pose = json.load(open(pred_json))

    # one output frame per keyframe (first occurrence), so held/padded frames are not double counted
    o2k = side["output_frame_to_keyframe"]
    first = {}
    for o, k in enumerate(o2k):
        first.setdefault(k, o)
    keys = sorted(first)
    skip = set(side.get("replaced_broken_keyframes", []))
    keys = [k for k in keys if k not in skip]
    src = s["sampled_source_frames"]

    R, RV, P, PV = [], [], [], []
    for k in keys:
        rp, rv = body_array(ref_pose["frames"][src[k]], rw, rh)
        R.append(ref_to_generated(rp, s))
        RV.append(rv)
        pp, pv = body_array(pred_pose["frames"][first[k]], gw, gh)
        P.append(pp)
        PV.append(pv)
    R, RV, P, PV = map(np.array, (R, RV, P, PV))
    rep = compute_pck(R, RV, P, PV, alphas)
    rep["velocity_cosine"], rep["velocity_pairs"] = velocity_cosine(R, RV, P, PV)
    rep.update({
        "generated": str(generated), "reference": str(reference), "keyframes_evaluated": len(keys),
        "source_frames": [src[k] for k in keys], "threshold": "alpha * max side of reference body box (px)",
        "pose_backend": pred_pose["frames"][0].get("pose_backend") if pred_pose["frames"] else None,
        "note": "body joints only (18). Missed generated joints count as wrong. Broken keyframes replaced by the engine are skipped.",
    })
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--reference", required=True, help="Student 1 pose_data.json of the same clip")
    ap.add_argument("--sidecar", default=None, help="default: <generated>.json written by the generation engine")
    ap.add_argument("--pred-json", default=None, help="existing DWPose json of the generated clip")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    rep = evaluate(a.generated, a.reference, a.sidecar, a.pred_json, a.device)
    txt = json.dumps(rep, indent=1)
    if a.report:
        Path(a.report).write_text(txt)
    print({k: rep[k] for k in ("pck@0.05", "pck@0.1", "pck@0.2", "velocity_cosine", "keyframes_evaluated")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
