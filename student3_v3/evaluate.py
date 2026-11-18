"""Student 3 v3 evaluation: technical fidelity, colour, temporal stability, mask coverage and PCK.

Run on the colourised *master* (same frames and fps as the generated input). Era-styled variants are
retimed and deliberately degraded, so luminance-fidelity checks do not apply to them.

PCK needs keypoints for the generated video. Produce them with Student 1's own estimator so the
backend matches the reference (README section 7):
    python pose_engine.py --input <generated.mp4> --output <generated_pose.json>
then pass --generated-pose and --reference-pose to the CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from student3_improved.evaluate import _local_ssim
from student3_improved.postprocess import read_video

from .student2_geometry import Student2Geometry, source_xy_to_generated

OPENPOSE_18 = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist",
               "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]
ARM_JOINTS = {"right": ["right_shoulder", "right_elbow", "right_wrist"], "left": ["left_shoulder", "left_elbow", "left_wrist"]}


def colourfulness(frame_bgr: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric M."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(frame_bgr)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def warping_error(prev_bgr: np.ndarray, cur_bgr: np.ndarray, region: np.ndarray | None = None) -> float:
    """Mean |chroma(cur) - warp(chroma(prev))| where forward/backward flow agree (lower = steadier colour)."""
    pl = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2LAB)
    cl = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2LAB)
    back = cv2.calcOpticalFlowFarneback(cl[..., 0], pl[..., 0], None, 0.5, 3, 19, 3, 5, 1.2, 0)
    fwd = cv2.calcOpticalFlowFarneback(pl[..., 0], cl[..., 0], None, 0.5, 3, 19, 3, 5, 1.2, 0)
    h, w = cl.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + back[..., 0], gy + back[..., 1]
    warped = cv2.remap(pl[..., 1:].astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb = np.linalg.norm(back + cv2.remap(fwd, mx, my, cv2.INTER_LINEAR), axis=2)
    valid = fb < 1.5
    if region is not None:
        valid &= region > 0
    if not valid.any():
        return float("nan")
    return float(np.abs(cl[..., 1:].astype(np.float32) - warped)[valid].mean())


def pck(ref: np.ndarray, pred: np.ndarray, ref_valid: np.ndarray, scale: np.ndarray, alpha: float) -> float:
    """Fraction of valid reference joints whose predicted joint lies within alpha * scale (per frame)."""
    dist = np.linalg.norm(ref - pred, axis=-1)
    ok = ref_valid & np.isfinite(dist)
    if not ok.any():
        return float("nan")
    return float((dist[ok] <= alpha * np.broadcast_to(scale[:, None], dist.shape)[ok]).mean())


def _keypoints(frame: dict[str, Any], key: str, w: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    pts = {p["joint"]: p for p in frame.get(key, [])}
    xy = np.full((len(OPENPOSE_18), 2), np.nan, np.float32)
    conf = np.zeros(len(OPENPOSE_18), np.float32)
    for j, name in enumerate(OPENPOSE_18):
        if name in pts:
            xy[j] = (pts[name]["x"] * w, pts[name]["y"] * h)
            conf[j] = pts[name].get("confidence", 0.0)
    return xy, conf


def compare_pose(reference_pose: dict[str, Any], generated_pose: dict[str, Any], geom: Student2Geometry, layout: list[int],
                 ref_size: tuple[int, int] = (512, 512), gen_size: tuple[int, int] = (384, 512), alphas=(0.05, 0.1, 0.2)) -> dict[str, Any]:
    """PCK of generated keypoints against Student 1's raw reference keypoints, in generated-frame pixels.

    ``layout[i]`` = keyframe index shown at output frame i; keyframe k came from source frame
    ``geom.sampled_source_indices[k]``. Only the first output frame of each keyframe is scored.
    """
    ref_frames = {f["frame_index"]: f for f in reference_pose["frames"]}
    gen_frames = generated_pose["frames"]
    seen: set[int] = set()
    refs, preds, valid = [], [], []
    for out_idx, k in enumerate(layout):
        if k in seen or out_idx >= len(gen_frames):
            continue
        seen.add(k)
        src = geom.sampled_source_indices[k]
        if src not in ref_frames:
            continue
        rxy, rconf = _keypoints(ref_frames[src], "raw_body_keypoints", *ref_size)
        rx, ry = source_xy_to_generated(rxy[:, 0], rxy[:, 1], geom)
        pxy, _ = _keypoints(gen_frames[out_idx], "raw_body_keypoints" if "raw_body_keypoints" in gen_frames[out_idx] else "body_keypoints", *gen_size)
        refs.append(np.stack([rx, ry], axis=1))
        preds.append(pxy)
        valid.append(rconf > 0.5)
    if not refs:
        return {"status": "no_overlapping_frames"}
    ref_a, pred_a, valid_a = np.stack(refs), np.stack(preds), np.stack(valid)
    neck, rhip, lhip = OPENPOSE_18.index("neck"), OPENPOSE_18.index("right_hip"), OPENPOSE_18.index("left_hip")
    torso = np.linalg.norm(ref_a[:, neck] - 0.5 * (ref_a[:, rhip] + ref_a[:, lhip]), axis=1)
    torso = np.where(np.isfinite(torso) & (torso > 1), torso, np.nanmedian(torso))
    result: dict[str, Any] = {"frames_scored": len(refs), "normaliser": "torso length (neck to mid-hip) in reference"}
    for a in alphas:
        result[f"pck@{a}"] = round(pck(ref_a, pred_a, valid_a, torso, a), 4)
    # Bowling arm = the arm whose wrist travels furthest vertically in the reference.
    travel = {}
    for side, joints in ARM_JOINTS.items():
        wy = ref_a[:, OPENPOSE_18.index(joints[2]), 1]
        travel[side] = float(np.nanmax(wy) - np.nanmin(wy)) if np.isfinite(wy).any() else 0.0
    arm = max(travel, key=travel.get)
    idx = [OPENPOSE_18.index(j) for j in ARM_JOINTS[arm]]
    result["active_arm"] = arm
    for a in alphas:
        result[f"active_arm_pck@{a}"] = round(pck(ref_a[:, idx], pred_a[:, idx], valid_a[:, idx], torso, a), 4)
    return result


def evaluate(input_path: Path, output_path: Path, layout: list[int], key_masks: list[dict[str, np.ndarray]],
             class_names: list[str]) -> dict[str, Any]:
    before, bspec = read_video(input_path)
    after, aspec = read_video(output_path)
    checks = {
        "resolution_preserved": (bspec.width, bspec.height) == (aspec.width, aspec.height),
        "frame_count_preserved": bspec.frame_count == aspec.frame_count,
        "fps_preserved": abs(bspec.fps - aspec.fps) < 0.05,
    }
    bl = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[..., 0] for f in before]
    al = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[..., 0] for f in after]
    err = [np.abs(a.astype(np.float32) - b.astype(np.float32)) for a, b in zip(bl, al)]
    mae = float(np.mean([e.mean() for e in err]))
    mse = float(np.mean([np.mean(e * e) for e in err]))
    ssim = float(np.mean([_local_ssim(a, b) for a, b in zip(bl, al)]))
    checks["luminance_mae_within_3"] = mae <= 3.0
    checks["luminance_ssim_at_least_0_98"] = ssim >= 0.98

    # Mask coverage on keyframes: how much of the player got a specific class (not just the base tint).
    coverage, class_area = [], {c: [] for c in class_names}
    for m in key_masks:
        player = m["player"] > 0
        if not player.any():
            continue
        assigned = np.zeros_like(player)
        for c in class_names:
            if c in m and c != "player":
                cm = (m[c] > 0) & player
                assigned |= cm
                class_area[c].append(float(cm.sum() / player.sum()))
        coverage.append(float(assigned.sum() / player.sum()))

    # Temporal: hold frames must be colour-identical; distinct keyframes scored by flow warping error.
    hold_delta, warp_err = [], []
    for i in range(1, len(after)):
        a_prev = cv2.cvtColor(after[i - 1], cv2.COLOR_BGR2LAB)[..., 1:].astype(np.float32)
        a_cur = cv2.cvtColor(after[i], cv2.COLOR_BGR2LAB)[..., 1:].astype(np.float32)
        if layout[i] == layout[i - 1]:
            hold_delta.append(float(np.abs(a_cur - a_prev).mean()))
        else:
            warp_err.append(warping_error(after[i - 1], after[i], key_masks[layout[i]]["player"]))
    return {
        "input_spec": bspec.__dict__,
        "output_spec": aspec.__dict__,
        "checks": checks,
        "luminance": {"mean_absolute_error": round(mae, 4), "psnr_db": round(10 * np.log10(255 ** 2 / max(mse, 1e-9)), 3), "mean_local_ssim": round(ssim, 5)},
        "colourfulness_hasler_suesstrunk": {"input": round(float(np.mean([colourfulness(f) for f in before])), 3),
                                             "output": round(float(np.mean([colourfulness(f) for f in after])), 3)},
        "mask_coverage": {"player_pixels_with_a_class_mean": round(float(np.mean(coverage)), 4) if coverage else None,
                          "class_share_of_player_mean": {c: round(float(np.mean(v)), 4) for c, v in class_area.items() if v}},
        "temporal": {"hold_frame_chroma_delta_mean": round(float(np.mean(hold_delta)), 4) if hold_delta else None,
                     "keyframe_player_warping_error_mean": round(float(np.nanmean(warp_err)), 4) if warp_err else None,
                     "keyframe_player_warping_error_max": round(float(np.nanmax(warp_err)), 4) if warp_err else None},
        "technical_verdict": "pass" if all(checks.values()) else "review",
        "visual_quality_verdict": "requires_human_review",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PCK between Student 1 reference keypoints and DWPose on a generated clip.")
    parser.add_argument("--processing", required=True, type=Path, help="processing.json written by student3_v3.run")
    parser.add_argument("--reference-pose", required=True, type=Path, help="Student 1 pose_data.json for the clip")
    parser.add_argument("--generated-pose", required=True, type=Path, help="pose_engine.py output on the generated video")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    proc = json.loads(args.processing.read_text(encoding="utf-8"))
    geom = Student2Geometry(**{k: (tuple(v) if k.startswith("crop") or k == "target_size" else v) for k, v in proc["student2_geometry"].items()})
    result = compare_pose(json.loads(args.reference_pose.read_text()), json.loads(args.generated_pose.read_text()), geom, proc["frame_layout"]["keyframe_of_output_frame"])
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
