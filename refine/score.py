"""Scores for one video against the skeleton it should follow and the reference photo it should look like.

    ~/gen/bin/python -m refine.score <video> <target_pose.json> <reference.png> [--out score.json]

Six groups (raw numbers, plus a 0-100 score per group so steps can be compared at a glance):

accuracy          PCK@0.1 of DWPose detections on the video vs the target skeleton (animate.evaluate.score)
limb_rigidity     per limb, detected length / target length per frame; its coefficient of variation (the
                  target already contains the true foreshortening, so a rigid limb keeps a constant ratio)
fluidity          detected-joint acceleration p99 / torso length; frame-to-frame colour flicker (change of the
                  frame's mean Lab colour) and background flicker (mean |difference| outside the player)
object_permanence frames with every limb joint present, frames with a face, player found in every frame,
                  outline (silhouette) drops below IoU 0.7
distortions       frames with two left/right joints merged; worst outline IoU; largest outline area jump
colour_realism    distance of the player's and the background's colours to the reference photo (mean Lab
                  delta E, histogram distance), share of "foreign" colours (pixels far from every colour of
                  the photo's palette), colour saturation vs the photo, gradient statistics vs the photo
                  (share of near-flat-but-banded steps)
"""
import argparse
import json

import cv2
import numpy as np

from animate.evaluate import LIMB_JOINTS, detect_video, merged_limbs, score as pck_score

LIMBS = [("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_elbow"),
         ("left_elbow", "left_wrist"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle")]
_YOLO = None


def yolo():
    global _YOLO
    if _YOLO is None:
        from pathlib import Path
        from ultralytics import YOLO
        _YOLO = YOLO(str(Path(__file__).resolve().parents[1] / "models" / "yolo26x-seg.pt"))
    return _YOLO


def read(path):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return fr


def player_mask(bgr):
    r = yolo().predict(bgr, classes=[0], conf=0.25, verbose=False, retina_masks=True, device="cpu")[0]
    if r.masks is None:
        return None
    ms = r.masks.data.cpu().numpy() > 0.5
    m = ms[int(np.argmax(ms.reshape(len(ms), -1).sum(1)))]
    if m.shape != bgr.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (bgr.shape[1], bgr.shape[0])) > 0
    return m


def lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def palette(ref_lab, mask, k=24):
    pts = ref_lab[mask].reshape(-1, 3)
    if len(pts) > 20000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 20000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(pts.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return centers


def foreign_share(lab_px, pal, thr=12.0):
    """Share of pixels whose colour is more than thr (delta E) from every palette colour of the photo."""
    if len(lab_px) == 0:
        return 0.0
    if len(lab_px) > 8000:
        lab_px = lab_px[np.random.default_rng(1).choice(len(lab_px), 8000, replace=False)]
    d = np.linalg.norm(lab_px[:, None, :] - pal[None], axis=2).min(1)
    return float((d > thr).mean())


def hist_dist(a, b, bins=16):
    """Chi-square-like distance between 3D Lab histograms (0 = same colours)."""
    rng = [[0, 100], [-60, 60], [-60, 60]]
    ha, _ = np.histogramdd(a, bins=bins, range=rng)
    hb, _ = np.histogramdd(b, bins=bins, range=rng)
    ha, hb = ha / max(ha.sum(), 1), hb / max(hb.sum(), 1)
    return float(0.5 * np.sum((ha - hb) ** 2 / (ha + hb + 1e-9)))


def banding(gray):
    """Share of small non-zero luminance steps that are exactly 1-2 levels (staircase gradients) among
    gentle-gradient pixels; smooth natural gradients have noise, banded ones have flat runs and steps."""
    g = gray.astype(np.int16)
    dx = np.abs(np.diff(g, axis=1))
    gentle = (dx <= 4)
    flat = (dx == 0) & gentle
    return float(flat.sum() / max(gentle.sum(), 1))


def run(video, target, ref, every=1):
    tg = json.load(open(target))
    W, H = tg["canvas"]
    frames = [cv2.resize(f, (W, H)) for f in read(video)]
    n = min(len(frames), len(tg["frames"]))
    frames = frames[:n]
    tframes = tg["frames"][:n]
    out = {"video": str(video), "frames": n, "size": [W, H]}
    # ---- detections
    pred = detect_video(video, (W, H))[:n]
    acc = pck_score(pred, tframes)
    # view-tolerant: per frame, the detection may be mirrored (DWPose cannot tell front from back on a
    # side-on body and then labels left as right); take the better of as-is / mirrored per frame
    sw = lambda j: j.replace("right", "@").replace("left", "right").replace("@", "left")
    mir = []
    for p, t in zip(pred, tframes):
        q = {sw(j): v for j, v in p.items()}
        a1 = pck_score([p], [t])["pck@0.1"]
        a2 = pck_score([q], [t])["pck@0.1"]
        mir.append(q if a2 > a1 else p)
    acc_m = pck_score(mir, tframes)
    strict_pred = pred
    pred = mir  # rigidity, fluidity and merged limbs on mirror-resolved detections: a detector that flips
    # front/back between frames would otherwise show up as huge "accelerations" and swapped limbs
    out["accuracy"] = {"pck@0.05": acc["pck@0.05"], "pck@0.1": acc["pck@0.1"], "pck@0.2": acc["pck@0.2"],
                       "pck@0.1_view_tolerant": acc_m["pck@0.1"],
                       "frames_detected_mirrored": int(sum(m is not p for m, p in zip(mir, pred)))}
    # ---- rigidity
    cvs = []
    for a, b in LIMBS:
        r = []
        for p, t in zip(pred, tframes):
            if a in p and b in p and t["body"].get(a) and t["body"].get(b):
                lt = np.hypot(*np.subtract(t["body"][a], t["body"][b]))
                if lt > 8:
                    r.append(np.hypot(*np.subtract(p[a], p[b])) / lt)
        if len(r) > 5:
            r = np.array(r)
            med = float(np.median(r))
            # robust CV (MAD): a single failed detection should not count as a stretched limb
            cvs.append(1.4826 * float(np.median(np.abs(r - med))) / max(med, 1e-6))
    out["limb_rigidity"] = {"length_ratio_robust_cv_median": round(float(np.median(cvs)), 3) if cvs else None,
                            "length_ratio_robust_cv_worst": round(float(np.max(cvs)), 3) if cvs else None}
    # ---- masks, colours
    masks = [player_mask(f) for f in frames]
    refb = cv2.resize(cv2.imread(str(ref)), (W, H))
    rmask = player_mask(refb)
    if rmask is None:
        rmask = np.zeros((H, W), bool)
    rl = lab(refb)
    pal_p = palette(rl, rmask) if rmask.sum() > 200 else palette(rl, np.ones_like(rmask))
    pal_b = palette(rl, ~rmask)
    ref_p, ref_b = rl[rmask], rl[~rmask]
    ref_sat = float(np.hypot(rl[..., 1], rl[..., 2]).mean())
    dE_p, dE_b, hd_p, hd_b, fo_p, fo_b, sat, band, means, bgdiff = [], [], [], [], [], [], [], [], [], []
    prev = None
    for k, (f, m) in enumerate(zip(frames, masks)):
        fl = lab(f)
        means.append(fl.reshape(-1, 3).mean(0))
        sat.append(float(np.hypot(fl[..., 1], fl[..., 2]).mean()))
        band.append(banding(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)))
        mm = m if m is not None else np.zeros((H, W), bool)
        if prev is not None:
            bg = ~(cv2.dilate(mm.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~prev[1]
            bgdiff.append(float(np.abs(f.astype(np.float32) - prev[0].astype(np.float32)).mean(2)[bg].mean()) if bg.any() else 0)
        prev = (f, cv2.dilate(mm.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
        if k % every:
            continue
        if mm.sum() > 200 and len(ref_p):
            pp = fl[mm]
            dE_p.append(float(np.linalg.norm(pp.mean(0) - ref_p.mean(0))))
            hd_p.append(hist_dist(pp[::4], ref_p[::4]))
            fo_p.append(foreign_share(pp, pal_p))
        bp = fl[~mm]
        dE_b.append(float(np.linalg.norm(bp.mean(0) - ref_b.mean(0))))
        hd_b.append(hist_dist(bp[::8], ref_b[::8]))
        fo_b.append(foreign_share(bp, pal_b))
    means = np.array(means)
    flick = np.linalg.norm(np.diff(means, axis=0), axis=1)
    # ---- fluidity
    P = np.full((n, 18, 2), np.nan)
    from animate.skeleton import BODY
    for t, p in enumerate(pred):
        for i, j in enumerate(BODY):
            if j in p:
                P[t, i] = p[j]
    torso = np.nanmedian([np.hypot(*np.subtract(t["body"]["neck"], np.mean([t["body"]["right_hip"], t["body"]["left_hip"]], 0)))
                          for t in tframes if t["body"].get("neck") and t["body"].get("right_hip") and t["body"].get("left_hip")])
    accn = np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=2) / torso
    out["fluidity"] = {"joint_accel_p90_per_torso": round(float(np.nanpercentile(accn, 90)), 3),
                       "joint_accel_spikes_share": round(float(np.nanmean(accn > 0.5)), 4),
                       "colour_flicker_mean_dE": round(float(flick.mean()), 3),
                       "colour_flicker_p99_dE": round(float(np.percentile(flick, 99)), 3),
                       "background_flicker_mean": round(float(np.mean(bgdiff)), 3) if bgdiff else None}
    # ---- permanence, distortions
    ious = []
    for a, b in zip(masks, masks[1:]):
        ious.append(0.0 if a is None or b is None else float((a & b).sum() / max((a | b).sum(), 1)))
    areas = [0 if m is None else int(m.sum()) for m in masks]
    out["object_permanence"] = {"complete_body_frames": acc["complete_body_frames"],
                                "face_frames": acc["face_detected_frames"],
                                "player_found_frames": round(sum(m is not None for m in masks) / n, 4),
                                "outline_drops_below_0.7": int(sum(i < 0.7 for i in ious))}
    mg = merged_limbs(pred, tframes)
    mg_strict = merged_limbs(strict_pred, tframes)
    out["distortions"] = {"merged_limb_frames": round(len(mg) / n, 4), "merged_limb_frames_strict": round(len(mg_strict) / n, 4),
                          "outline_iou_min": round(float(min(ious)), 4) if ious else None,
                          "outline_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
                          "outline_area_jump_max": round(float(max(abs(a - b) / max(a, b, 1) for a, b in zip(areas, areas[1:]))), 4)}
    out["colour_realism"] = {"player_mean_dE_to_photo": round(float(np.mean(dE_p)), 2) if dE_p else None,
                             "background_mean_dE_to_photo": round(float(np.mean(dE_b)), 2),
                             "player_hist_dist": round(float(np.mean(hd_p)), 3) if hd_p else None,
                             "background_hist_dist": round(float(np.mean(hd_b)), 3),
                             "player_foreign_colour_share": round(float(np.mean(fo_p)), 4) if fo_p else None,
                             "background_foreign_colour_share": round(float(np.mean(fo_b)), 4),
                             "saturation_video": round(float(np.mean(sat)), 2), "saturation_photo": round(ref_sat, 2),
                             "banding_flat_share_video": round(float(np.mean(band)), 3),
                             "banding_flat_share_photo": round(banding(cv2.cvtColor(refb, cv2.COLOR_BGR2GRAY)), 3)}
    out["scores_0_100"] = summary_scores(out)
    return out


def summary_scores(o):
    """0-100 per group (higher = better). Simple, documented mappings so steps can be compared; the raw
    numbers above are the real measurements."""
    c = lambda v: float(np.clip(v, 0, 100))
    a = o["accuracy"]["pck@0.1"] * 100
    r = c(100 - 300 * (o["limb_rigidity"]["length_ratio_robust_cv_median"] or 0.3))
    f = o["fluidity"]
    fl = c(100 - 200 * f["joint_accel_p90_per_torso"] - 300 * f["joint_accel_spikes_share"]
           - 20 * f["colour_flicker_mean_dE"] - 3 * (f["background_flicker_mean"] or 0))
    p = o["object_permanence"]
    pe = c(100 * (0.4 * p["complete_body_frames"] + 0.2 * p["face_frames"] + 0.2 * p["player_found_frames"])
           + 20 - 2 * p["outline_drops_below_0.7"])
    d = o["distortions"]
    di = c(100 - 300 * d["merged_limb_frames"] - 60 * (1 - (d["outline_iou_min"] or 0)) - 30 * d["outline_area_jump_max"])
    cr = o["colour_realism"]
    col = c(100 - 2.0 * (cr["player_mean_dE_to_photo"] or 0) - 1.5 * cr["background_mean_dE_to_photo"]
            - 100 * ((cr["player_foreign_colour_share"] or 0) + cr["background_foreign_colour_share"])
            - 0.5 * abs(cr["saturation_video"] - cr["saturation_photo"]))
    return {"accuracy": round(a, 1), "accuracy_view_tolerant": round(o["accuracy"]["pck@0.1_view_tolerant"] * 100, 1), "limb_rigidity": round(r, 1), "fluidity": round(fl, 1),
            "object_permanence": round(pe, 1), "distortions(no=100)": round(di, 1), "colour_realism": round(col, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("target")
    ap.add_argument("ref")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = run(a.video, a.target, a.ref)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
