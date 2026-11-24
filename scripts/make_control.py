"""Stage 2: turn raw.npz into a VACE control video (OpenPose skeleton on black, 480x832, 16 fps, 33 frames).

Steps: mask low-confidence points -> fill short gaps -> Savitzky-Golay smoothing -> resample to 16 fps
       -> (optional) per-bone retarget to the reference photo -> scale/centre on the canvas -> draw.

Usage:
  python scripts/make_control.py --pose data/pose/batter/raw.npz --out data/pose/batter
  python scripts/make_control.py --pose data/pose/batter/raw.npz --out data/pose/batter --start 0.4 --ref-photo data/refs/tendulkar.png

Outputs in --out:
  control_33.mp4          the control video for VACE (exactly --frames frames at --fps)
  control_33.npz          kpts[N,133,2] in canvas pixels (NaN = missing), used later by eval_metrics.py
  control_33_preview.jpg  6 sample frames side by side, for a quick check
  ref_pose.jpg            (only with --ref-photo) the reference photo with its detected skeleton
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

from pose_utils import (FACE, L_ANK, L_ELB, L_HIP, L_SHO, L_WRI, LHAND, NOSE, R_ANK, R_ELB, R_HIP, R_SHO, R_WRI,
                        RHAND, biggest_person, draw_openpose, load_dwpose, masked, write_video)


# ------------------------------------------------------------------ cleaning
def _runs(mask: np.ndarray):
    """Yield (start, end) of consecutive True runs in a 1-D boolean array (end exclusive)."""
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            s = i
            while i < n and mask[i]:
                i += 1
            yield s, i
        else:
            i += 1


def fill_gaps(k: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN gaps of up to max_gap frames (longer gaps stay missing)."""
    out = k.copy()
    T = out.shape[0]
    for j in range(out.shape[1]):
        for c in range(2):
            x = out[:, j, c]
            for s, e in _runs(np.isnan(x)):
                if s > 0 and e < T and (e - s) <= max_gap:
                    x[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
    return out


def smooth(k: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay filter on each continuous (non-NaN) stretch of each coordinate."""
    if window < 3:
        return k
    window += (window + 1) % 2                      # must be odd
    out = k.copy()
    for j in range(out.shape[1]):
        for c in range(2):
            x = out[:, j, c]
            for s, e in _runs(np.isfinite(x)):
                if e - s >= window:
                    x[s:e] = savgol_filter(x[s:e], window, order)
    return out


def sample_positions(n_src: int, src_fps: float, start: float, duration: float, n: int) -> np.ndarray:
    """Fractional SOURCE frame index for each of the n output frames (shared by pose and depth controls)."""
    pos = (start + np.linspace(0.0, duration, n)) * src_fps
    if pos[-1] > n_src - 1:
        print(f"WARNING: clip has {n_src / src_fps:.2f} s but {start + duration:.2f} s requested; last frames are held")
    return np.clip(pos, 0, n_src - 1)


def resample(k: np.ndarray, src_fps: float, start: float, duration: float, n: int) -> np.ndarray:
    """Sample n frames evenly over [start, start+duration] seconds of the source (linear interpolation)."""
    T = k.shape[0]
    pos = sample_positions(T, src_fps, start, duration, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    a = (pos - lo)[:, None, None]
    out = (1 - a) * k[lo] + a * k[hi]
    # where interpolation hit a NaN, fall back to the nearest frame if that one is valid
    nearest = k[np.where(a[:, 0, 0] < 0.5, lo, hi)]
    bad = np.isnan(out) & np.isfinite(nearest)
    out[bad] = nearest[bad]
    return out


# ------------------------------------------------------------------ per-bone retarget
# Working skeleton: 17 COCO body points + virtual NECK (17) and ROOT (18, mid-hip).
NECK, ROOT = 17, 18
BONES = [(ROOT, L_HIP), (ROOT, R_HIP), (L_HIP, 13), (13, L_ANK), (R_HIP, 14), (14, R_ANK),
         (ROOT, NECK), (NECK, L_SHO), (NECK, R_SHO), (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI),
         (NECK, NOSE), (NOSE, 1), (NOSE, 2), (1, 3), (2, 4)]       # parents always come before children
BONE_INDEX = {child: b for b, (_, child) in enumerate(BONES)}


def with_virtual(body17: np.ndarray) -> np.ndarray:
    """(..., 17, 2) -> (..., 19, 2) adding neck (mid-shoulders) and root (mid-hips)."""
    neck = (body17[..., L_SHO, :] + body17[..., R_SHO, :]) / 2
    root = (body17[..., L_HIP, :] + body17[..., R_HIP, :]) / 2
    return np.concatenate([body17, neck[..., None, :], root[..., None, :]], axis=-2)


def bone_lengths(p19: np.ndarray) -> np.ndarray:
    """Lengths of BONES for (..., 19, 2) points (NaN where a joint is missing)."""
    return np.stack([np.linalg.norm(p19[..., c, :] - p19[..., p, :], axis=-1) for p, c in BONES], axis=-1)


def retarget(k: np.ndarray, ref_k133: np.ndarray) -> np.ndarray:
    """Give the driving skeleton the reference photo's body proportions, keeping its per-frame directions.

    Each bone is scaled by (reference length / median driving length). Using the ratio (not a fixed
    length) keeps 2-D foreshortening. Feet, face and hands move rigidly with their anchor joint.
    """
    drive = with_virtual(k[:, :17])                       # (N, 19, 2)
    ref = with_virtual(ref_k133[None, :17])[0]            # (19, 2)
    d_len = np.nanmedian(bone_lengths(drive), axis=0)     # (B,)
    r_len = bone_lengths(ref)                             # (B,)
    torso = BONE_INDEX[NECK]
    if not (np.isfinite(r_len[torso]) and np.isfinite(d_len[torso])):
        print("WARNING: torso not visible in reference or clip; skipping retarget")
        return k
    g = d_len[torso] / r_len[torso]                       # keep the overall size of the driving player
    ratio = np.where(np.isfinite(r_len) & (d_len > 1e-3), r_len * g / d_len, 1.0)
    ratio = np.clip(ratio, 0.6, 1.6)                      # guard against bad detections in the photo
    print("Retarget ratios: " + ", ".join(f"{a}->{b}:{r:.2f}" for (a, b), r in zip(BONES, ratio)))

    out = k.copy()
    for t in range(k.shape[0]):
        old = drive[t]
        new = np.full_like(old, np.nan)
        if not np.all(np.isfinite(old[ROOT])):
            continue                                      # hips missing: leave this frame unchanged
        new[ROOT] = old[ROOT]
        for b, (p, c) in enumerate(BONES):
            if np.all(np.isfinite(new[p])) and np.all(np.isfinite(old[c])) and np.all(np.isfinite(old[p])):
                new[c] = new[p] + (old[c] - old[p]) * ratio[b]
        body = new[:17]
        # rigid groups follow their anchor joint, scaled like the bone that leads to it
        groups = [([17, 18, 19], L_ANK), ([20, 21, 22], R_ANK), (FACE, NOSE), (LHAND, L_WRI), (RHAND, R_WRI)]
        frame = k[t].copy()
        for idx, anchor in groups:
            if np.all(np.isfinite(body[anchor])) and np.all(np.isfinite(k[t, anchor])):
                frame[idx] = body[anchor] + (k[t, idx] - k[t, anchor]) * ratio[BONE_INDEX[anchor]]
            else:
                frame[idx] = np.nan
        frame[:17] = body
        out[t] = frame
    return out


# ------------------------------------------------------------------ canvas placement
def canvas_transform(k: np.ndarray, width: int, height: int, height_frac: float, top_margin: float = 0.0):
    """One scale s and offset for the whole clip, so that canvas = (src - centre) * s + offset.

    Body ~height_frac of the canvas height, everything inside. With top_margin > 0 the content is placed in the
    region below top_margin*height, and the box includes an estimate of the top of the head (above the eyes/ears),
    so the head never touches the top edge. top_margin = 0 gives exactly the original behaviour.
    """
    body = k[:, :17]
    h_per_frame = np.nanmax(body[..., 1], axis=1) - np.nanmin(body[..., 1], axis=1)
    n_valid = np.isfinite(body[..., 1]).sum(axis=1)
    body_h = np.nanmedian(h_per_frame[n_valid >= 8]) if (n_valid >= 8).any() else np.nanmedian(h_per_frame)
    x0, y0 = np.nanmin(body[..., 0]), np.nanmin(body[..., 1])
    x1, y1 = np.nanmax(body[..., 0]), np.nanmax(body[..., 1])
    if top_margin <= 0:
        s = min(height_frac * height / body_h, 0.95 * width / (x1 - x0), 0.95 * height / (y1 - y0))
        centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        return float(s), centre, np.array([width / 2, height / 2], np.float32)
    # head top: about 0.9 x (neck-to-nose distance) above the nose
    neck = (k[:, L_SHO] + k[:, R_SHO]) / 2
    head = np.nanmedian(np.linalg.norm(neck - k[:, NOSE], axis=1))
    head_top = np.nanmin(k[:, NOSE, 1] - 0.9 * head) if np.isfinite(head) else y0
    y0 = min(y0, head_top)
    # toes/heels sit below the ankles and fingers beyond the wrists (a bowler's raised hand!): keep them inside too
    extra = np.concatenate([k[:, 17:23], k[:, 91:133]], axis=1)
    if np.isfinite(extra).any():
        x0, x1 = min(x0, np.nanmin(extra[..., 0])), max(x1, np.nanmax(extra[..., 0]))
        y0, y1 = min(y0, np.nanmin(extra[..., 1])), max(y1, np.nanmax(extra[..., 1]))
    avail = height * (1 - top_margin)                     # content area below the margin
    s = min(height_frac * avail / body_h, 0.95 * width / (x1 - x0), 0.97 * avail / (y1 - y0))
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
    return float(s), centre, np.array([width / 2, top_margin * height + avail / 2], np.float32)


def apply_transform(k: np.ndarray, T) -> np.ndarray:
    s, centre, offset = T
    return (k - centre) * s + offset


def transform_matrix(T) -> np.ndarray:
    """2x3 affine for cv2.warpAffine that maps SOURCE image pixels onto the canvas (same as apply_transform)."""
    s, (cx, cy), (ox, oy) = T
    return np.float32([[s, 0, ox - s * cx], [0, s, oy - s * cy]])


def fit_to_canvas(k: np.ndarray, width: int, height: int, height_frac: float, top_margin: float = 0.0) -> np.ndarray:
    """Backwards-compatible wrapper: transform computed from k and applied to k."""
    return apply_transform(k, canvas_transform(k, width, height, height_frac, top_margin))


def clean_resampled(pose_npz: str, frames: int, fps: float, start: float, duration: float | None, thr: float = 0.3,
                    max_gap: int = 4, sg_window: int = 7, sg_order: int = 2):
    """raw.npz -> cleaned keypoints resampled to `frames` output frames (SOURCE pixel coordinates).

    Returns (kpts[frames,133,2], data, sample_pos, duration). sample_pos = fractional source frame per output frame.
    """
    data = np.load(pose_npz)
    src_fps = float(data["fps"])
    k = masked(data["kpts"], data["scores"], thr)
    k = smooth(fill_gaps(k, max_gap), sg_window, sg_order)
    duration = duration if duration is not None else (frames - 1) / fps
    pos = sample_positions(k.shape[0], src_fps, start, duration, frames)
    return resample(k, src_fps, start, duration, frames), data, pos, duration


def imread_any(path: str) -> np.ndarray:
    """cv2.imread that also works with non-ASCII Windows paths."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", required=True, help="raw.npz from extract_pose.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=33, help="output frames (VACE needs 4n+1: 17, 33, 49, 81)")
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the clip where the control starts")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds of source to use (default (frames-1)/fps = real speed; larger = sped up)")
    ap.add_argument("--thr", type=float, default=0.3, help="keypoint confidence threshold")
    ap.add_argument("--max-gap", type=int, default=4, help="fill gaps up to this many source frames")
    ap.add_argument("--sg-window", type=int, default=7, help="Savitzky-Golay window (source frames, odd)")
    ap.add_argument("--sg-order", type=int, default=2)
    ap.add_argument("--height-frac", type=float, default=0.85, help="body height as a fraction of the canvas")
    ap.add_argument("--top-margin", type=float, default=0.0,
                    help="empty band at the top (fraction of height) so the head never touches the edge, e.g. 0.08")
    ap.add_argument("--name", default=None, help="output base name (default control_<frames>)")
    ap.add_argument("--ref-photo", help="reference photo: scale bone lengths to this person's proportions")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--no-hands", action="store_true")
    args = ap.parse_args()
    if (args.frames - 1) % 4:
        print(f"WARNING: --frames {args.frames} is not 4n+1; VACE may trim it")

    k, data, _, duration = clean_resampled(args.pose, args.frames, args.fps, args.start, args.duration, args.thr,
                                           args.max_gap, args.sg_window, args.sg_order)
    src_fps = float(data["fps"])
    print(f"Source {data['kpts'].shape[0]} frames @ {src_fps:.2f} fps -> {args.frames} frames "
          f"covering {args.start:.2f}-{args.start + duration:.2f} s")
    # canvas placement is computed from the SOURCE geometry (before retargeting) so that depth controls made by
    # make_depth_control.py with the same settings line up with this pose control frame by frame
    T = canvas_transform(k, args.width, args.height, args.height_frac, args.top_margin)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.ref_photo:
        model = load_dwpose()
        photo = imread_any(args.ref_photo)
        found = biggest_person(model, photo)
        if found is None:
            print("WARNING: no person found in the reference photo; skipping retarget")
        else:
            ref_k, ref_s = found
            vis = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
            draw_openpose(vis, masked(ref_k, ref_s, args.thr), stick=max(2, photo.shape[1] // 200))
            cv2.imwrite(str(out / "ref_pose.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            k = retarget(k, masked(ref_k, ref_s, args.thr))

    k = apply_transform(k, T)

    frames = []
    for t in range(args.frames):
        canvas = np.zeros((args.height, args.width, 3), np.uint8)
        draw_openpose(canvas, k[t], hands=not args.no_hands, face=not args.no_face)
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    name = args.name or f"control_{args.frames}"
    write_video(out / f"{name}.mp4", frames, args.fps)
    np.savez_compressed(out / f"{name}.npz", kpts=k.astype(np.float32), fps=args.fps,
                        size=np.array([args.width, args.height]))
    picks = np.linspace(0, args.frames - 1, 6).astype(int)
    preview = np.hstack([cv2.resize(frames[i], (args.width // 2, args.height // 2)) for i in picks])
    cv2.imwrite(str(out / f"{name}_preview.jpg"), preview)

    missing = np.isnan(k[:, :17, 0]).mean()
    print(f"Body keypoints missing after cleaning: {missing:.1%}")
    print(f"Wrote {out / (name + '.mp4')} ({args.frames} frames, {args.width}x{args.height} @ {args.fps:g} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
