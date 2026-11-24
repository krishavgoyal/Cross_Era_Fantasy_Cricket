"""Depth control video for VACE (Depth Anything V2 Small, CPU), aligned frame-by-frame with make_control.py.

Why: a pose skeleton carries no bat, no clothing outline and no body volume. A depth map of the modern clip gives
VACE the bat and the body shape, while the reference photo still supplies the appearance.

Alignment: this script calls the SAME functions as make_control.py (clean_resampled -> canvas_transform) with the
same settings, so output frame i shows source frame round(sample_pos[i]) placed with the same scale/offset as the
skeleton in control_<N>.npz. Use identical --start/--duration/--frames/--height-frac/--top-margin in both scripts.

Depth handling
  * Depth Anything V2 outputs relative inverse depth: larger = nearer. We map near = bright, far = dark.
  * ONE global min/max over all frames (no per-frame normalisation) so the brightness does not pump/flicker.
  * --soften-head flattens and blurs the head region (from the head keypoints) so the modern helmet's shape
    (peak, grille) does not override the reference face; the region keeps roughly the right distance.
  * Areas outside the source frame (after the canvas placement) are black (= far).
  * --target-only keeps ONLY the tracked player (person mask per frame around his tracked box from raw.npz):
    other people (e.g. fielders' legs at the top of the frame) and the ground become flat black, so VACE does not
    turn them into extra players. The player's depth is normalised over his own pixels to 0.15-1.0.

Usage
  python scripts/make_depth_control.py --video data/clips/C2_coverdrive.mp4 --pose data/pose/batter/raw.npz ^
      --out data/pose/batter --start 0 --duration 5.6 --top-margin 0.08 --soften-head

Outputs (in --out)
  depth_33.mp4              the depth control (grayscale, 480x832, 16 fps) - goes to Kaggle, NO label (it is an input)
  depth_33.npz              the matching skeleton keypoints in canvas pixels (use as --control for PCK)
  depth_preview.mp4         source | pose | depth side by side (labelled) - check alignment here
  depth_align_check.jpg     depth with the skeleton drawn on top at 4 frames - the lines must sit on the body
  depth_head_check.jpg      (with --soften-head) head close-ups, top row before / bottom row after softening
Model: depth-anything/Depth-Anything-V2-Small-hf (Apache-2.0), downloaded by transformers on first use (~100 MB).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from era_filters import burn_label
from make_control import canvas_transform, apply_transform, clean_resampled, transform_matrix
from pose_utils import L_SHO, NOSE, R_SHO, draw_openpose, read_video, write_video

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthModel:
    """Depth Anything V2 Small on CPU. __call__(frame_bgr) -> relative inverse depth (H, W) float32, larger = nearer."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch, self.device = torch, device
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        inputs = self.proc(images=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            pred = self.model(**inputs).predicted_depth                # (1, h', w')
        pred = self.torch.nn.functional.interpolate(pred[:, None], size=(h, w), mode="bicubic", align_corners=False)
        return pred[0, 0].cpu().numpy().astype(np.float32)


def soften_head(depth: np.ndarray, k133: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Flatten + blur the head region of one normalised depth frame (canvas coordinates).

    Head size hn = neck-to-nose distance. The region is an ellipse centred a little above the nose
    (0.75 hn wide, 1.0 hn tall semi-axes), feathered. Inside it, depth becomes 70% the region's mean + 30% a heavy
    blur: the head stays at the right distance but loses the helmet's peak/grille detail.
    """
    nose, neck = k133[NOSE], (k133[L_SHO] + k133[R_SHO]) / 2
    if not (np.all(np.isfinite(nose)) and np.all(np.isfinite(neck))):
        return depth
    hn = float(np.linalg.norm(neck - nose))
    if hn < 2:
        return depth
    h, w = depth.shape
    ell = np.zeros((h, w), np.float32)
    cv2.ellipse(ell, (int(nose[0]), int(nose[1] - 0.3 * hn)), (int(0.75 * hn), int(1.0 * hn)), 0, 0, 360, 1.0, -1)
    region = depth[ell > 0.5]
    if region.size == 0:
        return depth
    # Only touch the HEAD itself (foreground), never the background around it: otherwise the flattening paints a
    # bright round halo into the sky that VACE can turn into a bald dome or an extra ball. The silhouette is kept.
    p20, p80 = np.percentile(region, [20, 80])
    fg = (depth > (p20 + p80) / 2) if p80 - p20 > 0.08 else np.ones_like(depth, bool)
    mask = cv2.GaussianBlur(ell, (0, 0), max(1.0, 0.15 * hn)) * fg.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.0) * fg * strength          # soft inside, hard at the silhouette
    inside = (mask > 0.5)
    if not inside.any():
        return depth
    flat = 0.7 * float(depth[inside].mean()) + 0.3 * cv2.GaussianBlur(depth, (0, 0), max(1.0, 0.4 * hn))
    return depth * (1 - mask) + flat * mask


def body_capsule(kpts: np.ndarray, scores: np.ndarray, shape: tuple[int, int], thr: float = 0.3) -> np.ndarray:
    """Soft mask of the area around one person's skeleton (source pixels): limb capsules + head disc + hands/feet."""
    h, w = shape
    cap = np.zeros((h, w), np.uint8)
    ok = scores > thr
    body = kpts[:17][ok[:17]]
    if len(body) < 4:
        return np.ones((h, w), np.float32)                     # too few points to trust: do not cut anything
    size = float(body[:, 1].max() - body[:, 1].min())          # rough body height in pixels
    thick = max(10, int(0.16 * size))
    limbs = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (15, 17), (15, 19), (16, 20), (16, 22)]
    for a, b in limbs:
        if ok[a] and ok[b]:
            cv2.line(cap, tuple(int(v) for v in kpts[a]), tuple(int(v) for v in kpts[b]), 1, thick)
    torso = [i for i in (5, 6, 12, 11) if ok[i]]
    if len(torso) >= 3:
        cv2.fillConvexPoly(cap, kpts[torso].astype(np.int32), 1)
    if ok[0] and ok[5] and ok[6]:                               # head disc around/above the nose
        neck = (kpts[5] + kpts[6]) / 2
        r = int(1.1 * np.linalg.norm(neck - kpts[0]))
        cv2.circle(cap, (int(kpts[0][0]), int(kpts[0][1] - 0.3 * r)), max(8, r), 1, -1)
    for i in range(91, 133):                                    # fingers (the ball hand!)
        if ok[i]:
            cv2.circle(cap, (int(kpts[i][0]), int(kpts[i][1])), max(6, thick // 3), 1, -1)
    return cv2.GaussianBlur(cap.astype(np.float32), (0, 0), max(2.0, thick / 6))


class TargetMasker:
    """Mask (source resolution, 0..1) of ONLY the tracked player in one frame.

    mode "seg": rembg u2net_human_seg on a crop around the tracked box (box grown by `pad`), then keep only the
                connected blobs that contain one of the player's confident body keypoints -> drops other people
                (e.g. fielders' legs) that reach into the crop.
    mode "box": the tracked box itself, grown by `pad` and feathered (no segmentation model needed).
    """

    def __init__(self, mode: str = "seg", pad: float = 0.15):
        self.mode, self.pad = mode, pad
        if mode == "seg":
            from rembg import new_session
            try:
                self.session = new_session("u2net_human_seg")           # person segmentation (downloaded once)
            except Exception as e:                                     # offline / download reset: use cached u2net
                print(f"WARNING: u2net_human_seg unavailable ({type(e).__name__}); using u2net (salient object). "
                      "The keypoint blob filter still keeps only the tracked player.")
                self.session = new_session("u2net")

    def __call__(self, frame: np.ndarray, box: np.ndarray, kpts: np.ndarray, scores: np.ndarray, thr: float = 0.3) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        px, py = self.pad * (x2 - x1), self.pad * (y2 - y1)
        X1, Y1, X2, Y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))
        mask = np.zeros((h, w), np.float32)
        if self.mode == "box":
            mask[Y1:Y2, X1:X2] = 1.0
            return cv2.GaussianBlur(mask, (0, 0), 3)
        from PIL import Image
        from rembg import remove
        crop = cv2.cvtColor(frame[Y1:Y2, X1:X2], cv2.COLOR_BGR2RGB)
        alpha = np.asarray(remove(Image.fromarray(crop), session=self.session))[..., 3].astype(np.float32) / 255.0
        n, labels = cv2.connectedComponents((alpha > 0.5).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(alpha, bool)
        pts = kpts[:17][scores[:17] > thr]
        for x, y in pts:                                        # blobs that contain one of HIS keypoints
            cx, cy = int(round(x)) - X1, int(round(y)) - Y1
            if 0 <= cx < alpha.shape[1] and 0 <= cy < alpha.shape[0] and labels[cy, cx] > 0:
                keep |= labels == labels[cy, cx]
        if not keep.any() and n > 1:                            # no keypoint landed inside: keep the biggest blob
            areas = [(labels == i).sum() for i in range(1, n)]
            keep = labels == 1 + int(np.argmax(areas))
        keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        mask[Y1:Y2, X1:X2] = alpha * keep
        # Other people can TOUCH his silhouette in 2-D (fielders' legs above his head), so also require every
        # pixel to lie near HIS skeleton: thick capsules along his bones plus a head disc.
        return mask * body_capsule(kpts, scores, (h, w), thr)


def save_head_check(before: list[np.ndarray], after: list[np.ndarray], k: np.ndarray, path: Path) -> None:
    """Head close-ups at 4 frames: top row before softening, bottom row after."""
    rows = [[], []]
    for t in np.linspace(0, len(before) - 1, 4).astype(int):
        nose, neck = k[t, NOSE], (k[t, L_SHO] + k[t, R_SHO]) / 2
        if not (np.all(np.isfinite(nose)) and np.all(np.isfinite(neck))):
            continue
        hn = max(8.0, float(np.linalg.norm(neck - nose)))
        x0, y0 = int(nose[0] - 2 * hn), int(nose[1] - 2.2 * hn)
        for r, d in enumerate((before[t], after[t])):
            pad = np.pad(d, int(3 * hn))
            crop = pad[y0 + int(3 * hn):y0 + int(3 * hn) + int(4 * hn), x0 + int(3 * hn):x0 + int(3 * hn) + int(4 * hn)]
            rows[r].append(cv2.resize((np.clip(crop, 0, 1) * 255).astype(np.uint8), (200, 200)))
    if rows[0]:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.vstack([np.hstack(rows[0]), np.hstack(rows[1])]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="the SAME clip that extract_pose.py processed")
    ap.add_argument("--pose", required=True, help="raw.npz from extract_pose.py for that clip")
    ap.add_argument("--out", required=True)
    # --- must match make_control.py ---
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--height-frac", type=float, default=0.85)
    ap.add_argument("--top-margin", type=float, default=0.0)
    ap.add_argument("--thr", type=float, default=0.3)
    # --- depth options ---
    ap.add_argument("--soften-head", action="store_true", help="flatten/blur the head so the helmet shape is removed")
    ap.add_argument("--head-strength", type=float, default=1.0, help="0-1, how strongly the head is flattened")
    ap.add_argument("--target-only", action="store_true",
                    help="keep ONLY the tracked player (others and the ground become flat black = far)")
    ap.add_argument("--target-mask", choices=["seg", "box"], default="seg",
                    help="seg = rembg u2net_human_seg around his tracked box (default), box = the dilated box itself")
    ap.add_argument("--target-pad", type=float, default=0.15, help="grow the tracked box by this fraction for the mask")
    ap.add_argument("--name", default=None, help="output base name (default depth_<frames>)")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    # 1) identical timing + placement to make_control.py
    k_src, data, pos, duration = clean_resampled(args.pose, args.frames, args.fps, args.start, args.duration, args.thr)
    T = canvas_transform(k_src, args.width, args.height, args.height_frac, args.top_margin)
    M = transform_matrix(T)
    k_canvas = apply_transform(k_src, T)
    frames, vfps = read_video(args.video)
    if abs(vfps - float(data["fps"])) > 0.01 or len(frames) != data["kpts"].shape[0]:
        print(f"WARNING: video ({len(frames)} frames @ {vfps:g}) differs from raw.npz "
              f"({data['kpts'].shape[0]} @ {float(data['fps']):g}); was raw.npz made with --start/--end trimming?")
    idx = np.clip(np.round(pos).astype(int), 0, len(frames) - 1)
    print(f"{args.frames} frames from source frames {idx[0]}..{idx[-1]} ({args.start:.2f}-{args.start + duration:.2f} s), "
          f"scale {T[0]:.3f}")

    # 2) depth per frame (source resolution), placed on the canvas with the same affine as the skeleton
    model = DepthModel(args.model)
    masker = TargetMasker(args.target_mask, args.target_pad) if args.target_only else None
    raw, valid, srcs, masks = [], [], [], []
    for t, i in enumerate(idx):
        d = model(frames[i])
        if masker is not None:
            m = masker(frames[i], data["boxes"][i], data["kpts"][i], data["scores"][i], args.thr)
            masks.append(cv2.warpAffine(m, M, (args.width, args.height), flags=cv2.INTER_LINEAR, borderValue=0))
        raw.append(cv2.warpAffine(d, M, (args.width, args.height), flags=cv2.INTER_LINEAR, borderValue=0))
        valid.append(cv2.warpAffine(np.ones_like(d), M, (args.width, args.height), flags=cv2.INTER_NEAREST, borderValue=0) > 0.5)
        srcs.append(cv2.warpAffine(frames[i], M, (args.width, args.height), borderValue=(0, 0, 0)))
        if t % 8 == 0:
            print(f"  depth frame {t + 1}/{args.frames}")

    # 3) ONE global normalisation over all frames (no per-frame stretch -> no brightness pumping)
    if masker is None:
        allv = np.concatenate([r[v] for r, v in zip(raw, valid)])
        lo, hi = float(allv.min()), float(allv.max())
        print(f"Global depth range: {lo:.3f} .. {hi:.3f} (near = bright)")
        depth = [np.where(v, (r - lo) / max(hi - lo, 1e-6), 0.0).astype(np.float32) for r, v in zip(raw, valid)]
    else:
        # only the player: one global range over HIS pixels, mapped to 0.15..1 so even his farthest parts stay
        # visibly brighter than the flat black (= far) background; other people and the ground are removed
        allv = np.concatenate([r[(m > 0.5) & v] for r, v, m in zip(raw, valid, masks)])
        lo, hi = float(np.percentile(allv, 1)), float(np.percentile(allv, 99))
        print(f"Target-only depth range: {lo:.3f} .. {hi:.3f} (player mapped to 0.15-1.0, background 0)")
        depth = [(m * np.where(v, 0.15 + 0.85 * np.clip((r - lo) / max(hi - lo, 1e-6), 0, 1), 0.0)).astype(np.float32)
                 for r, v, m in zip(raw, valid, masks)]
    if args.soften_head:
        before = depth
        depth = [soften_head(d, k_canvas[t], args.head_strength) for t, d in enumerate(depth)]
        save_head_check(before, depth, k_canvas, Path(args.out) / "depth_head_check.jpg")
    depth8 = [cv2.cvtColor(np.clip(d * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for d in depth]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"depth_{args.frames}"
    write_video(out / f"{name}.mp4", depth8, args.fps)
    np.savez_compressed(out / f"{name}.npz", kpts=k_canvas.astype(np.float32), fps=args.fps,
                        size=np.array([args.width, args.height]), source_frames=idx, depth_range=np.array([lo, hi]),
                        top_margin=args.top_margin, soften_head=args.soften_head, target_only=args.target_only)

    # 4) previews: source | pose | depth (labelled), plus an alignment still
    panels = []
    for t in range(args.frames):
        pose = np.zeros((args.height, args.width, 3), np.uint8)
        draw_openpose(pose, k_canvas[t])
        panels.append(np.hstack([srcs[t], cv2.cvtColor(pose, cv2.COLOR_RGB2BGR), depth8[t]]))
    write_video(out / "depth_preview.mp4", burn_label(panels), args.fps)
    checks = []
    for t in np.linspace(0, args.frames - 1, 4).astype(int):
        vis = cv2.cvtColor(depth8[t], cv2.COLOR_BGR2RGB)
        draw_openpose(vis, k_canvas[t], hands=False, face=False, stick=2)
        checks.append(cv2.resize(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), (args.width // 2, args.height // 2)))
    cv2.imwrite(str(out / "depth_align_check.jpg"), np.hstack(checks))
    print(f"Wrote {out / (name + '.mp4')}, {out / (name + '.npz')}, {out / 'depth_preview.mp4'}, "
          f"{out / 'depth_align_check.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
