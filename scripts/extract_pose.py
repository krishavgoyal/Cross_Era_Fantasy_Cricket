"""Stage 1: extract the target player's whole-body pose from a cricket clip (CPU, rtmlib DWPose).

Usage:
  # 1) see who was detected on frame 0 (writes frame0_boxes.jpg and prints x,y,w,h per person)
  python scripts/extract_pose.py --video data/clips/batter.mp4 --out data/pose/batter --list-boxes
  # 2) track the chosen player through the clip
  python scripts/extract_pose.py --video data/clips/batter.mp4 --out data/pose/batter --init-box 410,120,180,460

Outputs in --out:
  raw.npz      kpts[T,133,2] (pixels), scores[T,133], fps, size=[w,h], boxes[T,4], lost[T]
  overlay.mp4  source frames with the skeleton and tracked box drawn (check this by eye!)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from pose_utils import detect_people, draw_openpose, iou, load_dwpose, masked, pose_in_box, read_video, write_video


def parse_box(text: str) -> np.ndarray:
    """'x,y,w,h' -> array [x1, y1, x2, y2]."""
    x, y, w, h = [float(v) for v in text.split(",")]
    return np.array([x, y, x + w, y + h], np.float32)


def box_from_pose(kpts: np.ndarray, scores: np.ndarray, thr: float = 0.3) -> np.ndarray | None:
    """Bounding box of confident body keypoints, padded 15% (used when the detector misses the player)."""
    pts = kpts[:17][scores[:17] > thr]
    if len(pts) < 4:
        return None
    (x1, y1), (x2, y2) = pts.min(0), pts.max(0)
    pad = 0.15 * max(x2 - x1, y2 - y1)
    return np.array([x1 - pad, y1 - pad, x2 + pad, y2 + pad], np.float32)


def pick_target(dets: np.ndarray, prev: np.ndarray) -> np.ndarray | None:
    """Detection that best matches the previous target box: highest IoU, else nearest centre."""
    if len(dets) == 0:
        return None
    ious = np.array([iou(d, prev) for d in dets])
    if ious.max() >= 0.1:
        return dets[int(ious.argmax())]
    centre = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    dist = np.array([np.linalg.norm(centre(d) - centre(prev)) for d in dets])
    if dist.min() < 0.75 * (prev[3] - prev[1]):          # within ~3/4 of the player's height
        return dets[int(dist.argmin())]
    return None


def list_boxes(model, frame: np.ndarray, out_dir: Path) -> None:
    """Save frame 0 with numbered person boxes and print their x,y,w,h."""
    dets = detect_people(model, frame)
    img = frame.copy()
    print(f"{len(dets)} people detected on frame 0:")
    for i, (x1, y1, x2, y2) in enumerate(dets):
        text = f"{int(x1)},{int(y1)},{int(x2 - x1)},{int(y2 - y1)}"
        print(f"  [{i}] --init-box {text}")
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        cv2.putText(img, f"[{i}] {text}", (int(x1), max(15, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    path = out_dir / "frame0_boxes.jpg"
    cv2.imwrite(str(path), img)
    print(f"Saved {path}. Pick the player's box and re-run with --init-box x,y,w,h")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="trimmed clip with the target player")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds (optional trim)")
    ap.add_argument("--end", type=float, default=None, help="end time in seconds (optional trim)")
    ap.add_argument("--init-box", help="target player box on the first frame: x,y,w,h (pixels)")
    ap.add_argument("--list-boxes", action="store_true", help="only show detections on frame 0 and exit")
    ap.add_argument("--det-every", type=int, default=1, help="run the detector every N frames (higher = faster)")
    ap.add_argument("--kpt-thr", type=float, default=0.3, help="confidence threshold for drawing the overlay")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video(args.video, args.start, args.end)
    h, w = frames[0].shape[:2]
    print(f"Loaded {len(frames)} frames, {w}x{h} @ {fps:.2f} fps")

    model = load_dwpose()  # first run downloads ~220 MB of ONNX models to ~/.cache/rtmlib
    if args.list_boxes:
        list_boxes(model, frames[0], out)
        return 0

    # ---- initial target box
    dets0 = detect_people(model, frames[0])
    if args.init_box:
        prev = parse_box(args.init_box)
        match = pick_target(dets0, prev)
        if match is not None:
            prev = match                                       # snap to the detector's box
    else:
        if len(dets0) == 0:
            raise SystemExit("No person detected on frame 0; pass --init-box x,y,w,h")
        areas = (dets0[:, 2] - dets0[:, 0]) * (dets0[:, 3] - dets0[:, 1])
        prev = dets0[int(areas.argmax())]
        print("WARNING: no --init-box given, tracking the LARGEST person on frame 0")

    # ---- track + pose every frame
    T = len(frames)
    kpts = np.zeros((T, 133, 2), np.float32)
    scores = np.zeros((T, 133), np.float32)
    boxes = np.zeros((T, 4), np.float32)
    lost = np.zeros(T, bool)
    t0 = time.time()
    for i, frame in enumerate(frames):
        box = prev
        if i % args.det_every == 0:
            match = pick_target(detect_people(model, frame), prev)
            if match is not None:
                box = match
            else:
                lost[i] = True                                 # detector missed: reuse last box
        k, s = pose_in_box(model, frame, box)
        kpts[i], scores[i], boxes[i] = k, s, box
        # next frame searches around where the player's body actually is
        follow = box_from_pose(k, s) if lost[i] else None
        prev = follow if follow is not None else box
        if i % 20 == 0:
            print(f"  frame {i}/{T}  ({(time.time() - t0) / (i + 1):.2f} s/frame)")

    np.savez_compressed(out / "raw.npz", kpts=kpts, scores=scores, fps=fps, size=np.array([w, h]),
                        boxes=boxes, lost=lost, video=str(args.video), start=args.start)

    # ---- overlay for visual checking
    overlay = []
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        draw_openpose(rgb, masked(kpts[i], scores[i], args.kpt_thr), stick=max(2, w // 240))
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255) if lost[i] else (0, 255, 0), 2)
        cv2.putText(img, f"{i}{'  LOST' if lost[i] else ''}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        overlay.append(img)
    write_video(out / "overlay.mp4", overlay, fps)

    body = scores[:, :17]
    print(f"Done in {time.time() - t0:.0f} s. Frames: {T}, detector-lost frames: {int(lost.sum())}")
    print(f"Median body keypoint score: {np.median(body):.2f}  (want >= 0.5)")
    print(f"Wrote {out / 'raw.npz'} and {out / 'overlay.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
