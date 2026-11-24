"""Ground reference: per-frame 2x3 matrices mapping a processed clip's source_512 pixels into one fixed "world"
image (the first frame of the camera), measured on the background (pitch, crease, boards) with the players,
bats, black bars and burned-in captions masked out (animate.scene.track_camera).

After this, a point on the ground has the same world coordinates in every frame, so a planted foot must stay
still: that is what the 3D fit (motion3d/fit.py) holds the feet to.

    ~/gen/bin/python -m motion3d.camera <clip_dir> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

from animate.scene import crop_windows, track_camera

# burned-in overlays in the Cummins screen recording (caption band, corner logos): they move with the screen
IGNORE = {"bowler_new_cummins": [(0.0, 0.30, 1.0, 0.55), (0.9, 0.0, 1.0, 0.12), (0.9, 0.85, 1.0, 1.0)]}


def run(clip_dir, out):
    clip_dir = Path(clip_dir)
    meta = json.load(open(clip_dir / "metadata.json"))
    raw = Path(meta["source"]["path"])
    if raw.exists():
        # track on the raw clip (the 512 crop window moves with the bowler, the captions are fixed on screen)
        mats_raw, inl = track_camera(raw, ignore=IGNORE.get(clip_dir.name, []), bottom_bar=False)
        wins = crop_windows(clip_dir)
        mats = []
        for k, fr in enumerate(meta["frames"]):
            x0, y0, x1, y1 = wins[k]
            C = np.array([[(x1 - x0) / 512, 0, x0], [0, (y1 - y0) / 512, y0], [0, 0, 1]])  # 512 crop -> raw
            M = np.vstack([mats_raw[min(fr["source_frame_index"], len(mats_raw) - 1)], [0, 0, 1]]) @ C
            mats.append(M[:2])
        world = "raw clip frame 0 pixels"
    else:
        mats, inl = track_camera(clip_dir / "source_512.mp4")
        world = "source_512 frame 0 pixels"
    zoom = [float(np.sqrt(abs(np.linalg.det(np.asarray(m)[:, :2])))) for m in mats]
    rec = {"clip": clip_dir.name, "world": world, "matrices": [np.asarray(m, float).round(5).tolist() for m in mats],
           "inliers": [int(i) for i in inl], "zoom_range": [round(min(zoom), 3), round(max(zoom), 3)]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec))
    return {"frames": len(mats), "zoom_range": rec["zoom_range"], "min_inliers": min(inl[1:]) if len(inl) > 1 else None}


if __name__ == "__main__":
    print(run(sys.argv[1], sys.argv[2]))
