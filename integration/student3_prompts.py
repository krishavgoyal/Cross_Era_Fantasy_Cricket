"""Build a per-clip Student 3 config: SAM2 prompt boxes from the generation's own DensePose conditioning.

The base config (student3_improved/configs/1930s_archival.json) has boxes hand-placed for the
Bradman clip only. Every generated clip already knows where each body part is in frame 0: the
DensePose conditioning frame the generator used. So the boxes are read from that, plus the bat
line the engine drew (batsman).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# Student 3 class -> DensePose part ids
CLASS_PARTS = {
    "helmet": [23, 24],
    "skin": [3, 4, 19, 20, 21, 22],
    "shirt": [1, 2, 15, 16, 17, 18],
    "trousers": [7, 8, 9, 10],
    "pads": [11, 12, 13, 14],
    "shoes": [5, 6],
}


def parula_lut():
    v = (np.arange(25, dtype=np.float32) * 255.0 / 24.0).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(v, cv2.COLORMAP_PARULA).reshape(-1, 3)
    return bgr[:, ::-1].astype(np.int32)  # RGB, index = part id


def rgb_to_parts(rgb, max_dist=40):
    """PARULA control image (RGB) -> part ids; pixels not close to any part colour -> 0."""
    lut = parula_lut()[1:]
    px = rgb.reshape(-1, 3).astype(np.int32)
    d = ((px[:, None, :] - lut[None, :, :]) ** 2).sum(-1)
    idx = d.argmin(1)
    ok = (d[np.arange(len(px)), idx] <= max_dist ** 2) & (px.max(1) > 15)
    out = np.where(ok, idx + 1, 0).astype(np.uint8)
    return out.reshape(rgb.shape[:2])


def _box(mask, w, h, grow=0.04):
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None
    x1, x2 = np.percentile(xs, 1), np.percentile(xs, 99)
    y1, y2 = np.percentile(ys, 1), np.percentile(ys, 99)
    gx, gy = (x2 - x1) * grow + 2, (y2 - y1) * grow + 2
    return [round(float(np.clip((x1 - gx) / w, 0, 1)), 3), round(float(np.clip((y1 - gy) / h, 0, 1)), 3),
            round(float(np.clip((x2 + gx) / w, 0, 1)), 3), round(float(np.clip((y2 + gy) / h, 0, 1)), 3)]


def parts_in_generated_frame(parts_512, sampling):
    """Exact part ids for one DensePose frame, cropped/resized exactly like the generation engine did."""
    x1, y1, x2, y2 = sampling["crop_in_padded_px"]
    pad = sampling["pad_px"]
    gw, gh = sampling["generated_size"]
    p = np.pad(parts_512, pad) if pad > 0 else parts_512
    return cv2.resize(p[y1:y2, x1:x2], (gw, gh), interpolation=cv2.INTER_NEAREST)


def build_config(base_config: Path, conditioning_npz: Path, sidecar: Path, out_path: Path,
                 parts_npz: Path | None = None) -> dict:
    cfg = json.load(open(base_config))
    side = json.load(open(sidecar))
    first_key = side["output_frame_to_keyframe"][0]
    if parts_npz is not None and Path(parts_npz).exists():
        # exact: real DensePose part ids of the first sampled frame
        src = side["sampling"]["sampled_source_frames"][first_key]
        parts = parts_in_generated_frame(np.load(parts_npz)["part_index"][src], side["sampling"])
        cfg["_prompt_source"] = f"densepose_real_parts.npz frame {src}"
    else:
        # fallback: decode the PARULA colours of the conditioning frame (less exact)
        parts = rgb_to_parts(np.load(conditioning_npz)["frames"][first_key])
        cfg["_prompt_source"] = "conditioning frame colours"
    h, w = parts.shape[:2]

    prompts = {}
    player = _box(parts > 0, w, h, grow=0.06)
    if player:
        prompts["player"] = {"box": player}
        cfg["fallback_player_box"] = player
    for cls, ids in CLASS_PARTS.items():
        if cls == "pads" and side["role"] != "batsman":
            continue
        b = _box(np.isin(parts, ids), w, h)
        if b:
            prompts[cls] = {"box": b}
    bats = side.get("sampling", {}).get("bat_lines_generated_px")
    if side["role"] == "batsman" and bats:
        gx, gy, tx, ty = bats[first_key]
        m = np.zeros((h, w), np.uint8)
        cv2.line(m, (gx, gy), (tx, ty), 1, 12)
        b = _box(m > 0, w, h, grow=0.0)
        if b:
            prompts["bat"] = {"box": b}

    cfg["name"] = f"{cfg.get('name', 'student3')}_auto_{side['role']}"
    cfg["sam2_prompts"] = prompts
    cfg["expected_width"], cfg["expected_height"] = int(w), int(h)
    cfg["expected_fps"] = float(side["export_fps"])
    cfg["_generated_by"] = "integration/student3_prompts.py from the generation conditioning frame"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_path, "w"), indent=2)
    return cfg
