"""Shared helpers for the DensePose stage (no torch needed)."""
import json

import cv2
import numpy as np

PART_NAMES = {
    1: "torso_back", 2: "torso_front", 3: "right_hand", 4: "left_hand", 5: "left_foot", 6: "right_foot",
    7: "upper_leg_right_back", 8: "upper_leg_left_back", 9: "upper_leg_right_front", 10: "upper_leg_left_front",
    11: "lower_leg_right_back", 12: "lower_leg_left_back", 13: "lower_leg_right_front", 14: "lower_leg_left_front",
    15: "upper_arm_left_inside", 16: "upper_arm_right_inside", 17: "upper_arm_left_outside", 18: "upper_arm_right_outside",
    19: "lower_arm_left_inside", 20: "lower_arm_right_inside", 21: "lower_arm_left_outside", 22: "lower_arm_right_outside",
    23: "head_right", 24: "head_left",
}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def colorize(lab):
    """Part ids -> the PARULA-on-black control image used by every clip in this repo."""
    v = (lab.astype(np.float32) * 255.0 / 24.0).astype(np.uint8)
    img = cv2.applyColorMap(v, cv2.COLORMAP_PARULA)
    img[lab == 0] = 0
    return img


def expected_boxes(meta_path, size=512):
    """Tracked target box in crop (output) pixel coordinates, one per frame."""
    m = json.load(open(meta_path))
    out = []
    for f in m["frames"]:
        bx = f["bbox_source_xyxy_px"]
        cw = f["crop_window_source_xyxy_px"]
        k = size / (cw[2] - cw[0])
        out.append([(bx[0] - cw[0]) * k, (bx[1] - cw[1]) * k, (bx[2] - cw[0]) * k, (bx[3] - cw[1]) * k])
    return out
