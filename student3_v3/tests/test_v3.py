from __future__ import annotations

import cv2
import numpy as np
import pytest

from student3_v3 import densepose_parts as dpp
from student3_v3.colorize import colorize_keyframe, compose
from student3_v3.era_filters import apply_preset, retime
from student3_v3.evaluate import colourfulness, pck
from student3_v3.run import detect_frame_layout
from student3_v3.student2_geometry import compute_geometry, crop_like_student2, source_xy_to_generated


def _render_parts(parts: np.ndarray) -> np.ndarray:
    """Render a part map the way Student 1 does (PARULA on black), RGB."""
    rgb = np.zeros(parts.shape + (3,), np.uint8)
    for p in range(1, 25):
        rgb[parts == p] = dpp.PART_COLORS_RGB[p - 1].astype(np.uint8)
    return rgb


def test_decode_parts_round_trip_all_24_ids():
    parts = np.zeros((48, 240), np.uint8)
    for p in range(1, 25):
        parts[8:40, (p - 1) * 10 + 1:(p - 1) * 10 + 9] = p
    decoded = dpp.decode_parts(_render_parts(parts))
    assert np.array_equal(decoded, parts)


def test_decode_parts_tolerates_codec_noise():
    parts = np.zeros((64, 64), np.uint8)
    parts[10:50, 10:30], parts[10:50, 30:50] = 1, 15
    noisy = np.clip(_render_parts(parts).astype(int) + np.random.default_rng(0).integers(-6, 7, (64, 64, 3)), 0, 255).astype(np.uint8)
    decoded = dpp.decode_parts(noisy)
    assert (decoded[parts > 0] == parts[parts > 0]).mean() > 0.97


def test_head_split_back_view_uses_hair_only():
    parts = np.zeros((40, 20), np.uint8)
    parts[0:20, 5:15] = 23
    parts[20:40, 5:15] = 1
    masks = dpp.classes_from_parts(parts, {"shirt": [1, 2]}, {"top_fraction": 0.45, "top_class": "hair", "face_class": "skin", "back_of_head_class": "hair"}, face_visible=False)
    assert masks["hair"][0:20].sum() == 200
    assert masks.get("skin", np.zeros(1)).sum() == 0
    front = dpp.classes_from_parts(parts, {"shirt": [1, 2]}, {"top_fraction": 0.45, "top_class": "cap", "face_class": "skin"}, face_visible=True)
    assert front["cap"].sum() > 0 and front["skin"].sum() > 0 and front["shirt"].sum() == 200


def _synthetic_densepose(n: int = 40) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        parts = np.zeros((128, 128), np.uint8)
        x = 40 + i
        parts[30:60, x:x + 20], parts[60:100, x:x + 20], parts[20:30, x + 5:x + 15] = 1, 9, 23
        frames.append(_render_parts(parts))
    return frames


def test_geometry_role_sampling_and_crop_size():
    dp = _synthetic_densepose()
    bowler = compute_geometry(dp, "bowler", 16, "ahaan-main")
    batsman = compute_geometry(dp, "batsman", 16, "ahaan-main")
    assert bowler.sampled_source_indices[0] == 8 and bowler.sampled_source_indices[-1] == 38
    assert batsman.sampled_source_indices[0] == 24 and batsman.sampled_source_indices[-1] == 36  # min(total-4, 94)
    crop = crop_like_student2(dp[bowler.sampled_source_indices[0]], bowler)
    assert crop.shape == (512, 384, 3)


def test_coordinate_mapping_matches_crop():
    dp = _synthetic_densepose()
    geom = compute_geometry(dp, "bowler", 16, "ahaan-main")
    sx1, sy1, sx2, sy2 = geom.crop_xyxy_source
    gx, gy = source_xy_to_generated(np.array([sx1, sx2]), np.array([sy1, sy2]), geom)
    assert np.allclose(gx, [0, 384]) and np.allclose(gy, [0, 512])


def test_detect_layout_ahaan_padding():
    rng = np.random.default_rng(1)
    keys = [np.full((32, 24, 3), 20 + 12 * k, np.uint8) for k in range(16)]
    order = [0] * 3 + list(range(16)) + [15] * 7
    frames = [np.clip(keys[k] + rng.normal(0, 1.3, keys[k].shape), 0, 255).astype(np.uint8) for k in order]
    info = detect_frame_layout(frames, 16)
    assert info["method"] == "detected_holds" and info["keyframe_of_output_frame"] == order


def test_colourise_keeps_luminance_and_adds_colour():
    frame = np.full((64, 48, 3), 120, np.uint8)
    masks = {"player": np.zeros((64, 48), np.uint8), "skin": np.zeros((64, 48), np.uint8)}
    masks["player"][10:50, 10:38] = 1
    masks["skin"][10:20, 15:30] = 1
    cfg = {"palette": {"neutral": [128, 128], "skin": [141, 153], "grass": [103, 157], "shirt": [127, 138]},
           "class_order": ["skin"], "base_player_class": "shirt", "background": {"horizon": 0.0}}
    ab, luma = colorize_keyframe(frame, masks, cfg)
    out = compose(frame, ab)
    assert np.abs(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0].astype(int) - luma.astype(int)).max() <= 2
    assert colourfulness(out) > colourfulness(frame) + 3


def test_retime_preserves_duration_and_order():
    frames = [np.full((4, 4, 3), i, np.uint8) for i in range(14)]
    out = retime(frames, 7.0, 18.0, jitter=0.35, seed=3)
    assert len(out) == 36
    values = [int(f[0, 0, 0]) for f in out]
    assert values == sorted(values) and values[0] == 0 and values[-1] == 13


def test_era_presets_run_and_report_fps():
    frames = [np.full((64, 48, 3), 100 + i, np.uint8) for i in range(7)]
    for name, fps in [("newsreel_1930s", 18), ("broadcast_1970s", 25), ("remaster", 24)]:
        out, out_fps = apply_preset(name, frames, 7.0, {name: {"fps": fps}})
        assert out_fps == fps and len(out) == fps and out[0].shape == frames[0].shape


def test_pck_counts_joints_within_threshold():
    ref = np.zeros((2, 3, 2))
    pred = ref.copy()
    pred[0, 0] = [5, 0]
    pred[1, 2] = [20, 0]
    valid = np.ones((2, 3), bool)
    assert pck(ref, pred, valid, np.array([100.0, 100.0]), 0.1) == pytest.approx(5 / 6)
