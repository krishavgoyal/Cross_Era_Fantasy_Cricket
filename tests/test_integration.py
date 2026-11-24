"""Integration-layer tests: no GPU, no model downloads."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))
sys.path.insert(0, str(ROOT / "densepose"))
sys.path.insert(0, str(ROOT))

import eval_pck  # noqa: E402
import master_pipeline as mp  # noqa: E402
from patch_weak_frames import find_weak  # noqa: E402
from student3_prompts import parts_in_generated_frame, rgb_to_parts  # noqa: E402
from dp_common import colorize  # noqa: E402


def test_pck_perfect_and_missing():
    rng = np.random.default_rng(0)
    ref = rng.uniform(100, 300, (4, 18, 2)).astype(np.float32)
    vis = np.ones((4, 18), bool)
    r = eval_pck.compute_pck(ref, vis, ref.copy(), vis.copy())
    assert r["pck@0.05"] == 1.0
    pred_vis = vis.copy()
    pred_vis[:, :9] = False  # DWPose missed half the joints -> they count as wrong
    r = eval_pck.compute_pck(ref, vis, ref.copy(), pred_vis)
    assert abs(r["pck@0.1"] - 0.5) < 1e-6
    assert r["missed_by_dwpose"] == 4 * 9


def test_velocity_cosine_direction():
    t = np.arange(5, dtype=np.float32)[:, None, None]
    ref = np.zeros((5, 18, 2), np.float32) + t * np.array([5.0, 0.0])
    vis = np.ones((5, 18), bool)
    same, n = eval_pck.velocity_cosine(ref, vis, ref * 1.0, vis)
    opposite, _ = eval_pck.velocity_cosine(ref, vis, -ref, vis)
    assert same == 1.0 and opposite == -1.0 and n > 0


def test_ref_to_generated_matches_crop():
    s = {"crop_in_padded_px": [100, 50, 484, 562], "pad_px": 50, "generated_size": [384, 512]}
    pts = np.array([[50.0, 0.0], [434.0, 512.0]], np.float32)  # crop corners in unpadded coords
    g = eval_pck.ref_to_generated(pts, s)
    assert np.allclose(g, [[0, 0], [384, 512]])


def test_parts_crop_same_as_engine():
    parts = np.zeros((512, 512), np.uint8)
    parts[200:300, 250:260] = 9
    s = {"crop_in_padded_px": [100, 50, 484, 562], "pad_px": 50, "generated_size": [384, 512]}
    g = parts_in_generated_frame(parts, s)
    ys, xs = np.where(g == 9)
    assert g.shape == (512, 384)
    assert xs.min() == 200 and ys.min() == 200  # (250+50-100, 200+50-50)


def test_parula_roundtrip():
    lab = np.tile(np.arange(25, dtype=np.uint8), (4, 1))
    rgb = colorize(lab)[..., ::-1]
    assert (rgb_to_parts(rgb) == lab).all()


def test_find_weak_flags_collapsed_frame():
    P = np.zeros((9, 64, 64), np.uint8)
    for t in range(9):
        P[t, 10:50, 20:40] = 2
    P[4] = 0
    P[4, 10:20, 20:40] = 2  # lost the legs
    exp = [[20, 10, 40, 50]] * 9
    weak, _ = find_weak(P, exp)
    assert weak == [4]


def test_keyframe_fps_real_time():
    side = {"sampling": {"sampled_source_frames": list(range(8, 120, 8))[:15] + [120]}}
    side["sampling"]["sampled_source_frames"] = [int(x) for x in np.linspace(8, 119, 16).astype(int)]
    fps = mp.keyframe_fps(side, 24.0)
    assert 3.0 < fps < 3.6  # 16 keyframes over ~4.6 s of real motion


def test_master_dry_run(tmp_path):
    clip = tmp_path / "clips" / "demo_clip"
    clip.mkdir(parents=True)
    for f in ("source_512.mp4", "pose_data.json", "control_pose.mp4", "control_densepose_real.mp4"):
        (clip / f).write_bytes(b"x")
    (clip / "metadata.json").write_text(json.dumps({"role": "bowler", "processed": {"fps": 24}}))
    img = tmp_path / "p.jpg"
    img.write_bytes(b"x")
    code = mp.main(["--clip-id", "demo_clip", "--image", str(img), "--clips-root", str(tmp_path / "clips"),
                    "--runs-root", str(tmp_path / "runs"), "--dry-run"])
    assert code == 0


def test_master_missing_clip_needs_reference_video(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"x")
    code = mp.main(["--clip-id", "nope", "--image", str(img), "--role", "batsman",
                    "--clips-root", str(tmp_path / "clips"), "--runs-root", str(tmp_path / "runs")])
    assert code == mp.EXIT_ARGS


def test_action_window_finds_bowling_release():
    first, last, peak = mp.action_window(ROOT / "data/processed_clips/bowler_delivery_001/pose_data.json")
    assert 100 <= peak <= 118  # release is in the last second of the clip
    assert last - first >= 15


def test_student3_v3_uses_sidecar_geometry():
    sys.path.insert(0, str(ROOT))
    from student3_v3.run import geometry_from_sidecar
    from student3_v3.student2_geometry import Student2Geometry
    g = Student2Geometry(role="bowler", preset="ahaan-main", target_frames=16, active_source_indices=list(range(121)),
                         sampled_active_indices=list(range(16)), sampled_source_indices=list(range(8, 24)), pad=10,
                         crop_xyxy_padded=(0, 0, 10, 10), crop_xyxy_source=(-10, -10, 0, 0))
    side = {"sampling": {"sampled_source_frames": list(range(60, 124, 4)), "crop_in_padded_px": [100, 50, 484, 562], "pad_px": 50}}
    g, agree = geometry_from_sidecar(g, side)
    assert g.sampled_source_indices[0] == 60 and g.crop_xyxy_source == (50, 0, 434, 512)
    assert agree["used"] and not agree["rederived_frames_match"]


def test_force_all_keeps_processed_clip(tmp_path):
    a = mp.build_parser().parse_args(["--clip-id", "x", "--image", "i.jpg", "--force", "all"])
    assert "motion" not in mp.Pipeline(a).force and "generate" in mp.Pipeline(a).force


def test_keyframe_video_drops_holds(tmp_path):
    import cv2
    src = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 7, (64, 64))
    layout = [0, 0, 0] + list(range(4)) + [3, 3]
    for k in layout:
        w.write(np.full((64, 64, 3), 40 * (k + 1), np.uint8))
    w.release()
    assert mp.keyframe_video(src, tmp_path / "out.mp4", layout) == 4
