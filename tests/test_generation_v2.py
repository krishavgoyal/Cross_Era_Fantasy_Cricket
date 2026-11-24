"""CPU tests for generation_v2 helpers (no models; needs numpy, opencv, Pillow, and diffusers for the import)."""
import pytest

g2 = pytest.importorskip("generation_v2")


def test_back_view_when_right_side_on_image_right():
    back = {"right_shoulder": (300, 100), "left_shoulder": (200, 100), "right_hip": (290, 250), "left_hip": (210, 250)}
    front = {"right_shoulder": (200, 100), "left_shoulder": (300, 100), "right_hip": (210, 250), "left_hip": (290, 250)}
    assert g2.is_back_view(back) is True
    assert g2.is_back_view(front) is False
    assert g2.is_back_view({}) is None
    assert g2.clip_view([back, back, front]) == ("back", 0.667)
    neck = {"neck": (250, 60)}
    side = {"right_shoulder": (262, 100), "left_shoulder": (238, 100), "right_hip": (255, 250), "left_hip": (245, 250)}
    assert g2.clip_view([{**side, **neck}] * 3) == ("side", 1.0)          # narrow shoulders: side-on
    assert g2.clip_view([{**back, **neck}] * 3) == ("back", 1.0)          # wide shoulders: real back view
    assert g2.clip_view([front, {}])[0] == "front"


def test_map_to_generated_matches_sidecar_formula():
    s = {"crop_in_padded_px": [100, 50, 484, 562], "pad_px": 100, "generated_size": [384, 512]}
    assert g2.map_to_generated((0, 0), s) == (0.0, 50.0)
    assert g2.map_to_generated((384, 462), s) == (384.0, 512.0)


def test_draw_openpose_drops_face_points():
    pts = {"nose": (50, 20), "neck": (50, 40), "right_shoulder": (40, 40), "left_shoulder": (60, 40)}
    full = g2.draw_openpose(pts, (100, 100))
    no_face = g2.draw_openpose(pts, (100, 100), drop=g2.FACE_POINTS)
    assert full.shape == (100, 100, 3) and full.any()
    assert full[20, 50].any() and not no_face[20, 50].any()
