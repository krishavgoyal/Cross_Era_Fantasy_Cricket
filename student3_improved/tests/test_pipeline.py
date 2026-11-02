from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from student3_improved.evaluate import evaluate
from student3_improved.postprocess import _warp_previous_chroma, grade_frames, load_config, write_video

ROOT = Path(__file__).parents[1]


def config():
    return load_config(ROOT / "configs" / "1930s_archival.json")


def test_backward_flow_warp_uses_current_to_previous_mapping(monkeypatch):
    h, w = 24, 32
    previous = np.zeros((h, w), np.uint8)
    current = np.zeros((h, w), np.uint8)
    previous_ab = np.zeros((h, w, 2), np.float32)
    previous_ab[..., 0] = np.arange(w, dtype=np.float32)[None, :]

    def fake_flow(a, b, *_args):
        flow = np.zeros((h, w, 2), np.float32)
        if a is current:
            flow[..., 0] = -3.0  # current pixel samples 3 px left in previous frame
        else:
            flow[..., 0] = 3.0
        return flow

    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", fake_flow)
    warped, confidence = _warp_previous_chroma(previous, current, previous_ab)
    assert np.allclose(warped[10, 15, 0], 12.0, atol=0.1)
    assert confidence[10, 15] > 0.9


def test_colorization_preserves_dimensions_and_adds_color():
    frames = []
    masks = []
    for shift in (0, 2, 4):
        frame = np.full((512, 384, 3), 105, np.uint8)
        cv2.rectangle(frame, (100 + shift, 60), (270 + shift, 470), (220, 220, 220), -1)
        player = np.zeros((512, 384), np.uint8)
        cv2.rectangle(player, (100 + shift, 60), (270 + shift, 470), 1, -1)
        frames.append(frame)
        masks.append({"player": player})
    output = grade_frames(frames, config(), masks)
    assert len(output) == 3
    assert output[0].shape == frames[0].shape
    out_lab = cv2.cvtColor(output[0], cv2.COLOR_BGR2LAB)
    assert np.mean(np.abs(out_lab[..., 1:].astype(np.float32) - 128)) > 1.0


def test_evaluator_does_not_claim_visual_quality(tmp_path: Path):
    frames = [np.full((512, 384, 3), 100 + i, np.uint8) for i in range(3)]
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    write_video(frames, source, 7.0)
    write_video(grade_frames(frames, config()), output, 7.0)
    report = evaluate(source, output)
    assert report["technical_verdict"] in {"pass", "review"}
    assert report["visual_quality_verdict"] == "requires_human_review"
    assert "mean_local_ssim" in report["luminance"]


def test_sam2_prompt_boxes_are_validated():
    raw = json.loads((ROOT / "configs" / "1930s_archival.json").read_text())
    assert "player" in raw["sam2_prompts"]
    for prompt in raw["sam2_prompts"].values():
        x1, y1, x2, y2 = prompt["box"]
        assert 0 <= x1 < x2 <= 1
        assert 0 <= y1 < y2 <= 1
