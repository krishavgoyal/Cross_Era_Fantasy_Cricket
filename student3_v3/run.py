"""Student 3 v3 pipeline: DensePose-guided colourisation + era styling + evaluation, for batsman or bowler.

    python -m student3_v3.run \
        --input outputs/m3_bowler_delivery.mp4 \
        --densepose data/processed_clips/bowler_delivery_001/control_densepose_real.mp4 \
        --config student3_v3/configs/bowler_1930s.json \
        --out-dir data/student3/bowler_delivery_001

Writes into --out-dir:
  colourised_master.mp4      same frames/fps as the input, chroma only
  era_<preset>.mp4           one per preset listed in the config (retimed, styled)
  side_by_side.mp4           input | colourised | first era preset
  mask_overlay.mp4           class masks over the 16 keyframes, for reviewing segmentation
  processing.json            geometry, frame layout, alignment check, settings used
  evaluation.json            metrics (see evaluate.py)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from student3_improved.postprocess import read_video, write_video

from . import densepose_parts as dpp
from .colorize import colorize_keyframe, compose, temporal_blend
from .era_filters import apply_preset
from .evaluate import evaluate
from .student2_geometry import bridge_student2_masks, compute_geometry, crop_like_student2, read_rgb_frames


def detect_frame_layout(frames: list[np.ndarray], n_keys: int, hold_threshold: float = 2.5, fallback: dict[str, int] | None = None) -> dict[str, Any]:
    """Student 2 pads 16 keyframes with held copies (main: 4+16+6, Ahaan-main: 3+16+7). Find the runs."""
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    diffs = [float(np.abs(gray[i] - gray[i - 1]).mean()) for i in range(1, len(gray))]
    layout = [0]
    for d in diffs:
        layout.append(layout[-1] + (0 if d < hold_threshold else 1))
    method = "detected_holds"
    if layout[-1] + 1 != n_keys:
        fb = fallback or {"lead": 3, "tail": 7}
        lead, tail = int(fb["lead"]), int(fb["tail"])
        if lead + n_keys + tail != len(frames):
            if len(frames) == n_keys:
                lead = tail = 0
            else:
                raise ValueError(f"Cannot map {len(frames)} output frames to {n_keys} keyframes; set frame_layout in the config.")
        layout = [0] * lead + list(range(n_keys)) + [n_keys - 1] * tail
        method = f"config_fallback(lead={lead}, tail={tail})"
    return {"keyframe_of_output_frame": layout, "method": method, "adjacent_luma_mae": [round(d, 3) for d in diffs]}


def build_key_masks(dp_frames: list[np.ndarray], geom, cfg: dict[str, Any], bridge: dict[str, Any] | None,
                    exact_parts: np.ndarray | None = None) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray], dict[str, Any]]:
    head = cfg.get("head_split")
    face_visible = bool(head.get("face_visible", True)) if head else False
    masks, parts_list, ious = [], [], []
    use_bridge = bool(bridge and "body_soft" in bridge)
    for k, src in enumerate(geom.sampled_source_indices):
        if exact_parts is not None:
            parts = crop_like_student2(exact_parts[src], geom)  # NEAREST crop of the real part ids
        else:
            parts = dpp.decode_parts(crop_like_student2(dp_frames[src], geom))
        m = dpp.classes_from_parts(parts, cfg["class_parts"], head, face_visible)
        if use_bridge:
            body = (bridge["body_soft"][k] > 0.5).astype(np.uint8)
            inter, union = np.logical_and(body, m["player"]).sum(), np.logical_or(body, m["player"]).sum()
            ious.append(float(inter / max(1, union)))
            # Student 2 composites the generated player through body_soft, so it is the true silhouette.
            extra = (body > 0) & (m["player"] == 0)
            fill = cfg.get("silhouette_fill_class")
            ys = np.nonzero(body)[0]
            if fill and extra.any() and ys.size:
                # Student 2 only widens the lower legs (pads/shin bridge); keep the fill below this body fraction.
                cut = ys.min() + float(cfg.get("silhouette_fill_below", 0.55)) * (ys.max() - ys.min())
                extra &= (np.arange(body.shape[0])[:, None] >= cut)
                m[fill] = np.maximum(m.get(fill, np.zeros_like(body)), extra.astype(np.uint8))
            m["player"] = np.maximum(m["player"], body)
            if cfg.get("bat_from_student2") and np.max(bridge["bat_alpha"][k]) > 0:
                m["bat"] = (bridge["bat_alpha"][k] > 0.35).astype(np.uint8) * (body == 0)
        masks.append(m)
        parts_list.append(parts)
    alignment = {
        "student2_bridge": "ok" if use_bridge else (bridge.get("error") if bridge else "not_available"),
        "stubbed_modules": bridge.get("stubbed") if bridge else None,
        "decoded_vs_student2_silhouette_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
        "decoded_vs_student2_silhouette_iou_min": round(float(np.min(ious)), 4) if ious else None,
    }
    return masks, parts_list, alignment


def mask_overlay(keys: list[np.ndarray], masks: list[dict[str, np.ndarray]], cfg: dict[str, Any]) -> list[np.ndarray]:
    colours = cfg.get("overlay_colours_bgr", {})
    out = []
    for frame, m in zip(keys, masks):
        canvas = frame.copy().astype(np.float32)
        for label in cfg["class_order"]:
            if label in m and label in colours:
                sel = m[label] > 0
                canvas[sel] = 0.45 * canvas[sel] + 0.55 * np.asarray(colours[label], np.float32)
        img = canvas.astype(np.uint8)
        y = 18
        for label in cfg["class_order"]:
            if label in colours and label in m and m[label].any():
                cv2.putText(img, label, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tuple(int(c) for c in colours[label]), 1, cv2.LINE_AA)
                y += 16
        out.append(img)
    return out


def side_by_side(*clips: list[np.ndarray], labels: list[str]) -> list[np.ndarray]:
    n = min(len(c) for c in clips)
    out = []
    for i in range(n):
        tiles = []
        for clip, label in zip(clips, labels):
            t = clip[i].copy()
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(t, (4, 4), (12 + tw, 12 + th), (20, 20, 20), -1)
            cv2.putText(t, label, (8, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
            tiles.append(t)
        out.append(np.hstack(tiles))
    return out


def geometry_from_sidecar(geom, sidecar: dict[str, Any]):
    """Integration: use the exact sampling/crop Student 2 recorded (generated.mp4.json) instead of re-deriving it.

    Needed when generation used a non-default --window or frame count. Returns (geometry, agreement report)."""
    s = sidecar["sampling"]
    x1, y1, x2, y2 = s["crop_in_padded_px"]
    pad = int(s["pad_px"])
    agree = {
        "used": True,
        "rederived_frames_match": list(geom.sampled_source_indices) == list(s["sampled_source_frames"]),
        "rederived_crop_matches": tuple(geom.crop_xyxy_padded) == (x1, y1, x2, y2) and geom.pad == pad,
        "note": "False = generation used a custom window/frame count; the sidecar values were used, so masks stay aligned",
    }
    geom.sampled_source_indices = [int(v) for v in s["sampled_source_frames"]]
    geom.pad = pad
    geom.crop_xyxy_padded = (int(x1), int(y1), int(x2), int(y2))
    geom.crop_xyxy_source = (int(x1 - pad), int(y1 - pad), int(x2 - pad), int(y2 - pad))
    geom.target_frames = len(geom.sampled_source_indices)
    return geom, agree


def run(input_path: Path, densepose_path: Path, config_path: Path, out_dir: Path, ffmpeg: str | None = None, skip_bridge: bool = False, preset: str | None = None,
        sidecar_path: Path | None = None, parts_npz: Path | None = None) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if preset:
        cfg["student2_preset"] = preset
    role = cfg["role"]
    frames, spec = read_video(input_path)
    if (spec.width, spec.height) != (384, 512):
        raise ValueError(f"Expected Student 2's 384x512 output, got {spec.width}x{spec.height}")
    n_keys = int(cfg.get("keyframes", 16))
    dp_frames = read_rgb_frames(densepose_path)
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8")) if sidecar_path and Path(sidecar_path).exists() else None
    if sidecar:
        n_keys = int(sidecar.get("keyframes", n_keys))
    geom = compute_geometry(dp_frames, role, n_keys, cfg.get("student2_preset", "ahaan-main"))
    sidecar_agreement = None
    if sidecar:
        geom, sidecar_agreement = geometry_from_sidecar(geom, sidecar)
        layout_info = {"keyframe_of_output_frame": [int(k) for k in sidecar["output_frame_to_keyframe"]], "method": "student2_sidecar"}
        if len(layout_info["keyframe_of_output_frame"]) != len(frames):
            raise ValueError(f"sidecar lists {len(layout_info['keyframe_of_output_frame'])} frames, video has {len(frames)}")
    else:
        layout_info = detect_frame_layout(frames, n_keys, float(cfg.get("hold_threshold", 2.5)), cfg.get("frame_layout"))
    layout = layout_info["keyframe_of_output_frame"]
    window = (sidecar["sampling"]["sampled_source_frames"][0], sidecar["sampling"]["sampled_source_frames"][-1]) if sidecar else None
    bridge = None if skip_bridge else bridge_student2_masks(densepose_path, role, n_keys, window=window)
    exact_parts = np.load(parts_npz)["part_index"] if parts_npz and Path(parts_npz).exists() else None
    key_masks, _, alignment = build_key_masks(dp_frames, geom, cfg, bridge, exact_parts)
    alignment["sidecar"] = sidecar_agreement if sidecar else "not_used"
    alignment["part_source"] = "densepose_real_parts.npz (exact ids)" if exact_parts is not None else "decoded from DensePose video colours"

    first_of_key = {}
    for i, k in enumerate(layout):
        first_of_key.setdefault(k, i)
    keys = [frames[first_of_key[k]] for k in range(n_keys)]
    key_ab, key_luma = zip(*[colorize_keyframe(f, m, cfg) for f, m in zip(keys, key_masks)])
    key_ab, temporal_log = temporal_blend(keys, list(key_ab), list(key_luma), [m["player"] for m in key_masks], cfg)
    master = [compose(frames[i], key_ab[layout[i]]) for i in range(len(frames))]

    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = out_dir / "colourised_master.mp4"
    encoder = write_video(master, master_path, spec.fps, ffmpeg)
    era_outputs = {}
    era_frames = {}
    for preset in cfg.get("era_outputs", []):
        styled, fps = apply_preset(preset, master, spec.fps, cfg.get("era_presets", {}))
        path = out_dir / f"era_{preset}.mp4"
        write_video(styled, path, fps, ffmpeg)
        era_outputs[preset] = {"path": path.name, "fps": fps, "frames": len(styled)}
        era_frames[preset] = apply_preset(preset, master, spec.fps, {**cfg.get("era_presets", {}), preset: {**cfg.get("era_presets", {}).get(preset, {}), "fps": spec.fps}})[0]
    first_era = next(iter(era_frames), None)
    sbs_clips, sbs_labels = [frames, master], ["generated", "colourised"]
    if first_era:
        sbs_clips.append(era_frames[first_era])
        sbs_labels.append(first_era)
    write_video(side_by_side(*sbs_clips, labels=sbs_labels), out_dir / "side_by_side.mp4", spec.fps, ffmpeg)
    write_video(mask_overlay(keys, key_masks, cfg), out_dir / "mask_overlay.mp4", 2.0, ffmpeg)

    class_names = [c for c in cfg["class_order"]]
    evaluation = evaluate(input_path, master_path, layout, key_masks, class_names)
    evaluation["temporal"]["keyframe_blend_log"] = temporal_log
    (out_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    report = {
        "input": str(input_path),
        "densepose": str(densepose_path),
        "config": str(config_path),
        "role": role,
        "input_spec": spec.__dict__,
        "encoder": encoder,
        "student2_geometry": geom.to_json(),
        "student2_preset": cfg.get("student2_preset", "ahaan-main"),
        "frame_layout": {k: v for k, v in layout_info.items()},
        "alignment": alignment,
        "outputs": {"colourised_master": master_path.name, "era": era_outputs, "side_by_side": "side_by_side.mp4", "mask_overlay": "mask_overlay.mp4", "evaluation": "evaluation.json"},
        "notes": cfg.get("notes", []),
    }
    (out_dir / "processing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"processing": report, "evaluation": evaluation}


def main() -> int:
    p = argparse.ArgumentParser(description="Student 3 v3: DensePose-guided colourisation and era styling (batsman or bowler).")
    p.add_argument("--input", required=True, type=Path, help="Student 2 generated video (384x512)")
    p.add_argument("--densepose", required=True, type=Path, help="Student 1 control_densepose_real.mp4 used for that generation")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--student2-preset", choices=["ahaan-main", "main"], default=None, help="Which generation_engine.py version rendered the input (overrides config)")
    p.add_argument("--skip-student2-bridge", action="store_true", help="Do not import generation_engine.py for the silhouette/bat check")
    p.add_argument("--sidecar", type=Path, default=None, help="generated.mp4.json from Student 2 (exact frame mapping and crop)")
    p.add_argument("--parts-npz", type=Path, default=None, help="Student 1 densepose_real_parts.npz (exact part ids, no colour decoding)")
    a = p.parse_args()
    result = run(a.input, a.densepose, a.config, a.out_dir, a.ffmpeg, a.skip_student2_bridge, a.student2_preset, a.sidecar, a.parts_npz)
    ev = result["evaluation"]
    print(json.dumps({"technical_verdict": ev["technical_verdict"], "luminance": ev["luminance"], "colourfulness": ev["colourfulness_hasler_suesstrunk"],
                      "mask_coverage": ev["mask_coverage"]["player_pixels_with_a_class_mean"], "alignment": result["processing"]["alignment"],
                      "frame_layout": result["processing"]["frame_layout"]["method"], "out_dir": str(a.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
