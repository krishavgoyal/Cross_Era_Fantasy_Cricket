from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class SegmentationError(RuntimeError):
    pass


def _normalized_box(box: list[float], width: int, height: int) -> np.ndarray:
    if len(box) != 4:
        raise SegmentationError("Each SAM2 prompt box must contain [x1, y1, x2, y2].")
    x1, y1, x2, y2 = map(float, box)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise SegmentationError(f"Invalid normalized SAM2 box: {box}")
    return np.array([x1 * width, y1 * height, x2 * width, y2 * height], dtype=np.float32)


def run_sam2_video(
    frames: list[np.ndarray],
    config: dict[str, Any],
    sam2_config: str,
    checkpoint: str,
    device: str | None = None,
) -> list[dict[str, np.ndarray]]:
    """Propagate per-class box prompts through a clip with the official SAM2 video predictor.

    The boxes are normalized coordinates on frame 0. They must be reviewed for each
    materially different generated shot; SAM2 tracks prompted objects, it does not
    infer semantic parts such as skin versus shirt automatically.
    """
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as exc:  # pragma: no cover - depends on optional external package
        raise SegmentationError(
            "SAM2 is not installed in this Python environment. Follow the SAM2 setup "
            "instructions in student3_improved/README.md, or test with --segmentation grabcut."
        ) from exc

    if not Path(checkpoint).is_file():
        raise SegmentationError(f"SAM2 checkpoint not found: {checkpoint}")
    config_path = Path(sam2_config)
    if not config_path.is_file():
        try:
            import sam2
            package_root = Path(sam2.__file__).resolve().parent
            if not (package_root / sam2_config).is_file():
                raise SegmentationError(f"SAM2 model config not found: {sam2_config}")
        except ImportError as exc:
            raise SegmentationError(f"SAM2 model config not found and SAM2 is not installed: {sam2_config}") from exc

    height, width = frames[0].shape[:2]
    prompts = config.get("sam2_prompts", {})
    if not prompts:
        raise SegmentationError("No sam2_prompts configured. Add first-frame normalized boxes to the JSON config.")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(sam2_config, checkpoint, device=selected_device)

    with tempfile.TemporaryDirectory(prefix="student3_sam2_") as temp_dir:
        frame_dir = Path(temp_dir)
        for index, frame in enumerate(frames):
            # SAM2 video predictor expects sequentially named JPEG/PNG frames.
            if not cv2.imwrite(str(frame_dir / f"{index:05d}.jpg"), frame):
                raise SegmentationError(f"Could not write temporary frame {index} for SAM2.")

        state = predictor.init_state(video_path=str(frame_dir))
        object_ids: dict[int, str] = {}
        for obj_id, (label, prompt) in enumerate(prompts.items(), start=1):
            box = _normalized_box(prompt["box"], width, height)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                box=box,
            )
            object_ids[obj_id] = label

        results: list[dict[str, np.ndarray]] = [
            {label: np.zeros((height, width), dtype=np.uint8) for label in object_ids.values()}
            for _ in frames
        ]
        for frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            logits = out_mask_logits.detach().float().cpu().numpy()
            for local_index, obj_id in enumerate(out_obj_ids):
                label = object_ids.get(int(obj_id))
                if label is None:
                    continue
                mask = logits[local_index]
                if mask.ndim == 3:
                    mask = mask[0]
                results[int(frame_idx)][label] = (mask > 0.0).astype(np.uint8)
        predictor.reset_state(state)

    # SAM2's mask tensors may be lower resolution for some model variants.
    for frame_masks in results:
        for label, mask in list(frame_masks.items()):
            if mask.shape != (height, width):
                frame_masks[label] = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return results


def run_grabcut_fallback(frames: list[np.ndarray], config: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    """Dependency-free foreground approximation for pipeline testing, not SAM2 quality.

    GrabCut is reinitialized per frame using the previous frame's foreground mask
    warped by optical flow. This is intentionally labelled as a fallback in reports.
    """
    height, width = frames[0].shape[:2]
    box = config.get("fallback_player_box", [0.17, 0.025, 0.84, 0.99])
    x1, y1, x2, y2 = _normalized_box(box, width, height).astype(int)
    rect = (int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1)))
    outputs: list[dict[str, np.ndarray]] = []
    previous_gray: np.ndarray | None = None
    previous_mask: np.ndarray | None = None

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((height, width), np.uint8)
        if previous_mask is None:
            mask[:] = cv2.GC_BGD
            mask[y1:y2, x1:x2] = cv2.GC_PR_FGD
        else:
            backward = cv2.calcOpticalFlowFarneback(gray, previous_gray, None, 0.5, 3, 19, 3, 5, 1.2, 0)
            gx, gy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            warped = cv2.remap(previous_mask.astype(np.float32), gx + backward[..., 0], gy + backward[..., 1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            mask[:] = cv2.GC_PR_BGD
            mask[warped > 0.55] = cv2.GC_PR_FGD
            # Keep a conservative central region available for newly exposed body parts.
            mask[y1:y2, x1:x2][warped[y1:y2, x1:x2] > 0.15] = cv2.GC_PR_FGD

        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(frame, mask, rect, bg_model, fg_model, 2, cv2.GC_INIT_WITH_MASK)
            player = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
            # Remove isolated speckles and soften the boundary later in colorization.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(player, 8)
            if count > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                player = (labels == largest).astype(np.uint8)
        except cv2.error:
            player = np.zeros((height, width), np.uint8)
            player[y1:y2, x1:x2] = 1

        outputs.append({"player": player})
        previous_gray = gray
        previous_mask = player
    return outputs
