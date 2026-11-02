# Cross-Era Fantasy Cricket: Student 1 Motion Engine

This package implements Student 1's preprocessing contract:

`broadcast video -> tracked/cropped 512x512 clip -> DWPose keypoints -> confidence-aware 1-Euro smoothing -> canonical DWPose control MP4 + numerical vectors`

The project proposal defines Student 1 as responsible for clean reference footage and pure motion-vector extraction, with 512x512 and 3–5 second clips as the hardware bounds. The proposal describes a decoupled structure/appearance pipeline rather than a single mandatory conditioning implementation.

## 1. Supported environment

Use Python 3.10 or 3.11. `easy-dwpose` 1.0.2 requires Python >=3.9 and exposes the `DWposeDetector` interface used here.

Create the environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install a CUDA-enabled PyTorch build first using the official PyTorch selector. For this project, keep the PyTorch and ONNX Runtime CUDA major version aligned. PyTorch's official installation page recommends choosing the CUDA build suited to the installed NVIDIA driver/runtime.

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

System tools:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Verify the environment:

```bash
python verify_environment.py
```

Expected GPU checks:

```text
CUDA available: True
GPU: NVIDIA ...
ORT providers: [..., 'CUDAExecutionProvider', ...]
ffmpeg: /usr/bin/ffmpeg
```

If `CUDAExecutionProvider` is missing, DWPose is not using the NVIDIA GPU even if PyTorch says CUDA is available.

## 2. Models

The default actor tracker is `yolo26n-pose.pt`. Ultralytics currently documents YOLO26 pose models and ByteTrack tracking directly through the `model.track(..., tracker="bytetrack.yaml")` API.

The first run downloads the YOLO checkpoint automatically.

The DWPose wrapper automatically downloads its ONNX checkpoints on first use.

## 3. Single end-to-end command

```bash
python pipeline_cli.py \
  --reference-video data/raw/cover_drive.mp4 \
  --output-root data/processed_clips \
  --clip-id clip_001_cover_drive \
  --action-type cover_drive \
  --fps 24 \
  --device cpu \
  --dwpose-device cpu
```

Outputs:

```text
data/processed_clips/clip_001_cover_drive/
  source_512.mp4
  control_pose.mp4
  pose_data.json
  pose_vectors.npz
  metadata.json
```

## 4. Stage-by-stage execution

### A. Tracking + crop

```bash
python clip_processor.py \
  --input data/raw/cover_drive.mp4 \
  --output-dir data/processed_clips/clip_001_cover_drive \
  --clip-id clip_001_cover_drive \
  --action-type cover_drive \
  --fps 24 \
  --device cpu
```

Important options:

```text
--crop-margin 0.40
--pose-expand-margin 0.15
--bbox-smoothing 0.35
--max-missed-frames 8
--track-id <integer>       # use after visual inspection when auto-selection is ambiguous
```

The crop stage expands the selected person box using visible body keypoints before the square crop. This mitigates bat/follow-through clipping but does not detect the bat itself.

### B. DWPose

```bash
python pose_engine.py \
  --input data/processed_clips/clip_001_cover_drive/source_512.mp4 \
  --output data/processed_clips/clip_001_cover_drive/pose_data.json \
  --device cpu
```

The current `easy-dwpose` raw interface returns:

```text
bodies
body_scores
hands
hands_scores
faces
faces_scores
```

`body_scores` is an OpenPose-style subset/visibility mask. It is therefore stored as binary visibility in the canonical JSON rather than being mislabeled as calibrated probability. The hand and face detector scores remain continuous values.

DWPose is used here as a 2D whole-body estimator. `world_keypoints_3d` remains `null`; no synthetic depth is manufactured.

### C. Confidence-aware smoothing

```bash
python temporal_smoothing.py \
  --input data/processed_clips/clip_001_cover_drive/pose_data.json \
  --confidence-threshold 0.30 \
  --interpolation-gap 2
```

The default policy is:

```text
0–2 low-confidence frames between trustworthy observations -> interpolate
>2-frame occlusion -> do not invent trajectory; hold last trustworthy coordinate for filtering
```

Raw detector values remain available for evaluation.

### D. Control renderer

```bash
python render_control.py \
  --pose-json data/processed_clips/clip_001_cover_drive/pose_data.json \
  --output-video data/processed_clips/clip_001_cover_drive/control_pose.mp4 \
  --output-npz data/processed_clips/clip_001_cover_drive/pose_vectors.npz \
  --style dwpose
```

For the `dwpose` style the renderer calls the canonical Easy-DWPose OpenPose renderer. The generic `openpose` path exists as a fallback/debug renderer.

The downstream model still needs its documented preprocessing adapter. Do not assume that every video diffusion implementation accepts the same pose image normalization or keypoint subset.

## 5. Data contract

### `metadata.json`

Records:

- source video SHA-256
- source FPS/frame count/resolution
- target FPS and 512x512 resolution
- detector/tracker versions/configuration
- crop margins and smoothing parameters
- per-frame source frame index and timestamp
- tracker ID
- source bounding box and crop window
- per-frame crop-edge risk
- aggregate crop-edge risk rate

### `pose_data.json`

Each frame stores normalized coordinates in `[0,1]` with origin at top-left.

Body:

```text
18 OpenPose-compatible joints
```

Hands:

```text
21 joints per hand
```

Face:

```text
68 landmarks
```

Confidence semantics are explicit in the JSON:

```json
"confidence_semantics": {
  "body": "binary_visibility_from_easy_dwpose_openpose_subset",
  "hands": "detector_score_0_to_1",
  "face": "detector_score_0_to_1"
}
```

## 6. Student 2 handoff

`source_512.mp4` and `control_pose.mp4` have matching FPS, frame count and spatial resolution.

`pose_data.json` is the canonical numerical representation. `control_pose.mp4` is a renderer serialization and should be treated as an adapter artifact.

Student 2 should validate one short control sequence against the actual animation framework's documented input preprocessing before locking the integration. Resolution, normalization, keypoint subset and control-channel semantics are model-specific.

## 7. Student 3 handoff

Run the same pose backend/version on generated clips. Record the backend version in evaluation artifacts.

Recommended metrics:

- PCK with an explicitly reported threshold
- joint-trajectory cosine similarity after temporal alignment
- confidence/track coverage
- temporal second-order finite-difference magnitude

Use stored raw coordinates for reference truth and smoothed coordinates for the control trajectory. Do not silently mix detector versions.

## 8. Pranaav handoff

Required:

```text
--reference-video
--output-root
--clip-id
--action-type
```

Useful automation flags:

```text
--fps {24,30}
--detector-model
--tracker
--track-id
--device
--dwpose-device
--crop-margin
--pose-expand-margin
--bbox-smoothing
--max-missed-frames
--confidence-threshold
--interpolation-gap
--control-style
--skip-smoothing
--skip-render
```

Exit codes:

| Code | Meaning |
|---:|---|
| 0 | Pipeline completed successfully |
| 2 | CLI argument error |
| 3 | Missing dependency |
| 4 | Input/media validation failure |
| 5 | Model/device/inference failure |
| 6 | Output/FFmpeg failure |
| 70 | Unexpected runtime failure |

## 9. Cricket edge-case verification

Read `EDGE_CASE_VERIFICATION.md` before treating the module as handoff-ready.

Key points:

1. The bat and ball are not explicitly detected. The crop stage therefore uses pose-based expansion plus configurable safety margins and emits crop-edge-risk telemetry.
2. Short low-confidence wrist gaps are interpolated before the 1-Euro filter; longer occlusions are not hallucinated.
3. Initial target selection is dominated by bounding-box area and frame centrality. Tracker continuity is used after initialization, but the code can switch away from a weak current track. Manual `--track-id` remains available for ambiguous scenes.

## 10. Local tests

```bash
PYTHONPATH=. pytest -q
```

Expected result for the current package revision:

```text
8 passed
```
