# Cricket-Specific Edge-Case Verification

## 1. Disappearing bat

### Risk
DWPose tracks human keypoints, not the cricket bat or ball. The bat therefore cannot be guaranteed to remain inside the generated crop from pose data alone.

### Implemented mitigation
- YOLO pose tracking is used rather than box-only detection.
- The crop envelope is expanded using all visible target body keypoints before square cropping.
- An additional `--pose-expand-margin` is applied around that pose envelope.
- The default square crop margin was increased from `0.28` to `0.40`.
- Per-frame `crop_edge_risk` is recorded when visible pose points approach a crop boundary.
- Aggregate `crop_edge_risk_frames` and `crop_edge_risk_rate` are stored in `metadata.json`.
- `--crop-margin` remains configurable for action-specific tuning.

### Limitation
This is not bat tracking. A long bat can still extend beyond the canvas even when all human joints are safely inside it. A clip with non-zero crop-edge risk or an action with a very large bat arc should be manually reviewed before handing the control sequence to Student 2.

### Recommended cricket settings
```text
cover_drive:             crop_margin 0.40–0.50
pull/hook with wide arc:  crop_margin 0.45–0.60
fast bowling:             crop_margin 0.40–0.50
spin bowling:              crop_margin 0.35–0.45
```
These are starting points, not guarantees. Validate visually on the actual broadcast camera angle.

## 2. Fast-bowler wrist blur

### Risk
At 24/30 FPS, a release-frame wrist can have low detector confidence for 1–2 frames. Passing a zero/invalid point into a low-pass filter can create a false jump toward `(0,0)`.

### Implemented mitigation
- Confidence threshold defaults to `0.30`.
- Gaps of at most `2` consecutive low-confidence frames are linearly interpolated only when both neighboring frames are valid.
- Interpolated points are marked with `confidence_interpolated: true`.
- The original detector coordinates remain available through `raw_x` / `raw_y` and the pre-existing raw keypoint snapshots.
- Longer occlusions are not linearly hallucinated. The filter holds the last trustworthy coordinate so invalid detector zeros do not create a synthetic teleport.
- Smoothing configuration is recorded in `pose_data.json`.

### Important limitation
The body confidence exposed by `easy-dwpose==1.0.2` is an OpenPose-style visibility/subset mask, not a calibrated 0–1 probability. The implementation therefore records body confidence as binary visibility. Hand and face scores retain detector scores. This distinction is stored under `confidence_semantics` in `pose_data.json`.

## 3. Tracking distractions

### Risk
Broadcast frames can contain the umpire, non-striker, wicketkeeper, or fielders. Tracker IDs are not semantic identities.

### Implemented mitigation
Initial automatic selection uses a weighted spatial score dominated by:
- bounding-box area
- proximity to frame center
- detector confidence

After initialization, ByteTrack identity continuity is preferred. However, the code permits a switch when another candidate's spatial score is substantially stronger, preventing an incorrect ID from being followed indefinitely.

For ambiguous footage, `--track-id` can pin the intended actor after visual inspection.

### Human QA requirement
Automatic selection is not a semantic classifier. Clips containing two similarly sized players must be inspected before being treated as reference motion ground truth.

## Verification status

The automated edge-case test suite covers:
- large/central actor selection
- track switching when a distractor becomes implausible
- pose-based crop expansion
- crop-edge risk detection
- actual easy-dwpose raw dictionary structure
- 2-frame low-confidence interpolation
- protection against long-occlusion jumps to `(0,0)`

Local test result:

```text
8 passed
```
