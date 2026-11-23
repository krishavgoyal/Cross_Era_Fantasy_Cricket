# Phase 0: Technical research (summary)

Short version of the Phase 0 investigation, dated 2026-11-23. The final stack lives in `CLAUDE.md`; this page explains why.

## The proposal's named models, checked against their code
| Item | Finding | Verdict |
|---|---|---|
| LTX-2 | Has pose IC-LoRAs, but the reference photo only becomes the **first frame**. There is no identity encoder, and it needs 32 GB+ of VRAM. | Not the main animator |
| MagicAnimate | SD1.5 + DensePose, unmaintained since 2023, no pose alignment | Baseline only |
| arXiv 2510.03909 (CAMEO) | No code released | Related work |
| arXiv 2506.09411 | Needs a 360° video of the subject | Not usable |
| arXiv 2304.01186 (Follow-Your-Pose) | No reference-image input | Not usable |

## Models that actually do "one photo + pose sequence → video"
- Wan-family models (2025–26, mostly Apache-2.0): Wan2.2-Animate-14B, SCAIL, SteadyDancer, One-to-All 1.3B/14B, UniAnimate-DiT, Wan-Fun-Control, **Wan2.1-VACE-1.3B/14B**.
- Older SVD and SD1.5 models: MimicMotion, StableAnimator, Animate Anyone, MusePose, Champ.

## Pose
- DWPose (COCO-WholeBody 133 keypoints, drawn OpenPose-style) is the de-facto input format for pose-guided video models.
- **rtmlib** runs DWPose/RTMW through ONNX Runtime with no mmcv, so it is easy to install on Windows.
- Cricket-specific problems:
  - motion blur on the bowling arm
  - the bat hiding the hands
  - umpire and non-striker in frame, so the target player must be tracked
  - camera cuts

## Hardware
- RTX 4050 Laptop, 6 GB VRAM, 16 GB RAM. 14B video models are not practical locally.
- System RAM limits offloading as much as VRAM does.

## Risks noted
- The bat disappears or bends.
- Grayscale archival photos weaken identity.
- Body proportions differ between modern and historical players, so per-bone retargeting is needed.
- **Ethics**: realistic video of real people. Always label output as AI-generated and keep it academic.
- No current open model handles two-player interaction. **Generate bowler and batter separately and cut them together.**
