# Phase 0.1: Free alternatives (summary)

Short version of the free-resources investigation, dated 2026-11-23. The final stack lives in `CLAUDE.md`.

## Small reference-image + pose models (all Apache-2.0)
| Model | Notes |
|---|---|
| **Wan2.1-VACE-1.3B** | Takes a reference image plus a pose control video. 480p only (832x480 / 480x832), 16 fps, frame count must be 4n+1. Runs in ComfyUI with core nodes. **Chosen.** |
| One-to-All 1.3B | Best identity among the small models in its paper; runs on a 16 GB T4. Its official scripts are bf16-only with no offload. Backup. |
| Wan2.1-Fun-1.3B-Control | Reference image + pose; has offload modes. Backup. |
| MimicMotion | Needs 8 GB minimum; weights are non-commercial. Dropped. |

## Free GPUs
| Option | Notes |
|---|---|
| **Kaggle** | About 30 GPU-h/week; T4 (16 GB, **no bf16**, so use fp16); 12 h sessions; ~29 GB RAM; Internet needs phone verification. Run headless; keep notebooks private. **Chosen.** |
| Hugging Face ZeroGPU | Free tier is 5 min/day. |
| Modal | $30/month free credit. |
| Google Colab | Its FAQ bans "creating deepfakes" and web UIs on the free tier. **Avoid.** |
| SageMaker Studio Lab | Closed to new users. |

## Why ComfyUI and not the official repos
- The official Wan2.1/VACE code hard-requires `flash_attn`, which has no Windows wheels.
- It also loads the 7 GB fp32 model whole onto the GPU, and ignores `--frame_num` when given a pose video.
- ComfyUI's core nodes avoid all three problems, and it can be driven by API JSON.
- ComfyUI does **not** resample the control video, so we supply exactly 16 fps × 33 frames.

## Era styling helps quality
- A low frame rate, grain, B&W and blur all hide diffusion artifacts (flicker, broken hands, a bent bat).
- Cutting between shots keeps each generated segment short (about 2 s).

## Licences to state in the report
| Component | Licence |
|---|---|
| VACE, Wan2.1 | Apache-2.0 |
| rtmlib code | Apache-2.0 |
| DWPose weights | Trained on COCO-WholeBody, so **no direct commercial use** |
| ComfyUI | GPL-3.0 |
| DINOv2 | Apache-2.0 |
