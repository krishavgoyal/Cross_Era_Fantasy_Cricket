# Cross-Era Fantasy Cricket: why the output distorts, what to use instead

*Integration report (Pranahv), 23 Nov 2026. Covers the first full GPU run of `master_pipeline.py` on branch `Pranahv-main`.*

---

## 1. Summary

The pipeline now works end to end: modern clip → motion → DensePose → generation → colour → 24 fps → demo → evaluation. The first full run on the RTX 5060 produced a recognisable 1930s batsman following the modern batter's stance.

The distortions in that output (white sun hat, a sweater that isn't in the photo, flared trouser-legs instead of pads, a bat pointing the wrong way, cut-out edges, smeary motion) are **not integration bugs**. They come from four design choices inside the generation stage:

1. **An older, small model stack.** Stable Diffusion 1.5 + AnimateDiff, generating 16 frames at 384×512.
2. **Weak appearance transfer.** The IP-Adapter compresses the whole Bradman photo into one compact embedding, so the cap, shirt and face get lost.
3. **Hand-drawn fixes on top of the model.** A synthetic bat at fixed angles, painted shin lines, a pasted background plate and brightened clothes.
4. **Sparse frames.** Only 16 keyframes over the whole shot, with ffmpeg filling the gaps.

Newer methods replace each of these. In order of effort:

- **Path B (quick wins, fits the 8 GB laptop):** keep the current pipeline, replace the hand-drawn parts with real signals and generate more frames. This fixes the bat, the legs and most of the smear.
- **Path A (biggest quality jump):** "character replacement" with Wan2.2-Animate. It swaps the modern batter for Bradman *inside the original video*, so the real bat, ground and lighting stay. It needs a rented cloud GPU, or a quantised build on the laptop.
- **Path C (most robust, most engineering):** 3D. Recover the batter's body as a 3D model (GVHMR/SMPL), drive a rigged 1930s avatar, then style it.

**Recommendation:** do Path B now for the submission, and run Path A once as a comparison experiment (it's the strongest result to show in the presentation).

---

## 2. What the first full run actually produced

| Stage | Result |
|---|---|
| Motion + DensePose (Student 1) | Reused from the repo (`good_cover_drive_final`, 139 frames, 512×512 @ 24 fps) |
| Generation (Student 2) | 16 keyframes, 384×512, SD1.5 + AnimateDiff + DensePose ControlNet + IP-Adapter, on the RTX 5060 |
| Colour (Student 3 v3) | DensePose-guided colouring, mask coverage about 96% |
| Retime | 16 keyframes played at real-time rate (about 5.1 fps), interpolated to 24 fps |
| Demo | Source on the left, final on the right |

**What works well:**
- The body posture tracks the source: knee bend, trunk lean and head position all line up.
- The pipeline runs unattended and resumes after a crash.
- After the `.wslconfig` fix (12 GB RAM, 16 GB swap), the laptop has enough memory.

---

## 3. Distortion audit: what we see → why → fix

The reference photo (`data/historical_images/bradman.jpg`) shows Bradman in a **dark baggy green cap**, a **plain white open-neck shirt**, white pads, holding the bat down by his feet. Compare that with the output frame:

| # | What we see | Why it happens (with the code that causes it) | Fix |
|---|---|---|---|
| 1 | **White sun hat** instead of the dark cap | The IP-Adapter (`ip-adapter_sd15.bin`, scale 0.62) turns the whole photo into one global CLIP embedding. A cap only about 40 pixels across barely registers. SD 1.5's built-in idea of an "old cricketer" (sun hat) then wins, even though the prompt says "dark baggy green cap" and the negative prompt says "white hat". The post-process that brightens clothing toward white may also catch the head area. | Use IP-Adapter **Plus-Face** or **FaceID** with a *face-and-cap crop*. Keep the brightening mask off the head. Long-term: a reference-net model (MagicAnimate, Wan-Animate), which reads the photo at full detail. |
| 2 | **Patterned sweater** that isn't in the photo | Same cause: the IP-Adapter carries "vintage knitwear" as a style, not the actual garment. MagicAnimate's authors describe CLIP-style embeddings as "too sparse and compact to capture intricate details". | Same as 1. A reference-net or appearance encoder copies the real garment. |
| 3 | **Bat points the wrong way.** The source batter's bat is raised in the backlift; the generated bat hangs down-left. | The bat is **not tracked**. `generation_engine.py` draws it at fixed angles of 96° → 108° → 134° → 154° → 162° over the shot (`t_keys` / `ang_keys`), from a hand position guessed from colours. It then gets composited, and Student 3 colours it as wood. | Track the real bat in the source with SAM2 (Student 3 already uses SAM2). Take its angle per frame from the mask's main axis and feed that to the ControlNet. Path A keeps the real bat with no extra work. |
| 4 | **Legs balloon into flared trousers; pads disappear** | (a) The DensePose input comes from a small batter upscaled 2.5 to 4.5×, so the leg shapes are blobby. (b) The engine paints "shin bridges" in a colour (45,160,145) that **is not a DensePose part colour**, so the ControlNet treats those pixels as something unknown. (c) For the batsman, the leg mask is dilated with a 9×9 kernel before compositing, which widens the legs. (d) DensePose has no "pad" part, so pads and trousers look the same to the model. | Build the ControlNet input from `densepose_real_parts.npz` (exact part ids, real colour map), not from colour guesses plus painted lines. Add a second ControlNet with the **DWPose skeleton** (we already have `control_pose.mp4`) so knees and ankles are pinned. Drop the 9×9 dilation. |
| 5 | **Cut-out look, halo, player floats on the grass** | The generated person is pasted onto a blurred background plate using the DensePose silhouette as the cut-out mask. Where the generated body is thinner than the silhouette, the generated background shows through as a bright rim. There's no shadow or ground contact. | Matte the generated frame itself (Robust Video Matting), intersected with a slightly dilated DensePose mask. Add a soft contact shadow under the feet. Path A avoids pasting altogether, because it generates the person *into* the real scene with relighting. |
| 6 | **Smear and ghosting in fast motion** | Only 16 keyframes cover about 3 s of the shot, so about 5 per second. The 24 fps video is filled in by ffmpeg `minterpolate`, which blends pixels and ghosts on limbs that move a lot between keyframes. | Generate more frames with AnimateDiff's **sliding context window** (16-frame windows, 4 overlap), at 12 fps for example, then **RIFE** to 24 fps. RIFE is a learned frame-interpolation model and is much cleaner than `minterpolate`. |
| 7 | **Generic face, not Bradman** | At 384×512 the face is about 40 px, and the IP-Adapter carries no identity detail. | Higher resolution (512×768). A face-specific adapter (IP-Adapter FaceID, or StableAnimator's face encoder). Optionally a face-restore pass. |
| 8 | **Dark smear in the background** | The background plate is a heavily blurred crop of the reference photo's trees (`GaussianBlur(31,15)`). | Use a clean, period-correct background plate (an archival ground photo), or keep the real background (Path A). |
| 9 | **Framing differs from the source** | The source is a portrait phone clip with black bars. The generator crops around the DensePose body with padding, then resizes to 384×512. | Fine for the demo. Show both at the same body height in the side-by-side. |
| 10 | **Bowler: the body collapses at release** (Ahaan's bowler video) | Fast arm, tiny figure (90 to 170 px), DensePose weakest exactly at release (frame 118 needed patching). The built-in window gave the arm action only about 5 of 16 keyframes. | Use `--auto-window` (9 keyframes on the arm), use the skeleton ControlNet as well, and generate more frames (fix 6). Path A handles this far better. |

---

## 4. Root causes, grouped

1. **Model capacity and age.** SD 1.5 and AnimateDiff v1-5-2 date from 2023. They're trained around 512 px and 16-frame clips, and their sense of "1930s cricketer" is vague. Recent human-animation models are built on video diffusion (Stable Video Diffusion, Wan) and trained specifically on people moving.
2. **How appearance is transferred.** The IP-Adapter turns the reference into one global embedding. MagicAnimate, Animate Anyone and Wan-Animate instead read the reference image's *spatial features* at every step (a reference network or appearance encoder). That's how they keep the cap, the shirt and the face.
3. **How motion is given.** One DensePose ControlNet input, built from an upscaled, low-resolution source, with colour heuristics and painted patches. Better methods combine signals: MimicMotion uses DWPose with per-keypoint confidence, Champ uses SMPL depth, normals and semantic maps together, and Wan-Animate uses skeleton signals plus facial features.
4. **Hand-drawn fixes.** The synthetic bat, shin lines, brightened clothes and pasting were sensible stopgaps, but each one adds a failure of its own (wrong bat angle, flared legs, halo). Each should be replaced by a real signal (tracked bat, exact parts, proper matting).
5. **Temporal sampling.** Too few keyframes plus non-learned interpolation produces ghosting. Sliding-window generation plus RIFE fixes this.
6. **Source quality.** The bowler is 90 to 170 px tall in the broadcast. Every stage downstream inherits that blur. Higher-resolution source clips (or a closer broadcast angle) help everything.

---

## 5. Alternatives researched

| Method | Built on | Motion signal | How it keeps appearance | Resolution / length | Memory | Fit for us |
|---|---|---|---|---|---|---|
| **Current (Ahaan)** | SD 1.5 + AnimateDiff | DensePose ControlNet | IP-Adapter (global) | 384×512, 16 frames | Fits 8 GB with CPU offload | Works; the distortions above |
| **MagicAnimate** | SD 1.5 + temporal layers | DensePose | Appearance encoder (copy of the UNet reading the reference) | 512, overlapping segments for long clips | Similar to ours | The design Ahaan's engine approximates; its appearance encoder fixes issues 1, 2 and 7 |
| **Champ** | SD + 3D guidance | SMPL depth, normal, semantic map and skeleton together | Reference net | 512 | Higher | Better body shape and limbs (issue 4); needs SMPL fitting |
| **MimicMotion** | Stable Video Diffusion | DWPose, confidence-weighted | Reference image through SVD | Up to 72 frames at 576×1024; long clips by overlapping segments | 16-frame model about 8 GB; 72-frame model 16 GB | Closest drop-in upgrade: we already have DWPose; may fit the laptop at 16 frames |
| **StableAnimator** | Stable Video Diffusion | Pose | Face encoder + ID adapter + face optimisation while generating | 512+ | Higher than MimicMotion | Best for the face problem (issue 7) |
| **Wan2.2-Animate-14B** | Wan 2.2 (14B video model) | Skeleton + facial features | Reference conditioning; **replacement mode** with a relighting LoRA | 720p | Heavy: multi-GPU or a large single GPU; quantised builds (Wan2GP) on consumer cards | Best quality and keeps the real scene (Path A) |
| **Wan2.1-VACE-1.3B** | Wan 2.1 (1.3B) | Pose, depth or masked video editing | Reference image or video | 480p, up to 81 frames | The base 1.3B model needs about 8.2 GB | Borderline on 8 GB; good for masked replacement of just the batter |
| **3D route (GVHMR)** | Not diffusion | 3D body model (SMPL) per frame from the video | A rigged 1930s avatar | Any | GVHMR is fast | Perfect consistency and a real bat as a 3D prop; needs an avatar and Blender work (Path C) |

**Supporting tools:**
- **RIFE:** learned frame interpolation, replaces `minterpolate`.
- **Robust Video Matting:** temporally stable person matting, replaces the silhouette cut-out.
- **AnimateDiff sliding context windows:** generate more than 16 frames with the current model.

---

## 6. Recommended plan

### Path B: quick wins in the current pipeline (do now, 8 GB laptop)

In priority order. Each is a contained change in `generation_engine.py` or the master pipeline.

1. **Real ControlNet input.** Render the conditioning from `densepose_real_parts.npz` through `densepose/dp_common.colorize` (exact parts, consistent colours). Delete the shin-bridge painting and the colour-threshold hand and arm detection.
   *Fixes legs and pads (4). Prerequisite for everything else.*
2. **Add the skeleton ControlNet.** Use `lllyasviel/control_v11p_sd15_openpose` alongside the DensePose ControlNet (diffusers MultiControlNet), fed from Student 1's `control_pose.mp4`. Start at weights of about 0.6 (skeleton) and 0.9 (DensePose).
   *Pins knees, ankles and arms (4, 10).*
3. **Track the real bat.** Use SAM2 on `source_512.mp4`, prompted on the bat in frame 0, to get a mask per frame. Take the bat's centre line from the mask's main axis, map it into the generated frame with the recorded crop (`generated.mp4.json → sampling`), and replace the hardcoded `ang_keys`.
   *Fixes 3.*
4. **More frames.** Use AnimateDiff's sliding context (16-frame window, overlap 4) over every second source frame (12 fps, about 36 frames for the batting window), then RIFE to 24 fps in the retime stage.
   *Fixes 6.*
5. **Identity.** Use IP-Adapter Plus-Face (or FaceID) with a crop of Bradman's head and cap from the photo, plus the existing full-body adapter at a lower weight. Keep the brightening mask off the head.
   *Fixes 1, 2 and 7.*
6. **Resolution 512×768** if memory allows. Measure with `vram_report.json`, which every run already writes.
7. **Matting and shadow.** Run Robust Video Matting on the generated frames, intersected with a slightly dilated DensePose mask, and add a soft contact shadow.
   *Fixes 5.*
8. **Background.** Use a clean archival ground photo instead of the blurred reference crop.
   *Fixes 8.*
9. **Check the ControlNet colours once.** Confirm that `MnLgt/densepose` was trained on the same DensePose colour style we render. If not, render in its style. This is still unverified.

**How to judge each change:** keep the same seed, change one thing at a time, and compare against the current run on:
- PCK (motion accuracy; `evaluation.json` once DWPose is installed in the `gen` env)
- face similarity to the reference (ArcFace cosine)
- the Student 3 mask coverage
- a side-by-side look

### Path A: character replacement with Wan2.2-Animate (comparison experiment)

1. Inputs: `source_512.mp4` (or the original broadcast clip) plus `bradman.jpg`.
2. Run Wan2.2-Animate in **replacement** mode with the relighting LoRA on a rented GPU (an A100 or H100 hour), or try a quantised build through Wan2GP on the laptop.
3. Feed the result into the existing pipeline: `python master_pipeline.py ... --generated <wan_output.mp4>`. Colour, era styles, retime, demo and evaluation all run unchanged. This is exactly what `--generated` was added for.
4. Present it side by side with the Path B output. This shows the team understands the trade-off: modular, explainable and laptop-sized versus a state-of-the-art single model.

### Path C: 3D (future work, mention in the report)

GVHMR recovers the batter's 3D body (SMPL) from the video. That drives a rigged 1930s cricketer avatar in Blender, with the bat as a 3D prop attached to the hands, rendered and then styled by Student 3. It's the most controllable and consistent route, but needs avatar creation and rigging time.

---

## 7. Everything changed so far, and why

### Student 1 data (DensePose)
| Change | Reason |
|---|---|
| Real DensePose run for both clips on the laptop (detectron2 R50-FPN, CPU), via `densepose/run_densepose.sh` | The generator's ControlNet needs DensePose, not skeletons; it carries body shape and front/back |
| Picking the person by overlap with the tracked box (`dp_to_control.py --metadata`) | The bowler crop contains the umpire; the largest person isn't always the bowler |
| Bowler tracked and cropped dynamically (seeded on the yellow kit), 512×512 @ 24 fps, 121 frames | Isolate the bowler at the required resolution |
| Weak-frame repair (`patch_weak_frames.py`): frame 118 legs filled from the skeleton | Motion blur made DensePose lose the legs at release |

### Integration (`Pranahv-main`)
| Change | Reason |
|---|---|
| `master_pipeline.py`: one command, 7 stages, resumable, `manifest.json`, the exit codes Student 1 already uses | Nothing chained the students' modules; Student 3 ran from hardcoded Windows paths |
| `densepose/` scripts committed | The DensePose data was in the repo but couldn't be reproduced |
| Generation: fp32 on CPU, honours `--prompt / --frames / --fps / --seed / --steps / --window`, explicit role, frame alignment kept, `generated.mp4.json` sidecar | It crashed on CPU, ignored its CLI, guessed the role from filenames, and dropped frames silently |
| No frozen padding; keyframes played at real-time speed, then 24 fps | The output was 2.3 s of motion plus frozen frames at 7 fps |
| `--auto-window` (peak wrist speed) | The bowling arm got only about 5 of 16 keyframes |
| Student 3 v3 integrated as the default colour stage; given the sidecar and exact parts | v3 was unpushed; it re-derived the crop and decoded colours (coverage rose from 0.885 to 0.907 bowler, 0.945 to 0.964 batsman) |
| PCK evaluation (mine and Krishav's, from one DWPose run) | Motion accuracy had never been measured |
| `--generated <video>` (use a video Student 2 already made; frozen frames are dropped) | Lets the pipeline run without a GPU and accept outside generators (Path A) |
| `--force all` keeps the processed clip data | It wrongly forced the motion stage |
| Stable Diffusion 1.5 fallback to `stable-diffusion-v1-5/stable-diffusion-v1-5` | The `runwayml` repo was taken down (404) |
| `RUN_ON_GPU.md`: WSL RAM fix (12 GB + 16 GB swap), environment setup, run commands | The first GPU run was killed silently at 7.4 GB RAM |

---

## 8. Steps to reproduce the current output

```bash
# once (Windows PowerShell): WSL memory
#   [wsl2] memory=12GB  swap=16GB  in %USERPROFILE%\.wslconfig, then: wsl --shutdown
# once (Ubuntu): environment, see RUN_ON_GPU.md
source ~/gen/bin/activate
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
python master_pipeline.py --clip-id good_cover_drive_final --image data/historical_images/bradman.jpg --role batsman
python master_pipeline.py --clip-id bowler_delivery_001 --image data/historical_images/modern_bowler.jpeg --role bowler --auto-window
# results: runs/<clip>_<role>/final_24fps.mp4, demo_side_by_side.mp4, student3_v3/era_*.mp4, evaluation.json, manifest.json
```

Without a GPU, using an existing generated video:
```bash
python master_pipeline.py --clip-id bowler_delivery_001 --image data/historical_images/modern_bowler.jpeg --role bowler --generated outputs/m3_bowler_delivery.mp4
```

---

## 9. Open questions for the team

1. **Ahaan:** is the Path B list acceptable inside `generation_engine.py`, or should it become a separate `generation_v2` so Milestone 3 stays reproducible? (Suggested: a new module, selected with a master flag.)
2. **Budget:** can we rent one GPU hour for the Wan2.2-Animate comparison (Path A)?
3. **Presentation:** show Path B as "our pipeline" and Path A as "state of the art comparison"?
4. **Ethics line for the slides:** Bradman is a real, deceased public figure. Label every output as an AI reconstruction.

---

## Sources

- MagicAnimate (DensePose + appearance encoder; CLIP features "too sparse and compact"): [arXiv 2311.16498](https://ar5iv.labs.arxiv.org/html/2311.16498), [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_MagicAnimate_Temporally_Consistent_Human_Image_Animation_using_Diffusion_Model_CVPR_2024_paper.html)
- Champ (3D parametric guidance: depth, normal, semantic, skeleton): [arXiv 2403.14781](https://arxiv.org/abs/2403.14781v1), [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7192_ECCV_2024_paper.php)
- MimicMotion (SVD, confidence-aware DWPose, progressive latent fusion; VRAM figures): [GitHub](https://github.com/Tencent/MimicMotion), [arXiv 2406.19680](https://arxiv.org/html/2406.19680v2)
- StableAnimator (identity-preserving face encoder and ID adapter): [arXiv 2411.17697](https://arxiv.org/html/2411.17697v2), [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Tu_StableAnimator_High-Quality_Identity-Preserving_Human_Image_Animation_CVPR_2025_paper.html)
- Wan-Animate (animation and replacement, relighting LoRA): [arXiv 2509.14055](https://arxiv.org/abs/2509.14055), [model card](https://huggingface.co/wan-ai/wan2.2-animate-14b), [ComfyUI guide](https://docs.comfy.org/tutorials/video/wan/wan2-2-animate)
- Wan2.1-VACE-1.3B (pose/depth control, masked V2V, 480p; T2V-1.3B 8.19 GB): [model card](https://huggingface.co/wan-ai/wan2.1-vace-1.3b), [ComfyUI VACE](https://docs.comfy.org/tutorials/video/wan/vace)
- Wan2GP (low-VRAM Wan builds, quantisation, RTX 50xx): [GitHub](https://github.com/banodoco/Wan2GP)
- GVHMR (world-grounded SMPL from monocular video): [project page](https://zju3dv.github.io/gvhmr), [GitHub](https://github.com/zju3dv/GVHMR)
- RIFE (learned frame interpolation): [GitHub](https://github.com/hzwer/ECCV2022-RIFE), [arXiv 2011.06294](https://arxiv.org/abs/2011.06294v7)
- Robust Video Matting: [GitHub](https://github.com/PeterL1n/RobustVideoMatting), [arXiv 2108.11515](https://arxiv.org/pdf/2108.11515)
- AnimateDiff sliding context windows: [AnimateDiff-Evolved context options](https://comfy.icu/node/ADE_StandardUniformContextOptions)
- runwayml SD 1.5 takedown: [discussion](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/discussions/16)
