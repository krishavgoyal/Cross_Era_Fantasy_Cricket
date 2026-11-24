# Report notes: VACE pipeline (Krishav)

Working notes for the final report. Everything here is what the code in `scripts/` and `kaggle/` actually does. Fill in the **[TODO]** fields from `results/metrics.csv` and `results/kaggle_runs.csv` once the Kaggle outputs are back.

## 1. Method summary

```
modern clip --[1 pose]--> raw.npz --[2 control]--> control_33.mp4 --+
                                                                    +--[3 VACE on Kaggle T4]--> generated clip
reference photo -----------------------------(retarget, reference)--+
generated clip --[4 era filters]--> styled clips --[5 metrics]--> results/metrics.csv
bowler clip + batter clip --[6 ffmpeg edit]--> final.mp4
```

| Stage | Script | Runs on | What it does |
|---|---|---|---|
| 1. Pose | `scripts/extract_pose.py` | Laptop CPU | rtmlib: YOLOX-m person detector + **DWPose-l 384x288** (133 whole-body keypoints). Target player chosen by `--init-box` on frame 0, then followed by best IoU (centre-distance fallback). Output `raw.npz` + `overlay.mp4`. |
| 2. Control | `scripts/make_control.py` | Laptop CPU | Mask keypoints with score < 0.3 -> fill gaps <= 4 frames -> Savitzky-Golay (window 7, order 2) -> resample to 16 fps, 33 frames -> optional per-bone retarget to the reference photo (ratios clamped 0.6-1.6) -> one scale/offset so the body fills ~85% of a 480x832 canvas -> OpenPose-style skeleton (body, hands, face) on black. |
| 3. Generation | `kaggle/vace_generate.ipynb` | Kaggle T4 (16 GB) | **Wan2.1-VACE-1.3B** in headless ComfyUI (API only), reference photo + control video. Each output saved immediately; raw copy kept for metrics, labelled copy for display. |
| 4. Era styling | `scripts/era_filters.py` | Laptop CPU | `1930s_film` (grayscale, grain, blur, gate jitter, flicker, dust/scratches, vignette, 18 fps), `1970s_tv` (desaturation, chromatic aberration, scanlines, tape noise, 25 fps), `1990s_broadcast` (mild colour, chroma bleed, softness, 25 fps). Seeded. 1930s/1970s ported from the earlier Student 3 code. |
| 5. Metrics | `scripts/eval_metrics.py` | Laptop CPU | PCK@0.2 (DWPose re-run on the output vs the control skeleton, 17 body keypoints, torso-normalised), DINOv2-small cosine identity (mean/min, grayscale crops), flicker (std of per-frame mean brightness, mean frame-to-frame difference). On raw AND styled videos. |
| 6. Edit | `scripts/make_sequence.py` | Laptop (ffmpeg) | Bowler clip + batter clip (+ title cards) -> `final.mp4`, letterboxed to one size, 25 fps. |

Bowler and batter are generated **separately** and cut together like a broadcast; no open model reliably generates a two-player interaction.

## 2. Settings

| Setting | Value |
|---|---|
| Generator | Wan2.1-VACE-1.3B (`wan2.1_vace_1.3B_fp16.safetensors`) |
| Text encoder / VAE | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` / `wan_2.1_vae.safetensors` (Comfy-Org repackaged) |
| Resolution | **480x832** portrait (VACE-1.3B supports only 480x832 / 832x480) |
| Frame rate / length | **16 fps, 33 frames** (~2 s); smoke test 17 frames. Frame counts must be 4n+1 |
| Sampler | KSampler, **20 steps, cfg 5.0, uni_pc / simple, ModelSamplingSD3 shift 8**, denoise 1.0 |
| Seed | **42** |
| Precision / attention | **fp16** (`--force-fp16`, the T4 has no bf16), VAE in fp32 (`--fp32-vae`), PyTorch SDPA attention |
| Hardware | Kaggle **T4 16 GB**, one GPU (`CUDA_VISIBLE_DEVICES=0`); laptop RTX 4050 6 GB used only for CPU stages |
| ComfyUI | v0.46.0 tag (falls back to latest if the tag is missing; record the actual version from the notebook log) **[TODO]** |
| Prompt | "full-body photo of a cricketer in white flannels and cap holding a wooden cricket bat, plain sports ground background" |
| Measured | Smoke test: **17 frames in 265 s** on the T4. 33-frame time and peak VRAM: **[TODO from kaggle_runs.csv]** |

## 3. Kaggle bugs found and fixed

| # | Symptom | Cause | Fix (commit) |
|---|---|---|---|
| 1 | ComfyUI failed: `No module named 'torchsde'` | Cell 2 skipped every requirement line starting with "torch" (to keep Kaggle's preinstalled torch), which also dropped `torchsde` | Skip only exact `torch` / `torchvision` / `torchaudio`; `pip install torchsde` explicitly and import it (`cf2d320`) |
| 2 | `No input dataset found under /kaggle/input` | Kaggle mounts datasets at `/kaggle/input/datasets/<user>/<slug>/<folder>/`, deeper than the depth-3 `os.walk` | Locate `pairs.csv` with `find -L /kaggle/input` (no depth limit, follows symlinks), `os.walk` fallback (`cf2d320`) |
| 3 | Generation succeeded but `No such file: .../output/bradman_batting_<uid>_ctrl.mp4` | ComfyUI's `/history` "images" list also contains the LoadVideo **input** preview | Keep only entries with `type == "output"`; fall back to the newest `.mp4` in `output/cefc/` written during the run (`f79978c`) |

Also fixed before the first run: the burned-in label outline drifted out of line with the text (OpenCV spaces glyphs by stroke thickness), and the smoke test now also fails on a black/flat output (fp16 NaN symptom), not only on errors.

## 4. Why LTX-2 and MagicAnimate (named in the proposal) were replaced

From the Phase 0 research (`docs/research/PHASE0_RESEARCH.md`, `docs/research/PHASE0_1_FREE_ALTERNATIVES.md`):

- **LTX-2**: has pose control (IC-LoRA), but the reference photo is used only as the **first frame**. There is no identity encoder and no pose alignment, so the photo must already be in the first pose of the clip. The official target is 32 GB+ VRAM; the model is 19-22B parameters.
- **MagicAnimate**: SD1.5 + DensePose, code unchanged since Dec 2023, no DensePose extractor or pose alignment shipped, and DensePose is fragile on broadcast footage. It is kept only as a baseline (the team's AnimateDiff engine plays this role, see `docs/TEAM_INTEGRATION.md`).
- **VACE-1.3B** accepts a reference image plus a pose control video, is Apache-2.0, is small enough for a free T4, and runs with ComfyUI core nodes. The official Wan2.1 repo was not used: it hard-requires `flash_attn` and loads the fp32 model whole onto the GPU.
- Platform: Kaggle's free T4. Google Colab was avoided because its FAQ prohibits "creating deepfakes" and web UIs on the free tier.

## 5. Ablation: likeness and bat (2x2 + seeds)

**Problem seen in run A** (pose skeleton + padded `bradman.png`, seed 42, 17 frames): generic face, a wrong or missing bat (the skeleton carries no bat), extra balls, and the head cut off at the top of the frame.

**Two changes, tested separately and together:**

| Factor | Old | New | Script |
|---|---|---|---|
| Control | OpenPose skeleton (`control_33`, no margin) | **Depth** (Depth Anything V2 Small, Apache-2.0): carries the bat and the body volume. One global min/max over all frames (no flicker), near = bright, head region flattened (`--soften-head`) so the modern helmet does not override the face | `make_depth_control.py` |
| Reference | `bradman.png`: photo letterboxed with large white bands, face small | **`bradman_crop.png`**: background removed (rembg u2net, bat kept), on pure white, subject fills 90% of the height, head 3% from the top; up to 15% of the subject box trimmed per side (elbow and lap) | `prepare_ref.py` |
| Framing | Skeleton touched the top edge | `--top-margin 0.08` on both pose and depth controls (head-top estimate and feet included in the box), same canvas transform for both, so they line up frame by frame | `make_control.py` |

| | Padded ref `bradman.png` | Tight ref `bradman_crop.png` |
|---|---|---|
| **Pose control** | **A** (seed 42, done) | **C** (seed 42) |
| **Depth control** | **B** (seed 42) | **D1-D4** (seeds 42, 7, 123, 2024) |

- All new runs: 17 frames, 480x832, 16 fps, 20 steps, cfg 5, shift 8, new prompt ("archival 1930s photograph style, cricketer in white flannel shirt and trousers and a cap, holding one wooden cricket bat with both hands, playing a cover drive, natural motion"), negative prompt extended with "extra balls, multiple balls, two bats, deformed bat, hockey stick, extra limbs, extra fingers, cropped head, helmet, modern logos".
- Run A used the old prompt and the no-margin pose control, so A vs C mixes the reference change with the prompt and margin change. B vs D1 (same control, same prompt, same seed) is the clean test of the reference; C vs D1 is the clean test of the control.
- D1-D4 differ only in the seed: their spread shows how much of any difference is just seed noise. The best D seed is then re-run at 33 frames (`BEST_SEED` cell in the notebook) and used for the final video.

**Measured** with `scripts/score_ablation.py` (run list in `results/ablation_runs.csv`) into `results/ablation.csv`: PCK@0.2 against each run's own control (17-frame runs are scored on the first 17 control frames), DINOv2 identity against both references, flicker. **Also check by eye:** one bat (not two, not a hockey stick), no extra balls, the whole head in frame, and whether the face looks more like the photo.

**Baseline (run A and the earlier 33-frame run, scored 2026-11-24):**

| Run | Control | Ref | Frames | PCK@0.2 | DINO vs crop (mean) | DINO vs orig (mean) | Brightness std | Frame diff |
|---|---|---|---|---|---|---|---|---|
| A | pose (no margin) | padded | 17 | 0.934 | 0.205 | 0.158 | 2.27 | 10.10 |
| A, 33 frames | pose (no margin) | padded | 33 | 0.882 | 0.189 | 0.124 | 4.46 | 12.55 |
| B / C / D1-D4 | | | 17 | [TODO] | [TODO] | [TODO] | | |

Reading: motion following is already good (PCK 0.88-0.93), so the problem is appearance. The low identity scores (about 0.2) match the generic face seen by eye. These are whole-body crops: the reference shows a seated man in a suit and the output a standing batter in whites, so the absolute values are low for every run. **Compare runs with each other, not with 1.0.**

**Ablation outcome (Kaggle, 2026-11-24).** B, C and D1-D4 all finished with status "ok", about 280 s each for 17 frames on the T4. Visual check:

| Run | Control / ref / seed | Bat | Head in frame | Extra balls | Other |
|---|---|---|---|---|---|
| B | depth / padded / 42 | one proper bat, whole clip | yes | none | large dark cap hides the face |
| C | pose / crop / 42 | **lost** (like run A) | yes | none | |
| D1 | depth / crop / 42 | one proper bat, whole clip | yes | none | dark cap hides the face; logo on the cap. **Best overall** |
| D2 | depth / crop / 7 | one proper bat, whole clip | yes | none | dark cap hides the face |
| D3 | depth / crop / 123 | one proper bat, whole clip | yes | none | dark cap; modern black pads / black trousers |
| D4 | depth / crop / 2024 | one proper bat, whole clip | yes | none | dark cap; modern black pads / black trousers |

- **Depth control is what fixed the bat.** All depth runs (B, D1-D4) keep one proper bat for the whole clip, while pose + tight reference (C) loses it just like run A (pose + padded). The skeleton has no bat information; the depth map does.
- **The top margin fixed the cut-off head**, and the depth runs have **no extra balls**.
- **Remaining problem: the face.** A large dark cap hides it. Two causes: the prompt asks for "a cap", and the depth map still carries the modern helmet's outline (`--soften-head` flattens only the inside of the head). D1 adds a logo on the cap; D3/D4 drift to modern black pads and trousers, so seed matters for clothing.
- **Chosen:** D1 (seed 42). Final runs use depth + `bradman_crop.png`, 33 frames, seed 42, in two variants:
  - **F1 `final_cap`**: the ablation prompt.
  - **F2 `final_nocap`**: a bare-headed prompt ("bare-headed cricketer with short neatly combed hair, face clearly visible, white flannel shirt and white trousers, white leg pads, ..."), with the negative prompt extended by "cap, hat, helmet, logo, black pads, black trousers, dark pads".
  - The default negative now also includes "logo, black pads, black trousers".
  - Outputs: `final_cap_s42_33f.mp4`, `final_nocap_s42_33f.mp4`. Score both with `score_ablation.py` (control `data/pose/batter/depth_33.npz`) and compare face visibility by eye.

## 6. Results [TODO]

Paste from `results/metrics.csv` (raw vs styled, VACE vs AnimateDiff baseline):

| Clip | Kind | PCK@0.2 | DINO mean / min | Brightness std | Frame diff |
|---|---|---|---|---|---|
| bradman_vace | raw | | | | |
| bradman_vace | 1930s_film | | | | |
| animatediff_baseline | raw | (see TEAM_INTEGRATION.md) | | | |

How to read them: PCK@0.2 = share of body joints within 20% of the torso length of where the control put them. DINO = cosine similarity of image features (1 = identical). Lower brightness std / frame diff = less flicker. Styled versions are expected to score lower on PCK/DINO because grain, blur and jitter are added on purpose.

## 7. Limitations

- **Weak face likeness.** The reference is a single black-and-white archival photo, padded to the portrait canvas; at 480 px the face is a few dozen pixels, and VACE injects the reference rather than encoding identity strongly.
- **Modern look before styling.** Raw outputs show modern colour and modern shoes/kit; the era filters (grayscale, grain, 18 fps) hide much of this, which is part of the design.
- **Bat.** The bat held up in the 17-frame smoke test, but the skeleton carries no bat information, so bat quality is not guaranteed on longer or faster clips (bowling).
- **Single reference photo.** Per-bone retargeting from one photo is noisy (ratios clamped to 0.6-1.6); the photo's pose and framing strongly affect the result.
- **Short clips.** 33 frames at 16 fps is ~2 s per player; the final film is made from cuts, not one continuous shot.
- **One seed, few clips.** Results come from seed 42 and a handful of clips; no variance across seeds was measured.
- **Identity metric.** DINOv2 on whole-body crops also measures clothing and pose (the reference is seated, in a suit), so absolute values are low; use it to rank runs, not as a likeness score.
- **Metric caveats.** PCK uses the same DWPose model that produced the control (shared biases). DINOv2 measures general visual similarity, not face identity. The flicker metric is lowered by repeated (held) frames, which matters for the AnimateDiff baseline.
- **Precision.** fp16 on T4 with an fp8-stored text encoder (dequantised, the T4 has no native fp8).

## 8. Ethics

- Every generated video and title card carries a burned-in **"AI-generated - academic project"** label. Unlabelled copies are used only to compute metrics.
- The videos depict real cricketers (some living). They are synthetic, made for academic use only, and must not be distributed as real footage.
- Kaggle notebooks and datasets are kept **private**. Broadcast clips are used for private research only and are not redistributed.

## 9. Licences

| Component | Licence / terms |
|---|---|
| Wan2.1-VACE-1.3B weights, Wan2.1 | Apache-2.0 (Wan terms forbid content that spreads misinformation) |
| ComfyUI | GPL-3.0 |
| rtmlib (code) | Apache-2.0 |
| DWPose / RTMW weights | Trained on COCO-WholeBody: **no direct commercial use** |
| DINOv2-small | Apache-2.0 |
| Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) | Apache-2.0 |
| rembg (code) / u2net weights | MIT / Apache-2.0 |
| PyTorch / transformers | BSD-3 / Apache-2.0 |
| OpenCV / NumPy / SciPy | Apache-2.0 / BSD-3 / BSD-3 |
| FFmpeg (gyan.dev build) | GPL-3.0 |
| Kaggle | Kaggle Terms of Use (no deceptive content) |
| Reference photos / clips | Record the source and licence of each file in the report **[TODO]** |
