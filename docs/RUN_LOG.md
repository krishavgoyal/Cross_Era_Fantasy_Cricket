# Run log: GPU runs, failures and fixes

Newest first. One entry per run or change, with the evidence (log line, file) behind it.

## 2026-11-24 (evening): motion3d, rigid 3D skeletons from scratch (separate from the earlier results)
Full write-up: `docs/MOTION3D.md`. Skeleton outputs: `outputs/motion3d/`. Videos: `outputs/animate_v3/`.

- **Feedback:**
  - bat size changing at random;
  - arm and leg motion not smooth, joints speeding up or disappearing;
  - limbs not rigid;
  - feet not on a fixed ground;
  - asked for more reference videos and angles, a more complete stick figure, and the previous result kept.
- **New package `motion3d/`.** MediaPipe Pose heavy was added next to DWPose. In a separate venv `~/pose3d`
  (mediapipe needs numpy 2; libGLESv2 was extracted from the deb without root).
  - Ground reference by background tracking; one 3D fit per clip (PyTorch L-BFGS).
  - Renders from 6 angles; export with hands; new jobs `bradman_batting_3d`, `mcgrath_bowling_3d`.
- **Iterations on the fit, each checked on the multi-angle sheet:**
  1. A flat ground plane fitted to the planted feet tilted the side views (depth is the least certain axis).
     Changed to views turning about the camera vertical, plus a ground height per planted step.
  2. Bowling legs folded towards the camera. The Cummins clip cuts the feet off from frame ~36, and both
     detectors pinned the ankles to the frame edge. Fix: drop detections at the edge (40 px for feet); unseen
     limb bones stay near the image plane; unseen shins hang below the knee.
  3. The old delivery window 30–46 ended before the arm went over the top (it is vertical at 46–49 and over
     the top at 50–51). The blurred arm, frames 34–63, was annotated by hand
     (`keyframes_bowling_arm.json`). The clip now runs to frame 63 (follow-through); 34–58 is slowed 2.5×.
  4. Batting back arm swung in depth when hidden. Fix: hands-on-bat constraint plus the unseen-limb prior.
  5. Bowling shin fitted at 57 px against a 145 px thigh (rarely-seen limb). Fix: anthropometric proportion
     ranges.
- **Kaggle:** two pushes were cancelled while queued (jobs rebuilt after fixes 4 and 5).
- **Extra references** (`motion3d/extra.py`): Parry, Stokes and Onions deliveries (side-on) and the end-on bowler
  in the Amar Kala video, followed with YOLO from a hand-picked seed, DWPose 1024 + MediaPipe + fit + 6 views.

## 2026-11-24 (later): scenes tied to the ground, motion cleaning, real bat, rebuilt delivery
Full write-up: `docs/ANIMATION_REPORT.md`. Outputs: `outputs/animate_v2/`.

- **Feedback:** player in the middle of nowhere; limbs and bat moving randomly or merging; bat missing; bowling
  without a full step, arm rotation or release; overlaps; objects not persistent.
- **Batting scene:** the clip's own view. Camera zoom (0.91-1.0) measured on the background (ORB, people masked)
  and removed; plate = frame 0 with the batter removed (LaMa), visible stumps kept, letterbox bars filled.
  Bradman (bat removed from the photo by LaMa, u2net_human_seg cut-out) is placed on the batter's hips and feet.
  - A first try with a median plate over all frames ghosted, because the camera zooms. Replaced.
- **Bat:** composited after generation from the photo's bat sprite along the tracked direction, anchored at the
  wrists, minimum 55% length, darker willow tone plus a thin dark edge (it vanished against the shirt).
- **Kaggle `cricket-animate` v3** (bradman_batting + first mcgrath_bowling, 560x632, 30 steps, overlap 8): 57 min.
  - Bradman: PCK@0.1 0.978, complete body 0.986, merged limbs 0, silhouette IoU min 0.874, bat 139/139.
  - McGrath (that version): PCK@0.1 0.878, complete body 0.607, IoU min 0.593 with 6 drops. Causes: feet ~200 px
    above the ground (torso-only scaling); the stump duplicated and merged with the legs.
- **Bowling fixes:**
  - Stump as a fixed layer.
  - Per-bone retargeting on limb lengths only, left/right averaged: DWPose labelled the photo's legs the other way
    round from the clip's, and widths depend on the view.
  - Zoom-out with LaMa-extended grass so the action fits.
  - Second DWPose pass on raw frames merged in: frames with fewer than 12 joints went from 17 to 8.
  - Limb gaps filled by angle around the parent joint.
  - Delivery 30-46 slowed 2.5x, with a rebuilt straight bowling arm rotating -376 degrees (direction from the
    running direction).
  - Ball removed from the photo and composited: in the hand, released at the top of the arc, then thrown.
- **Kaggle v4:** 79 frames, rc=0, 28 min. PCK@0.1 0.896, complete body 0.684, merged limbs 0.063 (white blob in
  the bowling hand in frames 35-45: no hand points on the rebuilt arm; arm over the face at the end).
- **v5** was pushed with broken front-hand shapes (huge stretched hands passed the 12-point rule) and superseded.
- **v6:**
  - A measured hand shape rides along the rebuilt arm.
  - Hands bigger than 1.3x the forearm, or whose wrist point is off the body's wrist, are dropped (56 frames).
  - The clip ends at source frame 52.

## 2026-11-24

### MusePose run (Kaggle `cricket-animate` v2): rc=0, 73 min on a T4
- v1 failed at install: `av==11.0.0` has no Python 3.10 wheel and needs FFmpeg headers to build. Now
  `av==12.3.0`; decord dropped (training only); the script stops if the install fails.
- Install 1 min, weights 5 min, generation about 47 min for 265 frames (512x672, 20 steps, cfg 3.5,
  24-frame slices, overlap 4, seed 99).
- Outputs in `outputs/animate/`: `<job>.mp4`, `<job>_grid.mp4` (reference | skeleton | result),
  `<job>_demo.mp4` (motion source | result, labelled "AI reconstruction"), `<job>_eval.json`.

  | job | frames | PCK@0.05 | PCK@0.1 | PCK@0.2 | complete-body frames | face detected |
  |---|---|---|---|---|---|---|
  | bradman_cover_drive | 139 | 0.790 | 0.948 | 0.981 | 0.856 | 1.00 |
  | mcgrath_cummins | 45 | 0.763 | 0.854 | 0.896 | 0.578 | 1.00 |
  | mcgrath_runup | 81 | 0.945 | 0.976 | 0.991 | 0.938 | 1.00 |

  PCK here is against the retargeted target skeleton (what the player was asked to do), so it is not
  directly comparable to the earlier PCK against the source player. "Face detected" counts a DWPose nose,
  which DWPose also guesses on back views (mcgrath_runup), so it is weak evidence of a clear face.
- Visual: the photographed player in every frame, whole body, on the chosen ground; Bradman's cap, pads and
  gloves kept; bat visible in most frames but thin or merged with the gloves in some; McGrath-Cummins has a
  red smear at the ball and more missing limbs in the fast release frames (complete-body 0.58).

## 2026-11-23

### New approach: animate the reference player with MusePose (`animate/`)
Request: use only the skeleton motion (with the bat), make the player in the photo (Bradman batting,
McGrath bowling) perform it, complete body, no disappearing parts, clear face; backgrounds and extra
references from online; sort the files.

- Why: the SD 1.5 + IP-Adapter engine only takes a global "style" from the photo (generic, blurred face)
  and cut the body out with the DensePose silhouette (parts vanish where the model drew background).
  MusePose (TMElyralab, Animate-Anyone style reference UNet; code MIT, weights non-commercial research)
  generates the whole person from the photo along a DWPose skeleton, no cut-out.
- `lambdalabs/sd-image-variations-diffusers` is no longer public (401). MusePose only takes the UNet
  architecture from it (its own weights replace all UNet weights; reference UNet loaded strictly), so the
  SD 1.5 UNet config is used; image encoder = `openai/clip-vit-large-patch14` (the same CLIP ViT-L/14).
- Bat: `animate/bat_track.py`, YOLO26x-seg COCO "baseball bat" masks -> main axis, handle at the wrists,
  static vertical ground-level lines (stumps) rejected, gaps interpolated. good_cover_drive_final: detected
  in 122/139 frames, 139/139 after interpolation. Drawn as a light-wood stick in the pose video.
- Skeleton: body + hands from `pose_data.json`, joints missing for up to 8 frames filled from neighbours,
  3-tap smoothing, uniform scale + translation onto the reference player (`animate/skeleton.py`);
  back-view clips drop face points.
- Reference frame (`animate/reference.py`): rembg cut-out (u2net_human_seg for Bradman, whose whites on
  a light background came out half transparent with isnet; isnet for McGrath), holes filled, real gaps kept,
  placed on a Commons ground photo (SCG 1930s for Bradman, greyscale; Lord's 2017 for McGrath) with a
  contact shadow, 512x672.
- Jobs (`animate/build_jobs.py`): bradman_cover_drive (139 frames, bat), mcgrath_cummins (frames 0-44,
  side view), mcgrath_runup (bowler_delivery_001 frames 40-120, back view).
- Commons references (`tools/fetch_references.py`, `data/SOURCES_commons.md`): 3 Bradman photos, 3 grounds,
  3 bowling + 3 batting clips. The Commons clips are wide shots (players 30-250 px tall), too small for good
  skeletons, so they are kept as references only. Wikimedia throttled urllib (HTTP 429); curl works.
- Files sorted into `data/reference_players/`, `data/backgrounds/`, `data/motion_videos/` (`data/README.md`).
- Kaggle kernel `pranahvsubramaniam/cricket-animate`, job data as dataset `cricket-animate-extra-data`.

### Kaggle `cricket-clean` v2 (all three clips, CLEAN=1): rc=0, 28 min
- v1 of this kernel failed: `enable_free_noise_split_inference()` splits UNet activations but not the
  ControlNet residuals (768/1536 vs 3072). Removed; the 16-frame fallback now disables FreeNoise first.
- FreeNoise worked (no fallback): 32 frames, peak VRAM 9.22 GB, about 245 s generation per clip on the T4.
  Colour verdict pass for all three. Outputs in `kaggle_results/runs/colab_{bowler,batsman,bowler_new}_v2/`;
  earlier outputs unchanged in `kaggle_results_v1/`.

  | run | PCK@0.05 | PCK@0.1 | PCK@0.2 | torso PCK@0.1 | bowling/active arm @0.1 |
  |---|---|---|---|---|---|
  | bowler original, v1 | 0.154 | 0.375 | 0.721 | 0.125 | 0.000 |
  | **bowler original, v2 clean** | **0.561** | **0.804** | **0.919** | **0.638** | **0.614** |
  | batsman, v1 | 0.761 | 0.935 | 0.982 | 0.627 | 0.278 |
  | **batsman, v2 clean** | **0.833** | 0.934 | 0.969 | **0.696** | **0.414** |
  | bowler new (Cummins -> bowler_new.jpg), v2 clean | 0.376 | 0.604 | 0.713 | 0.314 | 0.244 |

- Visual: no doubled limbs (frames repeated, not blended); plain backdrop; crisp outline; bat drawn in
  every batting frame; original bowler now faces away like the source.
- Remaining defects: (1) grey translucent patches on legs/pads where the DensePose silhouette is wider
  than the person the model drew (the mask shows the generated background there; before, the pads lift
  turned these white); (2) the original bowler is flat grey with little detail; (3) new bowler: dark arms
  and a faded bowling arm at release, from dark, blurred, watermarked source footage
  (right elbow 0.17, right wrist 0.08 at PCK@0.1).
- Next: matte from the generated frames (person segmentation) intersected with the DensePose silhouette,
  so only pixels the model actually drew as the player are kept.

### Clean mode (v2) + new bowler clip; Kaggle kernel: all three clips, CLEAN=1
Request: batting cleaner (clear outline, joints, bat; no overlays or blur), player over background;
new bowler video mapped onto a new bowler photo; keep the earlier bowling outputs separate.

**Batting issues and causes (from kernel v4 frames):**
- Double/see-through legs and bat: only 16 frames, then `minterpolate` blends neighbours to 24 fps.
- White blobs at knees/ball: the engine brightens colour-guessed "pads"/"torso" masks and dilates the
  batsman's legs 9x9, so generated background inside the mask turns white.
- Soft, cut-out outline: 7x7-blurred DensePose mask pasted onto `GaussianBlur(31,15)` of the photo.
- Era styles add lens blur, softness, dust, scratches and gate weave.

**Changes (all opt-in; v1 defaults unchanged):**
- `generation_v2.py --clean` (`master_pipeline.py --gen-clean`): plain backdrop instead of the blurred
  plate; crisp DensePose silhouette (no leg dilation, 1 px erode + 1 px feather); pads/torso lifts off;
  prompt prefix "sharp focus on one full-body player, clean outline" and negatives for ghosting/blur;
  with `--frames 32` FreeNoise context windows (16, stride 4) so AnimateDiff stays consistent
  (falls back to 16-frame chunks if FreeNoise fails).
- `master_pipeline.py --retime-mode hold`: repeat generated frames to 24 fps, no blending.
- `student3_v3/configs/{batsman,bowler}_clean.json` (copies): no lens blur/softness (0.05, which
  OpenCV needs > 0), no dust/scratches/gate weave/jitter, light grain.
- View direction: back/front from shoulder order, plus **side** when a "back" clip has narrow shoulders
  (median shoulder width / torso < 0.5). Values: bowler_delivery_001 0.71 (back), good_cover_drive_final
  0.42 (front), bowler_new_cummins 0.35 (side; it is side-on with the face in profile).
- `--ref-box` in clean mode crops the player from any photo and pads it to a square for the IP-Adapter
  (the engine's default path squashed portrait photos and kept the watermark).

**New bowler clip `bowler_new_cummins`:**
- Source: `Pat Cummins Bowling Action Slow-Motion - YouTube - Google Chrome 2026-11-24 01-21-39.mp4`, a
  1920x1140 screen recording. 0-5.5 s is a paused frame; one clean delivery at 5.7-8.3 s; later a second
  partial delivery with play/pause icons, then a scene cut at 11.5 s.
- Cropped to the YouTube player (`crop=1316:664:24:196`) and trimmed to 5.7-8.3 s:
  `data/raw_clips/bowler_new_cummins.mp4` (72 frames at 27.7 fps). The "CRICKET MASTERCLASS" watermark
  is burned into the video and stays over the body.
- Motion (Student 1, CPU, `ultralytics` 8.4.217 installed into `~/gen`): 64 frames at 24 fps, 512x512.
- DensePose (CPU, `~/dp`): 35 of 64 frames patched from the skeleton (dark, blurred footage).
  Reliable to about frame 44 (release); generation window 0-44.
- Photo: `bowler-to bemapped.jpg` copied to `data/historical_images/bowler_new.jpg`;
  `--ref-box 0.06 0.13 0.96 0.95` (player, without the sporting-heroes.net watermark).
- Not on GitHub, so `run_on_kaggle.sh` sends it inside the kernel (`EXTRA_DATA=...`).

**Runner fixes:** every Kaggle call has a timeout (a `kernels status` call hung for 24 min);
`remote_run.sh` jobs `bowler | batsman | bowler_new | both | all`, `CLEAN=1` adds the v2 settings
and a `_v2` run-name suffix. Earlier results moved to `kaggle_results_v1/`.

**Push failures and fix:** three pushes after the local motion run hung (`kernels push` socket in
CLOSE-WAIT with data unsent). Cause: easy-dwpose had downloaded `checkpoints/dw-ll_ucoco_384.onnx` and
`checkpoints/yolox_l.onnx` (351 MB) and ultralytics `yolo26n-pose.pt` into the repo, and the code overlay
only excluded videos/npz/zip, so it was 334 MB (normally ~130 KB) over an uplink of 13-150 KB/s.
Fix: both runners exclude `checkpoints/`, `models/`, `kaggle_results*`, root images and weight files
(`*.onnx *.pt *.pth *.pkl *.bin *.safetensors`) and abort if the overlay is over 2 MB; `.gitignore` has
`checkpoints/` and `models/`. The new clip goes up as the private dataset
`pranahvsubramaniam/cricket-pipeline-extra-data` (`EXTRA_DATA`, or `EXTRA_DATASET` to reuse it).
The clean run is kernel `pranahvsubramaniam/cricket-clean` v1 (started without queueing).

**Kernel v5** (bowler, v2 skeleton only, not clean) sat in Kaggle's T4 queue for over an hour and was
superseded by the clean run.

### Kaggle kernel v4 (both clips): first complete run, rc=0
- T4 15 GB. Total about 8 min (install 1 min; bowler 19:31:58-19:36:17; batsman 19:36:18-19:38:52).
  Generation: 16 frames, 28 steps, about 78 s per clip, peak VRAM 6.37 GB. Colour stage verdict: pass.
- Motion PCK (DWPose on generated vs reference pose, threshold = alpha x reference body box):

  | clip | PCK@0.05 | PCK@0.1 | PCK@0.2 | velocity cosine |
  |---|---|---|---|---|
  | bowler (`bowler_delivery_001`, frames 63-120) | 0.154 | 0.375 | 0.721 | 0.464 |
  | batsman (`good_cover_drive_final`) | 0.761 | 0.935 | 0.982 | 0.304 |

  Torso-normalised PCK@0.1 (student3_v3 metric): bowler 0.125 (bowling arm 0.0), batsman 0.627.
- Bowler arms and hips are almost never matched (shoulders 0.07, elbows/wrists 0-0.07, right hip 0.0
  at PCK@0.1); only head and ankles track. Batsman tracks well; weakest are right elbow (0.6) and left ankle (0.63).
- Results: `kaggle_results/runs/colab_{bowler,batsman}/` (`final_24fps.mp4`, `demo_side_by_side.mp4`,
  `student3_v3/era_*.mp4`, `evaluation.json`, `manifest.json`).

### Fix: generation crashed at the first denoising step (all GPUs)
- **Symptom** (Kaggle T4, kernel v2 and v3, both clips):
  `RuntimeError: The size of tensor a (49152) must match the size of tensor b (3072) at non-singleton dimension 1`
  in `diffusers/models/attention.py:558`, inside the UNet's cross-attention.
- **Cause:** `generation_engine.py` loaded the *community* pipeline `pipeline_animatediff_controlnet`
  (`custom_pipeline=...`). With diffusers 0.32.2, `UNetMotionModel.forward` no longer repeats text
  embeddings per frame; the pipeline must (`pipelines/animatediff/pipeline_animatediff_controlnet.py:929`
  in the official one). The community file only repeats them for the ControlNet (its line 1056), so the
  UNet got batch 2 text embeds for batch 32 (2 x 16 frames) latents: 16 x 3072 = 49152.
  Not version drift: Kaggle and the laptop env (`~/gen`) both have torch 2.11.0+cu128, diffusers 0.32.2,
  transformers 4.57.6, accelerate 1.16.0, huggingface-hub 0.36.2.
- **Change:** `generation_engine.py` now uses diffusers' built-in `AnimateDiffControlNetPipeline`
  (same call arguments, IP-Adapter supported). `requirements-generation.txt` comment updated.
- **Also:** `generate_cli.py` prints the full traceback on generation errors (exit 5); 
  `colab/remote_run.sh` logs package versions.

### Kaggle runner added (`kaggle_runner/`)
- `run_on_kaggle.sh` pushes a private script kernel (`<user>/cricket-pipeline`, T4, internet on) that
  clones `Pranahv-main`, overlays local code (base64 in the script), runs `colab/remote_run.sh`, and
  downloads `job.log` + `results.zip` into `kaggle_results/` (git-ignored).
- Auth: `~/.kaggle/kaggle.json` or `~/.kaggle/access_token`; username from `kaggle config view`.
- Kaggle CLI 2.2.5 needs `uv tool install kaggle --with six` (missing dependency).
- Kernel v1: ran on CPU (`nvidia-smi` missing), 0 GPU hours used. Cause: account not phone-verified;
  Kaggle silently drops GPU. Fixed by verifying the phone. `kernel_body.py` now reports this as rc=3.
- Kernel v2 (both) and v3 (bowler, with traceback): T4 15 GB, install ~1 min, models load, generation
  fails with the 49152/3072 error above. rc=5.

### Colab runner fixes (`colab/run_on_colab.sh`)
- First T4 run: session created, code uploaded, then the launch step failed with
  `TimeoutError: no message from the kernel for 30s` and the session was lost (404/401).
  Balance was 0.00 compute units, so the VM was probably a free-tier T4.
- Changes: `colab exec --timeout` (300 s, 60 s for status polls); launch with
  `setsid nohup ... </dev/null &` so the call returns at once; `trap ... EXIT` stops the VM on any
  exit, not only Ctrl+C; the poll loop exits (rc 71) if the session is lost or 5 polls fail.
