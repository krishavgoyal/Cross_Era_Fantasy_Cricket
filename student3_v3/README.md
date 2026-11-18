# Student 3 v3: bowler + batsman colourisation, era styling and evaluation

v3 is a new folder. It does not overwrite `student3/` or `student3_improved/`, and it re-uses v2's video I/O, flow warping and SSIM code.

## What changed from v2

| | v2 (`student3_improved`) | v3 (`student3_v3`) |
| --- | --- | --- |
| Roles | Batsman only | Batsman and bowler (`"role"` in the config) |
| Masks | SAM2, one box per part on frame 0 (needs GPU + checkpoint) | Student 1's DensePose video, decoded to 24 body parts, cropped exactly like Student 2. Runs on CPU in seconds |
| Alignment check | None | Imports Student 2's own `generation_engine.py` (models not loaded) and reports silhouette IoU against it |
| Frame layout | Treats all frames alike | Detects Student 2's held frames (3+16+7 or 4+16+6) and colours each keyframe once, so holds cannot flicker |
| Palette | Near-neutral values | From reference colours (cream flannel, skin, turf, willow), luminance-tapered, one `saturation` knob |
| Era styling | None | `newsreel_1930s`, `broadcast_1970s`, `remaster`, each retimed (18 / 25 / 24 fps) |
| Evaluation | Luma MAE/PSNR/SSIM | Plus colourfulness, mask coverage, hold-frame stability, flow-warping error, PCK (overall + bowling arm) |
| Review files | Video + masks folder | Side-by-side MP4 and mask-overlay MP4 per run |

## Results so far (CPU, this repo's data)

| Run | Input | Luma SSIM | Colourfulness in → out | Player pixels with a class | Student 2 IoU |
| --- | --- | --- | --- | --- | --- |
| Batsman | Ahaan-main `data/generated_clips/test_run/m3_magicanimate.mp4` | 0.990 | 0.92 → 25.9 | 94% | 0.76 |
| Bowler (proxy) | Grayscale Starc footage cropped like Student 2 | 0.982 | 2.29 → 22.5 | 89% | 0.88 |
| v2 for comparison | `m3_magicanimate (2).mp4` | 0.989 | output 13.7 | 75% (min 66%), with parts drifting off the body | n/a |

The bowler run uses a **proxy** (real footage made to look like Student 2's output), because no generated bowler video has been pushed yet. Do not present the proxy as a generated result.

## Run it (Windows PowerShell, from the repo root)

```powershell
& .\.venv\Scripts\python.exe -m pip install opencv-python numpy pytest
$env:PYTHONPATH = "."

# Batsman: newest Student 2 output
& .\.venv\Scripts\python.exe -m student3_v3.run `
  --input data\generated_clips\test_run\m3_magicanimate.mp4 `
  --densepose data\processed_clips\good_cover_drive_final\control_densepose_real.mp4 `
  --config student3_v3\configs\batsman_1930s.json `
  --out-dir data\student3\good_cover_drive_final

# Bowler, once Student 2 has run: python generate_cli.py --role bowler ... --output outputs\m3_bowler_delivery.mp4
& .\.venv\Scripts\python.exe -m student3_v3.run `
  --input outputs\m3_bowler_delivery.mp4 `
  --densepose data\processed_clips\bowler_delivery_001\control_densepose_real.mp4 `
  --config student3_v3\configs\bowler_1930s.json `
  --out-dir data\student3\bowler_delivery_001

# Bowler proxy (works today, no GPU)
& .\.venv\Scripts\python.exe -m student3_v3.make_proxy `
  --source data\processed_clips\bowler_delivery_001\source_512.mp4 `
  --densepose data\processed_clips\bowler_delivery_001\control_densepose_real.mp4 `
  --role bowler --output data\student3\bowler_delivery_001_proxy\proxy_generated_input.mp4

& .\.venv\Scripts\python.exe -m pytest student3_v3\tests -q
```

If the input was rendered by the older `main`-branch engine, add `--student2-preset main`. Use the same `--densepose` file that Student 2 used for that generation.

## Motion fidelity (PCK)

This needs DWPose, which downloads models from Hugging Face, so run it on your machine:

```powershell
& .\.venv\Scripts\python.exe pose_engine.py --input data\student3\bowler_delivery_001\colourised_master.mp4 --output data\student3\bowler_delivery_001\generated_pose.json
& .\.venv\Scripts\python.exe -m student3_v3.evaluate `
  --processing data\student3\bowler_delivery_001\processing.json `
  --reference-pose data\processed_clips\bowler_delivery_001\pose_data.json `
  --generated-pose data\student3\bowler_delivery_001\generated_pose.json `
  --report data\student3\bowler_delivery_001\pck.json
```

It reports PCK@0.05/0.1/0.2 (torso-normalised) over all 18 joints and for the active arm (the arm whose wrist travels furthest, i.e. the bowling arm).

## Configs

- `configs/bowler_1930s.json`: shirt (torso + arms, sleeves down), skin (hands), trousers (all leg parts), white boots, hair (head, back view). Whole background treated as turf, because Student 2's bowler background is a blurred turf plate.
- `configs/batsman_1930s.json`: shirt, rolled-sleeve forearms as skin, white gloves, trousers, pads (lower legs, plus Student 2's widened shins), cap/face split, bat from Student 2's bat layer. A bright hat stays cream; a dark cap gets the baggy green.

Change colours in `palette` (OpenCV LAB a/b, neutral = 128), strength per class in `class_strength`, and the overall intensity with `saturation`.

## Known limits

- Student 2 samples the bowler evenly from frame 8 to 119, so only 3 of 16 keyframes (source 104, 111, 119) cover the release. That is a Student 2 setting (`load_aligned_densepose_sequence`), not fixable in post.
- DensePose masks follow the *control*, and the generated body is slightly wider. Student 2's composite silhouette fills that gap with the base shirt tint (or pads below the knees for the batsman).
- The shirt number in the proxy is real Starc footage. Student 2's bowler prompt removes it in generation; check that it does.
- Historical colours are informed choices, not verified facts.

## Integration options (added by the master pipeline)

- `--sidecar runs/<run>/generated.mp4.json`: use the frame mapping and crop Student 2 recorded, instead of re-deriving them. Required when generation used `--window`/`--auto-window`; `processing.json > alignment.sidecar` reports whether the re-derived values agreed.
- `--parts-npz data/processed_clips/<clip>/densepose_real_parts.npz`: exact DensePose part ids instead of decoding the PARULA colours.

`master_pipeline.py` passes both automatically (see `INTEGRATION.md`).
