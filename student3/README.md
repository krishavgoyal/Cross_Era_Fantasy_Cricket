# Student 3 Archival Finishing Stage

This is a downstream stage for the Cross-Era Fantasy Cricket pipeline. It does not modify Student 1 pose extraction or Student 2 motion generation. It takes the generated monochrome animation and produces a colour-graded archival derivative plus a preservation report.

## Current Bradman output

The supplied `m3_magicanimate (2).mp4` conforms to the Student 3 contract: 384x512, 7 FPS, 16 frames, 3:4 portrait, and neutral monochrome. The default preset is consequently `1930s_archival`.

## What the stage does

1. Validates geometry and frame rate.
2. Keeps the upstream luminance as the source of motion and detail.
3. Builds a restrained semantic palette for sky, turf, flannels, cap, skin, and bat.
4. Uses optical-flow warping plus conservative blending to carry chroma across frames.
5. Applies luma-only edge refinement and deterministic fine film grain.
6. Encodes an H.264/yuv420p master when FFmpeg is supplied.
7. Writes a JSON processing record and evaluates luma preservation, chroma introduction, and geometry/FPS preservation.

This is intentionally not a per-frame unconstrained colourizer. Such a colourizer can cause flicker and invent unstable colours when only a monochrome reference exists. The palette and all ROI values are in the JSON config, so they can be reviewed and tuned per historical era.

## Run

Use a virtual environment outside OneDrive if its sync client locks package files.

```powershell
$py = "C:\Users\ASUS V16\.venvs\cross-era-student3\Scripts\python.exe"
$ffmpeg = "C:\Users\ASUS V16\OneDrive\Documents\New project\video-tools\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"

& $py student3\postprocess.py `
  --input "C:\Users\ASUS V16\Downloads\m3_magicanimate (2).mp4" `
  --output data\student3\bradman_1930s_graded_master.mp4 `
  --ffmpeg $ffmpeg `
  --report data\student3\bradman_1930s_processing.json

& $py student3\evaluate.py `
  --input "C:\Users\ASUS V16\Downloads\m3_magicanimate (2).mp4" `
  --output data\student3\bradman_1930s_graded_master.mp4 `
  --report data\student3\bradman_1930s_evaluation.json
```

## Required review before submission

- Compare the original and final videos as a contact sheet and at playback speed.
- Confirm that the cap, flannels, turf, and sky retain consistent colours through the swing.
- Inspect frames 6–11 closely: post-processing can reduce soft edges but cannot restore a malformed bat or fingers.
- Keep the original 7-FPS master. Create a 24/30 FPS presentation variant only after approval, using an approved interpolation model such as RIFE.
- For PCK, run the same Student 1 DWPose backend on the final video and compare it with aligned source pose coordinates. The included evaluator intentionally does not claim PCK without matching pose data.
