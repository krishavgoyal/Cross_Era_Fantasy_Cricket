# Student 3 Improved — 1930s Archival Colorization v2

This is a **separate replacement candidate**. It does not overwrite the existing `student3/` directory or any previous rendered video.

## What changed

- Removed fixed-coordinate color painting from the old postprocessor.
- Added an optional **official SAM2 video predictor** integration. It propagates separately prompted masks for player, helmet, skin, shirt, trousers, pads, bat and shoes.
- Added a clearly labelled OpenCV GrabCut fallback so the pipeline can be exercised before SAM2 is installed. GrabCut is not equivalent to SAM2 and its output must be treated as a preview.
- Corrected temporal chroma remapping to use current-to-previous (backward) flow for sampling.
- Added forward/backward flow consistency confidence, border validity and scene-cut reset before temporal chroma blending.
- Made color masks edge-feathered and added restrained 1930s palette values.
- Disabled sharpening and grain by default. Add them only after approving the masks and colors.
- Replaced the old evaluator's FPS-only PASS with resolution, frame-count, FPS, luminance MAE, PSNR and local-window SSIM checks. It explicitly leaves visual quality and PCK as human/Student 1 follow-ups.
- Added tests for flow direction, geometry/colorization, evaluator honesty and prompt-box validation.

## 1. Install base requirements (Windows PowerShell)

Run from the repository root, where the `.venv` folder is located:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r student3_improved\requirements.txt
```

## 2. Make a coordinate-grid image for SAM2 prompts

```powershell
& .\.venv\Scripts\python.exe student3_improved\make_prompt_preview.py --input "$env:USERPROFILE\Downloads\m3_magicanimate (2).mp4" --output data\student3_improved\frame0_prompt_grid.jpg
Start-Process (Resolve-Path data\student3_improved\frame0_prompt_grid.jpg)
```

The grid is numbered 0–100 on both axes. `sam2_prompts` in `configs/1930s_archival.json` contains normalized `[x1,y1,x2,y2]` boxes for frame 0. Review and adjust these boxes to tightly cover the intended object/part. **Do not assume the starter boxes are perfect.** SAM2 propagates the objects you prompt; it does not automatically understand that a region is skin, cloth or bat.

## 3. Install SAM2 (optional but required for the selected SAM2 mode)

SAM2 needs the Python package and a model checkpoint. This is a separate, relatively large dependency and is not bundled in this ZIP.

1. Check PyTorch first:

```powershell
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

2. Follow the official SAM2 installation instructions for your OS/PyTorch/CUDA combination: https://github.com/facebookresearch/sam2

   Windows builds can be more involved than ordinary pip packages. If the official package fails to build on your Windows setup, use a compatible Linux/WSL2 environment rather than randomly changing the existing project environment.

3. Download a SAM2.1 checkpoint from the official Meta SAM2 repository/release links and keep it outside this folder, e.g. `models\sam2.1_hiera_tiny.pt`. Do not commit the checkpoint to Git.

4. The implementation expects the official model config identifier, for example `configs/sam2.1/sam2.1_hiera_t.yaml`, and the checkpoint path. Confirm that the selected config/checkpoint pair matches.

## 4. First run now — fallback preview (no SAM2 required)

This lets you test the improved color/temporal pipeline immediately. It uses GrabCut, **not SAM2**:

```powershell
New-Item -ItemType Directory -Force data\student3_improved | Out-Null
$inputVideo = "$env:USERPROFILE\Downloads\m3_magicanimate (2).mp4"
$outputVideo = "data\student3_improved\bradman_1930s_improved_grabcut_preview.mp4"
$ffmpeg = "C:\Users\ASUS V16\OneDrive\Documents\New project\video-tools\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"

& .\.venv\Scripts\python.exe student3_improved\postprocess.py `
  --input $inputVideo `
  --output $outputVideo `
  --segmentation grabcut `
  --ffmpeg $ffmpeg `
  --mask-dir data\student3_improved\grabcut_masks `
  --report data\student3_improved\grabcut_processing.json

& .\.venv\Scripts\python.exe student3_improved\evaluate.py `
  --input $inputVideo `
  --output $outputVideo `
  --report data\student3_improved\grabcut_evaluation.json

Start-Process (Resolve-Path $outputVideo)
```

If your input is not in Downloads, change `$inputVideo` to its actual location.

## 5. Run the selected SAM2 pipeline

After installing SAM2, downloading a compatible checkpoint and reviewing prompt boxes:

```powershell
$inputVideo = "$env:USERPROFILE\Downloads\m3_magicanimate (2).mp4"
$outputVideo = "data\student3_improved\bradman_1930s_improved_sam2.mp4"
$checkpoint = "models\sam2.1_hiera_tiny.pt"
$sam2Config = "configs/sam2.1/sam2.1_hiera_t.yaml"
$ffmpeg = "C:\Users\ASUS V16\OneDrive\Documents\New project\video-tools\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"

& .\.venv\Scripts\python.exe student3_improved\postprocess.py `
  --input $inputVideo `
  --output $outputVideo `
  --segmentation sam2 `
  --sam2-config $sam2Config `
  --sam2-checkpoint $checkpoint `
  --device cuda `
  --ffmpeg $ffmpeg `
  --mask-dir data\student3_improved\sam2_masks `
  --report data\student3_improved\sam2_processing.json

& .\.venv\Scripts\python.exe student3_improved\evaluate.py `
  --input $inputVideo `
  --output $outputVideo `
  --report data\student3_improved\sam2_evaluation.json

Start-Process (Resolve-Path $outputVideo)
```

Use `--device cpu` if CUDA is unavailable. If SAM2 config resolution fails, follow the installed SAM2 version's config-path instructions; do not substitute a different checkpoint/config pair.

## 6. Run tests

```powershell
$env:PYTHONPATH = "."
& .\.venv\Scripts\python.exe -m pytest student3_improved\tests -q
```

## Important limitations / review checklist

- The starter SAM2 prompt boxes are only initial guesses based on frame 0. Inspect the grid and refine them. A poor prompt creates a poor propagated mask.
- SAM2 tracks prompted visual objects; it does not guarantee fine-grained semantic separation of a person's skin, shirt, trousers and pads. Those prompts need visual validation, and masks may overlap.
- The original source clip is a palette reference, not a pixel-aligned replacement for the generated video. This v2 uses a restrained configurable 1930s palette; it does not yet automatically transfer source-video colors by matching corresponding objects/frames.
- The GrabCut preview is not the final SAM2 result.
- The generated input itself contains blur/deformation. Colorization cannot reconstruct missing anatomy, fingers, bat geometry or details. That requires improving Student 2 generation.
- Technical evaluation does not establish historical accuracy, semantic mask correctness, or PCK motion fidelity. Inspect all frames; run Student 1's pose estimator for PCK separately.
- Preserve the original 7 FPS master. Do not interpolate to 24/30 FPS until the colorized master is approved.
