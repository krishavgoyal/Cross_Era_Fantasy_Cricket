Cross-Era Fantasy Cricket 🏏
A computer-vision pipeline for bringing historical cricket footage into a modern, fantasy-style viewing experience.

This repository contains the project code currently consolidated in cross_era_motion_engine, including video processing utilities, motion/pose-related components, temporal smoothing, rendering controls, and Student 3's archival restoration/colorization pipeline.

Table of Contents
- Project Overview
- Objectives
- Pipeline at a Glance
- Repository Structure
- Student Contributions
- Student 3: Archival Restoration and Colorization
- Processing and Evaluation
- Technology Stack
- Environment Setup
- Running the Student 3 Pipeline
- Outputs and Artifacts
- Testing
- Reproducibility Notes
- Limitations and Future Work
- Team and Contributions
Project Overview
Cross-Era Fantasy Cricket is a computer-vision project centered on processing cricket footage across eras. The goal is to make archival or older cricket footage more visually engaging while retaining the original movement and visual structure of the clip.

The code currently present in this repository provides a foundation for:
- Video/clip handling and processing.
- Pose- and motion-related processing components.
- Temporal smoothing and rendering controls.
- Student 3's archival colorization/restoration workflow.
- Segmentation-mask generation and inspection.
- Output evaluation and edge-case testing.
The repository is organized as a shared project workspace. Individual modules are designed to be understandable and testable independently, while project-level scripts and shared utilities support integration.
Scope note: This README documents what is present in the repository and what has been confirmed during development. Student 1 and Student 2's exact implementation details and ownership should be filled in by the team; they are not guessed here.

Objectives
The project's broad objectives are:
1. Preserve source footage structure — retain the original frame dimensions, frame rate, and frame count wherever the processing workflow is intended to do so.
2. Improve visual presentation — apply controlled restoration/colorization to grayscale or archival footage.
3. Respect motion continuity — reduce frame-to-frame instability through temporal processing and motion-aware techniques.
4. Keep processing inspectable — produce masks, configuration files, and reports that help diagnose what the pipeline did.
5. Validate outputs — run automated checks and inspect results visually instead of treating successful execution as proof of quality.
6. Support team integration — keep components, configuration, tests, and documentation in a version-controlled repository.
Pipeline at a Glance
The high-level processing flow is:
Input cricket video
        │
        ▼
Clip / frame handling
        │
        ├── Motion and pose-related processing
        │
        ├── Temporal smoothing / consistency controls
        │
        └── Rendering controls and shared utilities
        │
        ▼
Student 3 archival processing
        │
        ├── Player and semantic-region segmentation
        ├── Mask generation and inspection
        ├── Region-specific colorization
        ├── Temporal / flow-aware stabilization
        └── Luminance preservation
        │
        ▼
Processed video + masks + processing report
        │
        ▼
Automated evaluation + human visual review
This is a conceptual overview of the components in the repository, not a claim that every module is automatically invoked in one end-to-end command. Check the relevant CLI and module documentation before running a specific workflow.
Repository Structure
cross_era_motion_engine/
├── README.md
├── requirements.txt
├── CPU_LAPTOP_PROFILE.md
├── EDGE_CASE_VERIFICATION.md
├── pipeline_cli.py
├── clip_processor.py
├── common.py
├── pose_engine.py
├── render_control.py
├── temporal_smoothing.py
├── verify_environment.py
├── schemas/
│   └── metadata.schema.json
├── student3/
│   ├── README.md
│   ├── postprocess.py
│   ├── evaluate.py
│   ├── configs/
│   │   └── 1930s_archival.json
│   └── tests/
├── student3_improved/
│   ├── README.md
│   ├── postprocess.py
│   ├── segmentation.py
│   ├── evaluate.py
│   ├── make_prompt_preview.py
│   ├── configs/
│   │   └── 1930s_archival.json
│   └── tests/
├── tests/
│   ├── test_edge_cases.py
│   ├── test_renderer.py
│   └── test_smoothing.py
├── data/
│   └── student3_improved/
│       ├── sam2_masks/
│       │   ├── player/
│       │   ├── helmet/
│       │   ├── skin/
│       │   ├── shirt/
│       │   ├── trousers/
│       │   ├── pads/
│       │   ├── bat/
│       │   └── shoes/
│       └── sam2_processing.json
├── student3.zip
└── checkpoints/                 # Local model files; not all are tracked on GitHub
Important files
File	Purpose
pipeline_cli.py	Project-level command-line entry point; inspect its help/options for supported commands.
clip_processor.py	Clip/video processing component.
pose_engine.py	Pose-related processing component.
temporal_smoothing.py	Temporal smoothing utilities.
render_control.py	Rendering/control utilities.
common.py	Shared helpers used across modules.
verify_environment.py	Environment/dependency verification helper.
schemas/metadata.schema.json	Metadata schema.
EDGE_CASE_VERIFICATION.md	Edge-case verification notes.
CPU_LAPTOP_PROFILE.md	CPU/laptop-oriented profile notes.


Student Contributions
The repository is a team project. The following section distinguishes confirmed repository work from details that still need to be supplied by the team.
Student 1 — Motion / Pose Work
Status: contribution details to be confirmed by the team.

The project contains pose- and motion-related components, including pose_engine.py, and the evaluation plan references DWPose/keypoint-based motion-fidelity checks. The exact Student 1 deliverables, model choices, inputs/outputs, and implementation ownership should be added here by the responsible student.

Suggested details to document:
- Problem handled:
- Model/algorithm:
- Input and output formats:
- Main files:
- Evaluation metric/results:
- Integration point:
Student 2 — Contribution to be documented
Status: contribution details to be confirmed by the team.

The current consolidated repository does not clearly identify a dedicated Student 2 folder or an authoritative contribution description. Add the student's actual module and responsibilities here rather than attributing unverified work.

Suggested details to document:
- Problem handled:
- Model/algorithm:
- Input and output formats:
- Main files:
- Evaluation metric/results:
- Integration point:
Student 3 — Archival Restoration and Colorization
Status: implementation and a sample processing run completed; visual-quality review remains important.

Student 3's work is available in two folders:
- student3/ — original implementation.
- student3_improved/ — improved pipeline with SAM2-based segmentation and additional temporal processing.
The improved implementation is described in detail below.
Student 3: Archival Restoration and Colorization
Goal
Process archival cricket footage while adding controlled color to semantic regions and retaining the source video's luminance and temporal structure as closely as possible.
Main implementation
The improved implementation is split into focused modules:
Component	File	Responsibility
Main postprocessing	student3_improved/postprocess.py	Coordinates the postprocessing/colorization workflow.
Segmentation	student3_improved/segmentation.py	SAM2-based segmentation support and semantic masks.
Evaluation	student3_improved/evaluate.py	Computes technical and visual-quality-related metrics/checks.
Prompt preview	student3_improved/make_prompt_preview.py	Helps inspect segmentation prompt placement.
Archival configuration	student3_improved/configs/1930s_archival.json	Stores color, strength, prompt, and processing parameters.


Semantic regions
The segmentation workflow uses prompts/masks for the following regions:
- Player — overall player region.
- Helmet — helmet/headgear region.
- Skin — exposed skin region.
- Shirt — upper clothing.
- Trousers — lower clothing.
- Pads — cricket protective pads.
- Bat — cricket bat.
- Shoes — footwear.
Masks are saved frame-by-frame under data/student3_improved/sam2_masks/, grouped by semantic class. This makes it possible to inspect segmentation quality independently of the final rendered video.
Colorization and temporal processing
The improved configuration and processing report identify these techniques:
- Semantic-mask-based region colorization.
- Class-specific color palette values.
- Separate strengths for the player, grass, sky, and clothing/equipment classes.
- Backward-flow warping.
- Forward/backward occlusion-confidence handling.
- Flow-consistency thresholding.
- Scene-cut detection and temporal reset.
- Luminance preservation.
- Temporal-strength control.
The configuration is intentionally separated from the implementation so that colors, strengths, prompt boxes, and temporal parameters can be adjusted without rewriting the pipeline.
Archival configuration
Configuration: student3_improved/configs/1930s_archival.json

It includes:
- Temporal and flow-consistency parameters.
- Scene-cut threshold.
- Player, grass, and sky colorization strengths.
- Per-class strengths for trousers, shirt, pads, skin, helmet, bat, and shoes.
- Edge sharpening and grain controls.
- Fallback player bounding box.
- SAM2 prompt boxes for semantic regions.
- LAB color values for the palette.
Historical authenticity note: palette values are processing choices, not proof of the original 1930s colors. Authenticity requires historical references or expert validation.
Sample processing run
A sample 16-frame video was processed successfully using:
- SAM2 segmentation backend.
- SAM2.1 Hiera-Tiny checkpoint.
- FFmpeg/libx264 output encoding.
- Resolution: 384 × 512.
- Frame rate: 7 FPS.
- Frame count: 16.
The generated sample output was:
data/student3_improved/bradman_1930s_improved_sam2.mp4
The processing report was:
data/student3_improved/sam2_processing.json
The evaluation report was generated separately by the evaluation script. Keep the output video and reports together when sharing a reproducible run.
Processing and Evaluation
The evaluation separates technical correctness from visual quality.
Technical checks
The sample evaluation reported:
Check / metric	Result
Resolution preserved	Pass
Frame count preserved	Pass
FPS preserved	Pass
Luminance MAE within configured limit	Pass
Mean local SSIM at/above configured threshold	Pass
Luminance MAE	1.7655
Luminance PSNR	41.495 dB
Mean local SSIM	0.98887


These results indicate that the sample output retained important technical properties and closely preserved luminance.
Visual review
The sample's technical verdict was pass, while its visual-quality verdict was requires human review.

This distinction matters: automated metrics cannot prove that:
- Segmentation masks follow the correct object boundaries.
- Colors are semantically correct or historically accurate.
- Color remains stable across all frames.
- No flicker, haloing, leakage, or visual artifacts are present.
- Player identity, pose, or motion is preserved perceptually.
A proper review should inspect the original and processed videos side-by-side, inspect semantic masks, and check difficult frames (occlusion, fast motion, scene cuts, and object boundaries).
Motion fidelity
Motion fidelity is not yet established by the Student 3 luminance/colorization metrics alone. The planned evaluation can use Student 1's pose/keypoint output and aligned keypoints to calculate a metric such as PCK (Percentage of Correct Keypoints). Record the exact procedure and results once that evaluation has been run.
Technology Stack
Technologies and tools used in the current repository/workflow include:
- Python — processing and evaluation code.
- OpenCV — video/frame and image-processing operations.
- NumPy — numerical/image-array operations.
- PyTorch — model execution.
- SAM2 / SAM 2.1 — prompt-based video/image segmentation.
- FFmpeg — video encoding.
- pytest — automated tests.
- Git and GitHub — version control and team collaboration.
The exact dependency versions are specified in the relevant requirements.txt files. SAM2 model weights are separate assets and are not automatically included in a Git clone.
Environment Setup
Prerequisites
- Python version compatible with the project's requirements.
- Git.
- FFmpeg available on PATH.
- A compatible PyTorch installation.
- SAM2 installed and its checkpoint available for the improved Student 3 workflow.
- NVIDIA GPU/CUDA is recommended for accelerated SAM2 inference; CPU execution may be substantially slower.
Clone the repository
git clone https://github.com/krishavgoyal/Cross_Era_Fantasy_Cricket.git
cd Cross_Era_Fantasy_Cricket
Create a virtual environment
Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
Linux / WSL:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
Install project dependencies:
pip install -r requirements.txt
pip install -r student3_improved/requirements.txt
Install SAM2 separately according to the official SAM2 installation instructions and the version/configuration used by this project. Place the compatible checkpoint at the path expected by the command/configuration. Model weights are not bundled in this repository.

Verify the environment:
python verify_environment.py
If a dependency is already installed in another environment (for example, WSL), it is not automatically available in Windows PowerShell's .venv. Install and verify dependencies in the environment in which you intend to run the pipeline.

Running the Student 3 Pipeline
The exact CLI arguments can vary with the script version. First inspect the available options:
python student3_improved/postprocess.py --help
python student3_improved/evaluate.py --help
Use the input video, archival configuration, SAM2 config/checkpoint, and output paths required by the displayed CLI.

A sample output from the completed run was:
data/student3_improved/bradman_1930s_improved_sam2.mp4
Evaluate a generated output using the arguments shown by:
python student3_improved/evaluate.py --help
Do not assume that the sample output video is present in a fresh clone: generated videos are local artifacts and may need to be produced again.
Outputs and Artifacts
Artifact	What it contains
Processed MP4	Rendered output video.
sam2_masks/<class>/frame_XXXX.png	Frame-level semantic mask for a class.
sam2_processing.json	Processing metadata/report for the SAM2 run.
Evaluation JSON	Technical metrics and evaluation verdicts.
Prompt preview	Visual aid for checking prompt boxes/regions.


Masks and reports are useful for debugging and reproducibility. They are not a substitute for the original input video or model checkpoint.
Testing
Run the repository tests from the project root:
python -m pytest
Run Student 3 tests specifically:
python -m pytest student3/tests student3_improved/tests
Run individual test files if debugging:
python -m pytest tests/test_edge_cases.py
python -m pytest tests/test_renderer.py
python -m pytest tests/test_smoothing.py
A passing test suite validates the behaviors covered by those tests; it does not guarantee perfect visual output for every video.
Reproducibility Notes
For a reproducible run, record:
- Input video filename and checksum.
- Python and dependency versions.
- Operating system and GPU/CUDA details.
- SAM2 repository/version and checkpoint filename.
- Archival configuration used.
- CLI command and output paths.
- Processing report and evaluation report.
- Any manual prompt/palette adjustments.
- Human visual-review notes.
The repository includes CPU_LAPTOP_PROFILE.md for hardware/profile notes. Use it when setting expectations for resource use and performance.
Limitations and Future Work
Potential next steps include:
- Complete and document Student 1 and Student 2 modules/contributions.
- Integrate all student components into a clearly documented end-to-end workflow.
- Add a single reproducible pipeline command or orchestration script.
- Validate pose/motion preservation quantitatively using aligned keypoints.
- Improve segmentation prompt/mask quality on challenging frames.
- Compare multiple palette configurations and document the rationale.
- Add side-by-side before/after visual examples and mask overlays.
- Expand tests for scene cuts, occlusions, fast movement, and boundary leakage.
- Document model downloads, expected checkpoint paths, and setup troubleshooting.
- Add a sample/demo video or lightweight screenshots where licensing permits.
Team and Contributions
This is a collaborative project. Add the final team roster and each member's verified contribution below.
Member	Contribution	Main files / deliverables
Student 1	Motion/pose work — please fill in verified details	pose_engine.py and/or confirmed files
Student 2	Please fill in verified contribution	Confirmed files/modules
Student 3	Archival restoration/colorization, SAM2 semantic masks, temporal/flow-aware processing, evaluation	student3/, student3_improved/, data/student3_improved/


Contributions should reflect actual ownership and work completed. Update this table after the team confirms the final responsibilities.
Acknowledgements
This project builds on open-source computer-vision tooling, including PyTorch, OpenCV, FFmpeg, and SAM2. Please retain and follow the licenses and attribution requirements of the tools, models, datasets, and footage used.
License
Add the project's chosen license here before distributing or reusing the code. Until a license is specified, do not assume that the repository grants permission for unrestricted reuse.