# Cross-Era Fantasy Cricket: animating Bradman and McGrath from motion clips

*Integration report (Pranahv), 24 Nov 2026. Everything here runs on Kaggle (free T4) plus CPU on the laptop.*
*Every video below shows a real player and is an **AI reconstruction**; label it as one wherever it is shown.*

---

## 1. Result: where the videos are

| clip | final video | side-by-side (source motion, then the result) | scores |
|---|---|---|---|
| **Don Bradman, batting** (motion from `good_cover_drive_final`) | `outputs/animate_v2/bradman_batting.mp4` | `outputs/animate_v2/bradman_batting_demo.mp4` | `outputs/animate_v2/bradman_batting_eval.json` |
| **Glenn McGrath, bowling** (motion from the Pat Cummins clip) | `outputs/animate_v2/mcgrath_bowling.mp4` | `outputs/animate_v2/mcgrath_bowling_demo.mp4` | `outputs/animate_v2/mcgrath_bowling_eval.json` |

Also in `outputs/animate_v2/`:
- `<job>_raw.mp4`: MusePose output **before** the bat, stump and ball are added. It has no bat or ball on purpose.
- `<job>_grid.mp4`: the reference frame, the skeleton and the raw output side by side.
- `<job>_sheet.jpg`: 8 frames from the final video.
- `summary.json`: all scores.

Earlier attempts are kept in `outputs/animate/` (first MusePose run), `outputs/animate_v2/previous/` (first McGrath scene run), `kaggle_results_v1/` and `kaggle_results_animate_v2/`.

**Before/after** (first MusePose run → this version):

| | PCK@0.1 | frames with the whole body | frames with merged limbs | worst outline overlap between frames | outline drops (<0.7) | bat / ball |
|---|---|---|---|---|---|---|
| Bradman, first run | 0.948 | 85.6% | 2.2% | 0.846 | 0 | drawn line, often merged or missing |
| **Bradman, now** | **0.978** | **98.6%** | **0%** | **0.874** | **0** | **real bat in 139/139 frames** |
| McGrath, first run | 0.854 | 57.8% | 4.4% | 0.669 | 1 | ball smeared |
| **McGrath, now** | **0.913** | **84.4%** | **2.6%** | 0.639 | 4 | **ball 59 frames (hand, release, flight); 1 fixed stump in 77 frames** |

The table is filled from `summary.json`; section 6 explains how each number is measured.

---

## 2. What was wrong, and what changed

| problem (as reported) | cause found | change |
|---|---|---|
| Player "in the middle of nowhere"; no wicket, crease or ground | the player was cut out of the photo and pasted on an unrelated ground photo | **Batting:** the motion clip's own camera view is the scene; the source batter is removed (LaMa) and Bradman stands exactly where he stood, in front of the real stumps and on the crease. **Bowling:** McGrath's own photo is the scene (crease, stump, pitch), and the motion is placed so his delivery stride lands on his photo pose. |
| Limbs move randomly, swap, merge | DWPose on blurred frames swaps left/right, loses joints, stretches bones, returns half hands | Skeleton cleaning (`animate/motion_clean.py`): left/right swap fix, limb gaps filled by **angle around the joint above** (not straight lines through the body), bone lengths kept steady, Savitzky-Golay smoothing, half-detected hands dropped. Generation with more steps (30) and longer overlap between slices (8). |
| Bat not clear, missing, merging with gloves | MusePose has no idea of a bat; the drawn bat line confused it | The bat is taken **out** of Bradman's photo and the skeleton, then added back after generation: his **real bat** from the photo, placed along the **tracked** bat direction, anchored in the hands, gloves drawn over the handle, darker willow tone and a thin edge so it shows against white flannels. |
| Bowling: no full step, no full arm rotation, no release | the arm rotation lasts 4-6 frames in the source, detections there are blurred and swap sides | Second DWPose pass on the raw full-resolution frames merged in by confidence. Delivery window **slowed 2.5x** and the bowling arm **rebuilt** as a straight arm rotating about the shoulder: down, back, over the top and through (about 376 degrees), eased so it is fastest at release. Run-up and step into the crease kept from the source. |
| Ball | MusePose smeared the ball in his hand | Ball removed from the photo and added as an **object**: in the bowling hand until the release frame, then thrown forward along the release direction with a slight drop. |
| Objects duplicating (stump appeared twice, merged with legs) | MusePose redraws every scene object every frame | The stump is a **fixed layer** cut from the photo and placed on top of every frame, so it never moves, duplicates or merges. |
| Player growing / sliding against the ground | the broadcast camera zooms (batting, about 10%) and pans (bowling, about 320 px) | Camera motion measured on the background (ORB features, people masked, watermark bands ignored) and removed from the skeleton, so the player moves relative to the ground, not the camera. |
| Body parts at the frame edge vanish | the skeleton left the canvas | The bowling scene is zoomed out just enough (LaMa extends the grass) for the whole action to fit; only frames with the whole body on the canvas are used. |

---

## 3. Pipeline, step by step

All commands from the repository root; generation on Kaggle (`kaggle_runner/run_on_kaggle.sh`), everything else on the laptop CPU in `~/gen`.

1. **References and sources** (sorted in `data/`, see `data/README.md`).
   - Photos: `data/reference_players/don_bradman/`, `data/reference_players/glenn_mcgrath/` (team photos; stock watermarks cropped out of what the model sees).
   - Motion clips: `data/motion_videos/`; Commons references downloaded by `python tools/fetch_references.py`, licences in `data/SOURCES_commons.md`.
2. **Motion clip processing** (Student 1, unchanged): `master_pipeline.py --only motion` and `--only densepose` give `data/processed_clips/<clip>/pose_data.json` (DWPose body + hands + face). The Cummins clip was cut from the screen recording: player area `crop=1316:664:24:196`, 5.7-8.3 s.
3. **Bat tracking** (batting): `python -m animate.bat_track data/processed_clips/good_cover_drive_final`. YOLO26x-seg "baseball bat" masks, main axis, handle at the wrists, stumps rejected (static, vertical, down to the feet), gaps interpolated: 122/139 frames detected, 139/139 after interpolation.
4. **Second pose pass** (bowling): `python -m animate.repose data/processed_clips/bowler_new_cummins`, giving `pose_data_merged.json`. Frames with fewer than 12 joints: 17 → 8.
5. **Jobs**: `python -m animate.build_scene_jobs`, giving `animate/jobs/<job>/`: `ref.png` (scene + player), `pose.mp4` (skeleton video, no bat), `target_pose.json` (skeleton + bat + ball per frame), `scene.json`, `job.json`, `preview.jpg`, plus `bat_sprite.npy`, `fg_layer.png` (stump), `ball_sprite.npy`.
   - batting scene (`animate/scene.py: batting_scene`): camera tracked; plate = frame 0 with the batter removed by LaMa (visible stumps kept, letterbox bars filled with grass), greyscale 1930s levels; Bradman cut out (u2net_human_seg, his bat removed by LaMa first), scaled to the batter's torso and placed on the batter's hips and feet.
   - bowling scene (`bowling_scene`): McGrath's photo below the watermark; stump and ball taken out (LaMa) into a layer and a sprite; source motion through Student 1's crop windows into the raw camera, camera pan removed; per-bone retargeting to McGrath's limb lengths (left/right averaged; widths and head keep the torso ratio); placed so at the best-matching stride frame his lower foot is on the photo's ground under his photo hips; zoom-out to fit.
   - cleaning (`animate/motion_clean.py`), delivery rebuild (`animate/delivery.py`), ball path (`build_scene_jobs.ball_path`).
6. **Generation** on Kaggle: `KERNEL=cricket-animate EXTRA_DATA="animate/jobs/<job>" MP_W=560 MP_H=632 MP_STEPS=30 MP_OVERLAP=8 JOBS="<job>" kaggle_runner/run_on_kaggle.sh animate`.
   - The job data goes up as the private dataset `cricket-animate-extra-data`.
   - `animate/remote_musepose.sh` builds a Python 3.10 env (torch 2.0.1 cu118, diffusers 0.27.2, av 12.3.0), downloads MusePose + SD 1.5 UNet config + CLIP ViT-L/14 + sd-vae-ft-mse, and runs `test_stage_2.py` (560x632, 30 steps, cfg 3.5, 24-frame slices, overlap 8).
7. **Finishing** on the laptop: `python -m animate.finish kaggle_results/animate outputs/animate_v2 <jobs>`. It adds the bat, stump and ball (`animate/composite.py`), scores (`animate/evaluate.py`), and writes the demo, sheet and summary.

---

## 4. Files added or changed

New (all mine; nobody else's files were rewritten):
- `animate/`
  - `skeleton.py`: DWPose drawing, retargeting
  - `bat_track.py`
  - `repose.py`
  - `motion_clean.py`
  - `delivery.py`
  - `scene.py`
  - `reference.py`
  - `build_jobs.py`: first run
  - `build_scene_jobs.py`
  - `composite.py`
  - `evaluate.py`
  - `finish.py`
  - `remote_musepose.sh`
- `tools/fetch_references.py`
- `data/README.md` and `data/SOURCES_commons.md`
- `docs/ANIMATION_REPORT.md`: this file
- `docs/RUN_LOG.md`: every run, failure and fix, newest first

Changed:
- `kaggle_runner/run_on_kaggle.sh` and `kernel_body.py`:
  - job data as a dataset;
  - `animate` mode;
  - MP_*/JOBS passed to the VM;
  - timeouts;
  - code overlay without weights or results, aborts if over 2 MB.
- `colab/remote_run.sh`: `animate` mode.
- `colab/run_on_colab.sh`: the same overlay exclusions.
- `.gitignore`: `checkpoints/`, `models/`, `data/motion_videos/`, `kaggle_results*/`.
- `CLAUDE.md`: layout lines.

Laptop environment `~/gen` gained `ultralytics`, `rembg`, `simple-lama-inpainting`. The last one pinned numpy 1.26.4, Pillow 9.5.0 and opencv-python 4.11; every module still imports and the v2 tests pass.

---

## 5. Honest limits

- **Reconstructed bowling arm.** The bowling arm in the delivery is a straight arm driven by a rotation model between the measured start and end positions, slowed 2.5x. The source frames there are too blurred to measure, so this is not the measured Cummins arm.
- **Batting background.** It comes from a modern broadcast frame turned into 1930s grey.
  - The uploader's name ("KLassic Rahul") is faint in the top-left corner; I left it rather than remove someone's credit.
  - The letterbox bars were filled with grass by LaMa, and seams are faint.
- **McGrath's photo** is a stock image (sporting-heroes.net); its watermark strip is cropped out of the scene. Do not publish the original photos.
- **Licences.**
  - MusePose weights: non-commercial research only.
  - Lord's photo and Commons clips: CC BY-SA, so attribution is needed.
- **Composited objects.** The bat, stump and ball are added in 2D. They do not cast shadows, and the bat's handle is hidden by drawing the generated gloves over it.
- **Run-up.** The bowler's run-up before the delivery stride is short: only what the clip shows inside the scene.
- **McGrath's outline.** It dips in 4 frame pairs (worst IoU 0.64), all in the fast part of the delivery where the
  arm sweeps across the body. In the last frame his arm comes across his face (merged-limb frames 43 and 76).
- **McGrath iterations.** Kaggle versions 3-6 are in `docs/RUN_LOG.md`; version 4's output is kept in
  `outputs/animate_v2/previous_v4/`. Version 6 added the measured hand shape on the rebuilt arm, removed broken
  hand detections, and ends at source frame 52.

---

## 6. How "body permanence" and "outlines" are checked (`animate/evaluate.py`)

DWPose is run again on the **final** video and compared with the skeleton the player was asked to follow.
- **PCK@a:** share of joints within a x (longest side of the target body box).
- **Frames with the whole body:** every shoulder, elbow, wrist, hip, knee and ankle that the target shows is detected, so no body part vanished.
- **Frames with merged limbs:** a left/right wrist, elbow, knee or ankle pair collapses onto one spot in the output while the target keeps them apart.
- **Outline stability:**
  - the player is segmented in every frame (YOLO26x-seg);
  - the overlap (IoU) between consecutive frames is measured: worst value, and number of drops below 0.7;
  - so is the largest change in body area from one frame to the next.
  A sudden drop means part of the body vanished or the outline broke.
- **Bat / ball frames:** how many frames got the composited object.
