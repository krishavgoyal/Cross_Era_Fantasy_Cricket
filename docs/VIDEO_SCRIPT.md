# Presentation video script (3:00 max)

Speakers: **[TODO: assign S1 / S2 / S3]**. Every generated clip already carries the burned-in "AI-generated - academic project" label; keep it visible (do not crop the bottom-left corner).

| Time | Section | On screen (real files) | Voice-over (approx.) | Speaker |
|---|---|---|---|---|
| 0:00-0:08 | Hook | `data/final/final_match.mp4` (Larwood vs Bradman, first 8 s) | "What if Harold Larwood bowled to Sir Don Bradman, using motion from today's cricket?" | [TODO] |
| 0:08-0:20 | Problem | `docs/report/figs/final_strip.png` (top row only: modern source) next to `data/refs/ref_compare.png` | "We have one old photo per player and modern broadcast clips. The goal: animate the historical player with real modern motion, then make it look like footage of his era, on free hardware: a 6 GB laptop and a free Kaggle GPU." | [TODO] |
| 0:20-0:35 | Pipeline overview | `docs/report/figs/pipeline.png` (slow zoom left to right) | "DWPose extracts the batter's motion on the laptop CPU. We render it as a control video, either a skeleton or a depth map, and Wan2.1-VACE generates the clip from the control plus the reference photo on a free Kaggle T4." | [TODO] |
| 0:35-0:50 | Pose and controls | `data/pose/batter/depth_preview.mp4` (source / pose / depth side by side) | "Pose and depth come from the same frames with the same placement, so they line up exactly. We keep the head in frame with a top margin." | [TODO] |
| 0:50-1:00 | Kaggle generation | Kaggle screenshots: notebook settings (T4, Internet on), smoke-test cell output, `kaggle_runs.csv` **[TODO: take screenshots]** | "ComfyUI runs headless; a 17-frame test takes about 4.5 minutes on the T4, then the notebook runs every job in pairs.csv." | [TODO] |
| 1:00-1:20 | Ablation: the problem | `docs/report/figs/ablation_strip.png`, rows A and C highlighted | "With a skeleton control the bat disappears or turns into something else: the skeleton simply has no bat in it." | [TODO] |
| 1:20-1:40 | Ablation: the fix | same figure, row D1 highlighted; then `results/ablation.csv` table (from the report) | "With depth, every run kept one proper bat for the whole clip, and brightness flicker dropped. Honest result: a tighter reference crop did not clearly improve likeness, and changing the seed mattered more than the crop." | [TODO] |
| 1:40-1:50 | Real speed | `data/final/compare_cap.mp4` | "The generated frames span 5.6 seconds of the source, so we retime them to real speed. Frame blending kept the bat intact; motion interpolation tore it." | [TODO] |
| 1:50-2:15 | Eras | `data/final/final_eras_cap.mp4` (1930s, 1970s, 1990s) | "The same clip as 1930s newsreel film, 1970s colour TV and 1990s broadcast. Grain, gate weave and lower frame rates also hide diffusion artefacts." | [TODO] |
| 2:15-2:40 | Larwood vs Bradman | `data/final/final_match.mp4` (full) | "And the bowler: Larwood from a public-domain 1929 photo, driven by a modern fast bowler. Only the bowler is kept in the depth map, so the fielders don't turn into extra players." | [TODO] |
| 2:40-2:52 | Limitations | `docs/report/figs/ablation_strip.png` (cap hiding the face) | "Limits: likeness is still weak, the modern helmet outline becomes a dark cap, and the model copies reference clothing." | [TODO] |
| 2:52-3:00 | Ethics + close | Title card with the AI-generated label (first 2 s of `final_match.mp4`) | "These are AI-generated videos of real people, labelled on every frame and made for academic use only. Thank you." | [TODO] |

Notes
- Record at 1080p; the clips are 480x832 portrait, so place them side by side or on a dark background, not stretched.
- The source cover-drive clip has social-media overlays: use `compare_cap.mp4` / the figures, where they are blurred, not the raw clip.
- If time is short, cut 0:35-0:50 first.
