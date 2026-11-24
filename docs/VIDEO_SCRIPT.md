# Presentation video script (2:30), following the 4 proposal milestones

- Speakers: **S1 [TODO name]** = Milestone 1, **S2 [TODO name]** = Milestone 2 + limitations, **S3 Krishav** = Milestones 3-4 (+ hook/close).
- Exact narration: `docs/VIDEO_NARRATION.md` (~330 words).
- Draft picture edit: `data/final/presentation_draft.mp4`, built by `python scripts/make_presentation.py` (1920x1080, 25 fps, no audio). Record the voice-over over it.
- Every generated clip carries the burned-in "AI-generated - academic project" label, and the draft adds one in the corner of every frame. Keep both visible. Source-clip social-media overlays are blurred.

| Time | Part | On screen (real files) | Caption in draft | Speaker |
|---|---|---|---|---|
| 0:00-0:08 | Hook | `data/final/final_match.mp4` from the Larwood clip on (4.0-12.0 s) | "What if Larwood bowled to Bradman?" | S3 |
| 0:08-0:25 | Idea + inputs | Modern batter clip `C2_coverdrive.mp4` (overlays blurred), `bradman_crop.png`, modern bowler clip `B1_bowler.mp4`, `larwood_crop.png` | "Modern motion + one archival photo" | S3 |
| 0:25-0:33 | Milestone 1: motion extraction | `data/pose/batter/overlay.mp4` (DWPose skeleton, tracked batter) | "Milestone 1: motion extraction - DWPose tracks one player" | S1 |
| 0:33-0:42 | Milestone 1 | `data/pose/batter/depth_preview.mp4` (source / pose / depth) | "Pose and depth controls, aligned frame by frame" | S1 |
| 0:42-0:50 | Milestone 1 | `data/pose/bowler/depth_preview.mp4` | "Bowler: target-only depth removes the fielders" | S1 |
| 0:50-1:02 | Milestone 2: running the model | `docs/report/figs/pipeline.png` + facts panel | "Milestone 2: Wan2.1-VACE-1.3B on a free Kaggle T4 - laptop 6 GB too small, 14.9 GB peak, ~9 min per clip" | S2 |
| 1:02-1:10 | Milestone 2 | `data/student3/bradman_1930s_graded_master.mp4` | "Team baseline (Student 2): AnimateDiff / MagicAnimate + DensePose" (clearly labelled as the baseline) | S2 |
| 1:10-1:20 | Milestone 3: generation | `docs/report/figs/ablation_strip.png` | "Skeleton -> bat lost / Depth -> bat kept" | S3 |
| 1:20-1:28 | Milestone 3 | `C_s42_17f.mp4` vs `D1_s42_17f.mp4` side by side (half speed) | "Skeleton -> bat lost" / "Depth -> bat kept" | S3 |
| 1:28-1:33 | Milestone 3 | Results card | "Depth kept the bat in 5/5 runs, skeleton lost it in 2/2 - PCK@0.2 0.90-1.00" | S3 |
| 1:33-1:45 | Milestone 3 | Raw `final_cap_s42_33f` (real speed) + raw `L3_s2024_33f` (slow motion) | "Raw VACE outputs: Bradman (F1) and Larwood (L3, slow motion)" | S3 |
| 1:45-1:50 | Milestone 4: cross-era | `docs/report/figs/era_strip.png` | "Milestone 4: the same moment in 1930s / 1970s / 1990s" | S3 |
| 1:50-1:59 | Milestone 4 | `data/final/final_eras_cap.mp4` at 2x speed | "1930s / 1970s / 1990s" | S3 |
| 1:59-2:15 | Milestone 4 highlight | `data/final/catalog/match_crossera.mp4` (full) | "Cross-era: Larwood in 1930s film vs Bradman in 1990s broadcast" | S3 |
| 2:15-2:23 | Limitations + ethics | Text card | "Limits: weak likeness, helmet outline -> dark cap, reference clothing copied. Ethics: AI-generated label on every frame, deceased public figures, public-domain photos, modern clips used only for motion" | S2 |
| 2:23-2:30 | Close | Team card + GitHub link | "Team: [TODO names] - github.com/krishavgoyal/Cross_Era_Fantasy_Cricket" | S3 |

Other material: `data/final/catalog/catalog_reel.mp4` (all four versions, 45.7 s) and the per-era matches can be shown at the live presentation or in the appendix. Kaggle screenshots (settings, smoke test, `kaggle_runs.csv`) are **[TODO: take them]**; they would replace the facts panel at 0:50-1:02.
