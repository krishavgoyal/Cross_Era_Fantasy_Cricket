# data/

| folder | what | source |
|---|---|---|
| `reference_players/don_bradman/` | photos of Don Bradman (appearance for batting) | `bradman_original_repo.jpg` = `historical_images/bradman.jpg`; others Wikimedia Commons (public domain / CC BY), see `SOURCES_commons.md` |
| `reference_players/glenn_mcgrath/` | photos of Glenn McGrath (appearance for bowling) | added by the team (stock photos with Getty / Alamy / sporting-heroes watermarks: crop the watermark out, do not publish the originals) |
| `backgrounds/` | cricket grounds the player is placed on | Wikimedia Commons: SCG 1930s (public domain), Lord's 2017 (CC BY-SA 4.0) |
| `motion_videos/bowling/`, `motion_videos/batting/` | clips used only for their **motion** (skeleton) | Pat Cummins slow motion (YouTube screen recording, added by the team); others Wikimedia Commons (CC BY-SA), see `SOURCES_commons.md` |
| `raw_clips/` | trimmed/cropped motion clips fed to Student 1's motion stage | `bowler_new_cummins.mp4` = Cummins recording, `crop=1316:664:24:196`, 5.7-8.3 s |
| `processed_clips/<clip>/` | Student 1 output per clip: `source_512.mp4`, `pose_data.json` (DWPose body + hands + face), DensePose, and `bat_track.json` (tracked bat, `animate/bat_track.py`) | pipeline |
| `motion3d/<clip>/` | motion3d output per clip: MediaPipe pass, camera (ground) tracking, 3D fit (`fit.npz`, `fit.json`), 6 camera-angle stick figures (`views/`), `overlay.mp4`, `multiview.mp4`, `clip/` (export read by the scene code); `bowler_new_cummins/keyframes_bowling_arm.json` = hand-annotated blurred bowling arm | `motion3d/` (see `docs/MOTION3D.md`) |
| `processed_clips/extra_*` , `raw_clips/extra_*` | extra reference deliveries cut from the Commons videos (Parry, Stokes, Onions side-on; Amar Kala end-on) | `motion3d/extra.py` |
| `historical_images/` | the original pipeline's reference images (kept for `master_pipeline.py` paths) | team |

`tools/fetch_references.py` re-downloads the Commons files and rewrites `SOURCES_commons.md`.
Large videos in `motion_videos/` are not committed (see `.gitignore`); run the script to get them.

Every output that shows a real player (Bradman, McGrath) is an **AI reconstruction** and must be labelled as one.
CC BY / CC BY-SA files need attribution wherever they, or videos made from them, are shown.
