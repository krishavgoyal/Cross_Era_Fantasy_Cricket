# bowler_delivery_001

Bowler isolated and dynamically cropped from the broadcast clip. 512x512, 24 fps, 121 frames. Every file is frame aligned (frame t in one file = frame t in all others).

| file | what it is |
|---|---|
| source_512.mp4 | cropped RGB of the bowler (the reference video) |
| control_pose.mp4 | DWPose skeleton control video (body, hands, face) on black |
| pose_data.json | DWPose keypoints per frame (OpenPose 18 body + 21 + 21 hands + 68 face), 512 crop pixels |
| pose_vectors.npz | same keypoints as a numpy matrix |
| control_densepose_real.mp4 | real DensePose (detectron2 R50 FPN) control video, PARULA colours on black |
| densepose_real_parts.npz | part_index (121,512,512) uint8, 0 = background, 1 to 24 = DensePose body parts |
| densepose_real_iuv.npz | u, v (121,512,512) uint8 surface coordinates for the same pixels |
| metadata.json | crop window and tracked bowler box in the original video for every frame |
| PATCHED_FRAMES.json | frame 118 lost its legs to motion blur; legs filled from the skeleton, u,v = 128 there |

Same format as good_cover_drive_final (batting). Use control_pose.mp4 and/or control_densepose_real.mp4 as the ControlNet conditioning.

Known limits: the bowler is small in the source broadcast (about 90 to 170 px tall), so the crop is upscaled 2.5 to 4.5x and softer than the batting clip. No bat or ball tracking.
