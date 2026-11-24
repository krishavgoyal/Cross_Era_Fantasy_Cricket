# Narration (2:30, ~330 words)

Timing matches `docs/VIDEO_SCRIPT.md` and `data/final/presentation_draft.mp4`. Speakers: **S1 [TODO name]**, **S2 [TODO name]**, **S3 Krishav**. Read at a calm pace (about 2.2 words per second).

**0:00-0:08 - S3 (hook)**
What if Harold Larwood bowled to Sir Don Bradman, filmed the way cricket looked in 1932?

**0:08-0:25 - S3 (idea + inputs)**
That's our project, Cross-Era Fantasy Cricket. We take the motion from a modern clip, a cover drive and a fast bowler's delivery, and one archival photo of each legend. The goal is a short video of the historical player performing the modern action, made only with free hardware.

**0:25-0:50 - S1 (Milestone 1: motion extraction)**
Milestone one is motion extraction. DWPose finds every joint, and we track just one player, so the umpire and fielders are ignored. We clean the keypoints and turn them into two kinds of control video: a skeleton, and a depth map from Depth Anything. Both are aligned frame by frame. For the bowler, we keep only his own depth, so the fielders behind him cannot become extra players.

**0:50-1:10 - S2 (Milestone 2: running the model)**
Milestone two is running the model. Our laptop GPU has six gigabytes; one clip needs almost fifteen, so we run Wan 2.1 VACE on a free Kaggle T4, about nine minutes per clip. We picked VACE because it accepts a reference photo plus a control video. LTX-2 needs over thirty gigabytes, and MagicAnimate is outdated. This is our team's own AnimateDiff and MagicAnimate baseline with DensePose.

**1:10-1:45 - S3 (Milestone 3: generation)**
Milestone three is generation. With the skeleton, the bat disappears, because the skeleton has no bat in it. With depth, every run kept one proper bat. Motion accuracy stayed above zero point nine. A tighter photo crop didn't clearly help the likeness; the random seed mattered more. Here are the raw outputs: Bradman, and Larwood in slow motion.

**1:45-2:15 - S3 (Milestone 4: cross-era)**
Milestone four is the cross-era look. Our filters recreate 1930s newsreel film, 1970s television and 1990s broadcast, and they keep the motion. And the highlight: Larwood in 1930s film, bowling to Bradman in 1990s broadcast.

**2:15-2:30 - S2, then S3 (limitations, ethics, close)**
S2: Limitations: the likeness is still weak, and the model copies helmets and clothing. These are AI-generated videos of real people, labelled on every frame, from public-domain photos, for academic use only.
S3: The code is on GitHub. Thank you.
