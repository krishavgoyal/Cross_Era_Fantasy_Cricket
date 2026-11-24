"""refine: post-processing on a COPY of the Kaggle/MusePose outputs (the originals in outputs/animate_v3/ are
never modified). Step by step, re-scored after every step (see docs/REFINE.md):
  step 1  conform to the project brief: 512 x 512 pixels, at most 100 frames (3-5 s at 24 fps)
  step 2  colours taken from the reference photo (sequence-level colour transfer, palette constraint)
  step 3  temporal colour stabilisation (no frame-to-frame colour flicker), gradient-safe output
"""
