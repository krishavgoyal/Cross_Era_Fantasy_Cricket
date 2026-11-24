"""motion3d: one rigid 3D skeleton per motion clip (fixed bone lengths, planted feet on a ground plane,
fixed-length bat), fitted to two 2D pose detectors (DWPose + MediaPipe) and rendered from several camera
angles as OpenPose stick figures. See docs/MOTION3D.md.

Built separately from animate/ (whose results are kept); animate/ scene + MusePose code is reused unchanged.
"""
