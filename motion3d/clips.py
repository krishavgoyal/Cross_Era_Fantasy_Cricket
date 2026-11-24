"""Motion clips handled by motion3d. Each writes to data/motion3d/<name>/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "data" / "processed_clips"

CLIPS = {
    # Bradman's motion: the cover drive (bat tracked by animate/bat_track.py)
    "good_cover_drive_final": dict(kind="batting", clip=PC / "good_cover_drive_final", pose_file="pose_data.json"),
    # McGrath's motion: Pat Cummins slow motion, the whole clip (frames 0-63: bound, gather, delivery stride,
    # arm over the top at 46-51, follow-through 52-63). The arm is blurred from frame 34 on: it comes from
    # hand-annotated keyframes (keyframes_bowling_arm.json); frames 34-58 are slowed down 2.5x.
    "bowler_new_cummins": dict(kind="bowling", clip=PC / "bowler_new_cummins", pose_file="pose_data_merged.json",
                               frames=(0, 63), delivery=(34, 58), stretch=2.5, bowling_arm="right",
                               arm_keyframes="keyframes_bowling_arm.json",
                               # Cummins' feet are off-screen from frame ~36: his unseen leg bones follow Ben Stokes'
                               # legs (side-on, feet visible). Matched by hand: Cummins output frame 71 (release,
                               # arm 15 deg past vertical) = Stokes frame 28 (widest delivery stride, front foot
                               # down); 1.75 Cummins output frames per Stokes frame (slow motion + 2.5x stretch).
                               leg_reference=dict(clip="extra_stokes_bowling", anchor=(71, 28), scale=1.75,
                                                  from_out=36)),
}

# More motions / camera angles (Commons videos, motion3d/extra.py): reference skeletons, not used for McGrath
for _n in ("extra_parry_bowling", "extra_stokes_bowling", "extra_onions_bowling", "extra_amar_endon_bowling"):
    CLIPS[_n] = dict(kind="bowling_reference", clip=PC / _n, pose_file="pose_data.json")


def out_dir(name):
    return ROOT / "data" / "motion3d" / name
