import argparse
import json
import sys
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description="Milestone 3 Cross-Era Motion Transfer CLI (Student 2)")
    p.add_argument("--image", required=True, help="historical player reference image")
    p.add_argument("--metadata", required=True, help="Student 1 clip metadata.json")
    p.add_argument("--densepose_video", "--densepose-video", dest="densepose_video", required=True,
                   help="control_densepose_real.mp4 of the same clip")
    p.add_argument("--role", default="auto", choices=["auto", "batsman", "bowler"])
    p.add_argument("--output", default="", help="output mp4 (default outputs/m3_<role>.mp4)")
    p.add_argument("--prompt", default=None, help="override the built-in role prompt")
    p.add_argument("--negative-prompt", default=None, help="override the built-in negative prompt")
    p.add_argument("--frames", type=int, default=16, help="keyframes to generate (motion adapter is trained on 16)")
    p.add_argument("--fps", type=float, default=7.0, help="export fps of the generated video")
    p.add_argument("--pad-start", type=int, default=3, help="frames the first keyframe is held (0 = none)")
    p.add_argument("--pad-end", type=int, default=7, help="frames the last keyframe is held (0 = none)")
    p.add_argument("--window", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"),
                   help="DensePose frame range to sample keyframes from (default: built-in per role)")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ref-box", type=float, nargs=4, default=None, metavar=("X1", "Y1", "X2", "Y2"),
                   help="bowler only: normalized box of the player in --image (default fits modern_bowler.jpeg)")
    p.add_argument("--report", default="", help="also write the run report json here (e.g. m3_vram_report.json)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    for path in (args.image, args.metadata, args.densepose_video):
        if not Path(path).exists():
            print(f"ERROR [4]: input does not exist: {path}", file=sys.stderr)
            return 4

    with open(args.metadata) as f:
        meta = json.load(f)
    role = args.role
    if role == "auto":
        role = meta.get("role") if meta.get("role") in ("batsman", "bowler") else None
        if role is None:
            action = str(meta.get("action_type", "")).lower()
            name = (args.densepose_video + args.image).lower()
            role = "bowler" if ("bowl" in action or "delivery" in action or "bowler" in name) else "batsman"

    output_path = Path(args.output) if args.output else Path(f"outputs/m3_{role}.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from generation_engine import Milestone3Engine
    except ModuleNotFoundError as e:
        print(f"ERROR [3]: missing dependency: {e}", file=sys.stderr)
        return 3

    engine = Milestone3Engine()
    if args.ref_box:
        engine.bowler_ref_box = tuple(args.ref_box)
    try:
        engine.generate(
            image=args.image,
            control_video=args.densepose_video,
            prompt=args.prompt,
            num_frames=args.frames,
            fps=args.fps,
            output_path=str(output_path),
            role=role,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            steps=args.steps,
            pad_start=args.pad_start,
            pad_end=args.pad_end,
            window=tuple(args.window) if args.window else None,
        )
    except ValueError as e:
        print(f"ERROR [4]: {e}", file=sys.stderr)
        return 4
    except Exception as e:  # model download / CUDA OOM / diffusers errors
        print(f"ERROR [5]: generation failed: {e}", file=sys.stderr)
        return 5

    if args.report:
        side = json.load(open(str(output_path) + ".json"))
        rep = {k: side[k] for k in ("role", "output_frame_count", "export_fps", "peak_vram_gb",
                                    "under_12gb_limit", "inference_wall_time_sec", "device")}
        rep.update({"milestone": 3, "clip_id": meta.get("clip_id"),
                    "resolution": "x".join(map(str, side["sampling"]["generated_size"]))})
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(args.report, "w"), indent=2)
    print(f"done: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
