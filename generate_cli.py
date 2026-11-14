import argparse
import json
from pathlib import Path
from generation_engine import Milestone3Engine

def main():
    parser = argparse.ArgumentParser(description="Milestone 3 Cross-Era Motion Transfer CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to player reference image")
    parser.add_argument("--metadata", type=str, required=True, help="Path to clip metadata.json")
    parser.add_argument("--densepose_video", type=str, required=True, help="Path to control_densepose_real.mp4")
    parser.add_argument("--role", type=str, default="auto", choices=["auto", "batsman", "bowler"], help="Player role")
    parser.add_argument("--output", type=str, default="", help="Optional custom output path")
    args = parser.parse_args()

    with open(args.metadata, "r") as f:
        meta = json.load(f)

    role = args.role
    if role == "auto":
        if "bowler" in args.densepose_video.lower() or "bowler" in args.image.lower():
            role = "bowler"
        else:
            role = "batsman"

    default_out = "outputs/m3_bowler_delivery.mp4" if role == "bowler" else "outputs/m3_magicanimate.mp4"
    output_path = Path(args.output) if args.output else Path(default_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt = meta.get("prompt", f"1930s archival film of a cricket {role}")
    num_frames = meta.get("num_frames", 16)
    fps = meta.get("fps", 8)

    engine = Milestone3Engine()
    engine.generate(
        image=args.image,
        control_video=args.densepose_video,
        prompt=prompt,
        num_frames=num_frames,
        fps=fps,
        output_path=str(output_path),
        role=role
    )

if __name__ == "__main__":
    main()