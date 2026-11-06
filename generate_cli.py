import sys
import json
import argparse
from pathlib import Path
import torch

import common
from ref_preprocessor import HistoricalImagePreprocessor
from generation_engine import Milestone3Engine

def run_pipeline(image_path: Path, metadata_path: Path, densepose_video_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / "m3_magicanimate.mp4"
    out_report = output_dir / "m3_vram_report.json"

    # 1. Parse Student 1's exact data contract
    try:
        meta = common.load_json(metadata_path)
        processed_meta = meta.get("processed", {})
        fps = float(processed_meta.get("fps", common.DEFAULT_FPS))
        num_frames = int(processed_meta.get("frame_count", 8))
    except common.PipelineError as e:
        print(f"Metadata parsing failed: {e}")
        sys.exit(2)

    # 2. Preprocess Reference Image
    preprocessor = HistoricalImagePreprocessor(target_size=(common.FRAME_SIZE, common.FRAME_SIZE))
    try:
        ref_image = preprocessor.process(str(image_path))
    except FileNotFoundError as e:
        print(e)
        sys.exit(4)

    # 3. Execute Generative Inference
    engine = Milestone3Engine()
    prompt = "Historical cricket batsman in white uniform playing a drive shot, 1930s vintage pitch"
    
    try:
        peak_vram, inference_sec = engine.generate(
            image=ref_image,
            control_video=str(densepose_video_path),
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            output_path=out_video
        )
    except torch.cuda.OutOfMemoryError:
        print("FATAL: Out of Memory. VRAM limit exceeded.")
        sys.exit(5)
    except Exception as e:
        print(f"Inference failure: {e}")
        sys.exit(70)

    # 4. Export Milestone 3 Hardware Proof Report
    report = {
        "milestone": 3,
        "clip_id": meta.get("clip_id", "unknown"),
        "resolution": f"{common.FRAME_SIZE}x{common.FRAME_SIZE}",
        "num_frames_generated": num_frames,
        "target_fps": fps,
        "peak_vram_gb": round(peak_vram, 2),
        "under_12gb_limit": peak_vram < 12.0,
        "inference_wall_time_sec": round(inference_sec, 2)
    }
    
    common.atomic_json_dump(report, out_report)
    print(f"\nSuccess. Peak VRAM: {peak_vram:.2f}GB (Limit < 12.0GB: {report['under_12gb_limit']})")
    print(f"Outputs saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student 2: Generative Video Inference")
    parser.add_argument("--image", type=Path, required=True, help="Historical player reference image")
    parser.add_argument("--metadata", type=Path, required=True, help="Student 1 metadata.json")
    parser.add_argument("--densepose_video", type=Path, required=True, help="Path to the extracted DensePose .mp4")
    parser.add_argument("--outdir", type=Path, default=Path("data/generated_clips/test_run"))

    args = parser.parse_args()

    run_pipeline(args.image, args.metadata, args.densepose_video, args.outdir)