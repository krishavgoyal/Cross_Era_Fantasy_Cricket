#!/usr/bin/env python3
"""Cross-Era Fantasy Cricket master pipeline (integration).

One command from a modern broadcast clip + a historical photo to the final archival clip:

  1 motion     Student 1  track, crop 512x512@24, DWPose, smoothing       -> data/processed_clips/<clip>/
  2 densepose  Student 1  real DensePose + weak-frame patch               -> control_densepose_real.mp4
  3 generate   Student 2  AnimateDiff + DensePose ControlNet + IP-Adapter -> runs/<run>/generated.mp4 (+ .json)
  4 colour     Student 3  v3 DensePose-guided colour + era styles (or v2) -> runs/<run>/coloured.mp4, student3_v3/
  5 retime     integration  keyframes -> real-time 24 fps                 -> runs/<run>/final_24fps.mp4
  6 demo       integration  source | final side by side                   -> runs/<run>/demo_side_by_side.mp4
  7 evaluate   Student 3 checks + PCK motion fidelity                      -> runs/<run>/evaluation.json
  -> runs/<run>/manifest.json

Every stage is skipped when its outputs already exist (use --force <stage> to redo it), so a
failed run resumes where it stopped. Heavy stages run as subprocesses so each student's module
keeps its own CLI and dependencies.

examples
  # existing processed clip (motion + DensePose already done):
  python master_pipeline.py --clip-id bowler_delivery_001 --image data/historical_images/modern_bowler.jpeg --role bowler
  # from a raw broadcast clip:
  python master_pipeline.py --reference-video data/raw/drive.mp4 --clip-id drive_002 --action-type cover_drive \
      --image data/historical_images/bradman.jpg --role batsman --track-id 1 --crop-margin 0.10
  # plan only, run nothing:
  python master_pipeline.py --clip-id good_cover_drive_final --image data/historical_images/bradman.jpg --dry-run

exit codes: 0 ok, 2 bad arguments, 3 missing dependency/model, 4 bad input, 5 model/inference failure,
            6 output/ffmpeg failure, 70 unexpected
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "integration"))

EXIT_OK, EXIT_ARGS, EXIT_DEP, EXIT_INPUT, EXIT_MODEL, EXIT_OUTPUT, EXIT_UNEXPECTED = 0, 2, 3, 4, 5, 6, 70
STAGES = ["motion", "densepose", "generate", "colour", "retime", "demo", "evaluate"]


class StageError(Exception):
    def __init__(self, stage, code, msg):
        super().__init__(msg)
        self.stage, self.code = stage, code


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group("inputs")
    io.add_argument("--clip-id", required=True)
    io.add_argument("--image", required=True, type=Path, help="historical player reference photo")
    io.add_argument("--role", choices=["auto", "batsman", "bowler"], default="auto")
    io.add_argument("--reference-video", type=Path, help="raw broadcast clip (only needed if the clip is not processed yet)")
    io.add_argument("--action-type", default=None, help="e.g. cover_drive, fast_bowling_delivery (motion stage)")
    io.add_argument("--clips-root", type=Path, default=ROOT / "data" / "processed_clips")
    io.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    io.add_argument("--run-name", default=None, help="default <clip-id>_<role>")

    m = p.add_argument_group("motion (Student 1)")
    m.add_argument("--device", default="cpu", help="YOLO device")
    m.add_argument("--dwpose-device", default="cpu")
    m.add_argument("--track-id", type=int, default=None)
    m.add_argument("--crop-margin", type=float, default=0.40)
    m.add_argument("--fps", type=float, default=24.0, choices=(24.0, 30.0))

    g = p.add_argument_group("generation (Student 2)")
    g.add_argument("--prompt", default=None)
    g.add_argument("--negative-prompt", default=None)
    g.add_argument("--frames", type=int, default=16)
    g.add_argument("--window", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"))
    g.add_argument("--auto-window", action="store_true",
                   help="sample keyframes around the fastest wrist motion (release / impact) instead of the built-in window; "
                        "gives the bowling arm ~9 of 16 keyframes instead of ~5")
    g.add_argument("--steps", type=int, default=28)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--ref-box", type=float, nargs=4, default=None)
    g.add_argument("--gen-engine", choices=["v1", "v2"], default="v1",
                   help="v2 = generation_v2.py: adds a skeleton ControlNet (pose_data.json) and back/front view control")
    g.add_argument("--pose-scale", type=float, default=0.8, help="v2: skeleton ControlNet weight")
    g.add_argument("--gen-clean", action="store_true",
                   help="v2: plain backdrop, crisp outline, no colour-guessed brightening (use with --frames 32)")
    g.add_argument("--generated", type=Path, default=None,
                   help="use an existing Student 2 video (e.g. outputs/m3_bowler_delivery.mp4) instead of generating; "
                        "the frame mapping is then taken from Student 3 v3")
    g.add_argument("--legacy-timing", action="store_true",
                   help="Milestone 3 timing: 3+7 held frames at 7 fps (default: no held frames, real-time keyframe fps)")

    c = p.add_argument_group("colour (Student 3)")
    c.add_argument("--colour-engine", choices=["v3", "v2"], default="v3",
                   help="v3 = Krishav's DensePose-guided colour + era styles (default, CPU); v2 = SAM2/GrabCut")
    c.add_argument("--segmentation", choices=["auto", "sam2", "grabcut", "none"], default="auto", help="v2 only")
    c.add_argument("--sam2-config", default=None)
    c.add_argument("--sam2-checkpoint", default=None)
    c.add_argument("--colour-config", type=Path, default=None,
                   help="fixed Student 3 config; default builds one per clip from the generation conditioning")

    o = p.add_argument_group("output / control")
    o.add_argument("--final-fps", type=float, default=24.0)
    o.add_argument("--retime-mode", choices=["interpolate", "hold"], default="interpolate",
                   help="interpolate = ffmpeg minterpolate (blends neighbours: ghosting on fast limbs); "
                        "hold = repeat each generated frame (no blending, every frame is a real one)")
    o.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    o.add_argument("--force", nargs="*", default=[], choices=STAGES + ["all"])
    o.add_argument("--only", nargs="*", default=None, choices=STAGES)
    o.add_argument("--python", default=sys.executable, help="interpreter for generation (diffusers env)")
    o.add_argument("--densepose-python", default=None, help="interpreter for the DensePose env (default: python on PATH in run_densepose.sh)")
    o.add_argument("--dry-run", action="store_true")
    return p


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_cmd(stage, cmd, dry, env=None, cwd=None):
    log(f"{stage}: $ {' '.join(shlex.quote(str(c)) for c in cmd)}")
    if dry:
        return
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env)
    if r.returncode != 0:
        code = r.returncode if r.returncode in (EXIT_DEP, EXIT_INPUT, EXIT_MODEL, EXIT_OUTPUT) else (
            EXIT_MODEL if stage in ("generate", "densepose", "colour") else EXIT_UNEXPECTED)
        raise StageError(stage, code, f"{stage} failed (exit {r.returncode})")


def ffmpeg(stage, args, dry):
    if not shutil.which("ffmpeg"):
        raise StageError(stage, EXIT_DEP, "ffmpeg not found on PATH")
    run_cmd(stage, ["ffmpeg", "-v", "error", "-y", *args], dry)


def action_window(pose_json, before_s=2.0, after_s=0.45, fps=24.0):
    """(first, last) frame around the peak wrist speed of either arm (bowling release / bat impact)."""
    d = json.load(open(pose_json))
    n = len(d["frames"])
    best, peak = 0.0, n - 1
    for j in (4, 7):  # right_wrist, left_wrist
        w = np.array([[f["body_keypoints"][j]["x"], f["body_keypoints"][j]["y"]] for f in d["frames"]], np.float32)
        ok = np.array([f["body_keypoints"][j].get("confidence", 0) >= 0.3 for f in d["frames"]])
        sp = np.r_[0.0, np.linalg.norm(np.diff(w, axis=0), axis=1)]
        sp[~(ok & np.r_[False, ok[:-1]])] = 0.0  # low-confidence frames jump around; ignore them
        sp = np.convolve(sp, np.ones(5) / 5, mode="same")  # ignore single-frame detector jumps
        if sp.max() > best:
            best, peak = float(sp.max()), int(sp.argmax())
    first = max(0, peak - int(round(before_s * fps)))
    last = min(n - 1, peak + int(round(after_s * fps)))
    if last - first < 15:
        first = max(0, last - 15)
    return first, last, peak


def write_derived_sidecar(proc, path, role):
    """Sidecar for a video generated without one (older Student 2 runs), from Student 3 v3's processing.json."""
    g = proc["student2_geometry"]
    side = {
        "role": role,
        "derived_by": "student3_v3 processing.json (frame layout detected from held frames, crop re-derived)",
        "keyframes": len(g["sampled_source_indices"]),
        "output_frame_to_keyframe": proc["frame_layout"]["keyframe_of_output_frame"],
        "replaced_broken_keyframes": [],
        "sampling": {"sampled_source_frames": g["sampled_source_indices"], "crop_in_padded_px": g["crop_xyxy_padded"],
                     "pad_px": g["pad"], "generated_size": g["target_size"]},
    }
    src = side["sampling"]["sampled_source_frames"]
    side["output_frame_to_source_frame"] = [src[k] for k in side["output_frame_to_keyframe"]]
    json.dump(side, open(path, "w"), indent=1)
    return side


def keyframe_video(src, dst, layout, fps=7):
    """Drop held/padded frames: keep the first output frame of every keyframe."""
    import cv2
    cap = cv2.VideoCapture(str(src))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    first = {}
    for i, k in enumerate(layout[:len(frames)]):
        first.setdefault(k, i)
    keep = [frames[first[k]] for k in sorted(first)]
    h, w = keep[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    for f in keep:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise StageError("retime", EXIT_OUTPUT, "ffmpeg failed writing keyframes")
    return len(keep)


def keyframe_fps(sidecar, src_fps):
    """fps at which the sampled keyframes play back in real time."""
    src = sidecar["sampling"]["sampled_source_frames"]
    if len(src) < 2:
        return 7.0
    spacing = (src[-1] - src[0]) / (len(src) - 1)
    return round(src_fps / max(spacing, 1e-6), 4)


class Pipeline:
    def __init__(self, a):
        self.a = a
        self.clip = a.clips_root / a.clip_id
        self.role = a.role
        self.run_dir = a.runs_root / (a.run_name or f"{a.clip_id}_{a.role}")
        self.gen = self.run_dir / "generated.mp4"
        self.col = self.run_dir / "coloured.mp4"
        self.final = self.run_dir / "final_24fps.mp4"
        self.demo_path = self.run_dir / "demo_side_by_side.mp4"
        self.eval = self.run_dir / "evaluation.json"
        self.manifest = {"clip_id": a.clip_id, "role": a.role, "image": str(a.image), "run_dir": str(self.run_dir),
                         "started": dt.datetime.now().isoformat(timespec="seconds"), "stages": {}}
        force = set(a.force)
        if "all" in force:
            # motion + DensePose can only be redone from the raw broadcast clip
            force = set(STAGES) if a.reference_video else set(STAGES) - {"motion", "densepose"}
        self.force = force

    # ---------- helpers
    def want(self, stage, outputs):
        if self.a.only is not None and stage not in self.a.only:
            return False
        if stage in self.a.skip:
            return False
        if stage in self.force:
            return True
        return not all(Path(o).exists() for o in outputs)

    def record(self, stage, status, t0=None, **extra):
        rec = {"status": status}
        if t0 is not None:
            rec["seconds"] = round(time.time() - t0, 1)
        rec.update(extra)
        self.manifest["stages"][stage] = rec
        log(f"{stage}: {status}")

    def meta(self):
        return json.load(open(self.clip / "metadata.json"))

    def resolve_role(self):
        if self.role != "auto":
            return
        role = None
        if (self.clip / "metadata.json").exists():
            m = self.meta()
            role = m.get("role") if m.get("role") in ("batsman", "bowler") else None
            act = str(m.get("action_type", "")).lower()
            if role is None and act:
                role = "bowler" if ("bowl" in act or "delivery" in act) else "batsman"
        if role is None and self.a.action_type:
            role = "bowler" if "bowl" in self.a.action_type.lower() else "batsman"
        self.role = role or ("bowler" if "bowl" in self.a.clip_id.lower() else "batsman")
        self.manifest["role"] = self.role
        log(f"role resolved to {self.role}")

    # ---------- stages
    def motion(self):
        outs = [self.clip / f for f in ("source_512.mp4", "pose_data.json", "metadata.json", "control_pose.mp4")]
        if not self.want("motion", outs):
            return self.record("motion", "skipped (outputs exist)" if all(o.exists() for o in outs) else "skipped")
        a = self.a
        if not a.reference_video:
            raise StageError("motion", EXIT_ARGS, f"{self.clip} is not processed yet: pass --reference-video and --action-type")
        if not a.reference_video.exists():
            raise StageError("motion", EXIT_INPUT, f"reference video does not exist: {a.reference_video}")
        t0 = time.time()
        cmd = [a.python, ROOT / "pipeline_cli.py", "--reference-video", a.reference_video, "--output-root", a.clips_root,
               "--clip-id", a.clip_id, "--action-type", a.action_type or "unknown", "--fps", a.fps,
               "--device", a.device, "--dwpose-device", a.dwpose_device, "--crop-margin", a.crop_margin]
        if a.track_id is not None:
            cmd += ["--track-id", a.track_id]
        run_cmd("motion", cmd, a.dry_run)
        self.record("motion", "done", t0)

    def densepose(self):
        outs = [self.clip / "control_densepose_real.mp4"]
        if not self.want("densepose", outs):
            return self.record("densepose", "skipped (outputs exist)" if outs[0].exists() else "skipped")
        t0 = time.time()
        env = None
        if self.a.densepose_python:
            import os
            env = dict(os.environ)
            env["PATH"] = str(Path(self.a.densepose_python).parent) + os.pathsep + env["PATH"]
        if not shutil.which("bash"):
            raise StageError("densepose", EXIT_DEP, "bash not found (run the DensePose stage in WSL/Linux)")
        run_cmd("densepose", ["bash", ROOT / "densepose" / "run_densepose.sh", self.clip], self.a.dry_run, env=env)
        self.record("densepose", "done", t0)

    def generate(self):
        if self.a.generated is not None:
            return self._use_existing_video()
        outs = [self.gen, Path(str(self.gen) + ".json")]
        if not self.want("generate", outs):
            return self.record("generate", "skipped (outputs exist)" if all(o.exists() for o in outs) else "skipped")
        a = self.a
        for f in (a.image, self.clip / "metadata.json", self.clip / "control_densepose_real.mp4"):
            if not a.dry_run and not Path(f).exists():
                raise StageError("generate", EXIT_INPUT, f"missing input: {f}")
        t0 = time.time()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [a.python, ROOT / "generate_cli.py", "--image", a.image, "--metadata", self.clip / "metadata.json",
               "--densepose_video", self.clip / "control_densepose_real.mp4", "--role", self.role,
               "--output", self.gen, "--frames", a.frames, "--steps", a.steps, "--seed", a.seed,
               "--report", self.run_dir / "vram_report.json"]
        if a.legacy_timing:
            cmd += ["--fps", 7, "--pad-start", 3, "--pad-end", 7]
        else:
            # real-time keyframe rate is only known after sampling; the engine records the mapping and
            # the retime stage uses it. Export without held frames at a provisional rate.
            cmd += ["--fps", 7, "--pad-start", 0, "--pad-end", 0]
        if a.prompt:
            cmd += ["--prompt", a.prompt]
        if a.negative_prompt:
            cmd += ["--negative-prompt", a.negative_prompt]
        window = a.window
        if window is None and a.auto_window and not a.dry_run:
            first, last, peak = action_window(self.clip / "pose_data.json", fps=self._src_fps())
            window = [first, last]
            log(f"generate: auto window {first}..{last} (peak wrist speed at frame {peak})")
        if window:
            cmd += ["--window", *window]
        if a.ref_box:
            cmd += ["--ref-box", *a.ref_box]
        if a.gen_engine == "v2":
            cmd += ["--engine", "v2", "--pose-json", self.clip / "pose_data.json", "--pose-scale", a.pose_scale]
            if a.gen_clean:
                cmd += ["--clean"]
        run_cmd("generate", cmd, a.dry_run)
        extra = {}
        if not a.dry_run:
            side = json.load(open(str(self.gen) + ".json"))
            extra = {k: side.get(k) for k in ("output_frame_count", "peak_vram_gb", "inference_wall_time_sec",
                                              "replaced_broken_keyframes", "device")}
            extra["source_frames"] = side["sampling"]["sampled_source_frames"]
        self.record("generate", "done", t0, **extra)

    def _use_existing_video(self):
        """--generated: take a video Student 2 already made. If it differs from the run's copy, redo everything after it."""
        import filecmp
        a = self.a
        if not a.dry_run and not a.generated.exists():
            raise StageError("generate", EXIT_INPUT, f"--generated video does not exist: {a.generated}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        changed = not self.gen.exists() or (not a.dry_run and not filecmp.cmp(a.generated, self.gen, shallow=False))
        if changed and not a.dry_run:
            shutil.copyfile(a.generated, self.gen)
            stale = Path(str(self.gen) + ".json")
            if stale.exists():
                stale.unlink()  # the mapping of an earlier run does not belong to this video
            self.force |= {"colour", "retime", "demo", "evaluate"}
        return self.record("generate", "used existing video" + (" (new)" if changed else " (unchanged)"), None,
                           source=str(a.generated), note="frame mapping comes from Student 3 v3 unless a sidecar exists")

    def colour(self):
        outs = [self.col]
        if not self.want("colour", outs):
            return self.record("colour", "skipped (outputs exist)" if self.col.exists() else "skipped")
        a = self.a
        t0 = time.time()
        if a.colour_engine == "v3":
            return self._colour_v3(t0)
        if not a.dry_run and not Path(str(self.gen) + ".json").exists():
            raise StageError("colour", EXIT_INPUT, "colour engine v2 needs the generation sidecar; use the default v3 for --generated videos")
        cfg = a.colour_config
        if cfg is None:
            cfg = self.run_dir / "student3_config.json"
            if not a.dry_run:
                from student3_prompts import build_config
                built = build_config(ROOT / "student3_improved" / "configs" / "1930s_archival.json",
                                     Path(str(self.gen) + ".conditioning.npz"), Path(str(self.gen) + ".json"), cfg,
                                     parts_npz=self.clip / "densepose_real_parts.npz")
                log(f"colour: SAM2 prompts from conditioning frame 0: {sorted(built['sam2_prompts'])}")
        cmd = [a.python, ROOT / "student3_improved" / "postprocess.py", "--input", self.gen, "--output", self.col,
               "--config", cfg, "--segmentation", a.segmentation, "--allow-nonstandard-fps",
               "--mask-dir", self.run_dir / "masks", "--report", self.run_dir / "colour_processing.json"]
        if shutil.which("ffmpeg"):
            cmd += ["--ffmpeg", shutil.which("ffmpeg")]
        if a.sam2_config and a.sam2_checkpoint:
            cmd += ["--sam2-config", a.sam2_config, "--sam2-checkpoint", a.sam2_checkpoint]
        run_cmd("colour", cmd, a.dry_run, cwd=ROOT)
        seg = None
        if not a.dry_run and (self.run_dir / "colour_processing.json").exists():
            seg = json.load(open(self.run_dir / "colour_processing.json")).get("segmentation_backend")
        self.record("colour", "done", t0, config=str(cfg), segmentation=seg)

    def _colour_v3(self, t0):
        """Student 3 v3: DensePose-guided colouring with the exact sidecar mapping + real part ids, plus era styles."""
        import os
        a = self.a
        out = self.run_dir / "student3_v3"
        cfg = a.colour_config or (ROOT / "student3_v3" / "configs" / f"{self.role}_1930s.json")
        cmd = [a.python, "-m", "student3_v3.run", "--input", self.gen,
               "--densepose", self.clip / "control_densepose_real.mp4", "--config", cfg, "--out-dir", out]
        sidecar = Path(str(self.gen) + ".json")
        if sidecar.exists() or a.dry_run:
            cmd += ["--sidecar", sidecar]
        parts = self.clip / "densepose_real_parts.npz"
        if parts.exists():
            cmd += ["--parts-npz", parts]
        if shutil.which("ffmpeg"):
            cmd += ["--ffmpeg", shutil.which("ffmpeg")]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        run_cmd("colour", cmd, a.dry_run, env=env, cwd=ROOT)
        extra = {"engine": "student3_v3", "config": str(cfg), "out_dir": str(out)}
        if not a.dry_run:
            shutil.copyfile(out / "colourised_master.mp4", self.col)
            proc = json.load(open(out / "processing.json"))
            ev = json.load(open(out / "evaluation.json"))
            if not sidecar.exists():
                write_derived_sidecar(proc, sidecar, self.role)
                log("colour: no generation sidecar; wrote one from Student 3 v3's detected frame layout")
            extra.update({"alignment": proc.get("alignment"), "era_outputs": proc["outputs"].get("era"),
                          "technical_verdict": ev.get("technical_verdict"),
                          "mask_coverage": ev.get("mask_coverage", {}).get("player_pixels_with_a_class_mean")})
        self.record("colour", "done", t0, **extra)

    def _src_fps(self):
        try:
            return float(self.meta()["processed"].get("fps", 24.0))
        except Exception:
            return 24.0

    def retime(self):
        outs = [self.final]
        if not self.want("retime", outs):
            return self.record("retime", "skipped (outputs exist)" if self.final.exists() else "skipped")
        t0 = time.time()
        src = self.col if (self.col.exists() or self.a.dry_run) else self.gen
        if self.a.legacy_timing:
            kf = 7.0
        elif self.a.dry_run:
            kf = 3.5
        else:
            side = json.load(open(str(self.gen) + ".json"))
            kf = keyframe_fps(side, self._src_fps())
            # held frames (Milestone 3 padding) would freeze the motion; keep one frame per keyframe
            keys = self.run_dir / "_keyframes.mp4"
            n = keyframe_video(src, keys, side["output_frame_to_keyframe"])
            log(f"retime: {n} keyframes at {kf} fps (real time) -> {self.a.final_fps} fps")
            src = keys
        F = self.a.final_fps
        # play keyframes at their real-time rate, then motion-interpolate up to F
        if self.a.retime_mode == "hold":
            vf = f"setpts=N/({kf}*TB),fps={F},format=yuv420p"
            note = "each generated frame is repeated to reach the final fps (no blending, no ghosting)"
        else:
            vf = (f"setpts=N/({kf}*TB),minterpolate=fps={F}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
                  f"format=yuv420p")
            note = "frames between keyframes are motion-interpolated by ffmpeg, not generated"
        ffmpeg("retime", ["-i", src, "-vf", vf, "-r", F, "-c:v", "libx264", "-crf", "16", self.final], self.a.dry_run)
        self.record("retime", "done", t0, input=str(src), keyframe_fps=kf, final_fps=F,
                    mode=self.a.retime_mode, note=note)

    def demo(self):
        outs = [self.demo_path]
        if not self.want("demo", outs):
            return self.record("demo", "skipped (outputs exist)" if self.demo_path.exists() else "skipped")
        t0 = time.time()
        if self.a.dry_run:
            first, last = 0, 0
        else:
            sf = json.load(open(str(self.gen) + ".json"))["sampling"]["sampled_source_frames"]
            first, last = sf[0], sf[-1]
        fps = self._src_fps()
        start, dur = first / fps, (last - first + 1) / fps
        F = self.a.final_fps
        fc = (f"[0:v]trim=start={start:.4f}:duration={dur:.4f},setpts=PTS-STARTPTS,fps={F},scale=-2:512[a];"
              f"[1:v]fps={F},scale=-2:512[b];[a][b]hstack=inputs=2:shortest=1,format=yuv420p[v]")
        ffmpeg("demo", ["-i", self.clip / "source_512.mp4", "-i", self.final, "-filter_complex", fc, "-map", "[v]",
                        "-c:v", "libx264", "-crf", "18", self.demo_path], self.a.dry_run)
        self.record("demo", "done", t0, source_window_frames=[first, last])

    def evaluate(self):
        outs = [self.eval]
        if not self.want("evaluate", outs):
            return self.record("evaluate", "skipped (outputs exist)" if self.eval.exists() else "skipped")
        t0 = time.time()
        if self.a.dry_run:
            log("evaluate: would collect Student 3 evaluation and run integration/eval_pck.py (+ Student 3 v3 PCK)")
            return self.record("evaluate", "planned")
        result = {}
        v3_dir = self.run_dir / "student3_v3"
        if self.a.colour_engine == "v3" and (v3_dir / "evaluation.json").exists():
            result["student3_v3"] = json.load(open(v3_dir / "evaluation.json"))
        elif self.col.exists():
            rep = self.run_dir / "colour_evaluation.json"
            try:
                run_cmd("evaluate", [self.a.python, ROOT / "student3_improved" / "evaluate.py", "--input", self.gen,
                                     "--output", self.col, "--report", rep], False, cwd=ROOT)
                result["student3"] = json.load(open(rep))
            except StageError as e:
                result["student3"] = {"error": str(e)}
        try:
            from eval_pck import evaluate as pck
            result["motion_fidelity"] = pck(self.gen, self.clip / "pose_data.json", device=self.a.dwpose_device)
        except ModuleNotFoundError as e:
            result["motion_fidelity"] = {"error": f"DWPose backend not installed in this env: {e}"}
        except Exception as e:  # keep the run usable even if PCK fails
            result["motion_fidelity"] = {"error": str(e)}
        # Krishav's PCK (torso-normalised, plus the bowling arm) on the same DWPose output
        pred = self.gen.with_suffix(".dwpose.json")
        if (v3_dir / "processing.json").exists() and pred.exists():
            try:
                sys.path.insert(0, str(ROOT))
                from student3_v3.evaluate import compare_pose
                from student3_v3.student2_geometry import Student2Geometry
                proc = json.load(open(v3_dir / "processing.json"))
                geom = Student2Geometry(**{k: (tuple(v) if k.startswith("crop") or k == "target_size" else v)
                                           for k, v in proc["student2_geometry"].items()})
                result["motion_fidelity_student3_v3"] = compare_pose(
                    json.load(open(self.clip / "pose_data.json")), json.load(open(pred)), geom,
                    proc["frame_layout"]["keyframe_of_output_frame"])
            except Exception as e:
                result["motion_fidelity_student3_v3"] = {"error": str(e)}
        json.dump(result, open(self.eval, "w"), indent=1)
        mf = result.get("motion_fidelity", {})
        self.record("evaluate", "done", t0, pck_at_0_1=mf.get("pck@0.1"), velocity_cosine=mf.get("velocity_cosine"))

    def run(self):
        self.resolve_role()
        self.run_dir = self.a.runs_root / (self.a.run_name or f"{self.a.clip_id}_{self.role}")
        self.gen, self.col = self.run_dir / "generated.mp4", self.run_dir / "coloured.mp4"
        self.final, self.demo_path = self.run_dir / "final_24fps.mp4", self.run_dir / "demo_side_by_side.mp4"
        self.eval = self.run_dir / "evaluation.json"
        self.manifest["run_dir"] = str(self.run_dir)
        if not self.a.image.exists() and not self.a.dry_run:
            raise StageError("inputs", EXIT_INPUT, f"image does not exist: {self.a.image}")
        for stage in STAGES:
            getattr(self, stage)()


def main(argv=None):
    a = build_parser().parse_args(argv)
    # absolute paths: some stages run with cwd=ROOT
    for k in ("image", "reference_video", "clips_root", "runs_root", "colour_config", "generated"):
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).expanduser().resolve())
    pl = Pipeline(a)
    code = EXIT_OK
    try:
        pl.run()
        pl.manifest["result"] = "ok"
    except StageError as e:
        log(f"ERROR [{e.code}] in {e.stage}: {e}")
        pl.record(e.stage, "failed", error=str(e), exit_code=e.code)
        pl.manifest["result"] = f"failed at {e.stage}"
        code = e.code
    except KeyboardInterrupt:
        pl.manifest["result"] = "interrupted"
        code = EXIT_UNEXPECTED
    except Exception as e:
        log(f"ERROR [70]: {type(e).__name__}: {e}")
        pl.manifest["result"] = f"unexpected: {e}"
        code = EXIT_UNEXPECTED
    pl.manifest["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    pl.manifest["files"] = {k: str(v) for k, v in {"generated": pl.gen, "coloured": pl.col, "final": pl.final,
                                                     "demo": pl.demo_path, "evaluation": pl.eval}.items() if Path(v).exists()}
    if not a.dry_run:
        pl.run_dir.mkdir(parents=True, exist_ok=True)
        json.dump(pl.manifest, open(pl.run_dir / "manifest.json", "w"), indent=1)
        log(f"manifest: {pl.run_dir / 'manifest.json'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
