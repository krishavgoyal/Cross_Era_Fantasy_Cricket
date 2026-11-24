"""Generation v2: Student 2's Milestone3Engine plus a skeleton ControlNet and view-direction control.

Why: the bowler clip is filmed from behind. With only the DensePose ControlNet, SD 1.5 drew a
front-facing bowler, so the generated body was left/right mirrored relative to the source and PCK
on shoulders, elbows, wrists and hips was about 0 (Kaggle run v4, see docs/RUN_LOG.md).

What v2 adds (generation_engine.py is unchanged; this subclasses it):
  1. A second ControlNet, lllyasviel/control_v11p_sd15_openpose, fed with an OpenPose-style skeleton
     drawn from Student 1's pose_data.json for exactly the sampled source frames, mapped through the
     crop the engine recorded (last_sampling). OpenPose colours encode left vs right limbs.
  2. View direction from the same pose data: if the person's right shoulder/hip is on the image
     right in most sampled frames, the clip is a back view. Then the prompt says so, the negative
     prompt rejects a visible face, and face points (nose, eyes) are left out of the skeleton, the
     way OpenPose itself sees a person from behind.
Everything else (reference prep, bat layer, post-processing, sidecar) is Student 2's engine.
"""
import json
import math

import cv2
import numpy as np
from PIL import Image

from generation_engine import Milestone3Engine

POSE_CONTROLNET = "lllyasviel/control_v11p_sd15_openpose"

# OpenPose 18-keypoint body order and drawing convention (as in controlnet_aux draw_bodypose).
OPENPOSE_JOINTS = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
                   "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip",
                   "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12),
         (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
           (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
           (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
FACE_POINTS = ("nose", "right_eye", "left_eye")

BACK_VIEW_PROMPT = "seen from behind, back view, facing away from the camera, back of the head, "
BACK_VIEW_NEGATIVE = ", face, facing the camera, front view, shirt buttons, collar"
SIDE_VIEW_PROMPT = "side-on view, body in profile, "
CLEAN_PROMPT = "sharp focus on one full-body player, clean outline, "
CLEAN_NEGATIVE = ", motion blur, double exposure, ghosting, extra limbs, duplicate person, blurry edges"


def plain_backdrop(h, w, sky=196.0, ground=118.0, horizon=0.40):
    """Clean grayscale backdrop (float32, h x w): flat sky over flat ground with a short smooth edge.
    Replaces the blurred photo plate so nothing in the background competes with the player."""
    y = np.arange(h, dtype=np.float32)[:, None]
    t = np.clip((y - horizon * h) / 10.0 + 0.5, 0.0, 1.0)          # 10 px transition, not a blur
    shade = 1.0 - 0.10 * np.clip((y - horizon * h) / ((1 - horizon) * h), 0, 1)  # ground slightly darker at the bottom
    plate = sky * (1 - t) + ground * shade * t
    return np.repeat(plate, w, axis=1).astype(np.float32)


def crisp_silhouettes(densepose_video, sampling):
    """Body masks (float 0..1, generated size) for the sampled frames, cut exactly like the engine's
    crop but without its leg dilation and 7x7 soft blur: filled, 1 px eroded, 1 px feather."""
    wanted = sampling["sampled_source_frames"]
    x1, y1, x2, y2 = sampling["crop_in_padded_px"]
    pad = sampling["pad_px"]
    gw, gh = sampling["generated_size"]
    frames, i = {}, 0
    cap = cv2.VideoCapture(str(densepose_video))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = f
        i += 1
    cap.release()
    out = []
    k3 = np.ones((3, 3), np.uint8)
    for idx in wanted:
        f = frames[idx]
        if pad > 0:
            f = cv2.copyMakeBorder(f, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = cv2.resize(f[y1:y2, x1:x2], (gw, gh), interpolation=cv2.INTER_NEAREST)
        fg = (crop.max(axis=2) > 10).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        solid = np.zeros_like(fg)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 150:
                cv2.drawContours(solid, [c], -1, 255, thickness=cv2.FILLED)
        solid = cv2.erode(solid, k3, iterations=1)
        out.append(cv2.GaussianBlur(solid.astype(np.float32) / 255.0, (3, 3), 0.8))
    return out


def player_reference(image_path, box, size=512):
    """IP-Adapter reference: the player box from the photo, padded (not squashed) to a square."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = box
    crop = img.crop((int(w * x1), int(h * y1), int(w * x2), int(h * y2)))
    cw, ch = crop.size
    side = max(cw, ch)
    edge = np.array(crop)[[0, -1], :, :].reshape(-1, 3).mean(axis=0).astype(np.uint8)  # pad with border colour
    sq = Image.new("RGB", (side, side), tuple(int(c) for c in edge))
    sq.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return sq.resize((size, size), Image.Resampling.LANCZOS), crop


def load_pose_frames(pose_json):
    """frame_index -> {joint: (x_px, y_px)} in the pose data's pixel space, visible joints only."""
    with open(pose_json) as f:
        data = json.load(f)
    w, h = data.get("resolution", [512, 512])
    out = {}
    for fr in data["frames"]:
        pts = {}
        for kp in fr.get("body_keypoints", []):
            if kp.get("confidence", 0) >= 0.3 or kp.get("confidence_interpolated"):
                pts[kp["joint"]] = (float(kp["x"]) * w, float(kp["y"]) * h)
        out[int(fr["frame_index"])] = pts
    return out


def is_back_view(pts):
    """True/False for one frame, None if it can't tell. From behind, the person's right is on image right."""
    votes = []
    for r, l in (("right_shoulder", "left_shoulder"), ("right_hip", "left_hip")):
        if r in pts and l in pts and abs(pts[r][0] - pts[l][0]) > 2:
            votes.append(pts[r][0] > pts[l][0])
    if not votes:
        return None
    return sum(votes) * 2 > len(votes) if sum(votes) * 2 != len(votes) else None


def shoulder_ratio(pts):
    """Horizontal shoulder width / torso length (neck to mid-hip); small = side-on. None if unknown."""
    if not all(k in pts for k in ("right_shoulder", "left_shoulder", "neck")):
        return None
    hips = [pts[k] for k in ("right_hip", "left_hip") if k in pts]
    if not hips:
        return None
    mh = np.mean(hips, axis=0)
    torso = math.hypot(pts["neck"][0] - mh[0], pts["neck"][1] - mh[1])
    return abs(pts["right_shoulder"][0] - pts["left_shoulder"][0]) / torso if torso > 1 else None


def clip_view(frames_pts, side_band=None, side_ratio=0.5):
    """Returns (view, back_fraction). 'back' if most decidable frames are back views, else 'front'.
    Shoulder left/right order is unreliable when the shoulders overlap, so a 'back' clip whose median
    shoulder width is under side_ratio x torso length is 'side' (bowler_new_cummins: 0.35; the true
    back view bowler_delivery_001: 0.71). side_band=(lo, hi): a mixed back fraction also means 'side'."""
    v = [is_back_view(p) for p in frames_pts]
    v = [x for x in v if x is not None]
    frac = (sum(v) / len(v)) if v else 0.0
    if side_band and side_band[0] <= frac <= side_band[1]:
        return "side", round(frac, 3)
    if frac > 0.5:
        ratios = [r for r in (shoulder_ratio(p) for p in frames_pts) if r is not None]
        if ratios and float(np.median(ratios)) < side_ratio:
            return "side", round(frac, 3)
        return "back", round(frac, 3)
    return "front", round(frac, 3)


def map_to_generated(pt, sampling):
    """Source/DensePose pixel -> generated pixel, the formula recorded in the engine sidecar."""
    x1, y1, x2, y2 = sampling["crop_in_padded_px"]
    pad = sampling["pad_px"]
    gw, gh = sampling["generated_size"]
    return ((pt[0] + pad - x1) * gw / (x2 - x1), (pt[1] + pad - y1) * gh / (y2 - y1))


def draw_openpose(pts, size, drop=(), stickwidth=4):
    """OpenPose-style RGB skeleton image (black background). pts: joint -> (x, y) in this canvas."""
    w, h = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    xy = [pts.get(j) if j not in drop else None for j in OPENPOSE_JOINTS]
    for i, (a, b) in enumerate(LIMBS):
        if xy[a] is None or xy[b] is None:
            continue
        (xa, ya), (xb, yb) = xy[a], xy[b]
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        length = math.hypot(xa - xb, ya - yb)
        angle = math.degrees(math.atan2(ya - yb, xa - xb))
        poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in COLOURS[i]])
    for i, p in enumerate(xy):
        if p is not None:
            cv2.circle(canvas, (int(p[0]), int(p[1])), 4, COLOURS[i], thickness=-1)
    return canvas


class SkeletonGuidedEngine(Milestone3Engine):
    """clean=True (recommended): plain backdrop, crisp silhouette, no colour-guessed brightening of
    "pads"/"torso" (it painted white blobs), and FreeNoise context windows when more than 16 frames
    are generated, so the final video can repeat frames instead of blending them."""

    def __init__(self, pose_json, pose_scale=0.8, view="auto", clean=False):
        super().__init__()
        self.pose_json = str(pose_json)
        self.pose_scale = float(pose_scale)
        self.view = view  # auto | back | front | side
        self.clean = bool(clean)
        self._real_pipe = None
        self._control_video = None

    def load_model(self):
        super().load_model()
        from diffusers import AnimateDiffControlNetPipeline, ControlNetModel

        print(f"5/5 Loading skeleton ControlNet ({POSE_CONTROLNET})...")
        try:
            pose_cn = ControlNetModel.from_pretrained(POSE_CONTROLNET, torch_dtype=self.dtype, variant="fp16")
        except Exception:
            pose_cn = ControlNetModel.from_pretrained(POSE_CONTROLNET, torch_dtype=self.dtype)
        old = self.pipe
        old.remove_all_hooks()  # undo the CPU offload hooks before re-assembling the pipeline
        comps = dict(old.components)
        comps["controlnet"] = [comps["controlnet"], pose_cn]  # the pipeline wraps a list in MultiControlNetModel
        pipe = AnimateDiffControlNetPipeline(**comps)  # same UNet (with IP-Adapter), scheduler, VAE
        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        pipe.enable_vae_slicing()
        self.pipe = pipe

    # ---- reference photo: crop the player box for any photo when --ref-box is given (clean mode)
    def prepare_bradman_and_clean_background(self, image, role="batsman"):
        if self.clean and self.bowler_ref_box and isinstance(image, str):
            clip_ref, crop = player_reference(image, self.bowler_ref_box)
            tw, th = self.target_size
            clean_pil = crop.resize(self.target_size, Image.Resampling.LANCZOS)
            plate = plain_backdrop(th, tw)
            return clip_ref, clean_pil, plate
        clip_ref, clean_pil, plate = super().prepare_bradman_and_clean_background(image, role=role)
        if self.clean:
            plate = plain_backdrop(*plate.shape[:2])
        return clip_ref, clean_pil, plate

    # ---- masks: crisp silhouette, no luminance lifting from colour guesses
    def load_aligned_densepose_sequence(self, video_path, target_frames=16, role="batsman", window=None):
        control, masks = super().load_aligned_densepose_sequence(video_path, target_frames, role, window)
        if self.clean:
            sil = crisp_silhouettes(video_path, self.last_sampling)
            for m, s in zip(masks, sil):
                m["body_soft"] = s
                m["torso"] = np.zeros_like(s)
                m["pads"] = np.zeros_like(s)
            self.last_sampling["clean_mode"] = {
                "backdrop": "plain (no blurred photo plate)",
                "mask": "DensePose silhouette, no leg dilation, 1 px erode + 1 px feather",
                "luminance_lifts": "off (pads/torso)",
            }
        return control, masks

    def generate(self, *args, **kwargs):
        if self.pipe is None:
            self.load_model()
        engine = self

        class _PipeWithSkeleton:
            """Stands in for self.pipe during Student 2's generate(): adds the skeleton ControlNet input."""
            def __getattr__(self, name):
                return getattr(engine._real_pipe, name)

            def __call__(self, **kw):
                return engine._call_with_skeleton(**kw)

        self._real_pipe = self.pipe
        self.pipe = _PipeWithSkeleton()
        try:
            return super().generate(*args, **kwargs)
        finally:
            self.pipe = self._real_pipe

    def skeleton_frames(self, sampling):
        """(list of PIL skeleton images, view, back_fraction, joints_per_frame) for the sampled frames."""
        poses = load_pose_frames(self.pose_json)
        src = sampling["sampled_source_frames"]
        frame_pts = [poses.get(i, {}) for i in src]
        view, frac = clip_view(frame_pts, side_band=(0.3, 0.7))
        if self.view in ("back", "front", "side"):
            view = self.view
        size = tuple(sampling["generated_size"])
        images, counts = [], []
        for pts in frame_pts:
            g = {j: map_to_generated(p, sampling) for j, p in pts.items()}
            drop = FACE_POINTS if view == "back" else ()
            images.append(Image.fromarray(draw_openpose(g, size, drop=drop)))
            counts.append(len([j for j in g if j not in drop]))
        return images, view, frac, counts

    def _call_with_skeleton(self, **kw):
        sampling = self.last_sampling
        images, view, frac, counts = self.skeleton_frames(sampling)
        kw["conditioning_frames"] = [kw["conditioning_frames"], images]
        dp_scale = kw.get("controlnet_conditioning_scale", 1.0)
        kw["controlnet_conditioning_scale"] = [float(dp_scale), self.pose_scale]
        prefix, neg = "", ""
        if view == "back":
            prefix, neg = BACK_VIEW_PROMPT, BACK_VIEW_NEGATIVE
        elif view == "side":
            prefix = SIDE_VIEW_PROMPT
        if self.clean:
            prefix, neg = CLEAN_PROMPT + prefix, neg + CLEAN_NEGATIVE
        kw["prompt"] = prefix + kw["prompt"]  # prefix: CLIP keeps only the first 77 tokens
        kw["negative_prompt"] = (kw.get("negative_prompt") or "") + neg
        n = int(kw["num_frames"])
        free_noise = n > 16
        sampling["generation_v2"] = {
            "pose_controlnet": POSE_CONTROLNET,
            "pose_source": self.pose_json,
            "controlnet_conditioning_scale": kw["controlnet_conditioning_scale"],
            "view": view,
            "back_view_fraction": frac,
            "skeleton_joints_per_frame": counts,
            "prompt_used": kw["prompt"],
            "negative_prompt_used": kw["negative_prompt"],
            "clean": self.clean,
            "frames": n,
            "free_noise": {"context_length": 16, "context_stride": 4} if free_noise else None,
        }
        self.skeleton_images = images  # kept for inspection / debugging
        print(f"v2: skeleton ControlNet scale {self.pose_scale}, view={view} (back fraction {frac}), "
              f"joints/frame {min(counts)}-{max(counts)}, frames {n}, clean={self.clean}")
        pipe = self._real_pipe
        if not free_noise:
            return pipe(**kw)
        # No enable_free_noise_split_inference(): it splits UNet activations but not the ControlNet
        # residuals added to them ("tensor a (768/1536) must match ... (3072)", Kaggle cricket-clean v1).
        # 32 frames fit a 15 GB T4 without it (16 frames peaked at 6.4 GB).
        def _plain():
            try:
                pipe.disable_free_noise()  # restore the plain motion blocks
            except Exception:
                pass
        try:
            pipe.enable_free_noise(context_length=16, context_stride=4)
            out = pipe(**kw)
            _plain()
            return out
        except Exception as e:  # fall back to consecutive 16-frame chunks so the run still finishes
            print(f"v2: FreeNoise failed ({type(e).__name__}: {e}); generating {n} frames in 16-frame chunks")
            sampling["generation_v2"]["free_noise_error"] = f"{type(e).__name__}: {e}"
            _plain()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._chunked(pipe, kw, n)

    @staticmethod
    def _chunked(pipe, kw, n, size=16):
        import types
        frames = []
        for a in range(0, n, size):
            b = min(n, a + size)
            part = dict(kw)
            part["num_frames"] = b - a
            part["conditioning_frames"] = [c[a:b] for c in kw["conditioning_frames"]]
            frames.extend(pipe(**part).frames[0])
        return types.SimpleNamespace(frames=[frames])
