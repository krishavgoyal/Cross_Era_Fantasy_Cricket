import time
import inspect
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, MotionAdapter, ControlNetModel, DDIMScheduler
from diffusers.utils import export_to_video

class Milestone3Engine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.pipe = None
        # Native 3:4 portrait resolution matching bradman.jpg
        self.target_size = (384, 512)  # (width, height)

    def load_model(self):
        print("1/4 Loading SD 1.5 DensePose ControlNet (MnLgt/densepose)...")
        controlnet = ControlNetModel.from_pretrained(
            "MnLgt/densepose",
            torch_dtype=self.dtype
        )

        print("2/4 Loading AnimateDiff Motion Adapter (v1-5-2)...")
        motion_adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=self.dtype
        )

        print("3/4 Building AnimateDiff + ControlNet Pipeline...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=motion_adapter,
            controlnet=controlnet,
            custom_pipeline="pipeline_animatediff_controlnet",
            torch_dtype=self.dtype
        )

        print("4/4 Loading IP-Adapter balanced for full hand/arm swing...")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
        # 0.56 keeps Bradman's identity, cap, and pads while freeing his arms/hands from the static photo pose!
        self.pipe.set_ip_adapter_scale(0.56)

        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False
        )

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()

    def prepare_bradman_and_clean_background(self, image):
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = Image.fromarray(image).convert("RGB")

        arr = np.array(pil_img)
        h, w, _ = arr.shape

        # Strip any pre-existing pillarbox bars from generate_cli.py
        col_std = arr.std(axis=(0, 2))
        valid_cols = np.where(col_std > 8.0)[0]
        if len(valid_cols) > w * 0.4:
            margin_x = int((valid_cols[-1] - valid_cols[0]) * 0.02)
            x1 = max(0, valid_cols[0] + margin_x)
            x2 = min(w, valid_cols[-1] - margin_x)
            arr = arr[:, x1:x2]

        row_std = arr.std(axis=(1, 2))
        valid_rows = np.where(row_std > 8.0)[0]
        if len(valid_rows) > h * 0.4:
            margin_y = int((valid_rows[-1] - valid_rows[0]) * 0.01)
            y1 = max(0, valid_rows[0] + margin_y)
            y2 = min(h, valid_rows[-1] - margin_y)
            arr = arr[y1:y2, :]

        clean_pil = Image.fromarray(arr).resize(self.target_size, Image.Resampling.LANCZOS)
        ref_gray = np.array(clean_pil.convert("L"), dtype=np.float32)

        # Build a clean background plate from the outer background strips of bradman.jpg
        tw, th = self.target_size
        left_strip = ref_gray[:, :int(tw * 0.24)]
        right_strip = ref_gray[:, int(tw * 0.88):]
        bg_cols = np.concatenate([left_strip, right_strip], axis=1)

        bg_plate = cv2.resize(bg_cols, (tw, th), interpolation=cv2.INTER_LINEAR)
        bg_plate = cv2.GaussianBlur(bg_plate, (35, 19), 0)

        # Crop CLIP reference focused on head, cap, and upper torso so CLIP doesn't lock his downward-pointing arms!
        cw, ch = clean_pil.size
        clip_ref = clean_pil.crop((0, 0, cw, int(ch * 0.62))).resize((512, 512), Image.Resampling.LANCZOS)

        return clip_ref, clean_pil, bg_plate

    def load_aligned_densepose_sequence(self, video_path, target_frames=16):
        """
        Loads 16 frames spanning the cover drive (frame 34 to frame 132) where the hand and arm
        movement is largest, and enhances the arm/glove segments so ControlNet tracks the hands accurately.
        """
        cap = cv2.VideoCapture(str(video_path))
        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_frames.append(frame)
        cap.release()

        total_frames = len(raw_frames)
        print(f"Total frames in DensePose video: {total_frames}")

        # Start at frame 34 (hands initiating downswing) to frame 132 (high follow-through with hands raised)
        start_f = min(34, max(0, total_frames - target_frames))
        end_f = min(total_frames - 1, 132)
        indices = np.linspace(start_f, end_f, target_frames, dtype=int)
        sampled = [raw_frames[i] for i in indices]

        stack = np.stack(sampled)
        mask = np.max(stack, axis=(0, 3)) > 20
        coords = np.argwhere(mask)

        h, w = sampled[0].shape[:2]
        target_w, target_h = self.target_size
        aspect = target_w / target_h  # 0.75 (3:4 portrait)

        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            box_h = int((y_max - y_min) * 1.24)
            box_w = int(box_h * aspect)

            # Give extra room on the left so the swinging hands and bat have full room to extend
            min_req_w = int((x_max - x_min) * 1.35)
            if box_w < min_req_w:
                box_w = min_req_w
                box_h = int(box_w / aspect)

            cx = int((x_min + x_max) * 0.47)
            cy = (y_min + y_max) // 2

            pad = max(box_h, box_w)
            y1 = cy - box_h // 2 + pad
            y2 = y1 + box_h
            x1 = cx - box_w // 2 + pad
            x2 = x1 + box_w
        else:
            pad = 0
            x1, y1, x2, y2 = 0, 0, w, h

        control_frames = []
        part_masks = []

        # Also build a wide global swing mask across all frames so there is NO tight moving halo around his back
        all_fg_accum = np.zeros((target_h, target_w), dtype=np.uint8)

        for frame in sampled:
            if pad > 0:
                padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                cropped = padded[y1:y2, x1:x2].copy()
            else:
                cropped = frame[y1:y2, x1:x2].copy()

            pil_img = Image.fromarray(cropped).resize(self.target_size, Image.Resampling.NEAREST)
            dp = np.array(pil_img, dtype=np.int32)
            r, g, b = dp[:, :, 0], dp[:, :, 1], dp[:, :, 2]

            # Identify arm & hand segments in DensePose:
            # Forearms/upper arms are yellow-green/orange (R>120, G>120, B<100 below the head)
            # Hands/gloves are bright blue (B>130, R<60 in the upper 65% of the body)
            fg_binary = ((r + g + b) > 35).astype(np.uint8) * 255
            all_fg_accum = np.maximum(all_fg_accum, fg_binary)

            ys, xs = np.where(fg_binary > 0)
            y_top = ys.min() if len(ys) > 0 else 0
            y_bot = ys.max() if len(ys) > 0 else target_h
            body_h = max(1, y_bot - y_top)

            # Head is in top 22% of body height; arms/hands are between 18% and 68% of body height
            arm_zone = np.zeros((target_h, target_w), dtype=bool)
            arm_zone[y_top + int(body_h * 0.16) : y_top + int(body_h * 0.68), :] = True

            yellow_arms = (r > 110) & (g > 115) & (b < 105) & arm_zone
            blue_hands = (b > 125) & (r < 65) & (g < 145) & arm_zone
            arm_hand_raw = (yellow_arms | blue_hands).astype(np.uint8)

            # Enhance the arm & hand segments slightly in the DensePose conditioning image so ControlNet
            # strongly prioritizes the moving arms/hands when they cross in front of the torso!
            dp_enhanced = np.array(pil_img).copy()
            arm_contours, _ = cv2.findContours(arm_hand_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dp_enhanced, arm_contours, -1, (255, 220, 60), thickness=2)
            control_frames.append(Image.fromarray(dp_enhanced))

            # Royal-blue torso mask EXCLUDING the moving arms and hands so arms aren't washed out!
            torso_raw = ((r < 75) & (g < 125) & (b > 105) & (~(yellow_arms | blue_hands))).astype(np.float32)
            torso_soft = cv2.GaussianBlur(torso_raw, (13, 13), 0)

            # Top of head cap mask
            head_raw = (r > 180) & (g > 180) & (b < 80) & (~arm_zone)
            cap_mask = np.zeros_like(torso_raw)
            hy, hx = np.where(head_raw)
            if len(hy) > 10:
                cap_cut = int(np.percentile(hy, 42))
                cap_pts_mask = head_raw.copy()
                cap_pts_mask[cap_cut:, :] = False
                cap_mask = cv2.GaussianBlur(cap_pts_mask.astype(np.float32), (11, 11), 0)

            part_masks.append({
                "fg_frame": fg_binary,
                "torso": torso_soft,
                "cap": cap_mask
            })

        # Create a wide, seamless swing-envelope mask that includes the entire left-side bat swing zone
        # so the bat and extended hands are NEVER clipped and there is NO tight halo around the player!
        kernel_wide = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (85, 85))
        wide_zone = cv2.dilate(all_fg_accum, kernel_wide, iterations=2)
        # Also open up the entire left batting arc (where the bat swings out)
        ys_all, xs_all = np.where(all_fg_accum > 0)
        if len(ys_all) > 0:
            y_mid_top = int(np.percentile(ys_all, 15))
            y_mid_bot = int(np.percentile(ys_all, 92))
            x_center = int(np.median(xs_all))
            wide_zone[y_mid_top:y_mid_bot, :x_center] = 255

        global_soft_mask = cv2.GaussianBlur(wide_zone.astype(np.float32) / 255.0, (95, 95), 0)
        for pm in part_masks:
            pm["global_fg"] = global_soft_mask

        return control_frames, part_masks

    def generate(self, image, control_video, prompt, num_frames, fps, output_path):
        if self.pipe is None:
            self.load_model()

        # 1. Prepare CLIP reference (cropped above the static wrists/bat so IP-Adapter doesn't freeze his hands down!)
        clip_ref_img, ref_portrait, clean_bg_plate = self.prepare_bradman_and_clean_background(image)

        # 2. Load 16 native frames with enhanced arm/hand conditioning
        target_frame_count = 16
        control_frames, part_masks = self.load_aligned_densepose_sequence(
            control_video, target_frames=target_frame_count
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        start_time = time.time()

        historical_prompt = (
            "1930s black and white archival film of cricketer Don Bradman dynamically swinging his arms and hands "
            "upward through a cricket cover drive shot, both hands in batting gloves gripping the handle of a "
            "wooden cricket bat and swinging it forward and up, wearing a small dark cricket cap, "
            "bright white collared shirt with rolled sleeves, white trousers, and white buckled batting leg pads, "
            "blurred grass cricket field and trees in background, sharp monochrome photography"
        )

        negative_prompt = (
            "static hands, stationary arms, one hand on hip, missing bat, broken bat, detached hands, "
            "white hat, sun hat, pith helmet, black shirt, dark shirt, bokeh dots, glowing circles, "
            "halo around body, cutout border, blurry, color"
        )

        # Boost ControlNet scale to 1.00 across all steps so the hands & forearms follow the full DensePose swing arc!
        pipe_kwargs = dict(
            prompt=historical_prompt,
            negative_prompt=negative_prompt,
            num_frames=len(control_frames),
            width=self.target_size[0],
            height=self.target_size[1],
            conditioning_frames=control_frames,
            ip_adapter_image=clip_ref_img,
            controlnet_conditioning_scale=1.00,
            num_inference_steps=28,
            guidance_scale=7.5,
            generator=torch.Generator(device="cpu").manual_seed(42)
        )

        print(f"Rendering {len(control_frames)} frames with full ControlNet hand-swing authority (1.00)...")
        output = self.pipe(**pipe_kwargs)
        raw_video_frames = output.frames[0]

        final_frames = []
        for idx, frame_pil in enumerate(raw_video_frames):
            gen_gray = np.array(frame_pil.convert("L"), dtype=np.float32)
            masks = part_masks[idx]

            # A. Lift shirt luminance ONLY on the torso (excluding the moving arms and hands so arm motion is crisp!)
            torso_w = masks["torso"]
            shirt_lifted = np.clip(gen_gray * 1.38 + 36.0, 0, 232)
            gen_gray = gen_gray * (1.0 - torso_w * 0.70) + shirt_lifted * (torso_w * 0.70)

            # B. Keep the cap dark
            cap_w = masks["cap"]
            cap_darkened = np.clip(gen_gray * 0.48, 25, 95)
            gen_gray = gen_gray * (1.0 - cap_w * 0.70) + cap_darkened * (cap_w * 0.70)

            # C. Seamless wide-envelope background blend (eliminates the tight body halo and never clips the swinging bat!)
            fg_w = np.clip(masks["global_fg"], 0.0, 1.0)
            composite = gen_gray * fg_w + clean_bg_plate * (1.0 - fg_w)

            out_u8 = np.clip(composite, 0, 255).astype(np.uint8)
            final_frames.append(Image.fromarray(out_u8).convert("RGB"))

        export_fps = 7
        export_to_video(final_frames, str(output_path), fps=export_fps)
        inference_time = time.time() - start_time

        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        print(f"Saved {len(final_frames)}-frame ({len(final_frames)/export_fps:.1f}s) video to {output_path} | Peak VRAM: {peak_vram:.2f} GB")
        return peak_vram, inference_time