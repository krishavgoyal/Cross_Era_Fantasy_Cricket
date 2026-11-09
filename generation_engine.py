import time
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

        print("4/4 Loading IP-Adapter...")
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
        # 0.62 preserves Bradman's real facial features, dark cap, plain white shirt, and buckled leg pads
        self.pipe.set_ip_adapter_scale(0.62)

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

        # Reconstruct clean 1930s oval background plate from outer background columns
        tw, th = self.target_size
        left_strip = ref_gray[:, :int(tw * 0.24)]
        right_strip = ref_gray[:, int(tw * 0.88):]
        bg_cols = np.concatenate([left_strip, right_strip], axis=1)

        bg_plate = cv2.resize(bg_cols, (tw, th), interpolation=cv2.INTER_LINEAR)
        bg_plate = cv2.GaussianBlur(bg_plate, (31, 15), 0)

        # Pass an UNMODIFIED, clean top-78% crop of bradman.jpg to IP-Adapter
        # (Eliminates the horizontal thigh line and "AIG" chest logo caused by rectangular patching)
        cw, ch = clean_pil.size
        clip_ref = clean_pil.crop((0, 0, cw, int(ch * 0.78))).resize((512, 512), Image.Resampling.LANCZOS)

        return clip_ref, clean_pil, bg_plate

    def build_connected_bat_blade_layer(self, h, w, gx, gy, tip_x, tip_y, y_head_bottom):
        """
        Builds a tight 13px tapered willow bat mask and shading extending strictly from the
        tracked hand coordinate (gx, gy) to (tip_x, tip_y) outside the body silhouette.
        """
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        vx = float(tip_x - gx)
        vy = float(tip_y - gy)
        length = max(1.0, np.hypot(vx, vy))
        ux, uy = vx / length, vy / length
        nx, ny = -uy, ux

        rx = xx - float(gx)
        ry = yy - float(gy)
        proj = rx * ux + ry * uy
        perp = rx * nx + ry * ny

        # Handle (0..26% length, half-width 3.2px) tapering into the willow blade (half-width 6.8px)
        shoulder_t = np.clip((proj - length * 0.24) / (length * 0.07 + 1e-4), 0.0, 1.0)
        half_w = 3.2 + 3.6 * shoulder_t
        toe_round = np.clip((length - proj) / 5.0, 0.0, 1.0)
        half_w = half_w * np.minimum(1.0, toe_round + 0.35)

        # Starts at the wrist/glove edge (proj >= 0) so it never overwrites the player's hands or body
        inside = (proj >= 0.0) & (proj <= length) & (np.abs(perp) <= half_w) & (yy >= y_head_bottom)
        bat_alpha = cv2.GaussianBlur(inside.astype(np.float32), (5, 5), 0)

        # 3D bevelled English willow shading matching bradman.jpg
        rel_perp = np.clip(np.abs(perp) / (half_w + 1e-4), 0.0, 1.0)
        blade_lum = 190.0 - 88.0 * (rel_perp ** 1.35)
        grip_lum = 110.0 - 25.0 * (rel_perp ** 1.2)
        bat_lum = np.where(proj < length * 0.25, grip_lum, blade_lum).astype(np.float32)

        return bat_alpha, bat_lum

    def load_aligned_densepose_sequence(self, video_path, target_frames=16):
        cap = cv2.VideoCapture(str(video_path))
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if np.count_nonzero(np.max(rgb, axis=2) > 15) > 1500:
                all_frames.append(rgb)
        cap.release()

        total_frames = len(all_frames)
        print(f"Total active frames in DensePose video: {total_frames}")

        # Sample the clean cover-drive stroke (frames 24 to 94):
        # Captures stance -> forward stride -> downswing -> chest-high cover drive extension to the left
        # Stops BEFORE the trailing frames (100+) where the player's elbows cross over his face!
        start_f = min(24, max(0, total_frames - target_frames))
        end_f = min(total_frames - 4, 94)
        indices = np.linspace(start_f, end_f, target_frames, dtype=int)
        sampled = [all_frames[i] for i in indices]

        stack = np.stack(sampled)
        mask = np.max(stack, axis=(0, 3)) > 15
        coords = np.argwhere(mask)

        h, w = sampled[0].shape[:2]
        target_w, target_h = self.target_size
        aspect = target_w / target_h  # 0.75 (3:4 portrait)

        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            box_h = int((y_max - y_min) * 1.24)
            box_w = int(box_h * aspect)

            min_req_w = int((x_max - x_min) * 1.36)
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

        cropped_pils = []
        raw_hands = []

        for frame in sampled:
            if pad > 0:
                padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                cropped = padded[y1:y2, x1:x2].copy()
            else:
                cropped = frame[y1:y2, x1:x2].copy()

            pil_img = Image.fromarray(cropped).resize(self.target_size, Image.Resampling.NEAREST)
            cropped_pils.append(pil_img)

            dp = np.array(pil_img, dtype=np.int32)
            r, g, b = dp[:, :, 0], dp[:, :, 1], dp[:, :, 2]
            fg = np.max(dp, axis=2) > 12
            ys, xs = np.where(fg)
            y_top = ys.min() if len(ys) > 0 else 0
            y_bot = ys.max() if len(ys) > 0 else target_h
            body_h = max(1, y_bot - y_top)

            # Locate actual hands/wrists BELOW the head (y_top + 18% .. y_top + 65%)
            hand_zone = np.zeros((target_h, target_w), dtype=bool)
            hand_zone[y_top + int(body_h * 0.18) : y_top + int(body_h * 0.65), :] = True

            blue_hands = (b > 115) & (r < 80) & (g < 155) & hand_zone
            yellow_arms = (r > 110) & (g > 115) & (b < 105) & hand_zone

            hy_b, hx_b = np.where(blue_hands)
            hy_a, hx_a = np.where(yellow_arms)

            # Compute coupled (gx, gy) from the SAME distal hand pixels (never independent X/Y percentiles!)
            if len(hx_b) >= 6:
                left_cut = np.percentile(hx_b, 40)
                keep = hx_b <= left_cut
                gx = float(np.mean(hx_b[keep]))
                gy = float(np.mean(hy_b[keep]))
            elif len(hx_a) >= 8:
                left_cut = np.percentile(hx_a, 20)
                keep = hx_a <= left_cut
                gx = float(np.mean(hx_a[keep]))
                gy = float(np.mean(hy_a[keep]))
            else:
                gx = float(target_w * 0.43)
                gy = float(y_top + body_h * 0.45)

            raw_hands.append((gx, gy, float(y_top), float(y_bot), float(body_h)))

        # Smooth hand trajectory across the 16 frames
        smooth_hands = []
        for i in range(target_frames):
            win = [raw_hands[max(0, min(target_frames - 1, j))] for j in range(i - 1, i + 2)]
            sgx = int(round(np.mean([p[0] for p in win])))
            sgy = int(round(np.mean([p[1] for p in win])))
            sy_top = int(round(np.mean([p[2] for p in win])))
            sy_bot = int(round(np.mean([p[3] for p in win])))
            sbh = int(round(np.mean([p[4] for p in win])))
            smooth_hands.append((sgx, sgy, sy_top, sy_bot, sbh))

        control_frames = []
        part_masks = []

        for idx, pil_img in enumerate(cropped_pils):
            dp = np.array(pil_img, dtype=np.int32)
            r, g, b = dp[:, :, 0], dp[:, :, 1], dp[:, :, 2]
            gx, gy, y_top, y_bot, body_h = smooth_hands[idx]

            dp_cond = np.array(pil_img).copy()
            fg_raw = (np.max(dp, axis=2) > 10).astype(np.uint8) * 255

            # 1. Bridge any shin gaps between knee and ankle so batting pads never vanish
            knee_y1, knee_y2 = y_top + int(body_h * 0.52), y_top + int(body_h * 0.68)
            foot_y1, foot_y2 = y_top + int(body_h * 0.82), min(target_h, y_bot + 4)
            _, knee_xs = np.where(fg_raw[knee_y1:knee_y2, :] > 0)
            _, foot_xs = np.where(fg_raw[foot_y1:foot_y2, :] > 0)

            pad_hull_mask = np.zeros((target_h, target_w), dtype=np.uint8)
            if len(knee_xs) > 10 and len(foot_xs) > 10:
                k_left, k_right = int(np.percentile(knee_xs, 15)), int(np.percentile(knee_xs, 85))
                f_left, f_right = int(np.percentile(foot_xs, 15)), int(np.percentile(foot_xs, 85))
                ky_mid = (knee_y1 + knee_y2) // 2
                fy_mid = (foot_y1 + foot_y2) // 2

                cv2.line(pad_hull_mask, (k_left, ky_mid), (f_left, fy_mid), 255, thickness=24)
                cv2.line(pad_hull_mask, (k_right, ky_mid), (f_right, fy_mid), 255, thickness=24)

                missing_shin = (pad_hull_mask > 0) & (fg_raw == 0)
                dp_cond[missing_shin] = (45, 160, 145)

            # 2. Cover-drive bat angle aligned with the forearms:
            # Starts at 96 deg (down-left stance), swings through 128 deg (downswing) to 162 deg (out-left extension)
            progress = idx / max(1, target_frames - 1)
            t_keys = [0.00, 0.20, 0.45, 0.72, 1.00]
            ang_keys = [96.0, 108.0, 134.0, 154.0, 162.0]
            angle_deg = float(np.interp(progress, t_keys, ang_keys))
            rad = np.deg2rad(angle_deg)
            bat_len = int(body_h * 0.42)
            tip_x = int(round(gx + bat_len * np.cos(rad)))
            tip_y = int(round(gy + bat_len * np.sin(rad)))

            # Draw bat conditioning into dp_cond ONLY on background pixels (fg_raw == 0)
            # so it NEVER overwrites the player's body, arms, or face in ControlNet!
            bat_cond_layer = dp_cond.copy()
            cv2.line(bat_cond_layer, (gx, gy), (tip_x, tip_y), (175, 190, 65), thickness=11, lineType=cv2.LINE_AA)
            bg_only = (fg_raw == 0)
            dp_cond[bg_only] = bat_cond_layer[bg_only]
            cv2.circle(dp_cond, (gx, gy), 7, (35, 90, 230), -1)
            control_frames.append(Image.fromarray(dp_cond))

            # 3. Tight, halo-free body + batting pads silhouette mask
            combined_fg = np.maximum(fg_raw, pad_hull_mask)
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            fg_closed = cv2.morphologyEx(combined_fg, cv2.MORPH_CLOSE, k_close)
            contours, _ = cv2.findContours(fg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            body_solid = np.zeros_like(fg_raw)
            for cnt in contours:
                if cv2.contourArea(cnt) > 150:
                    cv2.drawContours(body_solid, [cnt], -1, 255, thickness=cv2.FILLED)

            lower_leg_zone = np.zeros((target_h, target_w), dtype=np.uint8)
            lower_leg_zone[y_top + int(body_h * 0.50) :, :] = 255
            k_pad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            legs_dilated = cv2.dilate(body_solid & lower_leg_zone, k_pad, iterations=1)
            full_body_mask = np.maximum(body_solid, legs_dilated)
            body_soft = cv2.GaussianBlur(full_body_mask.astype(np.float32) / 255.0, (7, 7), 0)

            # 4. Connected bat blade layer extending from (gx, gy) outside the body
            y_head_bottom = y_top + int(body_h * 0.20)
            bat_alpha, bat_lum = self.build_connected_bat_blade_layer(
                target_h, target_w, gx, gy, tip_x, tip_y, y_head_bottom
            )

            # 5. Gentle anatomical masks for torso and lower legs (leaving face & arms 100% natural)
            arm_zone = np.zeros((target_h, target_w), dtype=bool)
            arm_zone[y_top + int(body_h * 0.16) : y_top + int(body_h * 0.68), :] = True
            yellow_arms = (r > 110) & (g > 115) & (b < 105) & arm_zone
            blue_hands_m = (b > 115) & (r < 80) & arm_zone

            torso_raw = ((r < 75) & (g < 125) & (b > 105) & (~(yellow_arms | blue_hands_m))).astype(np.float32)
            torso_soft = cv2.GaussianBlur(torso_raw, (15, 15), 0)

            pads_raw = ((full_body_mask > 0) & (lower_leg_zone > 0)).astype(np.float32)
            pads_soft = cv2.GaussianBlur(pads_raw, (13, 13), 0)

            part_masks.append({
                "body_soft": body_soft,
                "bat_alpha": bat_alpha,
                "bat_lum": bat_lum,
                "torso": torso_soft,
                "pads": pads_soft
            })

        return control_frames, part_masks

    def generate(self, image, control_video, prompt, num_frames, fps, output_path):
        if self.pipe is None:
            self.load_model()

        clip_ref_img, ref_portrait, clean_bg_plate = self.prepare_bradman_and_clean_background(image)

        target_frame_count = 16
        control_frames, part_masks = self.load_aligned_densepose_sequence(
            control_video, target_frames=target_frame_count
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        start_time = time.time()

        historical_prompt = (
            "1930s black and white archival film of cricketer Don Bradman playing a cover drive shot, "
            "both hands in white batting gloves gripping a wooden cricket bat, wearing a dark baggy green cricket cap, "
            "plain solid white button-up flannel shirt with rolled sleeves, white trousers, "
            "white buckled batting leg pads, sharp monochrome photography"
        )

        negative_prompt = (
            "text, logo, sponsor, letters, watermark, modern jersey, printed shirt, "
            "horizontal line on legs, missing bat, broken bat, missing legs, missing pads, "
            "white hat, sun hat, black shirt, blurry, color"
        )

        pipe_kwargs = dict(
            prompt=historical_prompt,
            negative_prompt=negative_prompt,
            num_frames=len(control_frames),
            width=self.target_size[0],
            height=self.target_size[1],
            conditioning_frames=control_frames,
            ip_adapter_image=clip_ref_img,
            controlnet_conditioning_scale=0.95,
            num_inference_steps=28,
            guidance_scale=7.5,
            generator=torch.Generator(device="cpu").manual_seed(42)
        )

        print(f"Rendering {len(control_frames)} native frames with coupled hand-bat tracking & untouched facial detail...")
        output = self.pipe(**pipe_kwargs)
        raw_video_frames = output.frames[0]

        gray_list = [np.array(f.convert("L"), dtype=np.float32) for f in raw_video_frames]
        valid_means = [g.mean() for g in gray_list if not np.isnan(g).any() and g.mean() > 45.0]
        median_mean = float(np.median(valid_means)) if valid_means else 120.0

        # Guard against any dark/corrupt frame drop
        for i in range(len(gray_list)):
            g = gray_list[i]
            if np.isnan(g).any() or g.mean() < median_mean * 0.80 or g.std() < 18.0:
                gray_list[i] = gray_list[i - 1].copy() if i > 0 else gray_list[min(1, len(gray_list) - 1)].copy()

        rendered_keyframes = []
        for idx, gen_gray in enumerate(gray_list):
            masks = part_masks[idx]

            # A. Keep torso shirt bright white without blurring or painting over the face/arms
            torso_w = masks["torso"]
            shirt_lifted = np.clip(gen_gray * 1.30 + 32.0, 155, 230)
            gen_gray = gen_gray * (1.0 - torso_w * 0.62) + shirt_lifted * (torso_w * 0.62)

            # B. Keep lower-leg batting pads solid and bright
            pads_w = masks["pads"]
            pads_lifted = np.clip(gen_gray * 1.20 + 28.0, 155, 232)
            gen_gray = gen_gray * (1.0 - pads_w * 0.50) + pads_lifted * (pads_w * 0.50)

            # C. Crisp unsharp mask across the player (preserves natural facial features, cap, and pad straps)
            blur_fine = cv2.GaussianBlur(gen_gray, (0, 0), 1.0)
            gen_sharp = np.clip(cv2.addWeighted(gen_gray, 1.24, blur_fine, -0.24, 0), 0, 255)

            # D. Composite player body on top of the connected bat blade (so the bat never overwrites the hands/body)
            body_w = masks["body_soft"]
            bat_w = masks["bat_alpha"]
            bat_pixels = 0.42 * gen_sharp + 0.58 * masks["bat_lum"]

            canvas = clean_bg_plate * (1.0 - bat_w) + bat_pixels * bat_w
            composite = gen_sharp * body_w + canvas * (1.0 - body_w)
            rendered_keyframes.append(composite)

        # Build a 26-frame (~3.7s at 7 FPS) sequence with ZERO cross-fade blending (100% sharp on every frame):
        # 4 frames opening stance lead-in + 16 active cover-drive frames + 6 frames completed shot hold
        frame_indices = [0] * 4 + list(range(target_frame_count)) + [target_frame_count - 1] * 6

        rng = np.random.default_rng(42)
        final_frames = []
        for f_idx in frame_indices:
            frame_mat = rendered_keyframes[f_idx]
            grain = rng.normal(0.0, 1.3, size=frame_mat.shape).astype(np.float32)
            out_u8 = np.clip(frame_mat + grain, 0, 255).astype(np.uint8)
            final_frames.append(Image.fromarray(out_u8).convert("RGB"))

        export_fps = 7
        export_to_video(final_frames, str(output_path), fps=export_fps)
        inference_time = time.time() - start_time

        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        print(f"Saved sharp {len(final_frames)}-frame ({len(final_frames)/export_fps:.2f}s) video to {output_path} | Peak VRAM: {peak_vram:.2f} GB | Time: {inference_time:.1f}s")
        return peak_vram, inference_time