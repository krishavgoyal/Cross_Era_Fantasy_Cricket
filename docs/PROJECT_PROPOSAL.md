

Project: Cross-Era Fantasy Cricket

 Project Overview & Background
In traditional sports analytics and gaming, "Fantasy Cricket" is limited to statistical simulations, pitching historical legends against modern players purely through numbers. Generative artificial intelligence offers a way to bring these cross-generational matchups to life visually—such as simulating how Sir Don Bradman would look facing a modern fast bowler.
Historically, animating human movement or creating high-fidelity sports footage from a single reference image required massive Hollywood rendering studios or prohibitive data-center computation. However, recent developments in open-source pose transfer models, latent video diffusion techniques, and lightweight audio-video diffusion transformers have democratized video synthesis. This project aims to build an AI-powered pipeline that extracts real motion from modern cricket broadcasts and maps it onto static historical images, generating short, plausible video clips of legendary players performing iconic actions. 

By executing this project, the students will demonstrate their abiity to Implement:
•	Motion Tracking Pipeline: Construct a reliable preprocessing engine to extract frame-by-frame structural movement or joint sequences from video clips of modern professional cricketers. [https://arxiv.org/html/2510.03909v1, https://arxiv.org/abs/2506.09411]
•	Controllable Image-to-Video Synthesis: Deploy a pre-trained generative video framework that takes a single static image of an era-specific player (appearance) and drives it using the extracted action sequence (motion). [https://arxiv.org/html/2510.03909v1, https://arxiv.org/html/2304.01186v2]
•	Historical Texture & Aesthetic Harmonization: Implement post-processing video modules to adjust color grading, add film grain, and scale frame rates to match the specific media era of the historical figure (e.g., 1930s black-and-white film vs. 1970s television broadcast).
•	Hardware-Constrained Optimization: Restructure the inference and tuning processes so that the entire architecture operates within the memory constraints of mid-tier consumer hardware. [https://arxiv.org/html/2304.01186v2]

 Methodology & System Architecture
The system operates as a decoupled generative pipeline, splitting human structure from appearance to ensure high fidelity without retraining massive foundation models:
A. Structure Extraction
A preprocessing engine will ingest short video clips of modern cricket players executing specific actions (e.g., a classic cover drive, a fast-bowling release, or a leg-spin delivery). Using open-source human pose-estimation models, the system extracts sequences of annotated joint coordinates or sparse 3D meshes. This outputs a normalized structural backbone devoid of background noise or specific player textures. 
B. Guided Latent Diffusion Transfer
To transfer this motion to a historical player, the system leverages a pre-trained open-source image-to-video latent diffusion framework or diffusion transformer (such as LTX-2 or MagicAnimate). 
•	Compression: Processing occurs within a low-dimensional compressed latent space to reduce computational overhead drastically compared to standard pixel-space generation. 
•	Conditioning: The static image of the historical cricket star acts as the appearance prompt, anchoring the facial features and vintage attire. A structural control adapter guides the temporal cross-attention layers, forcing the model to animate the historical figure along the exact path of the modern tracking coordinates. 

C. Era-Styling Filter Module
A specialized post-processing pipeline ensures visual plausibility by treating the output with era-specific artifacts. Clips generated for players from the 1920s–1940s are passed through grayscale mapping, artificial lens blur, and variable gate-shutter jitter filters. Clips for players from the 1970s–1980s receive chromatic aberration, subtle color desaturation, and tape-hiss visual overlays to mimic early analog television broadcasts.

Resource Mitigation & Feasibility
Standard text-to-video foundation models require immense clusters of industrial hardware for training. To make this project highly feasible for three students utilizing moderate computational resources (e.g., a single workstation with a consumer GPU like an NVIDIA RTX 3060 or 4060 with 12GB to 16GB of VRAM), the system implements the following resource limits:
•	Pre-trained Inference with Adapters: Students will not train a video model from scratch. The pipeline relies entirely on open-source, pre-trained base weights (like LTX-2) optimized for efficient local execution on consumer cards. [
•	Parameter-Efficient Tuning: Likeness variations (such as distinct historical batting pads, caps, or bats) are handled via lightweight Low-Rank Adaptation (LoRA) or structural adapters rather than full-parameter fine-tuning, keeping peak memory well below 12GB. 
•	Resolution and Clip Bounds: Training and inference bounds will be firmly locked to a resolution of 512x512 pixels and a temporal limit of 3 to 5 seconds per action clip (roughly 60–100 frames total), eliminating the memory scaling risks associated with longer or higher-resolution rendering.

Team Roles and Responsibilities

The technical responsibilities are divided evenly across three distinct engineering components:
Student Role	Primary Responsibilities
Student 1: Video Preprocessing & Pose Specialist	Responsible for compiling the clean reference video database and extracting pure motion vectors.
Student 2: Generative Video & Model Architect	Responsible for hosting the pre-trained diffusion transformer and integrating the control adapters.
Student 3: Era-Aesthetic & Evaluation Engineer	Focuses on color grading, texture filtering, and quantitative/qualitative assessment of video plausibility.
 
Project  Milestones
•	Milestone 1: Build the motion extraction toolkit. Ensure coordinates for key actions (such as a fast bowler's delivery stride) are successfully isolated and mapped.
•	Milestone 2: Deploy the pre-trained latent video diffusion framework locally. Run initial image-to-video tests on a standard consumer GPU to confirm execution without out-of-memory (OOM) errors.
•	Milestone 3 Connect the motion vectors to the video generation pipeline via control adapters. Generate the first unstyled, animated clips of historical players moving dynamically.
•	Milestone 4: Integrate the post-processing era filters. Run full pipeline evaluations to measure temporal smoothness and motion fidelity, and compile the final cross-era video catalog. [https://www.youtube.com/watch?v=ynT65hp92SE&t=121s, https://arxiv.org/abs/2506.09411, https://arxiv.org/html/2304.01186v2, https://arxiv.org/html/2510.03909v1]

Project  Deliverable
The project will conclude with the submission of software that is needed to generate video clips for prompts iike: “Sachin Tendulkar facing Michael Holding”.
