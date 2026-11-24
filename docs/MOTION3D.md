# motion3d: rigid 3D skeletons, planted feet, fixed bat, several camera angles

A new motion stage, built separately from `animate/`; earlier outputs and jobs are kept. It replaces the
2D skeleton cleaning (`animate/motion_clean.py`, `animate/delivery.py`) as the motion that drives MusePose.

Every output showing a real player (Bradman, McGrath) is an **AI reconstruction** and is labelled as one.

## Why

Feedback on the previous version (`outputs/animate_v2/`), and the cause of each problem:

| Feedback | Cause in the 2D pipeline |
|---|---|
| Bat size changes randomly | The tracked bat line was 14-178 px long (frame-to-frame jumps up to 60 px); its length was only clamped. |
| Arm and leg motion not smooth; joints speed up at random | Each joint was cleaned separately in 2D: detector glitches (39x the median speed) and left/right swaps leaked through, and nothing tied a joint's speed to its neighbours. |
| Limbs not rigid | Bones were clamped to 0.75-1.10x their median 2D length every frame, so they could still stretch and shrink by 35%. |
| Joints disappear | Hands and joints with low confidence were dropped (55 batting frames had a missing joint). |
| Leg not on a fixed ground | No ground model, so a planted foot could slide; the Cummins clip's camera pans and crops, so feet drifted. |
| Bowling incomplete | Release and follow-through are blurred, and the old window (frames 30-46) ended *before* the arm went over the top (frames 46-51). Frames 52-63 (follow-through) were cut. |

## Method

```
motion clip ──► DWPose 2D (Student 1, raw detections)  ┐
            └─► MediaPipe Pose heavy (2D + depth, heels/toes) ┤
camera tracking on the background (ORB, people masked) ──► one fixed "world" view (the ground does not move)
                                                        ▼
               3D fit (PyTorch L-BFGS over the whole clip at once)
                                                        ▼
        data/motion3d/<clip>/fit.npz  ──► render: 6 camera angles (OpenPose stick figures + JSON)
                                       └► export: 2D camera view + hands ──► scene + MusePose jobs
```

### The 3D fit (`motion3d/fit.py`)

**Unknowns:**
- 3D position of 21 joints in every frame: 17 body joints plus heels and toes;
- one length per bone, the same for left and right;
- for batting, the bat toe in every frame plus one bat length.

**Energy terms:**

| term | what it does |
|---|---|
| detections | Robust (Huber) distance from each projected joint to the DWPose and MediaPipe points, weighted by confidence. Outliers (glitches, swaps, blur) cost little, so they are ignored rather than followed. A missing detection has no term, but the joint still exists. |
| depth | Weak pull towards MediaPipe's per-joint depth, rescaled to the fitted bone lengths after pass 1. |
| rigid bones | \|a − b\| = bone length in every frame. Limbs, feet and head are strict; the torso is slightly soft (the spine bends). |
| proportions | Bone-length ratios stay in anthropometric ranges (shin ≈ thigh, forearm 0.70–0.92 × upper arm, foot vs shin …). Without them, a limb that is rarely seen could change size freely. |
| move as a unit | Small acceleration of the hip centre and of every **bone vector** (child − parent). A joint can only move with the joints it is attached to. |
| speed cap (pass 2) | Each bone vector's speed is capped at 1.15 × its 95th percentile from pass 1, so no sudden speed-ups. |
| ground (pass 2) | Planted feet are held still for each planted run. Planted means slow and lowest; a run splits after 2 % of body height of sliding. Each run is a ground reference point, and the ground height is interpolated between them. No foot that was seen may go below the ground. |
| unseen limbs | A limb joint no detector saw in that frame stays near the image plane. An unseen shin hangs below its knee and an unseen foot sits below its ankle (anatomy, not a guessed point). |
| frame edge | Detections within 10 px of the real camera frame edge are dropped (40 px for feet). The Cummins clip cuts the bowler's feet off from frame ~36, and both detectors pinned the ankles to the edge, which folded the legs. |
| bowling arm | Hand-annotated keyframes (shoulder, wrist) for the blurred delivery, source frames 34–63, see below. Straight arm of the fitted length, detections ignored there. |
| bat | Rigid: \|toe − grip\| = one length, with the grip between the wrists. Both hands stay on the handle (wrists within 0.35 forearm). The toe follows the tracked bat robustly, plus the tracked direction; smooth. |

**Passes:**
- Pass 1 has no ground terms. It finds the planted runs, the ground references and the speed caps.
- Pass 2 adds those terms.

**Bowling arm keyframes.** Source frames 34–63 are motion-blurred: DWPose and MediaPipe lose or swap the
arm.
- The bowling shoulder and wrist were annotated by hand on 2× brightened frames with a 32 px grid (accuracy
  ±10 px): `data/motion3d/bowler_new_cummins/keyframes_bowling_arm.json`.
- What they show:
  - frames 34–39: arm pointing back;
  - frames 40–45: rising;
  - frames 46–49: vertical;
  - frames 50–51: over the top;
  - frames 52–63: forward and down (follow-through).
- Source frames 34–58 are slowed down 2.5×.
- The release is where the fitted arm is 15° past vertical: output frame 71 (source frame 48.8).

### Ground reference (`motion3d/camera.py`)

The background (pitch, crease, boards) is tracked with ORB features, with players, bats, black bars and
burned-in captions masked out.
- Every frame is mapped into one fixed world view (the first frame), so a point on the ground keeps its
  coordinates.
- For the Cummins clip the raw camera is still (zoom 1.00); Student 1's crop window moves and zooms 1.2–2.3×.
- That world view is what the planted-feet terms use.

### Hands (`motion3d/export.py`)

DWPose hands are kept only as a shape, in the forearm's frame.
- Good detections are used; gaps are interpolated; otherwise the median shape is used. The shape is smoothed
  in time.
- The fitted forearm carries the hand, so a hand never vanishes or detaches.
- The blurred bowling hand uses the median shape.

### Angles (`motion3d/render.py`)

The 3D skeleton is turned about the vertical to 0° (the source camera), 45°, 90°, 135°, 180° and 270°.
- Each angle is drawn as an OpenPose stick figure (body, feet, bat) with the ground grid and planted feet
  marked.
- Each is also written as OpenPose BODY_18 JSON (+ feet + bat).

## More reference videos and angles (`motion3d/extra.py`)

Four more deliveries from the Commons videos in `data/motion_videos/`, as more motions and camera angles.

| clip | player | view | frames |
|---|---|---|---|
| `extra_parry_bowling` | Stephen Parry, 2012 (Nev1, CC BY-SA 3.0) | side-on | 38 |
| `extra_stokes_bowling` | Ben Stokes, 2012 (Nev1, CC BY-SA 3.0) | side-on | 36 |
| `extra_onions_bowling` | Graham Onions, 2012 (Nev1, CC BY-SA 3.0) | side-on, panning | 38 |
| `extra_amar_endon_bowling` | bowler in "Amar Kala Batting Video 2025" (DrRevathi25, CC BY-SA 4.0) | **end-on, behind the bowler** | 38 |

How each is processed:
1. Trim the delivery to 24 fps.
2. Follow the chosen player with YOLO person boxes from a seed picked on a contact sheet (other players are in
   frame).
3. Crop him square.
4. Run DWPose at 1024 px, then MediaPipe, camera tracking, the fit and the renders.

They are reference skeletons only; McGrath is driven by the Cummins motion.

The batting extras (club cover drives, pull shot) show players about 60 px tall among many others, so they were
not used.

## Results

### Fit quality (`data/motion3d/<clip>/fit.json`)

| | Cover drive (Bradman) | Cummins (McGrath) |
|---|---|---|
| frames (output) | 139 | 100 (0–63, delivery slowed 2.5×) |
| largest bone-length error, 3D | 0.40 px | 0.53 px |
| planted-foot drift within a planted run | 2.5 px | 0.17 px |
| foot below ground (seen feet) | 0.07 px | 0.02 px |
| joint acceleration p99, raw DWPose → fit | 7.9 → 1.6 px/frame² | 318 → 10 px/frame² |
| max joint speed / median, raw → fit | 17.5 → 13.5 (the shot itself is fast) | 26.8 → 5.5 |
| left/right swaps fixed | 0 | 23 |
| detections dropped at the frame edge | 0 | 422 |
| bat 3D length | 139.3 px, constant | n/a |
| bat on-screen length (raw tracking: 14–178 px) | 136.9–139.3 px | n/a |

### Skeleton that drives MusePose: before vs now (`outputs/motion3d/skeleton_comparison.json`)

| | batting before | batting now | bowling before | bowling now |
|---|---|---|---|---|
| frames with a missing body joint | 55 | **0** | 0 | **0** |
| bat length change per frame, p99 | 52.8 % | **0.1 %** | n/a | n/a |
| sudden limb stretch, p99 | 13.3 % | **8.7 %** | 41 % | **15.8 %** |
| joint acceleration p99 (canvas px/frame²) | 1.7 | 2.8 | 41.5 | **5.5** |
| max speed / median | 8.0 | 13.6 | 11.2 | **5.3** |
| on-screen limb-length variation | 0.11 | 0.19 | 0.15 | 0.17 |

**How to read the batting row:**
- Bradman's skeleton is drawn about 1.7× larger on the canvas now. Per source pixel, the acceleration
  matches the fit (1.6).
- The cover drive is still for most of the clip and fast in the shot, so max/median speed is high by nature.
- On-screen limb length now varies more because arms really foreshorten when the bat is lifted towards the
  end-on camera. The 3D bones stay within 0.4 px.
- The old clamping hid that by stretching the bones in 2D.

### Generated videos

See `docs/RUN_LOG.md` (Kaggle run) and `outputs/animate_v3/summary.json`.

## Files

| file | what |
|---|---|
| `motion3d/clips.py` | the clips and their settings |
| `motion3d/mediapipe_pass.py` | MediaPipe Pose heavy, video mode (runs in `~/pose3d`) |
| `motion3d/camera.py` | ground reference: background tracking into one world view |
| `motion3d/fit.py` | the 3D fit (energy terms above), diagnostics in `fit.json` |
| `motion3d/render.py` | six angles, overlay on the source, multiview video, OpenPose JSON |
| `motion3d/export.py` | camera view + hands → a processed-clip folder the scene code reads |
| `motion3d/build_jobs.py` | MusePose jobs `bradman_batting_3d`, `mcgrath_bowling_3d` (scenes as before) |
| `motion3d/extra.py` | extra reference deliveries from the Commons videos |
| `motion3d/compare.py` | before/now skeleton metrics and side-by-side videos |
| `motion3d/run_all.sh` | rebuild all skeletons |
| `data/motion3d/<clip>/` | mediapipe.json, camera.json, fit.npz/json, views/, overlay.mp4, multiview.mp4, clip/ |
| `outputs/motion3d/` | copies of the skeleton videos, angle views and comparisons |
| `outputs/animate_v3/` | the generated videos from these skeletons |

## Reproduce

```bash
# one-time: MediaPipe env (mediapipe needs numpy 2; ~/gen pins 1.26) + libGLESv2 without root
uv venv -p 3.11 ~/pose3d && uv pip install -p ~/pose3d/bin/python mediapipe opencv-python-headless scipy
curl -Lo ~/pose3d/models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
(cd ~/pose3d && mkdir -p debs && cd debs && apt-get download libgles2 libglvnd0 && for d in *.deb; do dpkg -x $d ~/pose3d/sysroot; done)

motion3d/run_all.sh                                  # all skeletons (~15 min CPU)
~/gen/bin/python -m motion3d.build_jobs              # MusePose jobs
KERNEL=cricket-animate EXTRA_DATA="animate/jobs/bradman_batting_3d animate/jobs/mcgrath_bowling_3d" \
  MP_W=560 MP_H=632 MP_STEPS=30 MP_OVERLAP=8 JOBS="bradman_batting_3d mcgrath_bowling_3d" \
  kaggle_runner/run_on_kaggle.sh animate
~/gen/bin/python -m animate.finish kaggle_results/animate outputs/animate_v3 bradman_batting_3d mcgrath_bowling_3d
```

## Limits

- **Monocular 3D.** Depth comes from one camera plus MediaPipe's estimate and the bone lengths. The side
  and back views are plausible, not measured; the 0° view is the measured one.
- **Hand-keyframed bowling arm.** It is annotated by hand (±10 px) where the footage is blurred, and the
  elbow is assumed straight.
- **Cummins feet.** The feet are off-screen for 74 of 100 frames. The legs there come from anatomy priors,
  not detections.
- **Ground.** The ground under off-screen feet is unknown, so planted-foot holding only applies where the
  feet were seen.
- **Extra clips.** Small players (150–250 px tall), so their skeletons are coarser. In Onions' first frames
  the box follows him past another player.
