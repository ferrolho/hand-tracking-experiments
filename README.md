# hand-tracking-experiments

Experiments in monocular hand tracking, aimed at **interactively teleoperating a 3D hand in [viser](https://github.com/nerfstudio-project/viser)** and, eventually, retargeting to a robot hand. Robotics-focused (joint positions over photorealistic mesh), running on an Apple Silicon MacBook Air.

## Method split

Two trackers for two jobs, behind one swappable `frame → tracker → 21 keypoints → viser` pipeline:

| | Live teleop | Offline policy data |
|---|---|---|
| Tracker | **MediaPipe Hands** | **WiLoR / HaMeR** (planned) |
| Speed (M2 Air) | ~40 fps ✅ | ~1–3 fps |
| Output | 21 keypoints | full MANO (rotations + mesh) |
| Why | human closes the loop; latency wins | no human in loop; accuracy + occlusion robustness win |

The live MediaPipe path is implemented in `track_hands.py`. WiLoR/HaMeR can drop into the same viewer later for the offline path.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/download_models.sh        # fetches hand_landmarker.task (gitignored)
```

## Usage

```bash
# Visualize a recorded clip in viser (loops the file)
python3 track_hands.py --source data/hand_recording.mp4

# Peak-throughput benchmark (no playback throttle)
python3 track_hands.py --source data/hand_recording.mp4 --max-speed

# Live teleop from the built-in webcam
python3 track_hands.py --source webcam --camera-index 0
```

Then open <http://localhost:8080>. The printed `infer_fps` is the teleop-relevant number (tracker throughput, independent of playback pacing).

## How it works

- **Articulation** comes from MediaPipe `hand_world_landmarks` (metric, center-relative).
- **Global placement** comes from `hand_landmarks` (image pixels) back-projected with the camera intrinsics: depth is estimated from known metric hand size vs. apparent pixel size (`Z = fx·size/pixels`), then the wrist pixel is back-projected. Without intrinsics, hands are spread on a plane by image position so they don't overlap.
- viser nodes are keyed by name; each frame re-adds `/hand_i/joints` (point cloud) and `/hand_i/bones` (line segments), which updates them in place.

## Camera calibration

Metric placement uses the bundled 1280×720 calibration of the built-in FaceTime HD camera (`calib/macbook_air_m2_1280x720.json`, the `--calib` default). **Intrinsics are resolution-specific** — capture at 1280×720 to match, or pass `--calib ""` to disable metric placement.

Known limitations (see inline comments in `track_hands.py`):
- Absolute depth scale is approximate (MediaPipe uses a generic hand model, not your measured hand size).
- Wrist pixels are not undistorted before back-projection; the camera has notable barrel distortion (k3 ≈ −0.23), so placement error grows toward frame edges. Run `cv2.undistortPoints` for the tight version.

## Roadmap

- [x] One-Euro filter to smooth per-frame jitter (`one_euro.py`, tune via `--min-cutoff` / `--beta`)
- [ ] `cv2.undistortPoints` for accurate metric placement
- [ ] Retarget keypoints → robot hand joint angles (`dex-retargeting`)
- [ ] WiLoR/HaMeR offline path for policy-learning data
