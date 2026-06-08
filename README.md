# hand-tracking-experiments

> **▶ Live browser demo: https://ferrolho.github.io/hand-tracking-experiments/** — runs entirely in your browser on your own webcam (MediaPipe + Three.js), no install.

Experiments in monocular hand tracking, aimed at **interactively teleoperating a 3D hand in [viser](https://github.com/nerfstudio-project/viser)** and, eventually, retargeting to a robot hand. Robotics-focused (joint positions over photorealistic mesh), running on an Apple Silicon MacBook Air.

## Method split

Two trackers for two jobs, behind one swappable `frame → tracker → 21 keypoints → viser` pipeline:

| | Live teleop | Offline policy data |
|---|---|---|
| Tracker | **MediaPipe Hands** | **WiLoR** (via [wilor-mini](https://github.com/warmshao/WiLoR-mini)) |
| Speed (M2 Air) | ~40 fps ✅ | ~4 fps (measured) |
| Output | 21 keypoints | full MANO (rotations + mesh) |
| Why | human closes the loop; latency wins | no human in loop; accuracy + occlusion robustness win |

The live MediaPipe path is `track_hands.py`. The offline WiLoR path is benchmarked in `bench_wilor.py` and rendered as a mesh-overlay video by `wilor_overlay.py` — see [`docs/wilor.md`](docs/wilor.md) for results (~4 fps, offline-only) and setup notes.

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

## Browser demo (`web/`)

A static, client-side port of the live path — [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe) (WASM) + [Three.js](https://threejs.org/), no Python or server. Deployed to **GitHub Pages** via Actions on every push to `web/**`.

**▶ https://ferrolho.github.io/hand-tracking-experiments/**

Same on-screen controls as the desktop app (smoothing, selfie mirror, FOV, appearance), and it uses the bundled M2 Air calibration so the skeleton overlays the feed. Run locally with `cd web && python3 -m http.server`, then open the printed URL (camera works on `localhost`).

## Licensing

MediaPipe's model is openly redistributable; **MANO** is non-commercial **and non-transferable**, and **WiLoR** is CC-BY-NC. That's *why* the browser demo is MediaPipe-only (a web app would re-serve MANO to every visitor) and WiLoR runs **locally only** (the model stays on your machine). Details + reasoning in [`docs/licensing.md`](docs/licensing.md). `MANO_RIGHT.pkl` is gitignored — download it per user from <https://mano.is.tue.mpg.de>.

## Roadmap

- [x] One-Euro filter to smooth per-frame jitter (`one_euro.py`, tune via `--min-cutoff` / `--beta`)
- [x] WiLoR offline path — benchmark (`bench_wilor.py`) + mesh-overlay renderer (`wilor_overlay.py`)
- [x] Browser demo on GitHub Pages (`web/`)
- [ ] `cv2.undistortPoints` for accurate metric placement
- [ ] Retarget keypoints → robot hand joint angles (`dex-retargeting`)
