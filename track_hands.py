#!/usr/bin/env python3
"""Real-time hand tracking -> viser, with FPS benchmarking.

One pipeline, swappable frame source:
  - a recorded video file (offline test / benchmark)
  - the live webcam (this is your teleop loop)

Tracker is MediaPipe Hands (fast enough for interactive teleop on Apple Silicon).
It outputs 21 3D keypoints per hand, which we draw in viser as a skeleton
(spheres for joints, line segments for bones).

The number that decides teleop viability is the printed `infer_fps` -- the rate
at which the tracker alone can process frames, independent of playback pacing.

Examples
--------
    # Benchmark + visualize on a recorded clip (loops the clip)
    python3 track_hands.py --source ~/Downloads/hand_recording.mp4

    # Run as fast as possible to measure peak tracker throughput
    python3 track_hands.py --source ~/Downloads/hand_recording.mp4 --max-speed

    # Live teleop from the built-in webcam
    python3 track_hands.py --source webcam --camera-index 0

Then open the printed viser URL (http://localhost:8080) in a browser.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import viser
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from one_euro import OneEuroFilter

# Standard MediaPipe 21-keypoint hand topology (joint index pairs = bones).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

DEFAULT_MODEL = str(Path(__file__).resolve().parent / "hand_landmarker.task")

# Bundled 1280x720 calibration of the built-in FaceTime HD camera. If present,
# hands are placed metrically; otherwise they're spread on a plane by image
# position so they don't overlap.
DEFAULT_CALIB = str(Path(__file__).resolve().parent / "calib" / "macbook_air_m2_1280x720.json")

# Color keyed by handedness so each hand keeps its color regardless of the other.
HAND_COLORS = {"Right": (80, 170, 255), "Left": (255, 140, 80)}  # RGB
DEFAULT_COLOR = (180, 180, 180)  # fallback if handedness is somehow unknown
ALL_LABELS = ("Left", "Right")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parent / "data" / "hand_recording.mp4"),
        help='Video file path, or "webcam" for the live camera.',
    )
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index when --source webcam.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to hand_landmarker.task model.")
    p.add_argument(
        "--calib",
        default=DEFAULT_CALIB,
        help="Camera calibration JSON (for metric hand placement). Must match capture resolution. "
        'Pass "" to disable and place hands by image position only.',
    )
    p.add_argument("--max-hands", type=int, default=2, help="Max hands to detect.")
    p.add_argument("--min-cutoff", type=float, default=1.0, help="One-Euro min cutoff (Hz); lower = smoother, more lag.")
    p.add_argument("--beta", type=float, default=0.5, help="One-Euro beta; higher = less lag during fast motion.")
    p.add_argument("--no-filter", action="store_true", help="Disable One-Euro smoothing (show raw keypoints).")
    p.add_argument("--show-camera", action="store_true",
                   help="Show the live camera feed as an image plane in the viser scene (nice for demos/recording).")
    p.add_argument(
        "--max-speed",
        action="store_true",
        help="Process frames as fast as possible (peak-throughput benchmark). "
        "Default throttles file playback to the source fps so the viser view is watchable.",
    )
    p.add_argument("--port", type=int, default=8080, help="viser server port.")
    return p.parse_args()


def open_source(source: str, camera_index: int) -> tuple[cv2.VideoCapture, bool, float]:
    """Return (capture, is_live, source_fps)."""
    is_live = source.lower() in {"webcam", "cam", "live"}
    if is_live:
        cap = cv2.VideoCapture(camera_index)
        # NOTE: OpenCV's camera index differs from ffmpeg's avfoundation index.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")
    return cap, is_live, fps


def load_intrinsics(path: str) -> tuple[float, float, float, float] | None:
    """Read (fx, fy, cx, cy) from the calibration JSON, or None to disable metric placement."""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        print(f"[calib] not found -> placing hands by image position only ({p})")
        return None
    c = json.loads(p.read_text())["calibration"]
    print(f"[calib] metric placement using {p.name} (fx={c['fx']:.1f}, cx={c['cx']:.1f})")
    return c["fx"], c["fy"], c["cx"], c["cy"]


def hand_offset(
    image_landmarks,
    joints: np.ndarray,
    img_w: int,
    img_h: int,
    K: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Translation to add to the (center-relative) joints so the hand sits where it really is.

    With intrinsics: estimate depth from known metric hand size vs. apparent pixel size
    (monocular size-based depth), then back-project the wrist pixel to a 3D point.
    Without intrinsics: just spread hands across a plane by their normalized image position.
    """
    u = np.array([lm.x * img_w for lm in image_landmarks], dtype=np.float32)
    v = np.array([lm.y * img_h for lm in image_landmarks], dtype=np.float32)

    if K is None:
        target_wrist = np.array(
            [(image_landmarks[0].x - 0.5) * 0.6, -(image_landmarks[0].y - 0.5) * 0.45, 0.0],
            dtype=np.float32,
        )
    else:
        fx, fy, cx, cy = K
        # Z = f * real_size / pixel_size, taken as the median over all bones for robustness.
        ratios = []
        for a, b in HAND_CONNECTIONS:
            pix = float(np.hypot(u[a] - u[b], v[a] - v[b]))
            if pix > 1.0:
                ratios.append(fx * float(np.linalg.norm(joints[a] - joints[b])) / pix)
        z = float(np.median(ratios)) if ratios else 0.5
        x = (u[0] - cx) * z / fx
        y = (v[0] - cy) * z / fy
        target_wrist = np.array([x, -y, -z], dtype=np.float32)  # camera -> viser axis flip

    return target_wrist - joints[0]


def landmarks_to_xyz(world_landmarks) -> np.ndarray:
    """MediaPipe world landmarks (metres, origin at hand center) -> (21, 3) array.

    MediaPipe axes: +x right, +y down, +z toward camera. Flip y and z so the
    hand sits upright and faces +z in viser's right-handed, y-up-ish scene.
    """
    pts = np.array([[lm.x, -lm.y, -lm.z] for lm in world_landmarks], dtype=np.float32)
    return pts


def draw_hand(server: viser.ViserServer, label: str, joints: np.ndarray) -> None:
    """(Re)draw one hand's joints and bones, keyed by handedness label (stable name)."""
    color = HAND_COLORS.get(label, DEFAULT_COLOR)

    server.scene.add_point_cloud(
        f"/hand_{label}/joints",
        points=joints,
        colors=np.tile(color, (joints.shape[0], 1)),
        point_size=0.006,
        point_shape="circle",
    )

    segments = np.array([[joints[a], joints[b]] for a, b in HAND_CONNECTIONS], dtype=np.float32)
    server.scene.add_line_segments(
        f"/hand_{label}/bones",
        points=segments,
        colors=np.tile(color, (segments.shape[0], 2, 1)),
        line_width=3.0,
    )


def clear_hand(server: viser.ViserServer, label: str) -> None:
    """Hide a hand that is no longer detected by emptying its scene nodes."""
    server.scene.add_point_cloud(
        f"/hand_{label}/joints", points=np.zeros((0, 3), np.float32), colors=np.zeros((0, 3), np.uint8)
    )
    server.scene.add_line_segments(
        f"/hand_{label}/bones", points=np.zeros((0, 2, 3), np.float32), colors=np.zeros((0, 2, 3), np.uint8)
    )


def main() -> None:
    args = parse_args()
    cap, is_live, source_fps = open_source(args.source, args.camera_index)
    frame_interval = 1.0 / source_fps if source_fps > 0 else 0.0
    K = load_intrinsics(args.calib)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")
    print(f"\nviser running -> open http://localhost:{args.port} in a browser\n")

    # Live filter controls (CLI flags seed the initial values; sliders override at runtime).
    # Detailed tuning guidance lives in per-control hover tooltips to keep the panel compact.
    with server.gui.add_folder("Smoothing (One-Euro)"):
        gui_enable = server.gui.add_checkbox(
            "enabled", initial_value=not args.no_filter, hint="Toggle One-Euro smoothing on/off."
        )
        gui_min_cutoff = server.gui.add_slider(
            "min_cutoff (Hz)",
            min=0.1,
            max=5.0,
            step=0.05,
            initial_value=args.min_cutoff,
            hint="Lower = smoother at rest but more lag. Hold your hand still and lower until the jitter is gone.",
        )
        gui_beta = server.gui.add_slider(
            "beta",
            min=0.0,
            max=5.0,
            step=0.05,
            initial_value=args.beta,
            hint="Higher = less lag during fast motion. Wave fast and raise until the lag disappears.",
        )

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=args.model),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=args.max_hands,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    infer_times: deque[float] = deque(maxlen=60)
    loop_times: deque[float] = deque(maxlen=60)
    frame_count = 0
    timestamp_ms = 0  # must be strictly increasing for VIDEO mode
    filters: dict[str, OneEuroFilter] = {}  # one One-Euro filter per hand, keyed by handedness

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if is_live:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the file
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if args.show_camera:
                # Camera feed as a 16:9 image plane behind the hands (re-add by name = live update).
                h, w = rgb.shape[:2]
                disp = cv2.resize(rgb, (640, int(640 * h / w)))
                server.scene.add_image(
                    "/camera_feed", disp,
                    render_width=0.64, render_height=0.64 * h / w,
                    position=(0.0, 0.0, -0.6), wxyz=(1.0, 0.0, 0.0, 0.0),
                )
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += max(1, int(frame_interval * 1000))

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            infer_times.append(time.perf_counter() - t0)

            img_h, img_w = frame.shape[:2]
            present = set()
            if result.hand_world_landmarks:
                for i, world in enumerate(result.hand_world_landmarks):
                    # Handedness drives both color and viser node name, so a hand keeps
                    # its identity whether or not the other hand is on screen.
                    label = result.handedness[i][0].category_name if result.handedness else f"hand{i}"
                    joints = landmarks_to_xyz(world)
                    joints = joints + hand_offset(result.hand_landmarks[i], joints, img_w, img_h, K)
                    if gui_enable.value:
                        filt = filters.get(label)
                        if filt is None:
                            filt = filters[label] = OneEuroFilter()
                        filt.min_cutoff = gui_min_cutoff.value
                        filt.beta = gui_beta.value
                        joints = filt(joints, loop_start)
                    draw_hand(server, label, joints)
                    present.add(label)
            # Clear hands that vanished, and drop their filter state so they re-init fresh.
            for label in set(filters) | set(ALL_LABELS):
                if label not in present:
                    clear_hand(server, label)
                    filters.pop(label, None)

            frame_count += 1
            if not args.max_speed and frame_interval > 0:
                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

            loop_times.append(time.perf_counter() - loop_start)
            if frame_count % 15 == 0 and infer_times:
                infer_fps = 1.0 / (sum(infer_times) / len(infer_times))
                loop_fps = 1.0 / (sum(loop_times) / len(loop_times))
                n_hands = len(result.hand_world_landmarks or [])
                print(
                    f"frame {frame_count:5d} | infer_fps={infer_fps:6.1f} (teleop-relevant) | "
                    f"loop_fps={loop_fps:6.1f} | hands={n_hands}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        landmarker.close()
        cap.release()


if __name__ == "__main__":
    main()
