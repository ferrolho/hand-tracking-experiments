#!/usr/bin/env python3
"""Benchmark WiLoR (via wilor-mini) end-to-end speed on this machine.

This is the *offline-path / "how slow is SOTA"* counterpart to track_hands.py.
It measures the full per-frame cost: hand detection (YOLO) + transformer-based
MANO reconstruction — i.e. what live teleop would actually pay per frame.

The wilor-mini pipeline auto-downloads its weights and a (chumpy-free) MANO model
from HuggingFace on first run, so the first invocation is slow (downloads).

Examples
--------
    python3 bench_wilor.py --source data/hand_recording.mp4 --frames 60
    python3 bench_wilor.py --source webcam --device mps
    python3 bench_wilor.py --device cpu          # compare CPU vs MPS
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported ops fall back to CPU

import cv2
import numpy as np
import torch

# wilor-mini targets torch<=2.5; under torch>=2.6 torch.load defaults to weights_only=True,
# which rejects the old ultralytics YOLO checkpoint. Restore the permissive default.
_orig_torch_load = torch.load
def _torch_load_compat(*a, **k):  # noqa: E306
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load_compat


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(Path(__file__).resolve().parent / "data" / "hand_recording.mp4"),
                   help='Video file path, or "webcam".')
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--frames", type=int, default=60, help="How many frames to time.")
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"device={device} dtype={dtype}")

    print("loading wilor-mini pipeline (first run downloads weights + MANO from HuggingFace)...")
    t_load = time.perf_counter()
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
    print(f"pipeline ready in {time.perf_counter() - t_load:.1f}s")

    # Open source
    is_live = args.source.lower() in {"webcam", "cam", "live"}
    cap = cv2.VideoCapture(args.camera_index if is_live else args.source)
    if is_live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {args.source}")

    times: list[float] = []
    n_with_hand = 0
    i = 0
    print(f"timing {args.frames} frames...")
    while i < args.frames:
        ok, frame = cap.read()
        if not ok:
            if is_live:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        outputs = pipe.predict(rgb)
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0

        # The first 1-2 frames include lazy init/compile; skip them from stats.
        if i >= 2:
            times.append(dt)
        if outputs:
            n_with_hand += 1
        i += 1
        if i % 10 == 0:
            print(f"  frame {i}: {dt * 1000:.0f} ms ({1.0 / dt:.2f} fps), hands={len(outputs)}", flush=True)

    cap.release()
    if times:
        arr = np.array(times)
        print("\n=== WiLoR end-to-end (detection + reconstruction) ===")
        print(f"  median: {np.median(arr) * 1000:.0f} ms  ->  {1.0 / np.median(arr):.2f} fps")
        print(f"  mean:   {arr.mean() * 1000:.0f} ms  ->  {1.0 / arr.mean():.2f} fps")
        print(f"  frames with >=1 hand: {n_with_hand}/{i}")


if __name__ == "__main__":
    main()
