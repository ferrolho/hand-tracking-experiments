#!/usr/bin/env python3
"""Live WiLoR mesh overlay from the webcam, in a local window — runs on YOUR machine.

This is the license-clean way to try WiLoR interactively: the model runs locally and the
MANO-based mesh never leaves your computer (unlike a public web demo). It's slow (~4 fps on
an M2), so expect a slideshow — but it's the real high-quality mesh, live.

Reuses the projection/rendering from wilor_overlay.py.

Examples
--------
    python3 wilor_live.py                       # webcam, selfie-mirrored, smoothing off
    python3 wilor_live.py --smooth              # One-Euro temporal smoothing
    python3 wilor_live.py --source data/hand_recording.mp4   # run on a clip instead

Press 'q' (or Esc) in the window to quit.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path

# wilor-mini logs an INFO/WARN line (and a YOLO summary) every frame — far too chatty for a live loop.
logging.disable(logging.WARNING)

import cv2
import numpy as np
import torch

# Importing wilor_overlay also applies its torch.load compat shim (needed for the old checkpoints).
from wilor_overlay import draw_mesh
from one_euro import OneEuroFilter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="webcam", help='"webcam" or a video file path.')
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--smooth", action="store_true", help="One-Euro temporal smoothing of the mesh.")
    p.add_argument("--min-cutoff", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--no-mirror", action="store_true", help="Disable the selfie mirror.")
    p.add_argument("--no-window", action="store_true", help="Headless: process without a display (for testing).")
    p.add_argument("--frames", type=int, default=0, help="Limit frames (0 = until quit / end).")
    args = p.parse_args()

    device = torch.device(args.device)
    print("loading WiLoR pipeline (first run downloads weights)...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32)
    pipe.verbose = False  # silence the per-frame YOLO summary + tqdm bars
    faces = pipe.wilor_model.mano.faces

    is_live = args.source.lower() in {"webcam", "cam", "live"}
    cap = cv2.VideoCapture(args.camera_index if is_live else args.source)
    if is_live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source: {args.source}")

    mirror = not args.no_mirror
    filters: dict[str, OneEuroFilter] = {}
    infer_times: deque[float] = deque(maxlen=30)
    i = 0
    print("running — press 'q' or Esc in the window to quit." if not args.no_window else "running headless…")
    while True:
        ok, frame = cap.read()
        if not ok:
            if is_live:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        # Mirror the input itself, so the mesh stays aligned with the (selfie) view for free.
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        outputs = pipe.predict(rgb)
        infer_times.append(time.perf_counter() - t0)

        t_now = time.perf_counter()
        for o in outputs:
            wp = o["wilor_preds"]
            verts = np.asarray(wp["pred_vertices"][0], dtype=np.float32)
            cam_t = np.asarray(wp["pred_cam_t_full"][0], dtype=np.float32)
            focal = float(np.asarray(wp["scaled_focal_length"]).reshape(-1)[0])
            if args.smooth:
                label = "right" if o.get("is_right", 1) else "left"
                filt = filters.get(label) or filters.setdefault(label, OneEuroFilter(args.min_cutoff, args.beta))
                draw_mesh(frame, filt(verts + cam_t, t_now), faces, np.zeros(3, np.float32), focal)
            else:
                draw_mesh(frame, verts, faces, cam_t, focal)

        fps = 1.0 / (sum(infer_times) / len(infer_times)) if infer_times else 0.0
        label = f"WiLoR  {fps:.1f} fps  ({len(outputs)} hand{'s' if len(outputs) != 1 else ''})"
        cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        i += 1
        if not args.no_window:
            cv2.imshow("WiLoR live", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
        elif i % 10 == 0:
            print(f"  frame {i}: {fps:.1f} fps", flush=True)
        if args.frames and i >= args.frames:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
