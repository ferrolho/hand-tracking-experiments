#!/usr/bin/env python3
"""Render WiLoR's 3D hand mesh as an overlay on a video — to *see* the higher-quality result.

WiLoR returns a full 778-vertex MANO mesh plus the camera translation and focal length,
so we can project the mesh onto each frame (no pyrender/OpenGL needed) and draw it as a
shaded, semi-transparent surface. Output is written as an MP4 you can play back at full speed,
even though inference itself runs at only ~4 fps on the M2.

Example
-------
    python3 wilor_overlay.py --source data/hand_recording.mp4 --frames 150
    # -> writes data/hand_recording_wilor.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# wilor-mini emits an INFO line + YOLO summary every frame; keep the console to a progress bar.
logging.disable(logging.WARNING)

import cv2
import numpy as np
import torch
from tqdm import tqdm

# wilor-mini targets torch<=2.5; restore the permissive torch.load default for its old checkpoints.
_orig_torch_load = torch.load
def _torch_load_compat(*a, **k):  # noqa: E306
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _torch_load_compat

from one_euro import OneEuroFilter

BASE_COLOR = np.array([235, 180, 90], dtype=np.float32)  # BGR (warm teal/cyan)


def open_ffmpeg_writer(out: Path, w: int, h: int, fps: float) -> subprocess.Popen:
    """H.264/yuv420p MP4 writer (Twitter/X-ready: +faststart and a silent AAC track)."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "pipe:0",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
        str(out),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def project(verts3d: np.ndarray, cam_t: np.ndarray, focal: float, center: np.ndarray) -> np.ndarray:
    """Perspective-project (N,3) mesh vertices to (N,2) image pixels."""
    p = verts3d + cam_t[None, :]
    z = np.clip(p[:, 2], 1e-4, None)
    u = focal * p[:, 0] / z + center[0]
    v = focal * p[:, 1] / z + center[1]
    return np.stack([u, v], axis=-1)


def draw_mesh(frame: np.ndarray, verts3d: np.ndarray, faces: np.ndarray,
              cam_t: np.ndarray, focal: float, alpha: float = 0.65) -> None:
    """Draw a shaded, depth-sorted, semi-transparent mesh onto `frame` in place."""
    center = np.array([frame.shape[1] / 2.0, frame.shape[0] / 2.0])
    p = verts3d + cam_t[None, :]
    v2d = project(verts3d, cam_t, focal, center)

    tri = p[faces]                                   # (F,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    shade = np.abs(n[:, 2])                           # facing-camera -> brighter
    depth = tri[:, :, 2].mean(axis=1)
    order = np.argsort(-depth)                        # painter's: far first

    overlay = frame.copy()
    v2d_i = v2d.astype(np.int32)
    for f in order:
        color = (BASE_COLOR * (0.35 + 0.65 * shade[f])).tolist()
        cv2.fillConvexPoly(overlay, v2d_i[faces[f]], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(Path(__file__).resolve().parent / "data" / "hand_recording.mp4"))
    p.add_argument("--out", default="", help="Output mp4 path (default: <source>_wilor.mp4).")
    p.add_argument("--frames", type=int, default=0, help="Limit frames (0 = whole clip).")
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--smooth", action="store_true", help="One-Euro temporal smoothing of the mesh (reduces jitter).")
    p.add_argument("--min-cutoff", type=float, default=1.0, help="One-Euro min cutoff (lower = smoother, more lag).")
    p.add_argument("--beta", type=float, default=0.5, help="One-Euro beta (higher = less lag on fast motion).")
    args = p.parse_args()

    src = Path(args.source).expanduser()
    out = Path(args.out) if args.out else src.with_name(src.stem + "_wilor.mp4")
    device = torch.device(args.device)

    print("loading WiLoR pipeline...")
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32)
    pipe.verbose = False  # silence the per-frame YOLO summary + internal tqdm bars
    faces = pipe.wilor_model.mano.faces

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = args.frames if args.frames > 0 else total
    writer = open_ffmpeg_writer(out, w, h, fps)

    print(f"processing {n_target} frames of {src.name} ({w}x{h}@{fps:.0f}) -> {out.name}")
    i, n_hands_total = 0, 0
    filters: dict[str, OneEuroFilter] = {}  # one per hand (keyed by handedness) when --smooth
    # tqdm gives a bar + ETA (the useful number) and a frame/s rate, on one self-updating line.
    pbar = tqdm(total=n_target, unit="frame", desc="WiLoR overlay", dynamic_ncols=True)
    while i < n_target:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = pipe.predict(rgb)
        t_now = i / fps  # video time, for the One-Euro filter
        for o in outputs:
            wp = o["wilor_preds"]
            verts = np.asarray(wp["pred_vertices"][0], dtype=np.float32)
            cam_t = np.asarray(wp["pred_cam_t_full"][0], dtype=np.float32)
            focal = float(np.asarray(wp["scaled_focal_length"]).reshape(-1)[0])
            if args.smooth:
                label = "right" if o.get("is_right", 1) else "left"
                filt = filters.get(label)
                if filt is None:
                    filt = filters[label] = OneEuroFilter(min_cutoff=args.min_cutoff, beta=args.beta)
                # Smooth the full camera-space mesh (pose + global translation) at once.
                p_cam = filt(verts + cam_t, t_now)
                draw_mesh(frame, p_cam, faces, np.zeros(3, dtype=np.float32), focal)
            else:
                draw_mesh(frame, verts, faces, cam_t, focal)
            n_hands_total += 1
        writer.stdin.write(np.ascontiguousarray(frame).tobytes())
        i += 1
        pbar.update(1)
        pbar.set_postfix(hands=n_hands_total)

    pbar.close()
    cap.release()
    writer.stdin.close()
    writer.wait()
    print(f"done: {out}  ({i} frames, {n_hands_total} hand detections)")


if __name__ == "__main__":
    main()
