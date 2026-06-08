# WiLoR on Apple Silicon — benchmark & notes

Quick experiment: how slow is WiLoR (SOTA hand mesh reconstruction) on the M2 Air?

## Result

| Tracker | M2 speed | Output | Use |
|---|---|---|---|
| MediaPipe | ~40 fps | 21 keypoints (2.5D) | live teleop |
| **WiLoR** (wilor-mini, MPS, fp32) | **~4.3 fps** (~232 ms/frame) | full MANO (joint rotations + 778-vtx mesh) | offline only |

~10× slower than MediaPipe → **offline use only** (policy-data generation, accuracy reference); not viable for interactive teleop on this hardware. Per-frame cost ≈ YOLO detect ~25 ms + ViT reconstruction ~200 ms. Hand detected in 40/40 frames of the test clip.

## Setup hacks (running on the pyenv 3.13 env)

[wilor-mini](https://github.com/warmshao/WiLoR-mini) targets torch ≤2.5 / older Python; on pyenv 3.13 + torch 2.11 + numpy 2 it needed:

1. `pip install --no-deps "git+…/WiLoR-mini.git"` — its `torch<=2.5` pin conflicts with torch 2.11.
2. Deps by hand: `smplx==0.1.28 timm einops ultralytics==8.1.34 scikit-image roma huggingface_hub`.
3. **chumpy** won't build on py3.13. Patched the sdist: the `from numpy import bool,…` line in `__init__.py`, and `inspect.getargspec`→`getfullargspec` in `ch.py`; then `pip install --no-build-isolation .`.
4. **torch.load**: torch ≥2.6 defaults `weights_only=True`, which rejects the old ultralytics checkpoint → monkeypatched to `weights_only=False` (in `bench_wilor.py`).
5. `MANO_RIGHT.pkl` from mano.is.tue.mpg.de placed in `mano_data/` (license-gated, gitignored). wilor-mini also auto-downloads its own copy.

## Cleaner alternative (recommended for real use)

A dedicated **micromamba/conda env: Python 3.10 + torch ≤2.5 + numpy <2 + ultralytics 8.1.34** avoids every hack above — chumpy builds natively and no monkeypatches are needed. Keeps this legacy stack isolated from the MediaPipe pyenv.

## Reproduce

```bash
python3 bench_wilor.py --source data/hand_recording.mp4 --device mps --frames 40
```
