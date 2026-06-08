# Licensing — and why WiLoR isn't in the web demo

A quick map of what each component allows, because it shaped the architecture. *(Not legal
advice — read the upstream licenses yourself.)*

| Component | License | Redistributable? | Commercial? |
|---|---|---|---|
| **MediaPipe** model | Apache-2.0 | ✅ yes (Google distributes it openly) | ✅ yes |
| **MANO** model | MPI non-commercial research license | ❌ **no** — "single-user, **non-transferable**", download-gated per user | ❌ no |
| **WiLoR** weights | CC-BY-NC | ✅ yes, with attribution | ❌ no |
| **wilor-mini** wrapper | (see repo) | — | — |

## Two separate constraints

It helps to keep two questions apart:

1. **Commercial use?** This is a personal, non-commercial research project, which satisfies the
   non-commercial clause of *both* MANO and WiLoR. ✅
2. **Redistribution?** MANO is **non-transferable** and gated behind per-user registration. You
   may use it on *your own machines*, but you may **not** re-serve it to others.

## Why the browser demo is MediaPipe-only

A client-side web app (GitHub Pages) **ships the model to every visitor's browser**. That's fine
for MediaPipe (openly redistributable) but would **breach MANO's non-transferable terms** — and
WiLoR's pipeline requires MANO, so WiLoR can't go on Pages even though the demo is non-commercial.
Baking MANO into ONNX weights wouldn't help (still a MANO-derived artifact).

## Why WiLoR runs locally only

Running WiLoR on **your own machine** (`wilor_live.py`, `wilor_overlay.py`, `bench_wilor.py`)
keeps the model on "your computers" per the MANO license and never transfers it to anyone — the
license-clean way to use it for non-commercial research. **Sharing rendered output videos is
fine** (that's output, not the model). `MANO_RIGHT.pkl` is therefore gitignored and must be
downloaded per user from <https://mano.is.tue.mpg.de>.

A *hosted* interactive WiLoR demo would need to be **server-side** (model stays on your hardware,
only rendered frames sent out) — not static hosting.
