"""One-Euro filter (Casiez et al., CHI 2012) for low-latency jitter smoothing.

A speed-adaptive low-pass filter: it smooths hard when the signal is slow (kills
jitter) and loosens when the signal moves fast (avoids lag). Vectorized here so a
single instance filters a whole (N, 3) keypoint array at once.

Tuning:
  min_cutoff  ↓ => more smoothing at rest (but more lag).  Start ~1.0 Hz.
  beta        ↑ => less lag during fast motion.            Start ~0.5.
  d_cutoff    cutoff for the derivative; 1.0 is usually fine.

Reference: https://gery.casiez.net/1euro/
"""

from __future__ import annotations

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Stateful One-Euro filter over an arbitrarily-shaped float array."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Filter sample `x` observed at time `t` (seconds). Returns the smoothed value."""
        x = np.asarray(x, dtype=np.float32)

        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x

        dt = t - self._t_prev
        if dt <= 0.0:
            dt = 1e-3  # guard against zero/negative timestep
        self._t_prev = t

        # Low-pass the derivative, then derive a signal-speed-dependent cutoff.
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat
