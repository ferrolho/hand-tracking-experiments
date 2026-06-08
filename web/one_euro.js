// One-Euro filter (Casiez et al., 2012) — JS port of one_euro.py.
// Speed-adaptive low-pass: smooths hard when slow (kills jitter), loosens when fast (avoids lag).
// Operates element-wise over a Float32Array, so one instance filters a whole 63-value hand.

function alpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

export class OneEuroFilter {
  constructor(minCutoff = 1.0, beta = 0.5, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = null;
    this.tPrev = null;
  }

  // x: Float32Array, t: seconds. Returns a smoothed Float32Array.
  filter(x, t) {
    if (this.xPrev === null) {
      this.xPrev = x.slice();
      this.dxPrev = new Float32Array(x.length);
      this.tPrev = t;
      return x;
    }
    let dt = t - this.tPrev;
    if (dt <= 0) dt = 1e-3;
    this.tPrev = t;

    const aD = alpha(this.dCutoff, dt);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const dx = (x[i] - this.xPrev[i]) / dt;
      const dxHat = aD * dx + (1 - aD) * this.dxPrev[i];
      const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
      const a = alpha(cutoff, dt);
      const xHat = a * x[i] + (1 - a) * this.xPrev[i];
      out[i] = xHat;
      this.xPrev[i] = xHat;
      this.dxPrev[i] = dxHat;
    }
    return out;
  }
}
