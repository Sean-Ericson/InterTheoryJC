import type { BC, Process } from './model';
import { Rng } from './rng';

export interface EnsembleConfig {
  n: number;
  x0: number;
  /** Standard deviation of the initial Gaussian. */
  sigma0: number;
  domain: [number, number];
  bc: [BC, BC];
  seed: number;
  /** Start from this density instead of the Gaussian N(x0, sigma0²). */
  initial?: CellDensity;
}

/** A density held as values on cells [lo + i h, lo + (i+1) h]. */
export interface CellDensity {
  lo: number;
  h: number;
  p: ArrayLike<number>;
}

export interface EnsembleStats {
  mean: number;
  variance: number;
  /** Fraction of paths not yet absorbed. */
  mass: number;
}

/**
 * An ensemble of independent sample paths integrated with the Milstein scheme
 *   x += a dt + b ΔW + ½ b ∂x b (ΔW² − dt),
 * which is strong order 1 for the Itô SDE (Stratonovich problems arrive here
 * already converted to Itô form by model.ts).
 */
export class Ensemble {
  readonly x: Float64Array;
  /** 1 while the path is in play; absorbing walls set it to 0. */
  readonly alive: Uint8Array;
  /** Time of absorption, NaN while alive. */
  readonly exitTime: Float64Array;
  aliveCount: number;
  t = 0;
  private readonly rng: Rng;

  constructor(
    private readonly proc: Process,
    private readonly cfg: EnsembleConfig,
  ) {
    const { n, x0, sigma0, domain, bc } = cfg;
    this.rng = new Rng(cfg.seed);
    this.x = new Float64Array(n);
    this.alive = new Uint8Array(n).fill(1);
    this.exitTime = new Float64Array(n).fill(NaN);
    this.aliveCount = n;
    // Same truncated Gaussian the Fokker–Planck solver starts from: resample
    // anything that lands beyond a wall (an open side is not a wall).
    const [lo, hi] = domain;
    const draw = cfg.initial ? cellSampler(cfg.initial, this.rng) : null;
    for (let i = 0; i < n; i++) {
      if (draw) {
        this.x[i] = draw();
        continue;
      }
      let xi = x0 + sigma0 * this.rng.normal();
      for (let tries = 0; tries < 100; tries++) {
        if ((xi >= lo || bc[0] === 'open') && (xi <= hi || bc[1] === 'open')) break;
        xi = x0 + sigma0 * this.rng.normal();
      }
      if (bc[0] !== 'open' && xi < lo) xi = lo;
      if (bc[1] !== 'open' && xi > hi) xi = hi;
      this.x[i] = xi;
    }
  }

  step(dt: number): void {
    const { a, b, db } = this.proc;
    const [lo, hi] = this.cfg.domain;
    const [bcL, bcR] = this.cfg.bc;
    const periodic = bcL === 'periodic';
    const L = hi - lo;
    const sq = Math.sqrt(dt);
    const t = this.t;
    const tEnd = t + dt;
    const { x, alive, rng } = this;

    for (let i = 0; i < x.length; i++) {
      if (!alive[i]) continue;
      const xi = x[i];
      const bi = b(xi, t);
      const dW = sq * rng.normal();
      let xn = xi + a(xi, t) * dt + bi * dW + 0.5 * bi * db(xi, t) * (dW * dW - dt);

      if (!Number.isFinite(xn)) {
        this.kill(i, tEnd, xi);
        continue;
      }
      if (periodic) {
        xn = lo + ((((xn - lo) % L) + L) % L);
      } else {
        if (xn < lo) {
          if (bcL === 'absorbing') {
            this.kill(i, tEnd, lo);
            continue;
          }
          if (bcL === 'reflecting') xn = 2 * lo - xn;
        }
        if (xn > hi) {
          if (bcR === 'absorbing') {
            this.kill(i, tEnd, hi);
            continue;
          }
          if (bcR === 'reflecting') xn = 2 * hi - xn;
        }
        // A huge step can bounce off one wall and past the other.
        if (bcL === 'reflecting' && xn < lo) xn = lo;
        if (bcR === 'reflecting' && xn > hi) xn = hi;
        // Brownian-bridge correction: the path may have touched an absorbing
        // wall between grid times even though both endpoints are inside.
        const b2dt = bi * bi * dt;
        if (b2dt > 0) {
          if (bcL === 'absorbing' && this.crossed(xi - lo, xn - lo, b2dt)) {
            this.kill(i, tEnd, lo);
            continue;
          }
          if (bcR === 'absorbing' && this.crossed(hi - xi, hi - xn, b2dt)) {
            this.kill(i, tEnd, hi);
            continue;
          }
        }
      }
      x[i] = xn;
    }
    this.t = tEnd;
  }

  stats(): EnsembleStats {
    const { x, alive } = this;
    let s = 0;
    let s2 = 0;
    let m = 0;
    for (let i = 0; i < x.length; i++) {
      if (!alive[i]) continue;
      s += x[i];
      m++;
    }
    const mean = m > 0 ? s / m : NaN;
    for (let i = 0; i < x.length; i++) {
      if (!alive[i]) continue;
      const d = x[i] - mean;
      s2 += d * d;
    }
    return { mean, variance: m > 1 ? s2 / (m - 1) : NaN, mass: m / x.length };
  }

  private crossed(d1: number, d2: number, b2dt: number): boolean {
    const z = (2 * d1 * d2) / b2dt;
    return z < 30 && this.rng.next() < Math.exp(-z);
  }

  private kill(i: number, time: number, at: number): void {
    this.alive[i] = 0;
    this.exitTime[i] = time;
    this.x[i] = at;
    this.aliveCount--;
  }
}

/** Inverse-CDF sampler for a density that is constant on each cell. */
function cellSampler({ lo, h, p }: CellDensity, rng: Rng): () => number {
  const cdf = new Float64Array(p.length);
  let total = 0;
  for (let i = 0; i < p.length; i++) cdf[i] = total += Math.max(0, p[i]);
  return () => {
    const u = rng.next() * total;
    let a = 0;
    let b = p.length - 1;
    while (a < b) {
      const m = (a + b) >> 1;
      if (cdf[m] > u) b = m;
      else a = m + 1;
    }
    const before = a > 0 ? cdf[a - 1] : 0;
    const w = cdf[a] - before;
    return lo + (a + (w > 0 ? (u - before) / w : 0.5)) * h;
  };
}
