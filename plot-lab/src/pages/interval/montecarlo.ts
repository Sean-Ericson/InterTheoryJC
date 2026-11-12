import { Rng } from '../../rng';
import type { Start } from './model';

export interface SamplerConfig {
  L: number;
  D: number;
  x0: number;
  start: Start;
  /** Number of paths. */
  n: number;
  seed: number;
}

/** Wall each path left through. */
export const WALL_0 = 0;
export const WALL_L = 1;

/**
 * Exit times of independent Brownian paths dx = √D dW from (0, L), simulated a
 * few at a time so the page stays responsive.
 *
 * Increments are exact Gaussians. A Brownian-bridge test catches crossings
 * between steps: a step from a to b inside the interval crossed the wall at 0
 * with probability exp(−2ab/(D dt)), and likewise for L. The step shrinks near
 * a wall (its s.d. is at most ⅕ of the distance), so exit times are resolved
 * where they cluster. The step size depends only on the current position, so
 * this adds no bias.
 */
export class ExitSampler {
  readonly times: Float64Array;
  readonly wall: Uint8Array;
  count = 0;
  private readonly rng: Rng;

  constructor(readonly cfg: SamplerConfig) {
    this.times = new Float64Array(cfg.n);
    this.wall = new Uint8Array(cfg.n);
    this.rng = new Rng(cfg.seed);
  }

  get done(): boolean {
    return this.count >= this.cfg.n;
  }

  /** Simulate further paths for about `budgetMs` milliseconds. */
  advance(budgetMs: number): void {
    const t0 = performance.now();
    while (this.count < this.cfg.n) {
      this.simulateOne();
      if ((this.count & 31) === 0 && performance.now() - t0 > budgetMs) break;
    }
  }

  private startPoint(): number {
    const { L, start, x0 } = this.cfg;
    switch (start) {
      case 'point':
        return x0;
      case 'uniform':
        return this.rng.next() * L;
      case 'qsd':
        // Inverse CDF of (π/2L) sin(πx/L).
        return (L / Math.PI) * Math.acos(1 - 2 * this.rng.next());
    }
  }

  private simulateOne(): void {
    const { L, D } = this.cfg;
    const rng = this.rng;
    const dtMax = (L * L) / (2000 * D);
    const dtMin = dtMax * 1e-7;
    let x = this.startPoint();
    let t = 0;
    for (;;) {
      const d = Math.min(x, L - x);
      const dt = Math.min(dtMax, Math.max(dtMin, (0.04 * d * d) / D));
      const xb = x + Math.sqrt(D * dt) * rng.normal();
      let hit = -1;
      if (xb <= 0) hit = WALL_0;
      else if (xb >= L) hit = WALL_L;
      else {
        const u = rng.next();
        const p0 = Math.exp((-2 * x * xb) / (D * dt));
        if (u < p0) hit = WALL_0;
        else if (u < p0 + Math.exp((-2 * (L - x) * (L - xb)) / (D * dt))) hit = WALL_L;
      }
      if (hit >= 0) {
        this.times[this.count] = t + dt / 2;
        this.wall[this.count] = hit;
        this.count++;
        return;
      }
      x = xb;
      t += dt;
    }
  }
}

export interface EmpiricalStats {
  n: number;
  mean: number;
  /** Standard error of the mean. */
  seMean: number;
  sd: number;
  median: number;
  piL: number;
  sePiL: number;
  mean0: number;
  meanL: number;
}

/** Sample moments of the first `sorted.length` exit times; `sorted` must be ascending. */
export function empiricalStats(times: Float64Array, wall: Uint8Array, sorted: Float64Array): EmpiricalStats {
  const n = sorted.length;
  let sum = 0;
  let sum2 = 0;
  let nL = 0;
  let sum0 = 0;
  let sumL = 0;
  for (let i = 0; i < n; i++) {
    const t = times[i];
    sum += t;
    sum2 += t * t;
    if (wall[i] === WALL_L) {
      nL++;
      sumL += t;
    } else sum0 += t;
  }
  const mean = sum / n;
  const sd = Math.sqrt(Math.max(0, (sum2 - n * mean * mean) / Math.max(1, n - 1)));
  const piL = nL / n;
  const median = n % 2 ? sorted[(n - 1) / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  return {
    n,
    mean,
    seMean: sd / Math.sqrt(n),
    sd,
    median,
    piL,
    sePiL: Math.sqrt((piL * (1 - piL)) / n),
    mean0: n - nL > 0 ? sum0 / (n - nL) : NaN,
    meanL: nL > 0 ? sumL / nL : NaN,
  };
}

/** Fraction of the sorted exit times greater than t. */
export function survivingFraction(sorted: Float64Array, t: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return (sorted.length - lo) / sorted.length;
}

export interface ExitHistogram {
  centers: Float64Array;
  /** Densities, normalised by the number of paths so that each integrates to its share of exits. */
  total: Float64Array;
  wall0: Float64Array;
  wallL: Float64Array;
}

/** Histogram of exit times over the bins between consecutive `edges`. */
export function exitHistogram(times: Float64Array, wall: Uint8Array, n: number, edges: Float64Array, log: boolean): ExitHistogram {
  const bins = edges.length - 1;
  const c0 = new Float64Array(bins);
  const cL = new Float64Array(bins);
  const lo = edges[0];
  const hi = edges[bins];
  for (let i = 0; i < n; i++) {
    const t = times[i];
    if (t < lo || t >= hi) continue;
    // Bisect for the bin, since log bins are not evenly spaced.
    let a = 0;
    let b = bins;
    while (b - a > 1) {
      const m = (a + b) >> 1;
      if (edges[m] <= t) a = m;
      else b = m;
    }
    if (wall[i] === WALL_L) cL[a]++;
    else c0[a]++;
  }
  const centers = new Float64Array(bins);
  const total = new Float64Array(bins);
  const wall0 = new Float64Array(bins);
  const wallL = new Float64Array(bins);
  for (let k = 0; k < bins; k++) {
    const w = edges[k + 1] - edges[k];
    centers[k] = log ? Math.sqrt(edges[k] * edges[k + 1]) : 0.5 * (edges[k] + edges[k + 1]);
    wall0[k] = c0[k] / (n * w);
    wallL[k] = cL[k] / (n * w);
    total[k] = wall0[k] + wallL[k];
  }
  return { centers, total, wall0, wallL };
}
