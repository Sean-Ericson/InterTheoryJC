import { FokkerPlanck, type GridConfig } from './fpe';
import type { Process } from './model';
import type { Recorder } from './recorder';

/** Mean and surviving probability at one lag, which turn moments of F into a covariance. */
export interface Survivors {
  mass: number;
  mean: number;
}

/** Mean of a density given at the cell centres xc. */
export function meanOf(p: ArrayLike<number>, xc: Float64Array): number {
  let m0 = 0;
  let m1 = 0;
  for (let i = 0; i < xc.length; i++) {
    m0 += p[i];
    m1 += p[i] * xc[i];
  }
  return m1 / m0;
}

/**
 * Two-time covariance ⟨X(s + τ), X(s)⟩ at lags τ_j = j Δτ, from the
 * Fokker–Planck equation. As in §3.7.4 (eq. 3.7.46), the Markov property
 * makes it a one-time problem: by linearity
 *   F(x, s + τ) = ∫ p(x, s + τ | y, s) (y − m_s) p(y, s) dy
 * solves the same equation from F(x, s) = (x − m_s) p(x, s), and
 *   ⟨X(s + τ), X(s)⟩ = (∫ x F dx − m ∫ F dx) / S,
 * with m and S the mean and surviving probability at s + τ. With absorbing
 * walls this is the covariance among the paths that survive to s + τ;
 * otherwise ∫ F dx = 0 and S = 1.
 *
 * Started from the stationary density it gives the stationary covariance
 * G(τ) (eq. 3.7.58), and from φ₀ its quasi-stationary analogue.
 */
export class Covariance {
  /** Covariance at lag j, filled below `count`. */
  readonly values: Float64Array;
  count = 0;
  private readonly fp: FokkerPlanck;

  /**
   * `p` is the density at the reference time s on the grid of `grid`;
   * `survivors(j)` gives the mean and mass at lag j, or null until they are known.
   */
  constructor(
    proc: Process,
    grid: GridConfig,
    p: ArrayLike<number>,
    s: number,
    lags: number,
    private readonly dtau: number,
    private readonly substeps: number,
    private readonly survivors: (j: number) => Survivors | null,
  ) {
    this.values = new Float64Array(lags).fill(NaN);
    const fp = (this.fp = new FokkerPlanck(proc, grid));
    const mean = meanOf(p, fp.xc);
    for (let i = 0; i < fp.n; i++) fp.p[i] = (fp.xc[i] - mean) * p[i];
    fp.t = s;
  }

  get done(): boolean {
    return this.count === this.values.length;
  }

  /** Extend to further lags until `deadline` (a performance.now() time) or until the survivors run out. */
  advance(deadline: number): void {
    const { fp } = this;
    while (this.count < this.values.length) {
      const j = this.count;
      const st = this.survivors(j);
      if (!st) return;
      if (j > 0) for (let i = 0; i < this.substeps; i++) fp.step(this.dtau / this.substeps);
      let f0 = 0;
      let f1 = 0;
      for (let i = 0; i < fp.n; i++) {
        f0 += fp.p[i];
        f1 += fp.p[i] * fp.xc[i];
      }
      this.values[j] = ((f1 - st.mean * f0) * fp.h) / st.mass;
      this.count++;
      if (performance.now() > deadline) return;
    }
  }
}

/**
 * Sample covariance of the path positions at snapshot j with those at
 * snapshot j + lag, among the paths still alive at j + lag. Fills `out` from
 * lag `from` up to the last recorded snapshot and returns how far it got.
 */
export function ensembleCovariance(rec: Recorder, j: number, out: Float64Array, from: number): number {
  const xs = rec.positionsAt(j).x;
  let lag = from;
  for (; lag < out.length && j + lag < rec.count; lag++) {
    const { x, alive } = rec.positionsAt(j + lag);
    // Centre on the recorded means first, to limit cancellation.
    const ms = rec.ens.mean[j];
    const mt = rec.ens.mean[j + lag];
    let n = 0;
    let a = 0;
    let b = 0;
    let ab = 0;
    for (let i = 0; i < x.length; i++) {
      if (!alive[i]) continue;
      const u = xs[i] - ms;
      const v = x[i] - mt;
      n++;
      a += u;
      b += v;
      ab += u * v;
    }
    out[lag] = n > 1 ? (ab - (a * b) / n) / (n - 1) : NaN;
  }
  return lag;
}
