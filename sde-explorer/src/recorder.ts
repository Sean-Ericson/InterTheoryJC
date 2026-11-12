import type { ExactModel } from './analytic';
import type { FokkerPlanck } from './fpe';
import type { Ensemble } from './sde';

export interface Moments {
  mean: Float64Array;
  variance: Float64Array;
  mass: Float64Array;
  /** Exit rate −Ṡ/S. */
  rate: Float64Array;
}

function moments(len: number): Moments {
  return {
    mean: new Float64Array(len).fill(NaN),
    variance: new Float64Array(len).fill(NaN),
    mass: new Float64Array(len).fill(NaN),
    rate: new Float64Array(len).fill(NaN),
  };
}

/** Snapshots spanned by the ensemble's exit-rate estimate. */
const RATE_WINDOW = 16;

/**
 * Everything needed to replay a run: W+1 evenly spaced snapshots of the
 * density and the ensemble, plus full-resolution traces of the paths on screen.
 */
export class Recorder {
  readonly snaps: number;
  readonly times: Float64Array;
  /**
   * snaps × nGrid, row-major by snapshot. Double precision so that p/S stays
   * meaningful after the surviving probability has fallen below ~1e-38.
   */
  readonly density: Float64Array;
  /** snaps × nPaths positions; alive flags alongside. */
  readonly positions: Float32Array;
  readonly alive: Uint8Array;
  /** nShow × (nSteps+1) traces of the displayed paths. */
  readonly traces: Float32Array;
  /** Step at which each displayed path was absorbed, −1 if never. */
  readonly traceExit: Int32Array;
  readonly ens: Moments;
  readonly fp: Moments;
  readonly exact: Moments;
  /** Snapshots recorded so far. */
  count = 0;
  /** Trace steps recorded so far. */
  traceCount = 0;

  constructor(
    readonly W: number,
    readonly nSteps: number,
    readonly nGrid: number,
    readonly nPaths: number,
    readonly nShow: number,
  ) {
    this.snaps = W + 1;
    this.times = new Float64Array(this.snaps);
    this.density = new Float64Array(this.snaps * nGrid);
    this.positions = new Float32Array(this.snaps * nPaths);
    this.alive = new Uint8Array(this.snaps * nPaths);
    this.traces = new Float32Array(nShow * (nSteps + 1));
    this.traceExit = new Int32Array(nShow).fill(-1);
    this.ens = moments(this.snaps);
    this.fp = moments(this.snaps);
    this.exact = moments(this.snaps);
  }

  recordTraces(ens: Ensemble): void {
    const s = this.traceCount;
    const stride = this.nSteps + 1;
    for (let j = 0; j < this.nShow; j++) {
      this.traces[j * stride + s] = ens.x[j];
      if (!ens.alive[j] && this.traceExit[j] < 0) this.traceExit[j] = s;
    }
    this.traceCount++;
  }

  snapshot(t: number, ens: Ensemble, fp: FokkerPlanck, exact: ExactModel | null): void {
    const k = this.count;
    this.times[k] = t;
    this.density.set(fp.p, k * this.nGrid);
    this.positions.set(ens.x, k * this.nPaths);
    this.alive.set(ens.alive, k * this.nPaths);
    const es = ens.stats();
    const fs = fp.stats();
    this.ens.mean[k] = es.mean;
    this.ens.variance[k] = es.variance;
    this.ens.mass[k] = es.mass;
    this.fp.mean[k] = fs.mean;
    this.fp.variance[k] = fs.variance;
    this.fp.mass[k] = fs.mass;
    this.fp.rate[k] = fs.rate;
    if (exact) {
      const ex = exact.at(t);
      this.exact.mean[k] = ex.mean;
      this.exact.variance[k] = ex.variance;
      this.exact.mass[k] = ex.mass;
      this.exact.rate[k] = ex.rate;
    }
    // The ensemble's −d ln S/dt, centred on a window of snapshots: too few
    // paths are absorbed between neighbouring snapshots to difference them.
    if (k >= RATE_WINDOW) {
      const j = k - RATE_WINDOW;
      const [s0, s1] = [this.ens.mass[j], this.ens.mass[k]];
      this.ens.rate[k - RATE_WINDOW / 2] = s1 > 0 ? Math.log(s0 / s1) / (t - this.times[j]) : NaN;
    }
    this.count++;
  }

  densityAt(k: number): Float64Array {
    return this.density.subarray(k * this.nGrid, (k + 1) * this.nGrid);
  }

  positionsAt(k: number): { x: Float32Array; alive: Uint8Array } {
    return {
      x: this.positions.subarray(k * this.nPaths, (k + 1) * this.nPaths),
      alive: this.alive.subarray(k * this.nPaths, (k + 1) * this.nPaths),
    };
  }

  trace(j: number): Float32Array {
    const stride = this.nSteps + 1;
    return this.traces.subarray(j * stride, (j + 1) * stride);
  }
}
