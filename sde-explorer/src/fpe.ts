import type { BC, Process } from './model';

export interface GridConfig {
  n: number;
  domain: [number, number];
  bc: [BC, BC];
  x0: number;
  /** Standard deviation of the initial Gaussian. */
  sigma0: number;
}

export interface DensityStats {
  mean: number;
  variance: number;
  /** Total probability left in the domain. */
  mass: number;
  /** Exit rate −Ṡ/S: current out through the absorbing walls over the mass left. */
  rate: number;
}

/**
 * The semi-discrete operator of dp/dt = L p for time-independent coefficients:
 * (L p)_i = sub_i p_{i−1} + diag_i p_i + sup_i p_{i+1}.
 */
export interface Operator {
  sub: Float64Array;
  diag: Float64Array;
  sup: Float64Array;
  /** Current out through the left and right walls per unit density in the end cell. */
  outL: number;
  outR: number;
}

/**
 * Finite-volume solver for ∂t p = −∂x J with probability current
 *   J = a p − ½ ∂x(b² p) = v p − d ∂x p,   v = a − b ∂x b,   d = ½ b².
 *
 * Face fluxes use the Scharfetter–Gummel (exponential-fitting) scheme, which
 * stays positive, upwinds automatically where drift dominates or b vanishes,
 * and reproduces zero-current stationary states exactly. Time stepping is
 * Crank–Nicolson, started with backward-Euler half steps (Rannacher) so the
 * sharp initial peak doesn't ring.
 */
export class FokkerPlanck {
  readonly n: number;
  readonly h: number;
  /** Cell centres. */
  readonly xc: Float64Array;
  /** Point values of the density at the cell centres. */
  readonly p: Float64Array;
  t = 0;
  private stepCount = 0;
  private readonly lo: number;
  private readonly bc: [BC, BC];
  // Semi-discrete operator: dp_i/dt = sub_i p_{i-1} + diag_i p_i + sup_i p_{i+1}
  // (+ corner entries linking the two ends when periodic).
  private readonly sub: Float64Array;
  private readonly diag: Float64Array;
  private readonly sup: Float64Array;
  private cornerTop = 0; // row 0, column n−1
  private cornerBottom = 0; // row n−1, column 0
  private assembled = false;
  private readonly flux = { alpha: 0, beta: 0 };
  // Scratch space for the linear solve.
  private readonly ma: Float64Array;
  private readonly mb: Float64Array;
  private readonly mc: Float64Array;
  private readonly rhs: Float64Array;
  private readonly cp: Float64Array;
  private readonly dp: Float64Array;
  private readonly u: Float64Array;
  private readonly z: Float64Array;
  private readonly alpha: Float64Array;
  private readonly beta: Float64Array;

  constructor(
    private readonly proc: Process,
    cfg: GridConfig,
  ) {
    const n = cfg.n;
    const [lo, hi] = cfg.domain;
    this.n = n;
    this.lo = lo;
    this.bc = cfg.bc;
    this.h = (hi - lo) / n;
    this.xc = new Float64Array(n);
    for (let i = 0; i < n; i++) this.xc[i] = lo + (i + 0.5) * this.h;
    this.p = initialDensity(this.xc, this.h, cfg.x0, cfg.sigma0);
    const arr = () => new Float64Array(n);
    this.sub = arr();
    this.diag = arr();
    this.sup = arr();
    this.ma = arr();
    this.mb = arr();
    this.mc = arr();
    this.rhs = arr();
    this.cp = arr();
    this.dp = arr();
    this.u = arr();
    this.z = arr();
    this.alpha = new Float64Array(n + 1);
    this.beta = new Float64Array(n + 1);
  }

  step(dt: number): void {
    if (this.stepCount < 2) {
      // Rannacher start-up: two backward-Euler half steps per step.
      this.theta(dt / 2, 1);
      this.theta(dt / 2, 1);
    } else {
      this.theta(dt, 0.5);
    }
    this.stepCount++;
  }

  stats(): DensityStats {
    const { p, xc, h, n } = this;
    let m = 0;
    let s = 0;
    for (let i = 0; i < n; i++) {
      const pi = Math.max(p[i], 0);
      m += pi;
      s += pi * xc[i];
    }
    const mean = m > 0 ? s / m : NaN;
    let v = 0;
    for (let i = 0; i < n; i++) {
      const d = xc[i] - mean;
      v += Math.max(p[i], 0) * d * d;
    }
    return { mean, variance: m > 0 ? v / m : NaN, mass: m * h, rate: m > 0 ? this.outflow() / (m * h) : NaN };
  }

  /** The operator L at t = 0. Meaningful for time-independent coefficients and non-periodic boundaries. */
  operator(): Operator {
    this.assemble(0);
    this.assembled = true;
    const [bcL, bcR] = this.bc;
    return {
      sub: this.sub.slice(),
      diag: this.diag.slice(),
      sup: this.sup.slice(),
      outL: bcL === 'absorbing' ? this.beta[0] : 0,
      outR: bcR === 'absorbing' ? this.alpha[this.n] : 0,
    };
  }

  /** Probability current out through the absorbing walls at the current time. */
  private outflow(): number {
    const { n, h, p, flux } = this;
    const [bcL, bcR] = this.bc;
    let out = 0;
    if (bcL === 'absorbing') {
      this.face(0, h / 2, this.t);
      out += flux.beta * p[0];
    }
    if (bcR === 'absorbing') {
      this.face(n, h / 2, this.t);
      out += flux.alpha * p[n - 1];
    }
    return out;
  }

  /** One θ-scheme step: (I − θ dt L) pⁿ⁺¹ = (I + (1−θ) dt L) pⁿ. */
  private theta(dt: number, th: number): void {
    const tau = this.t + th * dt;
    if (!this.assembled || this.proc.timeDependent) {
      this.assemble(tau);
      this.assembled = true;
    }
    const { n, p, sub, diag, sup, ma, mb, mc, rhs } = this;
    const ex = (1 - th) * dt;
    const im = th * dt;
    for (let i = 0; i < n; i++) {
      let lp = diag[i] * p[i];
      if (i > 0) lp += sub[i] * p[i - 1];
      if (i < n - 1) lp += sup[i] * p[i + 1];
      rhs[i] = p[i] + ex * lp;
      ma[i] = -im * sub[i];
      mb[i] = 1 - im * diag[i];
      mc[i] = -im * sup[i];
    }
    if (this.bc[0] === 'periodic') {
      rhs[0] += ex * this.cornerTop * p[n - 1];
      rhs[n - 1] += ex * this.cornerBottom * p[0];
      this.solveCyclic(-im * this.cornerBottom, -im * this.cornerTop);
    } else {
      this.solveTridiagonal(rhs, p);
    }
    this.t += dt;
  }

  /** Scharfetter–Gummel coefficients of face j (at lo + j h) across a gap `dist`, into this.flux. */
  private face(j: number, dist: number, tau: number): void {
    const { a, b, db } = this.proc;
    const x = this.lo + j * this.h;
    const bx = b(x, tau);
    sgFlux(a(x, tau) - bx * db(x, tau), 0.5 * bx * bx, dist, this.flux);
  }

  private assemble(tau: number): void {
    const { n, h, alpha, beta, sub, diag, sup, flux } = this;
    // Current through face j (between cells j−1 and j): J_j = α_j p_{j−1} − β_j p_j.
    for (let j = 1; j < n; j++) {
      this.face(j, h, tau);
      alpha[j] = flux.alpha;
      beta[j] = flux.beta;
    }
    const [bcL, bcR] = this.bc;
    alpha[0] = beta[0] = alpha[n] = beta[n] = 0; // reflecting / open: no current
    if (bcL === 'periodic') {
      this.face(0, h, tau); // face 0 ≡ face n, joining cell n−1 to cell 0
      alpha[0] = alpha[n] = flux.alpha;
      beta[0] = beta[n] = flux.beta;
    } else {
      // Absorbing: p = 0 on the wall, half a cell from the first centre.
      if (bcL === 'absorbing') {
        this.face(0, h / 2, tau);
        beta[0] = flux.beta;
      }
      if (bcR === 'absorbing') {
        this.face(n, h / 2, tau);
        alpha[n] = flux.alpha;
      }
    }
    // dp_i/dt = (J_i − J_{i+1}) / h
    for (let i = 0; i < n; i++) {
      sub[i] = alpha[i] / h;
      diag[i] = -(beta[i] + alpha[i + 1]) / h;
      sup[i] = beta[i + 1] / h;
    }
    this.cornerTop = bcL === 'periodic' ? alpha[0] / h : 0;
    this.cornerBottom = bcL === 'periodic' ? beta[n] / h : 0;
  }

  /** Thomas algorithm on (ma, mb, mc) with right-hand side r, writing into out. */
  private solveTridiagonal(r: Float64Array, out: Float64Array): void {
    const { n, ma, mb, mc, cp, dp } = this;
    cp[0] = mc[0] / mb[0];
    dp[0] = r[0] / mb[0];
    for (let i = 1; i < n; i++) {
      const m = mb[i] - ma[i] * cp[i - 1];
      cp[i] = mc[i] / m;
      dp[i] = (r[i] - ma[i] * dp[i - 1]) / m;
    }
    out[n - 1] = dp[n - 1];
    for (let i = n - 2; i >= 0; i--) out[i] = dp[i] - cp[i] * out[i + 1];
  }

  /**
   * Cyclic tridiagonal solve by Sherman–Morrison. `bottomLeft` is the matrix
   * entry (n−1, 0) and `topRight` is (0, n−1).
   */
  private solveCyclic(bottomLeft: number, topRight: number): void {
    const { n, mb, rhs, u, z, p } = this;
    const gamma = -mb[0];
    const b0 = mb[0];
    const bn = mb[n - 1];
    mb[0] = b0 - gamma;
    mb[n - 1] = bn - (bottomLeft * topRight) / gamma;
    this.solveTridiagonal(rhs, p);
    u.fill(0);
    u[0] = gamma;
    u[n - 1] = bottomLeft;
    this.solveTridiagonal(u, z);
    const fact = (p[0] + (topRight * p[n - 1]) / gamma) / (1 + z[0] + (topRight * z[n - 1]) / gamma);
    for (let i = 0; i < n; i++) p[i] -= fact * z[i];
    mb[0] = b0;
    mb[n - 1] = bn;
  }
}

/** Bernoulli function z / (eᶻ − 1). */
function bernoulli(z: number): number {
  if (Math.abs(z) < 1e-6) return 1 - z / 2;
  return z / Math.expm1(z);
}

/**
 * Scharfetter–Gummel coefficients for J = v p − d p′ across a gap `dist`:
 * J = α p_left − β p_right.
 */
export function sgFlux(v: number, d: number, dist: number, out: { alpha: number; beta: number }): void {
  const pe = d > 0 ? (v * dist) / d : Infinity;
  if (!Number.isFinite(pe) || Math.abs(pe) > 600) {
    // Diffusion negligible across the gap: pure upwinding.
    out.alpha = Math.max(v, 0);
    out.beta = Math.max(-v, 0);
    return;
  }
  const k = d / dist;
  out.alpha = k * bernoulli(-pe);
  out.beta = k * bernoulli(pe);
}

/** Gaussian N(x0, s²) sampled at the cell centres, normalised to unit mass on the grid. */
export function initialDensity(xc: Float64Array, h: number, x0: number, s: number): Float64Array {
  const p = new Float64Array(xc.length);
  let m = 0;
  for (let i = 0; i < xc.length; i++) {
    const z = (xc[i] - x0) / s;
    p[i] = Math.exp(-0.5 * z * z);
    m += p[i];
  }
  if (m > 0) for (let i = 0; i < p.length; i++) p[i] /= m * h;
  return p;
}
