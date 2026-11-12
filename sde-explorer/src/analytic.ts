import type { BC, Interp, Process } from './model';
import type { ExactKind } from './presets';

/** Exact transition density at one time, for an initial Gaussian N(x0, s0²). */
export interface ExactSnapshot {
  density: (x: number) => number;
  mean: number;
  variance: number;
  /** Probability still in the domain (below 1 only for absorbing walls). */
  mass: number;
  /** Exit rate −Ṡ/S (zero without absorbing walls). */
  rate: number;
}

export interface ExactModel {
  at(t: number): ExactSnapshot;
  /** Two-time covariance ⟨X(t), X(s)⟩ for t ≥ s, where known. */
  covariance?(s: number, t: number): number;
}

/**
 * Regression theorem (§3.7.4, eq. 3.7.61): when the Itô drift is linear,
 * a = A + Bx, the conditional mean obeys d⟨X⟩/dt = A + B⟨X⟩, so
 * ⟨X(t), X(s)⟩ = e^{B(t−s)} var X(s).
 */
function withRegression(model: ExactModel, B: number): ExactModel {
  return { ...model, covariance: (s, t) => Math.exp(B * (t - s)) * model.at(s).variance };
}

export interface ExactInput {
  kind: ExactKind;
  params: Record<string, number>;
  interp: Interp;
  x0: number;
  s0: number;
  domain: [number, number];
}

const SQRT2PI = Math.sqrt(2 * Math.PI);

function gaussian(mean: number, variance: number): ExactSnapshot {
  const s = Math.sqrt(variance);
  return {
    density: (x) => {
      const z = (x - mean) / s;
      return Math.exp(-0.5 * z * z) / (s * SQRT2PI);
    },
    mean,
    variance,
    mass: 1,
    rate: 0,
  };
}

/** Closed-form solution for one of the presets, or null if it doesn't apply. */
export function exactModel(input: ExactInput): ExactModel | null {
  const { kind, params: P, x0, s0 } = input;
  const s02 = s0 * s0;
  switch (kind) {
    case 'wiener':
      // §3.8.1: Gaussian with linearly growing mean and variance.
      return withRegression({ at: (t) => gaussian(x0 + P.mu * t, s02 + P.sigma ** 2 * t) }, 0);
    case 'ou': {
      // §4.4.4
      const { k, m, sigma } = P;
      const model: ExactModel = {
        at: (t) => {
          const e = Math.exp(-k * t);
          return gaussian(m + (x0 - m) * e, s02 * e * e + ((sigma * sigma) / (2 * k)) * (1 - e * e));
        },
      };
      return withRegression(model, -k);
    }
    case 'gbm': {
      // The Itô drift is μx, or (μ + σ²/2)x for a Stratonovich SDE.
      const B = input.interp === 'ito' ? P.mu : P.mu + 0.5 * P.sigma * P.sigma;
      return withRegression(gbmModel(P.mu, P.sigma, input.interp, x0, s0), B);
    }
    case 'walls':
      return wallsModel(P.mu, P.sigma, x0, s0, input.domain);
    case 'halfline':
      return halfLineModel(P.mu, P.sigma, x0, s0, input.domain);
  }
}

/**
 * dx = μx dt + σx dW (§4.4.2): ln x(t) = ln x(0) + N(c t, σ² t) with
 * c = μ − σ²/2 (Itô) or c = μ (Stratonovich). The initial Gaussian is
 * integrated over by quadrature.
 */
function gbmModel(mu: number, sigma: number, interp: Interp, x0: number, s0: number): ExactModel {
  const c = interp === 'ito' ? mu - 0.5 * sigma * sigma : mu;
  const muIto = c + 0.5 * sigma * sigma;
  const K = 81;
  const nodes: number[] = [];
  const weights: number[] = [];
  let wsum = 0;
  for (let k = 0; k < K; k++) {
    const z = -5 + (10 * k) / (K - 1);
    const xk = x0 + s0 * z;
    if (xk <= 0) continue;
    const w = Math.exp(-0.5 * z * z);
    nodes.push(xk);
    weights.push(w);
    wsum += w;
  }
  for (let k = 0; k < weights.length; k++) weights[k] /= wsum;
  const m1 = x0;
  const m2 = x0 * x0 + s0 * s0;
  return {
    at: (t) => {
      const s2 = sigma * sigma * t;
      if (s2 < 1e-12) return gaussian(x0, s0 * s0);
      const s = Math.sqrt(s2);
      const mean = m1 * Math.exp(muIto * t);
      const second = m2 * Math.exp((2 * muIto + sigma * sigma) * t);
      return {
        density: (x) => {
          if (x <= 0) return 0;
          const lx = Math.log(x);
          let sum = 0;
          for (let k = 0; k < nodes.length; k++) {
            const z = (lx - Math.log(nodes[k]) - c * t) / s;
            sum += weights[k] * Math.exp(-0.5 * z * z);
          }
          return sum / (x * s * SQRT2PI);
        },
        mean,
        variance: second - mean * mean,
        mass: 1,
        rate: 0,
      };
    },
  };
}

/**
 * Constant drift μ and noise σ between absorbing walls (§5.2.7–5.2.8).
 * Writing p = e^{κ(x−lo) − μ²t/4d} q with κ = μ/2d, d = σ²/2 turns the problem
 * into pure diffusion of q with q = 0 on the walls, solved by a sine series.
 */
function wallsModel(mu: number, sigma: number, x0: number, s0: number, domain: [number, number]): ExactModel | null {
  const [lo, hi] = domain;
  const L = hi - lo;
  const d = 0.5 * sigma * sigma;
  const kappa = mu / (2 * d);
  if (Math.abs(kappa) * L > 300) return null; // series would over/underflow
  const lambda = (mu * mu) / (4 * d);
  const N = Math.min(600, Math.ceil((8 * L) / (Math.PI * Math.max(s0, 1e-3))) + 20);
  const M = Math.max(2000, 12 * N);
  const hq = L / M;

  // Truncated, normalised initial Gaussian, and q₀ = p₀ e^{−κ(x−lo)}.
  const q0 = new Float64Array(M);
  let norm = 0;
  for (let j = 0; j < M; j++) {
    const x = lo + (j + 0.5) * hq;
    const z = (x - x0) / s0;
    const p0 = Math.exp(-0.5 * z * z);
    norm += p0 * hq;
    q0[j] = p0 * Math.exp(-kappa * (x - lo));
  }
  const kn = new Float64Array(N);
  const cn = new Float64Array(N);
  for (let n = 0; n < N; n++) {
    const k = ((n + 1) * Math.PI) / L;
    kn[n] = k;
    let s = 0;
    for (let j = 0; j < M; j++) s += q0[j] * Math.sin(k * (j + 0.5) * hq);
    cn[n] = ((2 / L) * s * hq) / norm;
  }

  return {
    at: (t) => {
      // Modes with d k² t > 50 have decayed to nothing.
      const decay = new Float64Array(N);
      let nMax = 0;
      for (let n = 0; n < N; n++) {
        const r = d * kn[n] * kn[n] * t;
        if (r > 50) break;
        decay[n] = cn[n] * Math.exp(-r);
        nMax = n + 1;
      }
      const f = (x: number) => {
        if (x <= lo || x >= hi) return 0;
        let s = 0;
        for (let n = 0; n < nMax; n++) s += decay[n] * Math.sin(kn[n] * (x - lo));
        return Math.max(0, Math.exp(kappa * (x - lo) - lambda * t) * s);
      };
      // Exit current d ∂x p out through each wall, where p = 0; cos(k_n L) = ±1.
      let gl = 0;
      let gr = 0;
      for (let n = 0; n < nMax; n++) {
        gl += decay[n] * kn[n];
        gr += decay[n] * kn[n] * (n % 2 === 0 ? -1 : 1);
      }
      const outflow = d * Math.exp(-lambda * t) * (gl - Math.exp(kappa * L) * gr);
      return snapshotByQuadrature(f, lo, hi, outflow);
    },
  };
}

/**
 * Constant drift −μ toward an absorbing wall at lo, with nothing beyond the
 * domain on the other side (the half-line). By the method of images, a path
 * started a distance y from the wall has, at distance x,
 *   p = G(x − y + μt) (1 − e^{−2xy/Dt}),   G(u) = e^{−u²/2Dt} / √(2πDt),
 * with D = σ² as in the notes (B = D). The initial Gaussian is integrated
 * over by quadrature. Conditioned on survival this tends to
 * (μ/D)² x e^{−μx/D}, the quasi-stationary density at the edge of the
 * continuum λ₀ ≤ μ²/2D.
 */
function halfLineModel(mu: number, sigma: number, x0: number, s0: number, domain: [number, number]): ExactModel {
  const [lo, hi] = domain;
  const D = sigma * sigma;
  const K = 81;
  const ys: number[] = [];
  const weights: number[] = [];
  let wsum = 0;
  for (let k = 0; k < K; k++) {
    const z = -5 + (10 * k) / (K - 1);
    const y = x0 + s0 * z - lo;
    if (y <= 0) continue;
    const w = Math.exp(-0.5 * z * z);
    ys.push(y);
    weights.push(w);
    wsum += w;
  }
  for (let k = 0; k < weights.length; k++) weights[k] /= wsum;
  return {
    at: (t) => {
      if (t < 1e-12) return gaussian(x0, s0 * s0);
      const twoDt = 2 * D * t;
      const norm = 1 / Math.sqrt(Math.PI * twoDt);
      const f = (x: number) => {
        const X = x - lo;
        if (X <= 0) return 0;
        let s = 0;
        for (let k = 0; k < ys.length; k++) {
          const u = X - ys[k] + mu * t;
          s += weights[k] * Math.exp(-(u * u) / twoDt) * -Math.expm1((-2 * X * ys[k]) / (D * t));
        }
        return s * norm;
      };
      // Exit current ½D ∂x p at the wall.
      let outflow = 0;
      for (let k = 0; k < ys.length; k++) {
        const u = mu * t - ys[k];
        outflow += (weights[k] * Math.exp(-(u * u) / twoDt) * ys[k]) / t;
      }
      return snapshotByQuadrature(f, lo, hi, outflow * norm);
    },
  };
}

/** Mass and moments of a density on [lo, hi] by midpoint quadrature. */
function snapshotByQuadrature(f: (x: number) => number, lo: number, hi: number, outflow: number): ExactSnapshot {
  const Q = 800;
  const hx = (hi - lo) / Q;
  let m0 = 0;
  let m1 = 0;
  let m2 = 0;
  for (let j = 0; j < Q; j++) {
    const x = lo + (j + 0.5) * hx;
    const p = f(x);
    m0 += p;
    m1 += p * x;
    m2 += p * x * x;
  }
  const mean = m1 / m0;
  const mass = m0 * hx;
  return { density: f, mean, variance: m2 / m0 - mean * mean, mass, rate: outflow / mass };
}

/**
 * Stationary solution of the Fokker–Planck equation on the grid (§5.2.2), or
 * null when there isn't a meaningful one (time-dependent coefficients,
 * absorbing walls, vanishing diffusion).
 *
 * Zero-current boundaries: p_s ∝ exp(∫ 2a/b²) / b², evaluated as exp ∫ v/d
 * with v = a − b ∂x b and d = ½b².
 * Periodic: p_s(x) ∝ ∫ₓ^{x+L} exp(Φ(x) − Φ(y)) / d(y) dy with Φ = ∫ v/d, the
 * constant-current solution.
 */
export function stationaryDensity(proc: Process, xc: Float64Array, h: number, bc: [BC, BC]): Float64Array | null {
  if (proc.timeDependent || bc.includes('absorbing')) return null;
  const n = xc.length;
  const v = (x: number) => {
    const bx = proc.b(x, 0);
    return proc.a(x, 0) - bx * proc.db(x, 0);
  };
  const dd = (x: number) => {
    const bx = proc.b(x, 0);
    return 0.5 * bx * bx;
  };
  const ratio = (x: number) => v(x) / dd(x);

  // Φ at the cell centres by Simpson's rule on each interval.
  const phi = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const xa = xc[0] + i * h;
    phi[i + 1] = phi[i] + (h / 6) * (ratio(xa) + 4 * ratio(xa + h / 2) + ratio(xa + h));
  }
  const logp = new Float64Array(n);

  if (bc[0] === 'periodic') {
    const phiL = phi[n]; // increase of Φ over one period
    const dy = new Float64Array(n);
    for (let j = 0; j < n; j++) dy[j] = dd(xc[j]);
    const terms = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) {
      // y runs over x_i, x_{i+1}, … one full period, wrapping round.
      let mx = -Infinity;
      for (let k = 0; k <= n; k++) {
        const j = (i + k) % n;
        const phiY = phi[j] + (i + k >= n ? phiL : 0);
        terms[k] = phi[i] - phiY - Math.log(dy[j]);
        if (terms[k] > mx) mx = terms[k];
      }
      let s = 0;
      for (let k = 0; k <= n; k++) s += (k === 0 || k === n ? 0.5 : 1) * Math.exp(terms[k] - mx);
      logp[i] = mx + Math.log(s);
    }
  } else {
    for (let i = 0; i < n; i++) logp[i] = phi[i];
  }

  let mx = -Infinity;
  for (let i = 0; i < n; i++) if (logp[i] > mx) mx = logp[i];
  const p = new Float64Array(n);
  let m = 0;
  for (let i = 0; i < n; i++) {
    p[i] = Math.exp(logp[i] - mx);
    m += p[i];
  }
  if (!Number.isFinite(m) || m <= 0) return null;
  for (let i = 0; i < n; i++) p[i] /= m * h;
  return p;
}
