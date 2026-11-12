import { lgamma } from '../special';
import { linspace, memo, simpson } from './num';
import { eigenvalues, eigenvector } from './tridiag';

/**
 * The later parts of the notes, with D = 1: the overdamped limit (toolkit),
 * the shape-invariant potentials of Part IV, and the double well of Part V.
 * Ported from make_figures.py.
 */

// ---------------------------------------------------------------- where the equation comes from

/**
 * Stationary position correlation C(t)/C(0) of the damped harmonic oscillator,
 * in units where the Smoluchowski time γ/κ = 1, against the damping ratio
 * ζ = γ/2√(mκ): ω = 2ζ and γ/2m = 2ζ².
 */
export function dampedC(t: number, zeta: number): number {
  const g2 = 2 * zeta * zeta;
  const w = 2 * zeta;
  if (Math.abs(zeta - 1) < 1e-12) return (1 + w * t) * Math.exp(-w * t);
  if (zeta > 1) {
    const O = Math.sqrt(g2 * g2 - w * w);
    return 0.5 * (1 + g2 / O) * Math.exp(-(g2 - O) * t) + 0.5 * (1 - g2 / O) * Math.exp(-(g2 + O) * t);
  }
  const w1 = Math.sqrt(w * w - g2 * g2);
  return Math.exp(-g2 * t) * (Math.cos(w1 * t) + (g2 / w1) * Math.sin(w1 * t));
}

/** Long-time decay rate of C(t), in units of the Smoluchowski rate κ/γ. */
export function dampedRate(zeta: number): number {
  return zeta >= 1 ? 2 * zeta * zeta * (1 - Math.sqrt(Math.max(0, 1 - 1 / (zeta * zeta)))) : 2 * zeta * zeta;
}

// ---------------------------------------------------------------- special functions

/** e^{−x} I_ν(x) for real ν ≥ −½ and x > 0: the series in log form, or the large-x expansion. */
export function besselIve(nu: number, x: number): number {
  if (x === 0) return nu === 0 ? 1 : nu > 0 ? 0 : Infinity;
  if (x > 40) {
    const mu = 4 * nu * nu;
    let term = 1;
    let sum = 1;
    for (let k = 1; k < 40; k++) {
      const next = (-term * (mu - (2 * k - 1) ** 2)) / (k * 8 * x);
      if (Math.abs(next) > Math.abs(term)) break;
      term = next;
      sum += term;
      if (Math.abs(term) < 1e-17 * Math.abs(sum)) break;
    }
    return sum / Math.sqrt(2 * Math.PI * x);
  }
  const lh = Math.log(x / 2);
  let sum = 0;
  for (let k = 0; k < 400; k++) {
    const term = Math.exp((2 * k + nu) * lh - lgamma(k + 1) - lgamma(k + nu + 1) - x);
    sum += term;
    if (k > x && term < 1e-17 * sum) break;
  }
  return sum;
}

/** The beta function B(a, b). */
export const betaFn = (a: number, b: number) => Math.exp(lgamma(a) + lgamma(b) - lgamma(a + b));

/** Jacobi P_n^(α,α)(y), by the three-term recurrence. */
export function jacobiSym(n: number, alpha: number, y: number): number {
  if (n === 0) return 1;
  let p0 = 1;
  let p1 = (alpha + 1) * y;
  for (let k = 2; k <= n; k++) {
    const ab = 2 * alpha;
    const c = 2 * k + ab;
    const p2 = ((c - 1) * c * (c - 2) * y * p1 - 2 * (k + alpha - 1) * (k + alpha - 1) * c * p0) / (2 * k * (k + ab) * (c - 2));
    p0 = p1;
    p1 = p2;
  }
  return p1;
}

/** Gegenbauer C_n^(λ)(y) for every n ≤ N, by the three-term recurrence. */
export function gegenbauerAll(N: number, lambda: number, y: number): Float64Array {
  const out = new Float64Array(N + 1);
  out[0] = 1;
  if (N > 0) out[1] = 2 * lambda * y;
  for (let n = 2; n <= N; n++) out[n] = (2 * y * (n + lambda - 1) * out[n - 1] - (n + 2 * lambda - 2) * out[n - 2]) / n;
  return out;
}

// ---------------------------------------------------------------- Part IV: radial motion

/** Radius of a d-dimensional free (k = 0) or OU motion from r₀: (r/σ²)(r/m)^μ e^{−(r−m)²/2σ²} Ĩ_μ(rm/σ²), μ = d/2 − 1. */
export function radialP(r: number, t: number, r0: number, d: number, k = 0): number {
  const mu = d / 2 - 1;
  const [m, s2] = k === 0 ? [r0, t] : [r0 * Math.exp(-k * t), (0.5 / k) * -Math.expm1(-2 * k * t)];
  return (r / s2) * (r / m) ** mu * Math.exp(-((r - m) ** 2) / (2 * s2)) * besselIve(mu, (r * m) / s2);
}

/** Chi density: the stationary radius of a d-dimensional OU motion (ℓ = 1). */
export function radialPs(r: number, d: number): number {
  return Math.exp(Math.log(2) + (d - 1) * Math.log(r) - r * r - lgamma(d / 2));
}

/** Probability that free d-dimensional motion ever comes within a of the origin, from r₀ > a. */
export const radialReach = (a: number, r0: number, d: number) => (d <= 2 ? 1 : (a / r0) ** (d - 2));

/** The same as a splitting probability: reach a before R (Gardiner 5.2.190, corrected), with ψ = r^{d−1}. */
export function radialReachBefore(a: number, r0: number, d: number, R = 1e6): number {
  if (Math.abs(d - 2) < 1e-12) return Math.log(R / r0) / Math.log(R / a);
  const g = (r: number) => r ** (2 - d);
  return (g(r0) - g(R)) / (g(a) - g(R));
}

// ---------------------------------------------------------------- Part IV: the soft V, U = a ln cosh(x/w)

/** 1/cosh u without overflow. */
export function sech(u: number): number {
  const e = Math.exp(-Math.abs(u));
  return (2 * e) / (1 + e * e);
}

export const lncoshPs = (x: number, s: number, w: number) => sech(x / w) ** (2 * s) / (w * betaFn(0.5, s));

/** Discrete rates (D/2w²) n(2s − n), n < s. */
export function lncoshLevels(s: number, w: number): number[] {
  const out: number[] = [];
  for (let n = 0; n < Math.ceil(s); n++) out.push((n * (2 * s - n)) / (2 * w * w));
  return out;
}

/** Forward bound state n < s: sech^{2s−n}(x/w) P_n^(s−n, s−n)(tanh(x/w)), unnormalised. */
export const lncoshP = (x: number, n: number, s: number, w: number) => sech(x / w) ** (2 * s - n) * jacobiSym(n, s - n, Math.tanh(x / w));

// ---------------------------------------------------------------- Part IV: a well with its own walls, U = −a ln cos(x/w)

export const lncosPs = (x: number, s: number, w: number) => Math.cos(x / w) ** (2 * s) / (w * betaFn(0.5, s + 0.5));

export const lncosLam = (n: number, s: number, w: number) => (n * (n + 2 * s)) / (2 * w * w);

/**
 * The Gegenbauer propagator pₛ(x) Σ C_n^(s)(sin x/w) C_n^(s)(sin x₀/w) e^{−λ_n t}/h_n,
 * with h_n = ∫pₛ C_n² dx = π2^{1−2s}Γ(n+2s)/[n!(n+s)Γ(s)²] / B(½, s+½) in closed form.
 */
export function lncosP(xs: ArrayLike<number>, t: number, x0: number, s: number, w: number): Float64Array {
  let N = 2;
  while (N < 400 && lncosLam(N, s, w) * t < 40) N++;
  const C0 = gegenbauerAll(N, s, Math.sin(x0 / w));
  const B = betaFn(0.5, s + 0.5);
  const coef = new Float64Array(N + 1);
  for (let n = 0; n <= N; n++) {
    const hn = Math.exp(Math.log(Math.PI) + (1 - 2 * s) * Math.LN2 + lgamma(n + 2 * s) - lgamma(n + 1) - 2 * lgamma(s)) / ((n + s) * B);
    coef[n] = (C0[n] / hn) * Math.exp(-lncosLam(n, s, w) * t);
  }
  const out = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    const C = gegenbauerAll(N, s, Math.sin(xs[i] / w));
    let sum = 0;
    for (let n = 0; n <= N; n++) sum += coef[n] * C[n];
    out[i] = lncosPs(xs[i], s, w) * sum;
  }
  return out;
}

// ---------------------------------------------------------------- Part V: the double well

/**
 * Symmetrised generator of a jump process on n cells of [a, b] with rates
 * (D/2h²) e^{−(U_j − U_i)/D} between neighbours (jump_spectrum in the notes).
 * Detailed balance with π ∝ e^{−2U/D} is exact, so the zero mode is exact and
 * exponentially small rates come out with relative accuracy.
 */
export function jumpOperator(U: (x: number) => number, a: number, b: number, n: number): { x: Float64Array; d: Float64Array; e: Float64Array } {
  const h = (b - a) / n;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = a + h * (i + 0.5);
  const u = x.map(U);
  const c = 1 / (2 * h * h);
  const d = new Float64Array(n);
  for (let i = 0; i < n - 1; i++) {
    d[i] += c * Math.exp(-(u[i + 1] - u[i]));
    d[i + 1] += c * Math.exp(-(u[i] - u[i + 1]));
  }
  return { x, d, e: new Float64Array(n - 1).fill(-c) };
}

/** Quartic double well with barrier ΔU = β/2 between minima at ±a (D = 1). */
export const dwU = (beta: number, a = 1) => (x: number) => (beta / 2) * ((x / a) ** 2 - 1) ** 2;

/** U″(a) at the minima and |U″(0)| at the barrier top. */
export const dwCurvatures = (beta: number, a = 1): [number, number] => [(4 * beta) / (a * a), (2 * beta) / (a * a)];

/** Kramers: λ₁ = 2r, r = √(U″_a |U″_b|)/2π e^{−2ΔU/D} (G 5.2.174 with D → D/2). */
export function dwKramers(beta: number, a = 1): number {
  const [ka, kb] = dwCurvatures(beta, a);
  return ((2 * Math.sqrt(ka * kb)) / (2 * Math.PI)) * Math.exp(-beta);
}

/** Gardiner (9.2.18) with D → D/2: λ₁ = (D/2)/(n_a n_c ∫ₐᶜ dx/pₛ), here n_a = n_c = Z/2. */
export function dwGardiner9218(beta: number, a = 1): number {
  const U = dwU(beta, a);
  const w = (x: number) => Math.exp(-2 * (U(x) - U(0)));
  const Z = simpson(w, -4 * a, 4 * a, 40000);
  const I = simpson((x) => 1 / w(x), -a, a, 20000);
  return 0.5 / (0.25 * Z * I);
}

/** The slowest rates λ₀ = 0, λ₁, λ₂, … on [−3a, 3a]. */
export const dwLevels = memo((beta: number, count: number, n = 3000) => {
  const op = jumpOperator(dwU(beta), -3, 3, n);
  return eigenvalues(op.d, op.e, 0, count - 1);
}, 512);

/** Every mode on a coarser grid, for the time evolution from a point. */
const dwAllModes = memo((beta: number, n = 400) => {
  const op = jumpOperator(dwU(beta), -3, 3, n);
  const lam = eigenvalues(op.d, op.e, 0, n - 1);
  const vecs: Float64Array[] = [];
  lam.forEach((l) => {
    const near = vecs.filter((_, j) => Math.abs(lam[j] - l) < 1e-6 * (1 + Math.abs(l)));
    vecs.push(eigenvector(op.d, op.e, l, near));
  });
  return { x: op.x, lam, vecs };
}, 16);

/** Probability on the right of the barrier at times t, from a point start at the left minimum. */
export function dwPopulation(beta: number, ts: ArrayLike<number>): Float64Array {
  const { x, lam, vecs } = dwAllModes(beta);
  const U = dwU(beta);
  const sq = x.map((y) => Math.exp(-(U(y) - U(1)))); // √π up to a constant
  let j = 0;
  for (let i = 1; i < x.length; i++) if (Math.abs(x[i] + 1) < Math.abs(x[j] + 1)) j = i;
  const coef = vecs.map((v) => {
    let right = 0;
    for (let i = 0; i < x.length; i++) if (x[i] > 0) right += sq[i] * v[i];
    return (right * v[j]) / sq[j];
  });
  return Float64Array.from(ts, (t) => coef.reduce((s, c, k) => s + c * Math.exp(-Math.max(0, lam[k]) * t), 0));
}

/** The slowest backward eigenfunction Q₁, scaled to 1 at x = a. */
export const dwQ1 = memo((beta: number) => {
  const op = jumpOperator(dwU(beta), -3, 3, 3000);
  const [, l1] = eigenvalues(op.d, op.e, 0, 1);
  const v = eigenvector(op.d, op.e, l1);
  const U = dwU(beta);
  const Q = v.map((y, i) => y / Math.exp(-(U(op.x[i]) - U(1))));
  let j = 0;
  for (let i = 1; i < op.x.length; i++) if (Math.abs(op.x[i] - 1) < Math.abs(op.x[j] - 1)) j = i;
  const scale = Q[j];
  return { x: op.x, Q: Q.map((q) => q / scale) };
}, 32);

/** π_c(x): probability of reaching +a before −a (corrected G 5.2.190 / 9.1.10). */
export const dwCommittor = memo((beta: number) => {
  const U = dwU(beta);
  const y = linspace(-1, 1, 4001);
  const f = y.map((v) => Math.exp(2 * (U(v) - U(0))));
  const cum = new Float64Array(y.length);
  for (let i = 1; i < y.length; i++) cum[i] = cum[i - 1] + 0.5 * (f[i] + f[i - 1]) * (y[i] - y[i - 1]);
  const total = cum[cum.length - 1];
  return (x: number) => {
    if (x <= -1) return 0;
    if (x >= 1) return 1;
    const u = ((x + 1) / 2) * (y.length - 1);
    const i = Math.min(y.length - 2, Math.floor(u));
    return (cum[i] + (u - i) * (cum[i + 1] - cum[i])) / total;
  };
}, 32);

/** Symmetrised potential V = U′²/2D − U″/2 from U′ and U″. */
export const vFrom = (U1: (x: number) => number, U2: (x: number) => number) => (x: number) => (U1(x) * U1(x)) / 2 - U2(x) / 2;
