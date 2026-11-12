import { kummerNeg, lgamma } from '../special';
import { Rng } from '../rng';
import { memo, simpson } from './num';
import { besselIve } from './potentials';

/**
 * Part VI of the notes: interfaces and singular boundaries. D = k = 1 where
 * they appear, so ℓ² = D/k = 1. Ported from make_figures.py.
 */

// ---------------------------------------------------------------- a step in the diffusion coefficient

/** The density jump p(0⁺)/p(0⁻) for the drift χB′/2 of a smoothed step: χ = 1 thermal, ½ Stratonovich, 0 Itô. */
export const ifaceK = (chi: number, Dm: number, Dp = 1) => (Dm / Dp) ** (1 - chi);

/** Weight of the image; the transmitted Gaussian has 1 − r. η = K√(D₊/D₋). */
export function ifaceR(K: number, Dm: number, Dp = 1): number {
  const eta = K * Math.sqrt(Dp / Dm);
  return (eta - 1) / (eta + 1);
}

/** Propagator from x₀ > 0: incident and image Gaussians (variance D₊t) on the right, the transmitted one on the left. */
export function ifaceP(x: number, t: number, x0: number, K: number, Dm: number, Dp = 1): number {
  const r = ifaceR(K, Dm, Dp);
  const g = (y: number, Dd: number) => Math.exp(-(y * y) / (2 * Dd * t)) / Math.sqrt(2 * Math.PI * Dd * t);
  return x > 0 ? g(x - x0, Dp) + r * g(x + x0, Dp) : (1 - r) * g(x - x0 * Math.sqrt(Dm / Dp), Dm);
}

/** π₊(x₀) on (−a, a), both walls absorbing: piecewise linear, continuous, with KD₊π′(0⁺) = D₋π′(0⁻). */
export function ifaceSplit(x0: number, K: number, Dm: number, Dp = 1, a = 1): number {
  const sp = 1 / (a + (K * Dp * a) / Dm);
  return x0 > 0 ? 1 - sp * (a - x0) : ((K * Dp * sp) / Dm) * (x0 + a);
}

/** Mean exit time from (−a, a): −x²/D± + slope·x + T(0), continuous, with KD₊T′(0⁺) = D₋T′(0⁻). */
export function ifaceT(x0: number, K: number, Dm: number, Dp = 1, a = 1): number {
  const sp = (a * a * (1 / Dp - 1 / Dm)) / (a + (K * Dp * a) / Dm);
  const T0 = (a * a) / Dp - sp * a;
  return x0 > 0 ? -(x0 * x0) / Dp + sp * x0 + T0 : -(x0 * x0) / Dm + ((K * Dp) / Dm) * sp * x0 + T0;
}

// ---------------------------------------------------------------- where B vanishes: the square-root process

/**
 * Regularised lower incomplete gamma P(a, z) = γ(a, z)/Γ(a), a > 0: the series
 * for z < a + 1, Lentz's continued fraction for Q = 1 − P otherwise.
 */
export function gammainc(a: number, z: number): number {
  if (z <= 0) return 0;
  if (!Number.isFinite(z)) return 1;
  const lead = a * Math.log(z) - z - lgamma(a);
  if (z < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let n = 1; n < 1000; n++) {
      term *= z / (a + n);
      sum += term;
      if (term < 1e-17 * sum) break;
    }
    return Math.min(1, sum * Math.exp(lead));
  }
  const tiny = 1e-300;
  let b = z + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 1000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-16) break;
  }
  return Math.max(0, 1 - Math.exp(lead) * h);
}

/** ȳ_t = y₀e^{−2t} and σ_t² = ½(1 − e^{−2t}), the per-component OU variance. */
export function sqrtScales(t: number, y0: number): [number, number] {
  return [y0 * Math.exp(-2 * t), -0.5 * Math.expm1(-2 * t)];
}

/** p₊ (I_μ: reflected, or the only solution for d ≥ 2) or p₋ (I₋μ: absorbed at 0), μ = d/2 − 1. */
export function sqrtP(y: number, t: number, y0: number, d: number, sign: 1 | -1 = 1): number {
  const mu = d / 2 - 1;
  const [m, s2] = sqrtScales(t, y0);
  return ((y / m) ** (mu / 2) * Math.exp(-((Math.sqrt(y) - Math.sqrt(m)) ** 2) / (2 * s2)) * besselIve(sign * mu, Math.sqrt(y * m) / s2)) / (2 * s2);
}

export const sqrtZ = (t: number, y0: number) => y0 / Math.expm1(2 * t);

/** Survival of the absorbed process, P(1 − d/2, z_t); 1 for d ≥ 2, where the origin is never reached. */
export const sqrtS = (t: number, y0: number, d: number) => (d >= 2 ? 1 : gammainc(1 - d / 2, sqrtZ(t, y0)));

/** QSD with decay rate λ in (0, (2 − d)]: M(1 + λ/2, 2 − d/2, −y), normalised by ∫M(a, b, −z) dz = (b − 1)/(a − 1). */
export function sqrtQsd(y: number, lam: number, d: number): number {
  const a = 1 + lam / 2;
  const b = 2 - d / 2;
  return (kummerNeg(a, b, y) * (a - 1)) / (b - 1);
}

/** Mean time to reach 0 from y₀, ∫S dt (finite for d < 2), with t = ln(1 + 1/u)/2 so that z = y₀u. */
export const sqrtT = memo((y0: number, d: number): number => {
  if (d >= 2) return Infinity;
  const a = 1 - d / 2;
  // ∫₀^∞ S dt = ½ ∫₀^∞ P(a, y₀u) du/(u(1 + u)); split at u = 1 and map each half to a finite interval.
  const f = (u: number) => gammainc(a, y0 * u) / (u * (1 + u));
  const lower = simpson((s) => f(Math.exp(s)) * Math.exp(s), -60, 0, 4000); // u = e^s on (0, 1]
  const upper = simpson((w) => (w <= 0 ? 1 : f(1 / w) / (w * w)), 0, 1, 4000); // u = 1/w on [1, ∞); → P(a, ∞) = 1
  return 0.5 * (lower + upper);
}, 128);

/** Poisson(λ) by counting unit-rate exponential arrivals in [0, λ]: exact, O(λ). */
function poisson(rng: Rng, lam: number): number {
  let n = 0;
  let s = -Math.log(1 - rng.next());
  while (s < lam) {
    n++;
    s -= Math.log(1 - rng.next());
  }
  return n;
}

/** Gamma(n, scale) for integer n ≥ 1, as a sum of exponentials. */
function gammaInt(rng: Rng, n: number, scale: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) s -= Math.log(1 - rng.next());
  return s * scale;
}

/**
 * Paths for the figure from y₀ = 1 on kt ∈ [0, 2.5]: d = 3 and d = 1 as squared
 * radii of exact OU steps, the first zero of the d = 1 path (where absorption
 * would stop it), and d = 0 by exact branching steps (a Poisson number of
 * gamma shapes). Not the notes' random stream: the notes' path is numpy's.
 */
export const sqrtPaths = memo((seed: number) => {
  const T = 2.5;
  const dt = 0.002;
  const n = Math.round(T / dt);
  const rng = new Rng(seed);
  const t = new Float64Array(n + 1);
  const y3 = new Float64Array(n + 1);
  const y1 = new Float64Array(n + 1);
  const y0b = new Float64Array(n + 1);
  const a = Math.exp(-dt);
  const s = Math.sqrt(-0.5 * Math.expm1(-2 * dt));
  const x = [1, 0, 0, 1];
  y3[0] = y1[0] = y0b[0] = 1;
  let tHit = Infinity;
  const [mFac, s2] = sqrtScales(dt, 1);
  for (let i = 1; i <= n; i++) {
    t[i] = i * dt;
    const prev = x[3];
    for (let j = 0; j < 4; j++) x[j] = a * x[j] + s * rng.normal();
    y3[i] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    y1[i] = x[3] * x[3];
    if (tHit === Infinity && Math.sign(x[3]) !== Math.sign(prev)) tHit = t[i];
    const yb = y0b[i - 1];
    const N = yb > 0 ? poisson(rng, (yb * mFac) / (2 * s2)) : 0;
    y0b[i] = N > 0 ? gammaInt(rng, N, 2 * s2) : 0;
  }
  return { t, y3, y1, tHit, y0b };
}, 8);

// ---------------------------------------------------------------- a current through infinity: the QIF neuron

/** Mean period 1/J_s = √(2π/D) ∫₀^∞ u^{−1/2} exp[−(2/D)(u³/12 + Iu)] du, with u = w². */
export const qifT = memo((I: number, D: number): number => {
  const c2 = Math.max(0, -I);
  // Far enough that (2/D)(w⁶/12 + Iw²) > 60 beyond W.
  let W = Math.max((360 * D) ** (1 / 6), (24 * c2) ** 0.25 + 1);
  while ((2 / D) * (W ** 6 / 12 + I * W * W) < 60) W *= 1.2;
  const f = (w: number) => 2 * Math.exp(-(2 / D) * (w ** 6 / 12 + I * w * w));
  return Math.sqrt((2 * Math.PI) / D) * simpson(f, 0, W, 4000);
}, 512);

const QIF_X = 30;
const QIF_N = 30001;

/**
 * inner(x) = ∫ₓ^∞ e^{φ(x)−φ(y)} dy on a grid over [−X, X], by the stable
 * backward recursion inner(xᵢ) = h(1 − e^{−Δφ})/Δφ + e^{−Δφ} inner(xᵢ₊₁),
 * started from the far-field value 1/φ′(X).
 */
const qifInner = memo((I: number, D: number): Float64Array => {
  const h = (2 * QIF_X) / (QIF_N - 1);
  const phi = (x: number) => (2 / D) * (x ** 3 / 3 + I * x);
  const inner = new Float64Array(QIF_N);
  inner[QIF_N - 1] = D / (2 * (QIF_X * QIF_X + I));
  for (let i = QIF_N - 2; i >= 0; i--) {
    const x = -QIF_X + i * h;
    const dphi = phi(x + h) - phi(x);
    const seg = Math.abs(dphi) > 1e-12 ? (-h * Math.expm1(-dphi)) / dphi : h;
    inner[i] = seg + Math.exp(-dphi) * inner[i + 1];
  }
  return inner;
}, 16);

/** p_s(x) = (2J_s/D) ∫ₓ^∞ e^{φ(x)−φ(y)} dy, φ = (2/D)(x³/3 + Ix); beyond ±30 the far field J_s/(x² + I). */
export function qifPs(x: number, I: number, D: number): number {
  const J = 1 / qifT(I, D);
  if (Math.abs(x) >= QIF_X) return J / (x * x + I);
  const inner = qifInner(I, D);
  const u = ((x + QIF_X) / (2 * QIF_X)) * (QIF_N - 1);
  const i = Math.min(QIF_N - 2, Math.floor(u));
  const f = u - i;
  return ((2 * J) / D) * ((1 - f) * inner[i] + f * inner[i + 1]);
}

/** The same density in θ = 2 arctan x: p_θ = p_x (1 + x²)/2. */
export function qifPsTheta(theta: number, I: number, D: number): number {
  const x = Math.tan(theta / 2);
  return Math.abs(theta) >= Math.PI ? 0.5 / qifT(I, D) : (qifPs(x, I, D) * (1 + x * x)) / 2;
}
