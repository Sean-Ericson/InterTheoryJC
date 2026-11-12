import { erfc, erfcx, logNormalCdf } from '../special';
import { flux0, heat, interval } from './free';
import { brent, cumtrapz, geomspace, interp, linspace, memo, roots, trapz } from './num';

/**
 * Part II of the notes: constant drift, A = v and B = D. With D = 1 and, on the
 * interval, L = 1, v stands for vL/D and a = v/D = v. The gauge
 * p = e^{a(x−x₀) − v²t/2D} u carries every result over from free diffusion:
 * an absorbing wall stays u = 0, a reflecting one becomes u′ = a u.
 */

const coth = (z: number) => 1 / Math.tanh(z);

// ---------------------------------------------------------------- interval, reflecting walls

export function drStationary(x: number, v: number): number {
  const a = v;
  if (Math.abs(a) < 1e-12) return 1;
  if (a > 0) return (2 * a * Math.exp(2 * a * (x - 1))) / -Math.expm1(-2 * a);
  return (-2 * a * Math.exp(2 * a * x)) / -Math.expm1(2 * a);
}

export function drMeanS(v: number): number {
  const a = v;
  return Math.abs(a) < 1e-12 ? 0.5 : 1 / -Math.expm1(-2 * a) - 1 / (2 * a);
}

const drPhi = (y: number, k: number, a: number) => Math.cos(k * y) + (a / k) * Math.sin(k * y);

/** Modes needed at time t: e^{−½k²t} below 1e−17. */
const modeCount = (t: number) => Math.min(2000, Math.ceil(Math.sqrt(80 / Math.max(t, 1e-6)) / Math.PI) + 3);

/** p(x, t | x₀, 0) with reflecting walls: pₛ plus the gauged cosine–sine modes. */
export function drDensity(x: number, t: number, x0: number, v: number): number {
  const a = v;
  let s = 0;
  for (let n = 1, N = modeCount(t); n <= N; n++) {
    const k = n * Math.PI;
    s += (drPhi(x, k, a) * drPhi(x0, k, a)) / (1 + (a / k) ** 2) * Math.exp(-0.5 * k * k * t);
  }
  return drStationary(x, v) + 2 * Math.exp(a * (x - x0) - (v * v * t) / 2) * s;
}

/**
 * ⟨x(t)⟩ with reflecting walls. The overlaps cₖ = ∫₀¹ x e^{ax} φₖ dx are done
 * in closed form: ∫₀¹ x e^{zx} dx = e^z(1/z − 1/z²) + 1/z² with z = a + ik.
 */
export function drMean(t: number, x0: number, v: number): number {
  const a = v;
  let s = 0;
  for (let n = 1, N = Math.max(60, modeCount(t)); n <= N; n++) {
    const k = n * Math.PI;
    // 1/z and 1/z² for z = a + ik
    const d = a * a + k * k;
    const ir = a / d;
    const ii = -k / d;
    const i2r = ir * ir - ii * ii;
    const i2i = 2 * ir * ii;
    const ea = Math.exp(a);
    const er = ea * Math.cos(k);
    const ei = ea * Math.sin(k);
    const br = ir - i2r;
    const bi = ii - i2i;
    const Ir = er * br - ei * bi + i2r;
    const Ii = er * bi + ei * br + i2i;
    const c = Ir + (a / k) * Ii;
    s += ((c * drPhi(x0, k, a)) / (1 + (a / k) ** 2)) * Math.exp(-(v * v / 2 + 0.5 * k * k) * t);
  }
  return drMeanS(v) + 2 * Math.exp(-a * x0) * s;
}

// ---------------------------------------------------------------- interval, absorbing walls

export function daDensity(x: number, t: number, x0: number, v: number): number {
  return Math.exp(v * (x - x0) - (v * v * t) / 2) * interval(x, t, x0, 'absorbing');
}

export function daSurvival(t: number, x0: number, v: number): number {
  const a = v;
  let s = 0;
  for (let n = 1, N = modeCount(t) + 4; n <= N; n++) {
    const k = n * Math.PI;
    s += (Math.sin(k * x0) * k * (1 - (n % 2 ? -1 : 1) * Math.exp(a))) / (a * a + k * k) * Math.exp(-0.5 * k * k * t);
  }
  return 2 * Math.exp(-a * x0 - (v * v * t) / 2) * s;
}

export const daFlux0 = (t: number, x0: number, v: number) => Math.exp(-v * x0 - (v * v * t) / 2) * flux0(t, x0);
export const daFluxL = (t: number, x0: number, v: number) => Math.exp(v * (1 - x0) - (v * v * t) / 2) * flux0(t, 1 - x0);

export function daPiL(x0: number, v: number): number {
  return Math.abs(v) < 1e-12 ? x0 : Math.expm1(-2 * v * x0) / Math.expm1(-2 * v);
}

export function daT(x0: number, v: number): number {
  return Math.abs(v) < 1e-12 ? x0 * (1 - x0) : (daPiL(x0, v) - x0) / v;
}

export function daTL(x0: number, v: number): number {
  if (Math.abs(v) < 1e-12) return (1 - x0 * x0) / 3;
  if (x0 < 1e-12) return (coth(v) - 1 / v) / v;
  return (coth(v) - x0 * coth(v * x0)) / v;
}

export const daT0 = (x0: number, v: number) => daTL(1 - x0, -v);

export function daQsd(x: number, v: number): number {
  const a = v;
  const k = Math.PI;
  return (Math.exp(a * x) * Math.sin(k * x) * (a * a + k * k)) / (k * (1 + Math.exp(a)));
}

/** Slowest rate with two walls of the same type (absorbing; the reflecting gap is the same). */
export const daLam1 = (v: number) => (v * v) / 2 + (Math.PI * Math.PI) / 2;

// ---------------------------------------------------------------- interval, reflecting at 0, absorbing at L

export interface MixedMode {
  lambda: number;
  /** u(y): cos ky + (a/k) sin ky, or cosh qy + (a/q) sinh qy for the bound state. */
  u(y: number): number;
}

/**
 * Lowest modes, ascending. Real k solve k cos k + a sin k = 0; for a < −1 the
 * ground state is the bound state with q cosh q + a sinh q = 0 (G 5.2.5).
 */
export const dmModes = memo((v: number, count: number): MixedMode[] => {
  const a = v;
  const out: MixedMode[] = [];
  if (a < -1) {
    const q = brent((q) => q * Math.cosh(q) + a * Math.sinh(q), 1e-9, Math.abs(a));
    out.push({ lambda: (v * v) / 2 - 0.5 * q * q, u: (y) => Math.cosh(q * y) + (a / q) * Math.sinh(q * y) });
  }
  const g = (k: number) => k * Math.cos(k) + a * Math.sin(k);
  const ks = roots(g, linspace(1e-9, (count + 2) * Math.PI, 60 * (count + 2)), count);
  for (const k of ks) out.push({ lambda: (v * v) / 2 + 0.5 * k * k, u: (y) => drPhi(y, k, a) });
  return out.slice(0, count);
}, 512);

/** The quasi-stationary density e^{ax} u₀(x), normalised. */
export const dmQsd = memo((v: number): ((x: number) => number) => {
  const { u } = dmModes(v, 1)[0];
  const xs = linspace(0, 1, 4001);
  const norm = trapz(xs.map((x) => Math.exp(v * x) * u(x)), xs);
  return (x: number) => (Math.exp(v * x) * u(x)) / norm;
});

export function dmT(x0: number, v: number): number {
  const a = v;
  if (Math.abs(a) < 1e-12) return 1 - x0 * x0;
  return (1 - x0 + (Math.exp(-2 * a) - Math.exp(-2 * a * x0)) / (2 * a)) / v;
}

// ---------------------------------------------------------------- half-line, reflecting at 0

/** The Robin image (with D = 1): Gardiner's reflecting propagator for constant drift. */
export function dhrDensity(x: number, t: number, x0: number, v: number): number {
  const a = v;
  const s = Math.sqrt(2 * t);
  const z = (x + x0 + v * t) / s;
  const third = z > 0 ? erfcx(z) * Math.exp(2 * a * x - z * z) : Math.exp(2 * a * x) * erfc(z);
  return heat(x - x0 - v * t, t) + Math.exp(-2 * a * x0) * heat(x + x0 - v * t, t) - a * third;
}

/**
 * ⟨x(t)⟩ on a time grid. With J(0) = 0, d⟨x⟩/dt = ∫₀^∞ J dx = v + ½ p(0, t),
 * so ⟨x(t)⟩ = x₀ + vt + ½∫₀ᵗ p(0, s) ds, integrated on a fine log grid.
 */
export const dhrMean = memo((x0: number, v: number, tMax: number): { t: Float64Array; mean: Float64Array } => {
  const ts = geomspace(1e-5 * x0 * x0, tMax, 4000);
  const p0 = ts.map((t) => dhrDensity(0, t, x0, v));
  const I = cumtrapz(p0, ts);
  // The integrand vanishes like e^{−x₀²/2s} below the first grid point.
  return { t: ts, mean: ts.map((t, i) => x0 + v * t + 0.5 * I[i]) };
});

export function dhrMeanAt(t: number, x0: number, v: number, tMax: number): number {
  const m = dhrMean(x0, v, tMax);
  return interp(t, m.t, m.mean);
}

// ---------------------------------------------------------------- half-line, absorbing at 0

export function dhaDensity(x: number, t: number, x0: number, v: number): number {
  return heat(x - x0 - v * t, t) - Math.exp(-2 * v * x0) * heat(x + x0 - v * t, t);
}

export function dhaSurvival(t: number, x0: number, v: number): number {
  const s = Math.sqrt(2 * t);
  return 0.5 * (erfc(-(x0 + v * t) / s) - Math.exp(-2 * v * x0) * erfc((x0 - v * t) / s));
}

/** First-passage density to 0 from x₀ > 0 with drift v (v < 0 is toward 0). */
export function inverseGaussian(t: number, x0: number, v: number): number {
  return (x0 / Math.sqrt(2 * Math.PI * t ** 3)) * Math.exp(-((x0 + v * t) ** 2) / (2 * t));
}

/** Quasi-stationary density with decay rate 0 < λ ≤ v²/2 (v < 0). */
export function dhaQsd(x: number, lambda: number, v: number): number {
  const disc = v * v - 2 * lambda;
  if (disc < 1e-12) return v * v * x * Math.exp(v * x);
  const rp = v + Math.sqrt(disc);
  const rm = v - Math.sqrt(disc);
  return (Math.exp(rp * x) - Math.exp(rm * x)) / (1 / rm - 1 / rp);
}

/** Decay rate selected by an initial tail e^{−βx}: βℓ(2 − βℓ)/2τ, up to the edge v²/2. */
export function dhaLamOfTail(beta: number, v: number): number {
  const b = Math.min(beta / Math.abs(v), 1);
  return (b * (2 - b) * v * v) / 2;
}

/**
 * log p(x, t) for v < 0 from p₀ = β₀e^{−β₀y}: the gauge carries the free
 * half-line over, and the Gaussian integrals over the gauged start are done
 * in closed form.
 */
export function dhaLogPExp(x: number, t: number, beta0: number, v: number): number {
  const a = v;
  const g = -a - beta0;
  const sd = Math.sqrt(t);
  const A1 = g * x + 0.5 * g * g * t + logNormalCdf((x + g * t) / sd);
  const A2 = -g * x + 0.5 * g * g * t + logNormalCdf((g * t - x) / sd);
  return a * x - (v * v * t) / 2 + Math.log(beta0) + A1 + Math.log(-Math.expm1(A2 - A1));
}

/** Conditioned density q and log S from the exponential start, on a grid x starting at 0. */
export function dhaConditionedExp(xs: Float64Array, t: number, beta0: number, v: number): { q: Float64Array; logS: number } {
  const lp = xs.map((x) => dhaLogPExp(x, t, beta0, v));
  let m = -Infinity;
  for (const l of lp) if (l > m) m = l;
  const e = lp.map((l) => (Number.isFinite(l) ? Math.exp(l - m) : 0));
  const Z = trapz(e, xs);
  return { q: e.map((y) => y / Z), logS: Math.log(Z) + m };
}
