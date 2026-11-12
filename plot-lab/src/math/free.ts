import { erf, erfc } from '../special';

/**
 * Part I of the notes: free diffusion, A = 0 and B = D, so ∂ₜp = ½D ∂ₓ²p.
 * Ported from make_figures.py with D = 1: lengths are in units of L (or x₀ on
 * the half-line) and times in L²/D (or x₀²/D), the units of the figures.
 *
 * Image sums and mode sums are truncated where their terms drop below double
 * precision, rather than at the fixed counts of the script.
 */

const SQRT_2PI = Math.sqrt(2 * Math.PI);

/** Free kernel of ∂ₜp = ½∂ₓ²p: a Gaussian of variance t. */
export function heat(y: number, t: number): number {
  return Math.exp(-(y * y) / (2 * t)) / Math.sqrt(2 * Math.PI * t);
}

/** Images m = −M..M needed on a period 2L at time t. */
function imageCount(t: number, period: number): number {
  return Math.ceil((9 * Math.sqrt(t) + 2 * period) / period);
}

/** Modes needed for e^{−½(nπ/L)²t} to fall below 1e−17 (L = 1). */
function modeCount(t: number, k1 = Math.PI): number {
  return Math.ceil(Math.sqrt(80 / t) / k1) + 2;
}

export type Walls = 'reflecting' | 'absorbing' | 'mixed';

/** p(x, t | x₀, 0) on (0, 1): images at early times, modes at late (G 5.2.100, 5.2.103). */
export function interval(x: number, t: number, x0: number, walls: Walls): number {
  if (walls === 'mixed') return t < 0.05 ? mixImages(x, t, x0) : mixModes(x, t, x0);
  const sign = walls === 'reflecting' ? 1 : -1;
  if (t < 0.05) {
    const M = imageCount(t, 2);
    let s = 0;
    for (let m = -M; m <= M; m++) s += heat(x - x0 - 2 * m, t) + sign * heat(x + x0 - 2 * m, t);
    return s;
  }
  const N = modeCount(t);
  let s = walls === 'reflecting' ? 1 : 0;
  for (let n = 1; n <= N; n++) {
    const k = n * Math.PI;
    const e = Math.exp(-0.5 * k * k * t);
    s += 2 * e * (walls === 'reflecting' ? Math.cos(k * x) * Math.cos(k * x0) : Math.sin(k * x) * Math.sin(k * x0));
  }
  return s;
}

/** Reflecting at 0, absorbing at 1: images with alternating signs. */
export function mixImages(x: number, t: number, x0: number): number {
  const M = imageCount(t, 2);
  let s = 0;
  for (let m = -M; m <= M; m++) s += (m % 2 === 0 ? 1 : -1) * (heat(x - x0 + 2 * m, t) + heat(x + x0 + 2 * m, t));
  return s;
}

export function mixModes(x: number, t: number, x0: number): number {
  const N = modeCount(t, Math.PI / 2);
  let s = 0;
  for (let n = 0; n < N; n++) {
    const k = (n + 0.5) * Math.PI;
    s += 2 * Math.cos(k * x) * Math.cos(k * x0) * Math.exp(-0.5 * k * k * t);
  }
  return s;
}

/** λₙ = ½(nπ)², and the mixed interval's ½((n + ½)π)². */
export const lam = (n: number) => 0.5 * (n * Math.PI) ** 2;
export const lamMix = (n: number) => 0.5 * ((n + 0.5) * Math.PI) ** 2;

// ---------------------------------------------------------------- reflecting walls

export function reflMean(t: number, x0: number): number {
  const N = modeCount(t) + 4;
  let s = 0;
  for (let n = 1; n <= N; n += 2) s += (Math.cos(n * Math.PI * x0) * Math.exp(-lam(n) * t)) / (n * n);
  return 0.5 - (4 / Math.PI ** 2) * s;
}

export function reflSecond(t: number, x0: number): number {
  const N = modeCount(t) + 4;
  let s = 0;
  for (let n = 1; n <= N; n++) s += ((n % 2 ? -1 : 1) * Math.cos(n * Math.PI * x0) * Math.exp(-lam(n) * t)) / (n * n);
  return 1 / 3 + (4 / Math.PI ** 2) * s;
}

/** Stationary covariance ⟨x(τ), x(0)⟩ₛ (G 5.2.110). */
export function reflAutocov(tau: number): number {
  let s = 0;
  for (let n = 1; n <= 401; n += 2) s += Math.exp(-lam(n) * tau) / n ** 4;
  return (8 / Math.PI ** 4) * s;
}

// ---------------------------------------------------------------- absorbing walls

export function survival(t: number, x0: number): number {
  const N = modeCount(t) + 4;
  let s = 0;
  for (let n = 1; n <= N; n += 2) s += (Math.sin(n * Math.PI * x0) * Math.exp(-lam(n) * t)) / n;
  return (4 / Math.PI) * s;
}

/** Probability current out through x = 0, by images (L = 1). */
export function flux0(t: number, x0: number, L = 1): number {
  const M = imageCount(t, 2 * L);
  let s = 0;
  for (let m = -M; m <= M; m++) {
    const y = x0 + 2 * m * L;
    s += (y / Math.sqrt(2 * Math.PI * t ** 3)) * Math.exp(-(y * y) / (2 * t));
  }
  return s;
}

export const fluxL = (t: number, x0: number) => flux0(t, 1 - x0);

export const T_mean = (x0: number) => x0 * (1 - x0);
export const T_var = (x0: number) => (x0 * (1 - x0) * (x0 * x0 + (1 - x0) ** 2)) / 3;
export const T_given0 = (x0: number) => (x0 * (2 - x0)) / 3;
export const T_givenL = (x0: number) => (1 - x0 * x0) / 3;
export const qsd = (x: number) => (Math.PI / 2) * Math.sin(Math.PI * x);

// ---------------------------------------------------------------- reflecting at 0, absorbing at L

export function mixSurvival(t: number, x0: number): number {
  const N = modeCount(t, Math.PI / 2) + 4;
  let s = 0;
  for (let n = 0; n < N; n++) s += ((n % 2 ? -1 : 1) / (2 * n + 1)) * Math.cos(((2 * n + 1) * Math.PI * x0) / 2) * Math.exp(-lamMix(n) * t);
  return (4 / Math.PI) * s;
}

/** Exit-time density (every exit is at L), by images. */
export function mixFlux(t: number, x0: number): number {
  const M = imageCount(t, 2);
  let f = 0;
  for (let m = -M; m <= M; m++) {
    const y = 2 * m + 1 - x0;
    f += ((m % 2 === 0 ? 1 : -1) * y) / Math.sqrt(2 * Math.PI * t ** 3) * Math.exp(-(y * y) / (2 * t));
  }
  return f;
}

export const mixT = (x0: number) => 1 - x0 * x0;
export const mixTVar = (x0: number) => (2 * (1 - x0 ** 4)) / 3;
export const mixQsd = (x: number) => (Math.PI / 2) * Math.cos((Math.PI * x) / 2);

// ---------------------------------------------------------------- half-line and line (x₀ = 1)

/** sign = +1 reflecting, −1 absorbing at x = 0. */
export function halfLine(x: number, t: number, x0: number, sign: number): number {
  return heat(x - x0, t) + sign * heat(x + x0, t);
}

/** ⟨x(t)⟩ with reflection at 0: mean of the folded normal |x₀ + √t Z|. */
export function hlMean(t: number, x0: number): number {
  const s = Math.sqrt(t);
  return x0 * erf(x0 / (Math.SQRT2 * s)) + s * Math.sqrt(2 / Math.PI) * Math.exp(-(x0 * x0) / (2 * s * s));
}

export function hlSurvival(t: number, x0: number): number {
  return erf(x0 / Math.sqrt(2 * t));
}

/** First-passage density to a level a distance a away. */
export function levySmirnov(t: number, a: number): number {
  return (a / (SQRT_2PI * Math.sqrt(t ** 3))) * Math.exp(-(a * a) / (2 * t));
}

/** S for the two walls taken separately, valid while each is far from the other (early times). */
export function wallsSeparately(t: number, x0: number): number {
  return 1 - erfc(x0 / Math.sqrt(2 * t)) - erfc((1 - x0) / Math.sqrt(2 * t));
}

// ---------------------------------------------------------------- circle (length 1)

/** Density on the circle of length L: the line's kernel summed over lifts, with drift v. */
export function circle(x: number, t: number, x0: number, v = 0, L = 1): number {
  const d = x - x0 - v * t;
  if (t > 0.04 * L * L) {
    const N = Math.ceil(Math.sqrt(80 / t) / ((2 * Math.PI) / L)) + 2;
    let s = 1 / L;
    for (let m = 1; m <= N; m++) {
      const q = (2 * Math.PI * m) / L;
      s += (2 / L) * Math.cos(q * d) * Math.exp(-0.5 * q * q * t);
    }
    return s;
  }
  const w = d - L * Math.round(d / L);
  const M = imageCount(t, L);
  let s = 0;
  for (let n = -M; n <= M; n++) s += heat(w + n * L, t);
  return s;
}
