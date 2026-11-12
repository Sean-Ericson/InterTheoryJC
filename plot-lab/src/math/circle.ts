import { erf, erfi } from '../special';
import { memo } from './num';
import { levels, type Problem } from './spectrum';
import { complexTridiagSolve } from './tridiag';
import { ouLevels } from './ou';

/**
 * The circles of Part III, in OU units (k = D = 1, ℓ = 1). S¹ = ℝ/Lℤ has no
 * boundary, so nothing is imposed anywhere; an even potential splits the
 * spectrum into sectors even and odd about the bottom, which are intervals of
 * half the length with reflecting (even) or absorbing (odd) ends.
 */

// ---------------------------------------------------------------- sawtooth: A = −kx on |x| < L/2, repeated

export const sawWrap = (x: number, L: number) => ((((x + L / 2) % L) + L) % L) - L / 2;

export function sawPs(x: number, L: number): number {
  const w = sawWrap(x, L);
  return Math.exp(-w * w) / (Math.sqrt(Math.PI) * erf(L / 2));
}

/** Lifson–Jackson for the sawtooth. */
export function sawDeff(L: number): number {
  const z = L / 2;
  return (L * L) / (Math.PI * erf(z) * erfi(z));
}

/** Levels of the sawtooth: even sector = OU reflecting on (0, L/2), odd = absorbing. */
export const sawLevels = (L: number, sector: 'even' | 'odd', count: number, from = 0) =>
  ouLevels(L / 2, sector === 'even' ? 'reflecting' : 'absorbing', count, from);

// ---------------------------------------------------------------- cosine well: A = v − kR sin(x/R), R = L/2π

/** κ = 2R²/ℓ²: R = √(κ/2). */
export const cosR = (kappa: number) => Math.sqrt(kappa / 2);

/** Symmetrised potential, units of k (no tilt): (κ/4) sin²(x/R) − ½ cos(x/R). */
export function cosV(kappa: number): (x: number) => number {
  const R = cosR(kappa);
  return (x) => (kappa / 4) * Math.sin(x / R) ** 2 - 0.5 * Math.cos(x / R);
}

function cosProblem(kappa: number, sector: 'even' | 'odd'): Problem {
  const R = cosR(kappa);
  const wall = sector === 'even' ? 'reflecting' : 'absorbing';
  return { A: (x) => -R * Math.sin(x / R), dA: (x) => -Math.cos(x / R), D: 1, a: 0, b: Math.PI * R, left: wall, right: wall };
}

/** Levels without tilt, units of k: even (cos) and odd (sin) sectors of the half circle. */
export const cosLevels = memo((kappa: number, sector: 'even' | 'odd', count: number) => {
  const R = cosR(kappa);
  const n = Math.round(Math.min(1500, Math.max(300, 60 * Math.PI * R)));
  return levels(cosProblem(kappa, sector), count, n);
}, 1024);

export interface Transport {
  /** Winding velocity of the lift. */
  V: number;
  /** Effective diffusion of the lift. */
  Deff: number;
  /** Fourier coefficients c_m (m = −N..N) of pₛ(θ), θ = x/R, with 2πc₀ = 1. */
  re: Float64Array;
  im: Float64Array;
  N: number;
}

/**
 * V and D_eff of the lift with tilt f = v/kR, by perturbation theory in the
 * counting field s of the tilted generator L₀ + sL₁ + s²D_θ/2, in Fourier
 * space (cos_transport in make_figures.py):
 *   dc_m/dt = −[D_θ m²/2 + i m v/R] c_m + (k m/2)(c_{m−1} − c_{m+1}).
 * Both matrices are tridiagonal in m, and so is L₀ with the m = 0 row and
 * column removed, so each solve is O(N).
 */
export const cosTransport = memo((kappa: number, f: number, N = 100): Transport => {
  const R = cosR(kappa);
  const v = f * R;
  const Dth = 1 / (R * R);
  const M = 2 * N + 1;
  const m = (i: number) => i - N;
  // L0 entries: diag, sub (row i+1, col i) = m_{i+1}/2, sup (row i, col i+1) = −m_i/2.
  const d0r = new Float64Array(M);
  const d0i = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    d0r[i] = -0.5 * Dth * m(i) ** 2;
    d0i[i] = (-m(i) * v) / R;
  }
  const sub0 = (i: number) => m(i + 1) / 2;
  const sup0 = (i: number) => -m(i) / 2;
  // Reduced system: indices without N, in order.
  const keep: number[] = [];
  for (let i = 0; i < M; i++) if (i !== N) keep.push(i);
  const K = keep.length;
  const sr = new Float64Array(K - 1);
  const si = new Float64Array(K - 1);
  const ur = new Float64Array(K - 1);
  const ui = new Float64Array(K - 1);
  const dr = new Float64Array(K);
  const di = new Float64Array(K);
  for (let r = 0; r < K; r++) {
    dr[r] = d0r[keep[r]];
    di[r] = d0i[keep[r]];
    if (r < K - 1) {
      const a = keep[r];
      const b = keep[r + 1];
      if (b === a + 1) {
        sr[r] = sub0(a);
        ur[r] = sup0(a);
      }
    }
  }
  const solve = (rr: Float64Array, ri: Float64Array) => complexTridiagSolve(sr, si, dr, di, ur, ui, rr, ri);
  // L1 applied to a vector: diag v/R − i m Dth, sub iK/2, sup −iK/2 (K = k = 1).
  const applyL1 = (pr: Float64Array, pi: Float64Array): [Float64Array, Float64Array] => {
    const or = new Float64Array(M);
    const oi = new Float64Array(M);
    for (let i = 0; i < M; i++) {
      const dR = v / R;
      const dI = -m(i) * Dth;
      or[i] = dR * pr[i] - dI * pi[i];
      oi[i] = dR * pi[i] + dI * pr[i];
      if (i > 0) {
        // + (i/2) p_{i−1}
        or[i] += -0.5 * pi[i - 1];
        oi[i] += 0.5 * pr[i - 1];
      }
      if (i < M - 1) {
        // − (i/2) p_{i+1}
        or[i] += 0.5 * pi[i + 1];
        oi[i] += -0.5 * pr[i + 1];
      }
    }
    return [or, oi];
  };
  const pr = new Float64Array(M);
  const pi = new Float64Array(M);
  pr[N] = 1 / (2 * Math.PI);
  {
    const rr = new Float64Array(K);
    const ri = new Float64Array(K);
    // −L0[keep, N] p_N: rows m = −1 (sup of row N−1) and m = +1 (sub into row N+1), both 1/2.
    rr[N - 1] = -sup0(N - 1) * pr[N];
    rr[N] = -sub0(N) * pr[N];
    const [xr, xi] = solve(rr, ri);
    keep.forEach((i, r) => {
      pr[i] = xr[r];
      pi[i] = xi[r];
    });
  }
  const [l1r, l1i] = applyL1(pr, pi);
  const lam1r = 2 * Math.PI * l1r[N];
  const lam1i = 2 * Math.PI * l1i[N];
  const rr = new Float64Array(K);
  const ri = new Float64Array(K);
  keep.forEach((i, r) => {
    rr[r] = lam1r * pr[i] - lam1i * pi[i] - l1r[i];
    ri[r] = lam1r * pi[i] + lam1i * pr[i] - l1i[i];
  });
  const [xr, xi] = solve(rr, ri);
  const p1r = new Float64Array(M);
  const p1i = new Float64Array(M);
  keep.forEach((i, r) => {
    p1r[i] = xr[r];
    p1i[i] = xi[r];
  });
  const [l2r] = applyL1(p1r, p1i);
  const lam2r = 0.5 * Dth + 2 * Math.PI * l2r[N];
  return { V: R * lam1r, Deff: 2 * R * R * lam2r, re: pr, im: pi, N };
}, 2048);

/** pₛ(x) from the Fourier coefficients (x measured from the bottom). */
export function cosDensity(tr: Transport, x: number, kappa: number): number {
  const R = cosR(kappa);
  const th = x / R;
  let s = 0;
  for (let i = 0; i < tr.re.length; i++) {
    const m = i - tr.N;
    s += tr.re[i] * Math.cos(m * th) - tr.im[i] * Math.sin(m * th);
  }
  return s / R;
}
