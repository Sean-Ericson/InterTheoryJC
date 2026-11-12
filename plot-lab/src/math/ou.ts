import { erf, erfcx, erfi, kummerNeg } from '../special';
import { brent, geomspace, linspace, memo, simpson, trapz } from './num';
import { Modes, levels, type Problem, type Wall } from './spectrum';

/**
 * Part III of the notes: Ornstein–Uhlenbeck, A = −kx and B = D, in the
 * process's own units ℓ = √(D/k) and τ = 1/k (k = D = 1). x = 0 is the bottom
 * of the well. Gardiner's substitution y = x e^{kt} (4.4.23) maps the process
 * onto free diffusion run for θ(t) = (e^{2kt} − 1)/2k, so walls at 0 carry the
 * images of Part I over unchanged; two walls need the spectral solver.
 */

export function ouMeanVar(t: number, x0: number): [number, number] {
  return [x0 * Math.exp(-t), -0.5 * Math.expm1(-2 * t)];
}

export function gauss(x: number, m: number, v: number): number {
  return Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
}

export function ouLine(x: number, t: number, x0: number): number {
  const [m, v] = ouMeanVar(t, x0);
  return gauss(x, m, v);
}

/** Wall at the bottom of the well: sign +1 reflecting, −1 absorbing. */
export function ouHalf(x: number, t: number, x0: number, sign: number): number {
  const [m, v] = ouMeanVar(t, x0);
  return gauss(x, m, v) + sign * gauss(x, -m, v);
}

/** pₛ on the line, e^{−x²}/√π. */
export const ouPs = (x: number) => Math.exp(-x * x) / Math.sqrt(Math.PI);

export function ouHlSurvival(t: number, x0: number): number {
  return erf(x0 / Math.sqrt(Math.expm1(2 * t)));
}

/** Mean time to reach the bottom of the well, √π ∫₀^{x₀} erfcx (G 5.2.161 with b → ∞). */
export const ouHlT = memo((x0: number): number => {
  const f = (u: number) => erfcx(u);
  if (x0 <= 3) return Math.sqrt(Math.PI) * simpson(f, 0, x0, 400);
  // erfcx ~ 1/(u√π) past 3: integrate in log u.
  const tail = simpson((s) => f(Math.exp(s)) * Math.exp(s), Math.log(3), Math.log(x0), 400);
  return Math.sqrt(Math.PI) * (simpson(f, 0, 3, 400) + tail);
}, 1024);

/** The Rayleigh QSD 2x e^{−x²}, decay rate k. */
export const ouHlQsd = (x: number) => 2 * x * Math.exp(-x * x);

/**
 * The QSD with decay rate νk, 0 < ν ≤ 1: 2ν x M(1 + ν/2, 3/2, −x²). Its
 * tail is x^{−1−ν}; ν = 1 is the Rayleigh density.
 */
export function ouQsdNu(x: number, nu: number): number {
  return 2 * nu * x * kummerNeg(1 + nu / 2, 1.5, x * x);
}

/** Lomax start ν₀(1 + x)^{−1−ν₀}, tail x^{−1−ν₀}. */
export const lomax = (nu0: number) => (y: number) => nu0 * (1 + y) ** (-1 - nu0);

/**
 * Density and survival on the half-line absorbed at the bottom, from the
 * Lomax start, through the map onto the free half-line:
 *   p(x, t) = e^t ∫ p₀(y) [g(X − y) − g(X + y)] dy,  X = e^t x, g of variance θ(t),
 *   S(t) = ∫ p₀(y) erf(y/√2θ) dy,
 * the y-integrals on a log grid (where p₀ varies) merged with a linear one
 * (where the kernel does, a width √θ about X); the survival tail past the
 * grid is done exactly.
 */
export function ouHlEvolveLomax(nu0: number, xs: ArrayLike<number>, t: number): { p: Float64Array; S: number } {
  const p0 = lomax(nu0);
  const s2 = 0.5 * Math.expm1(2 * t);
  const s = Math.sqrt(s2);
  const et = Math.exp(t);
  let Xmax = 0;
  for (let i = 0; i < xs.length; i++) Xmax = Math.max(Xmax, et * xs[i]);
  const Ymax = Xmax + 14 * s + 50;
  const grid = mergeGrids(geomspace(1e-9, Ymax, 6000), linspace(0, Ymax, 6000));
  const w = grid.map(p0);
  const p = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    // Only |y − X| < 12√θ contributes (e^{−72} beyond): trapezoids over that window of the grid.
    const X = et * xs[i];
    let j = lowerBound(grid, X - 12 * s);
    const end = Math.min(grid.length - 1, lowerBound(grid, X + 12 * s));
    let sum = 0;
    let prev = w[j] * Math.exp(-((X - grid[j]) ** 2) / (2 * s2)) * -Math.expm1((-2 * X * grid[j]) / s2);
    for (j++; j <= end; j++) {
      const f = w[j] * Math.exp(-((X - grid[j]) ** 2) / (2 * s2)) * -Math.expm1((-2 * X * grid[j]) / s2);
      sum += 0.5 * (f + prev) * (grid[j] - grid[j - 1]);
      prev = f;
    }
    p[i] = (et * sum) / Math.sqrt(2 * Math.PI * s2);
  }
  const S = trapz(
    grid.map((y, j) => w[j] * erf(y / Math.sqrt(2 * s2))),
    grid,
  ) + (1 + Ymax) ** -nu0;
  return { p, S };
}

/** First index with a[i] ≥ v (a ascending), clamped to a valid index. */
function lowerBound(a: Float64Array, v: number): number {
  let lo = 0;
  let hi = a.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return Math.max(0, lo - 1);
}

function mergeGrids(a: Float64Array, b: Float64Array): Float64Array {
  const all = Float64Array.from([...a, ...b]).sort();
  const out: number[] = [0];
  for (const v of all) if (v > out[out.length - 1] * (1 + 1e-12) + 1e-15) out.push(v);
  return Float64Array.from(out);
}

/** ⟨x⟩ with a reflecting wall at the bottom: mean of the folded Gaussian. */
export function ouFoldedMean(t: number, x0: number): number {
  const [m, v] = ouMeanVar(t, x0);
  const sd = Math.sqrt(v);
  return m * erf(m / (Math.SQRT2 * sd)) + sd * Math.sqrt(2 / Math.PI) * Math.exp(-(m * m) / (2 * v));
}

// ---------------------------------------------------------------- the interval (0, L)

export type OuWalls = 'reflecting' | 'absorbing' | 'mixed';

/** Bottom wall, top wall. 'mixed' is reflecting at the bottom, absorbing at L. */
const WALLS: Record<OuWalls, [Wall, Wall]> = {
  reflecting: ['reflecting', 'reflecting'],
  absorbing: ['absorbing', 'absorbing'],
  mixed: ['reflecting', 'absorbing'],
};

export function ouProblem(L: number, walls: OuWalls): Problem {
  const [left, right] = WALLS[walls];
  return { A: (x) => -x, dA: () => -1, D: 1, a: 0, b: L, left, right };
}

/** Modes on (0, L) (cached): densities, survival, QSDs. */
export const ouModes = memo((L: number, walls: OuWalls, count = 120, n = 1200) => {
  const m = new Modes(ouProblem(L, walls), count, n);
  if (walls === 'mixed') m.lambda[0] = ouEscapeRate(L, 0.5 * (m.lambda[0] + m.lambda[1]));
  return m;
}, 48);

/**
 * Decay rates on (0, L), from index `from`. The grid scales with the
 * interval so the low rates keep their accuracy at every L; the escape rate
 * of the mixed interval comes from ouEscapeRate.
 */
export const ouLevels = memo((L: number, walls: OuWalls, count: number, from = 0): number[] => {
  const n = Math.round(Math.min(900, Math.max(400, 150 + 160 * L)));
  const out = levels(ouProblem(L, walls), count + (from === 0 ? 1 : 0), n, from);
  if (from === 0 && walls === 'mixed') out[0] = ouEscapeRate(L, 0.5 * (out[0] + out[1]));
  return out.slice(0, count);
}, 4096);

/** Kummer's M(a, b, z) by its power series. */
function kummer(a: number, b: number, z: number): number {
  let term = 1;
  let sum = 1;
  for (let n = 0; n < 3000; n++) {
    term *= ((a + n) / (b + n)) * (z / (n + 1));
    sum += term;
    if (n > z && Math.abs(term) < 1e-17 * Math.abs(sum)) break;
  }
  return sum;
}

/**
 * Escape rate (units of k) from the well reflecting at the bottom and
 * absorbing at L: the lowest ν with Q(L) = M(−ν/2, ½, L²) = 0, Q the even
 * backward eigenfunction (make_figures.ou_Q). For 0 < ν < 2 every term of the
 * series after the first has the same sign, so the root keeps full relative
 * accuracy even at Kramers rates of 1e−9, which finite differences (an
 * absolute O(h²) error in the energy) cannot resolve. `above` is any rate
 * between the first and second roots.
 */
export const ouEscapeRate = memo((L: number, above: number): number => {
  const f = (nu: number) => kummer(-nu / 2, 0.5, L * L);
  return f(above) < 0 ? brent(f, 0, above, 0) : NaN;
}, 4096);

/** G(ξ) = √π ∫₀^ξ e^{u²} erf u du = 2 Σ 2ⁿ ξ^{2n+2}/((2n+2)(2n+1)!!), so F(x) = τ G(x/ℓ). */
export function ouG(xi: number): number {
  let term = xi * xi; // n = 0: 2·ξ²/2
  let sum = term;
  for (let n = 0; n < 2000; n++) {
    term *= (2 * xi * xi * (2 * n + 2)) / ((2 * n + 4) * (2 * n + 3));
    sum += term;
    if (term < 1e-17 * sum) break;
  }
  return sum;
}

export const ouPiL = (x0: number, L: number) => erfi(x0) / erfi(L);

/** Mean exit time: absorbing T = π_L F(L) − F(x₀); reflecting at the bottom T = F(L) − F(x₀). */
export function ouT(x0: number, L: number, walls: 'absorbing' | 'mixed'): number {
  const GL = ouG(L);
  const G0 = ouG(x0);
  return walls === 'absorbing' ? ouPiL(x0, L) * GL - G0 : GL - G0;
}

/** Leading Arrhenius–Kramers rate for the half-well, in units of k. */
export const ouKramers = (L: number) => (2 * L * Math.exp(-L * L)) / Math.sqrt(Math.PI);

/** Symmetrised potential V = A²/2D + A′/2 of OU, in units of k. */
export const ouV = (x: number) => 0.5 * (x * x - 1);
