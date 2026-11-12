import type { Operator } from './fpe';

export interface QuasiStationary {
  /** φ₀ at the cell centres, normalised to unit mass on the grid. */
  density: Float64Array;
  /** Decay rate of the surviving probability once q = p/S has settled into φ₀. */
  lambda0: number;
  /** λ₁ − λ₀: the rate at which q approaches φ₀. */
  gap: number;
}

/**
 * Quasi-stationary density of the discretised Fokker–Planck operator: the
 * positive solution of L φ₀ = −λ₀ φ₀ with φ₀ = 0 on the absorbing walls
 * (§5.2.5). It is the stationary state of the density conditioned on
 * survival, q = p/S, which obeys ∂t q = L q + κ(t) q with κ = −Ṡ/S.
 *
 * λ₀ and λ₁ come from lowestEigenvalues, and φ₀ from inverse iteration with
 * a shift just below λ₀, where (A − μ)⁻¹ is entrywise positive (A = −L is an
 * M-matrix). λ₀ is then recomputed as the exit current of φ₀ over its mass,
 * which keeps its relative accuracy even when it is tiny (a deep metastable
 * well).
 *
 * Null if no probability ever leaves.
 */
export function quasiStationary(g: Operator, h: number): QuasiStationary | null {
  const { outL, outR } = g;
  const n = g.diag.length;
  if (!(outL + outR > 0)) return null;
  const [l0, l1] = lowestEigenvalues(g, 2);

  const phi = inverseIteration(g, Math.max(0, l0 - 0.01 * (l1 - l0))) ?? inverseIteration(g, 0);
  if (!phi) return null;
  let m = 0;
  for (let i = 0; i < n; i++) m += phi[i];
  for (let i = 0; i < n; i++) phi[i] /= m * h;
  const lambda0 = outL * phi[0] + outR * phi[n - 1];
  if (!(lambda0 > 0)) return null;
  return { density: phi, lambda0, gap: l1 - lambda0 };
}

/**
 * The `count` smallest eigenvalues of A = −L, ascending: the decay rates of
 * the modes of the discretised Fokker–Planck equation (§5.2.5). A is a
 * tridiagonal M-matrix whose off-diagonal products are ≥ 0, so it is similar
 * to a symmetric matrix, its eigenvalues are real and ≥ 0, and Sturm-sequence
 * bisection finds them to machine precision.
 */
export function lowestEigenvalues(g: Operator, count: number): number[] {
  const { sub, diag, sup } = g;
  const n = diag.length;
  // Gershgorin bound on the spectrum of A, and LAPACK's guard for zero pivots.
  let top = 0;
  let off2Max = 0;
  for (let i = 0; i < n; i++) {
    top = Math.max(top, -diag[i] + sub[i] + sup[i]);
    if (i > 0) off2Max = Math.max(off2Max, sub[i] * sup[i - 1]);
  }
  const pivmin = 1e-300 * Math.max(1, off2Max);

  /** Number of eigenvalues of A below x: negative pivots of the LDLᵀ factorisation of A − x. */
  const below = (x: number) => {
    let neg = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      q = -diag[i] - x - (i > 0 ? (sub[i] * sup[i - 1]) / q : 0);
      if (Math.abs(q) < pivmin) q = -pivmin;
      if (q < 0) neg++;
    }
    return neg;
  };
  const eigenvalue = (k: number) => {
    let lo = 0;
    let hi = top;
    for (let it = 0; it < 128; it++) {
      const mid = 0.5 * (lo + hi);
      if (mid <= lo || mid >= hi) break;
      if (below(mid) > k) hi = mid;
      else lo = mid;
    }
    return 0.5 * (lo + hi);
  };
  return Array.from({ length: Math.min(count, n) }, (_, k) => eigenvalue(k));
}

/**
 * Inverse iteration for the eigenvector of A = −L nearest the shift μ, which
 * must lie below λ₀. Null if A − μ turns out not to be an M-matrix (a
 * non-positive pivot) or the iteration breaks down.
 */
function inverseIteration(g: Operator, mu: number): Float64Array | null {
  const { sub, diag, sup } = g;
  const n = diag.length;
  // LU factors of the tridiagonal A − μ (Thomas algorithm), computed once.
  const piv = new Float64Array(n);
  const cp = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    piv[i] = -diag[i] - mu - (i > 0 ? -sub[i] * cp[i - 1] : 0);
    if (!(piv[i] > 0)) return null;
    cp[i] = -sup[i] / piv[i];
  }
  let v = new Float64Array(n).fill(1);
  let w = new Float64Array(n);
  for (let it = 0; it < 100; it++) {
    for (let i = 0; i < n; i++) w[i] = (v[i] + (i > 0 ? sub[i] * w[i - 1] : 0)) / piv[i];
    for (let i = n - 2; i >= 0; i--) w[i] -= cp[i] * w[i + 1];
    let s = 0;
    for (let i = 0; i < n; i++) s += w[i];
    if (!(s > 0) || !Number.isFinite(s)) return null;
    let change = 0;
    let big = 0;
    for (let i = 0; i < n; i++) {
      w[i] = Math.max(0, (w[i] * n) / s);
      change = Math.max(change, Math.abs(w[i] - v[i]));
      big = Math.max(big, w[i]);
    }
    [v, w] = [w, v];
    if (change <= 1e-14 * big) break;
  }
  return v;
}
