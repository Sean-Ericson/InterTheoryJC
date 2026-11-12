import { eigenvalues, eigenvector } from './tridiag';

/**
 * Spectral solution of ∂ₜp = −∂ₓ(A p) + ½D ∂ₓ²p on an interval, through the
 * symmetrised (Schrödinger) form of the notes: p = √pₛ ψ turns the forward
 * operator into −H with
 *
 *   H = −(D/2) ∂ₓ² + V,   V = A²/2D + A′/2,
 *
 * an absorbing wall into ψ = 0 and a reflecting one into ψ′ = (A/D) ψ. H is
 * discretised by second-order finite differences on n + 1 nodes, a Robin end
 * by a ghost point with trapezoid weight ½ (as in make_figures.h_interval), so
 * the matrix is symmetric tridiagonal and its levels come from Sturm bisection.
 */

export type Wall = 'reflecting' | 'absorbing';

export interface Problem {
  A(x: number): number;
  /** A′(x); a central difference of A when left out. */
  dA?(x: number): number;
  D: number;
  a: number;
  b: number;
  left: Wall;
  right: Wall;
}

interface Discrete {
  /** Every node a + ih, i = 0..n. */
  x: Float64Array;
  /** First and last node kept as an unknown (Dirichlet ends are dropped). */
  lo: number;
  hi: number;
  d: Float64Array;
  e: Float64Array;
  /** √(trapezoid weight) of each kept node: √½ at a Robin end, else 1. */
  sw: Float64Array;
  h: number;
}

function discretise(p: Problem, n: number): Discrete {
  const h = (p.b - p.a) / n;
  const x = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) x[i] = p.a + i * h;
  const dA = p.dA ?? ((y: number) => (p.A(y + 1e-6) - p.A(y - 1e-6)) / 2e-6);
  const lo = p.left === 'reflecting' ? 0 : 1;
  const hi = p.right === 'reflecting' ? n : n - 1;
  const m = hi - lo + 1;
  const d = new Float64Array(m);
  const e = new Float64Array(m - 1);
  const sw = new Float64Array(m).fill(1);
  const D = p.D;
  for (let k = 0; k < m; k++) {
    const xi = x[lo + k];
    const A = p.A(xi);
    d[k] = D / (h * h) + (A * A) / (2 * D) + 0.5 * dA(xi);
  }
  e.fill(-0.5 * D / (h * h));
  if (p.left === 'reflecting') {
    d[0] += p.A(p.a) / h;
    if (m > 1) e[0] = -D / (Math.SQRT2 * h * h);
    sw[0] = Math.SQRT1_2;
  }
  if (p.right === 'reflecting') {
    d[m - 1] -= p.A(p.b) / h;
    if (m > 1) e[m - 2] = -D / (Math.SQRT2 * h * h);
    sw[m - 1] = Math.SQRT1_2;
  }
  return { x, lo, hi, d, e, sw, h };
}

/**
 * The decay rates λ_k for k = from..from+count−1, ascending. With both walls
 * reflecting λ₀ = 0 exactly (pₛ), which the discretisation only gets to O(h²).
 */
export function levels(p: Problem, count: number, n = 600, from = 0): number[] {
  const g = discretise(p, n);
  const out = eigenvalues(g.d, g.e, from, from + count - 1);
  if (from === 0 && conserving(p)) out[0] = 0;
  return out;
}

const conserving = (p: Problem) => p.left === 'reflecting' && p.right === 'reflecting';

/**
 * The lowest `count` modes, for densities, survival and quasi-stationary
 * states. ψ_k is stored on every node (zero on an absorbing wall), normalised
 * with the trapezoid rule, ∫ψ_k² dx = 1.
 */
export class Modes {
  readonly x: Float64Array;
  readonly lambda: number[];
  readonly psi: Float64Array[];
  /** Φ(x) = ∫ₐˣ 2A/D, so pₛ ∝ e^Φ and the forward modes are e^{Φ/2} ψ_k. */
  readonly Phi: Float64Array;
  private readonly w: Float64Array;

  constructor(readonly problem: Problem, count: number, n = 1000) {
    const g = discretise(problem, n);
    this.x = g.x;
    const nodes = g.x.length;
    this.lambda = eigenvalues(g.d, g.e, 0, count - 1);
    const vecs: Float64Array[] = [];
    this.psi = this.lambda.map((lam, k) => {
      const near = vecs.filter((_, j) => Math.abs(this.lambda[j] - lam) < 1e-7 * (1 + Math.abs(lam)));
      const y = eigenvector(g.d, g.e, lam, near);
      vecs[k] = y;
      const psi = new Float64Array(nodes);
      for (let i = 0; i < y.length; i++) psi[g.lo + i] = y[i] / (g.sw[i] * Math.sqrt(g.h));
      return psi;
    });
    if (conserving(problem)) this.lambda[0] = 0;
    // Φ by Simpson on each cell.
    this.Phi = new Float64Array(nodes);
    const { A, D } = problem;
    for (let i = 1; i < nodes; i++) {
      const a = g.x[i - 1];
      const b = g.x[i];
      this.Phi[i] = this.Phi[i - 1] + ((2 / D) * (b - a) * (A(a) + 4 * A(0.5 * (a + b)) + A(b))) / 6;
    }
    this.w = new Float64Array(nodes).fill(g.h);
    this.w[0] = this.w[nodes - 1] = 0.5 * g.h;
    if (conserving(problem)) {
      // The exact ground state ψ₀ ∝ e^{Φ/2}.
      const psi0 = this.Phi.map((f) => Math.exp(0.5 * f));
      let m = 0;
      for (let i = 0; i < nodes; i++) m += this.w[i] * psi0[i] * psi0[i];
      this.psi[0] = psi0.map((v) => v / Math.sqrt(m));
    }
  }

  /** Linear interpolation of a node array at y. */
  at(arr: Float64Array, y: number): number {
    const { x } = this;
    const n = x.length - 1;
    const u = ((y - x[0]) / (x[n] - x[0])) * n;
    if (u <= 0) return arr[0];
    if (u >= n) return arr[n];
    const i = Math.floor(u);
    const f = u - i;
    return arr[i] * (1 - f) + arr[i + 1] * f;
  }

  /** How many modes a mode sum at time t needs: e^{−(λ_k − λ_0)t} below 1e-16. */
  private cutoff(t: number): number {
    const l0 = this.lambda[0];
    let k = 1;
    while (k < this.lambda.length && (this.lambda[k] - l0) * t < 37) k++;
    return k;
  }

  /** p(x, t | x₀, 0) on the nodes. */
  density(x0: number, t: number): Float64Array {
    const n = this.x.length;
    const out = new Float64Array(n);
    const K = this.cutoff(t);
    const Phi0 = this.at(this.Phi, x0);
    for (let k = 0; k < K; k++) {
      const c = this.at(this.psi[k], x0) * Math.exp(-this.lambda[k] * t);
      const psi = this.psi[k];
      for (let i = 0; i < n; i++) out[i] += c * psi[i];
    }
    for (let i = 0; i < n; i++) out[i] *= Math.exp(0.5 * (this.Phi[i] - Phi0));
    return out;
  }

  /** Coefficients c_k with S(t | x₀) = Σ c_k e^{−λ_k t}. */
  survivalCoeffs(x0: number): number[] {
    const Phi0 = this.at(this.Phi, x0);
    return this.psi.map((psi) => {
      let m = 0;
      for (let i = 0; i < psi.length; i++) m += this.w[i] * Math.exp(0.5 * (this.Phi[i] - Phi0)) * psi[i];
      return m * this.at(psi, x0);
    });
  }

  /** The k-th forward eigenfunction e^{Φ/2} ψ_k, normalised to unit mass (k = 0: the QSD or pₛ). */
  forward(k = 0): Float64Array {
    const psi = this.psi[k];
    const f = new Float64Array(psi.length);
    let m = 0;
    for (let i = 0; i < psi.length; i++) {
      f[i] = Math.exp(0.5 * this.Phi[i]) * psi[i];
      m += this.w[i] * f[i];
    }
    for (let i = 0; i < psi.length; i++) f[i] /= m;
    return f;
  }
}
