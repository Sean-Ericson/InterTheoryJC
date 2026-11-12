/**
 * Tridiagonal linear algebra: eigenvalues and eigenvectors of real symmetric
 * tridiagonal matrices (Sturm bisection and inverse iteration), and solves of
 * general complex tridiagonal systems with partial pivoting.
 */

/**
 * Eigenvalues with indices k0..k1 (ascending order, inclusive) of the
 * symmetric tridiagonal matrix with diagonal d and off-diagonal e (e[i]
 * couples i and i + 1), by Sturm-sequence bisection to machine precision.
 */
export function eigenvalues(d: ArrayLike<number>, e: ArrayLike<number>, k0: number, k1: number): number[] {
  const n = d.length;
  let lo = Infinity;
  let hi = -Infinity;
  let e2max = 0;
  for (let i = 0; i < n; i++) {
    const r = (i > 0 ? Math.abs(e[i - 1]) : 0) + (i < n - 1 ? Math.abs(e[i]) : 0);
    lo = Math.min(lo, d[i] - r);
    hi = Math.max(hi, d[i] + r);
    if (i < n - 1) e2max = Math.max(e2max, e[i] * e[i]);
  }
  const pivmin = 1e-300 * Math.max(1, e2max);
  const e2 = new Float64Array(Math.max(0, n - 1));
  for (let i = 0; i < n - 1; i++) e2[i] = e[i] * e[i];
  /** Number of eigenvalues below x: negative pivots of the LDLᵀ factorisation of T − x. */
  const below = (x: number) => {
    let neg = 0;
    let q = d[0] - x;
    if (Math.abs(q) < pivmin) q = -pivmin;
    if (q < 0) neg++;
    for (let i = 1; i < n; i++) {
      q = d[i] - x - e2[i - 1] / q;
      if (Math.abs(q) < pivmin) q = -pivmin;
      if (q < 0) neg++;
    }
    return neg;
  };
  const out: number[] = [];
  let floor = lo;
  for (let k = k0; k <= Math.min(k1, n - 1); k++) {
    let a = floor;
    let b = hi;
    for (let it = 0; it < 200; it++) {
      const mid = 0.5 * (a + b);
      if (mid <= a || mid >= b) break;
      if (below(mid) > k) b = mid;
      else a = mid;
    }
    const v = 0.5 * (a + b);
    out.push(v);
    floor = a;
  }
  return out;
}

/** Banded LU factors of a tridiagonal matrix with partial pivoting (LAPACK dgttrf). */
interface TriLU {
  dl: Float64Array;
  d: Float64Array;
  du: Float64Array;
  du2: Float64Array;
  piv: Uint8Array;
}

function factor(sub: ArrayLike<number>, diag: ArrayLike<number>, sup: ArrayLike<number>): TriLU {
  const n = diag.length;
  const dl = Float64Array.from(sub);
  const d = Float64Array.from(diag);
  const du = Float64Array.from(sup);
  const du2 = new Float64Array(Math.max(0, n - 2));
  const piv = new Uint8Array(n);
  for (let i = 0; i < n - 1; i++) {
    if (Math.abs(d[i]) >= Math.abs(dl[i])) {
      if (d[i] === 0) d[i] = 1e-300;
      const f = dl[i] / d[i];
      dl[i] = f;
      d[i + 1] -= f * du[i];
    } else {
      const f = d[i] / dl[i];
      d[i] = dl[i];
      dl[i] = f;
      const t = du[i];
      du[i] = d[i + 1];
      d[i + 1] = t - f * d[i + 1];
      if (i < n - 2) {
        du2[i] = du[i + 1];
        du[i + 1] = -f * du[i + 1];
      }
      piv[i] = 1;
    }
  }
  if (d[n - 1] === 0) d[n - 1] = 1e-300;
  return { dl, d, du, du2, piv };
}

function solveFactored(f: TriLU, b: Float64Array): void {
  const { dl, d, du, du2, piv } = f;
  const n = d.length;
  for (let i = 0; i < n - 1; i++) {
    if (piv[i]) {
      const t = b[i];
      b[i] = b[i + 1];
      b[i + 1] = t - dl[i] * b[i + 1];
    } else b[i + 1] -= dl[i] * b[i];
  }
  b[n - 1] /= d[n - 1];
  if (n > 1) b[n - 2] = (b[n - 2] - du[n - 2] * b[n - 1]) / d[n - 2];
  for (let i = n - 3; i >= 0; i--) b[i] = (b[i] - du[i] * b[i + 1] - du2[i] * b[i + 2]) / d[i];
}

/**
 * Unit eigenvector of the symmetric tridiagonal (d, e) for an eigenvalue
 * known to machine precision, by inverse iteration. `against` are vectors of
 * nearby eigenvalues to orthogonalise against (near-degenerate levels).
 */
export function eigenvector(d: ArrayLike<number>, e: ArrayLike<number>, lambda: number, against: Float64Array[] = []): Float64Array {
  const n = d.length;
  const scale = Math.max(1e-300, Math.abs(lambda) + 1);
  const shift = lambda + 1e-13 * scale;
  const diag = new Float64Array(n);
  for (let i = 0; i < n; i++) diag[i] = d[i] - shift;
  const lu = factor(e, diag, e);
  let v = new Float64Array(n);
  // A deterministic start with no special symmetry.
  for (let i = 0; i < n; i++) v[i] = 1 + 0.37 * Math.sin(1.7 * i + 0.3);
  for (let it = 0; it < 4; it++) {
    solveFactored(lu, v);
    for (const u of against) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += u[i] * v[i];
      for (let i = 0; i < n; i++) v[i] -= dot * u[i];
    }
    let norm = 0;
    for (let i = 0; i < n; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    for (let i = 0; i < n; i++) v[i] /= norm;
  }
  // Fix the sign: positive at its largest entry nearest the left.
  let big = 0;
  for (let i = 0; i < n; i++) if (Math.abs(v[i]) > Math.abs(v[big]) * (1 + 1e-9)) big = i;
  let first = 0;
  while (first < n && Math.abs(v[first]) < 1e-3 * Math.abs(v[big])) first++;
  if (v[Math.min(first, n - 1)] < 0) v = v.map((x) => -x);
  return v;
}

/**
 * Solve a complex tridiagonal system A z = r with partial pivoting. A has
 * sub-diagonal (sr, si), diagonal (dr, di) and super-diagonal (ur, ui): row i
 * reads sub[i−1] z[i−1] + diag[i] z[i] + sup[i] z[i+1]. Returns [re, im].
 */
export function complexTridiagSolve(
  sr: ArrayLike<number>,
  si: ArrayLike<number>,
  dr: ArrayLike<number>,
  di: ArrayLike<number>,
  ur: ArrayLike<number>,
  ui: ArrayLike<number>,
  rr: ArrayLike<number>,
  ri: ArrayLike<number>,
): [Float64Array, Float64Array] {
  const n = dr.length;
  // Dense band storage per row: columns i, i+1, i+2 (after pivoting a row may reach i+2).
  const a0r = Float64Array.from(dr);
  const a0i = Float64Array.from(di);
  const a1r = new Float64Array(n);
  const a1i = new Float64Array(n);
  const a2r = new Float64Array(n);
  const a2i = new Float64Array(n);
  for (let i = 0; i < n - 1; i++) {
    a1r[i] = ur[i];
    a1i[i] = ui[i];
  }
  const lr = Float64Array.from(sr); // sub-diagonal still to eliminate
  const li = Float64Array.from(si);
  const br = Float64Array.from(rr);
  const bi = Float64Array.from(ri);
  const abs2 = (r: number, i: number) => r * r + i * i;
  for (let k = 0; k < n - 1; k++) {
    // Row k (pivot candidate) vs row k+1, whose entry in column k is l[k].
    // Row k+1 occupies columns k+1 (a0[k+1]) and k+2 (a1[k+1]).
    if (abs2(lr[k], li[k]) > abs2(a0r[k], a0i[k])) {
      // swap rows k and k+1
      const r0r = lr[k],
        r0i = li[k];
      const r1r = a0r[k + 1],
        r1i = a0i[k + 1];
      const r2r = a1r[k + 1],
        r2i = a1i[k + 1];
      const s0r = a0r[k],
        s0i = a0i[k];
      const s1r = a1r[k],
        s1i = a1i[k];
      const s2r = a2r[k],
        s2i = a2i[k];
      a0r[k] = r0r;
      a0i[k] = r0i;
      a1r[k] = r1r;
      a1i[k] = r1i;
      a2r[k] = r2r;
      a2i[k] = r2i;
      lr[k] = s0r;
      li[k] = s0i;
      a0r[k + 1] = s1r;
      a0i[k + 1] = s1i;
      a1r[k + 1] = s2r;
      a1i[k + 1] = s2i;
      let t = br[k];
      br[k] = br[k + 1];
      br[k + 1] = t;
      t = bi[k];
      bi[k] = bi[k + 1];
      bi[k + 1] = t;
    }
    // m = l[k] / a0[k]
    const den = abs2(a0r[k], a0i[k]) || 1e-300;
    const mr = (lr[k] * a0r[k] + li[k] * a0i[k]) / den;
    const mi = (li[k] * a0r[k] - lr[k] * a0i[k]) / den;
    a0r[k + 1] -= mr * a1r[k] - mi * a1i[k];
    a0i[k + 1] -= mr * a1i[k] + mi * a1r[k];
    a1r[k + 1] -= mr * a2r[k] - mi * a2i[k];
    a1i[k + 1] -= mr * a2i[k] + mi * a2r[k];
    br[k + 1] -= mr * br[k] - mi * bi[k];
    bi[k + 1] -= mr * bi[k] + mi * br[k];
  }
  const xr = new Float64Array(n);
  const xi = new Float64Array(n);
  for (let k = n - 1; k >= 0; k--) {
    let tr = br[k];
    let ti = bi[k];
    if (k + 1 < n) {
      tr -= a1r[k] * xr[k + 1] - a1i[k] * xi[k + 1];
      ti -= a1r[k] * xi[k + 1] + a1i[k] * xr[k + 1];
    }
    if (k + 2 < n) {
      tr -= a2r[k] * xr[k + 2] - a2i[k] * xi[k + 2];
      ti -= a2r[k] * xi[k + 2] + a2i[k] * xr[k + 2];
    }
    const den = abs2(a0r[k], a0i[k]) || 1e-300;
    xr[k] = (tr * a0r[k] + ti * a0i[k]) / den;
    xi[k] = (ti * a0r[k] - tr * a0i[k]) / den;
  }
  return [xr, xi];
}
