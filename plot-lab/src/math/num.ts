/** Grids, quadrature, root finding and caching shared by the figure models. */

/** n points from a to b inclusive, evenly spaced. */
export function linspace(a: number, b: number, n: number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = n === 1 ? a : a + ((b - a) * i) / (n - 1);
  return out;
}

/** n points from a to b inclusive (both > 0), evenly spaced in log. */
export function geomspace(a: number, b: number, n: number): Float64Array {
  const out = new Float64Array(n);
  const la = Math.log(a);
  const lb = Math.log(b);
  for (let i = 0; i < n; i++) out[i] = Math.exp(n === 1 ? la : la + ((lb - la) * i) / (n - 1));
  return out;
}

export function map(x: ArrayLike<number>, f: (v: number, i: number) => number): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = f(x[i], i);
  return out;
}

/** Trapezoid rule on samples y(x). */
export function trapz(y: ArrayLike<number>, x: ArrayLike<number>): number {
  let s = 0;
  for (let i = 1; i < x.length; i++) s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
  return s;
}

/** Running trapezoid integral, starting at 0. */
export function cumtrapz(y: ArrayLike<number>, x: ArrayLike<number>): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 1; i < x.length; i++) out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
  return out;
}

/** Composite Simpson rule for ∫ₐᵇ f with n (made even) panels. */
export function simpson(f: (x: number) => number, a: number, b: number, n = 200): number {
  const m = n % 2 === 0 ? n : n + 1;
  const h = (b - a) / m;
  let s = f(a) + f(b);
  for (let i = 1; i < m; i++) s += (i % 2 === 1 ? 4 : 2) * f(a + i * h);
  return (s * h) / 3;
}

/** Linear interpolation in ascending samples, clamped at the ends. */
export function interp(x: number, xs: ArrayLike<number>, ys: ArrayLike<number>): number {
  const n = xs.length;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const u = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + u * (ys[hi] - ys[lo]);
}

/** A root of f in [a, b], where f changes sign, by Brent's method. */
export function brent(f: (x: number) => number, a: number, b: number, tol = 1e-14): number {
  let fa = f(a);
  let fb = f(b);
  if (fa === 0) return a;
  if (fb === 0) return b;
  if (fa * fb > 0) return NaN;
  let c = a;
  let fc = fa;
  let d = b - a;
  let e = d;
  for (let it = 0; it < 200; it++) {
    if (fb * fc > 0) {
      c = a;
      fc = fa;
      d = e = b - a;
    }
    if (Math.abs(fc) < Math.abs(fb)) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    const tol1 = 2e-16 * Math.abs(b) + 0.5 * tol;
    const xm = 0.5 * (c - b);
    if (Math.abs(xm) <= tol1 || fb === 0) return b;
    if (Math.abs(e) >= tol1 && Math.abs(fa) > Math.abs(fb)) {
      const s = fb / fa;
      let p: number;
      let q: number;
      if (a === c) {
        p = 2 * xm * s;
        q = 1 - s;
      } else {
        const qq = fa / fc;
        const r = fb / fc;
        p = s * (2 * xm * qq * (qq - r) - (b - a) * (r - 1));
        q = (qq - 1) * (r - 1) * (s - 1);
      }
      if (p > 0) q = -q;
      p = Math.abs(p);
      if (2 * p < Math.min(3 * xm * q - Math.abs(tol1 * q), Math.abs(e * q))) {
        e = d;
        d = p / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }
    a = b;
    fa = fb;
    b += Math.abs(d) > tol1 ? d : xm > 0 ? tol1 : -tol1;
    fb = f(b);
  }
  return b;
}

/** Every sign change of f on a grid, refined by Brent: the roots in [a, b], ascending. */
export function roots(f: (x: number) => number, grid: ArrayLike<number>, max = Infinity): number[] {
  const out: number[] = [];
  let x0 = grid[0];
  let f0 = f(x0);
  for (let i = 1; i < grid.length && out.length < max; i++) {
    const x1 = grid[i];
    const f1 = f(x1);
    if (f0 === 0) out.push(x0);
    else if (f0 * f1 < 0) out.push(brent(f, x0, x1));
    x0 = x1;
    f0 = f1;
  }
  return out;
}

/**
 * Memoise a function of a few numbers/strings. Figures recompute their model
 * on every slider move, so the expensive pieces (spectra, sweeps) are cached
 * by their own arguments and survive changes to unrelated controls.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function memo<F extends (...args: any[]) => any>(fn: F, size = 64): F {
  const cache = new Map<string, ReturnType<F>>();
  return ((...args: Parameters<F>) => {
    const key = args.join('|');
    if (cache.has(key)) return cache.get(key);
    const value = fn(...args);
    if (cache.size >= size) cache.delete(cache.keys().next().value!);
    cache.set(key, value);
    return value;
  }) as F;
}
