import { describe, expect, it } from 'vitest';
import { Interval, qsdDensity, type IntervalParams, type Start } from '../src/pages/interval/model';
import { erf, erfc } from '../src/special';

/** Composite Simpson's rule with n (even) intervals. */
function simpson(fn: (t: number) => number, a: number, b: number, n: number): number {
  const h = (b - a) / n;
  let sum = fn(a) + fn(b);
  for (let i = 1; i < n; i++) sum += (i % 2 ? 4 : 2) * fn(a + i * h);
  return (sum * h) / 3;
}

/** ∫₀^∞ in time: fine steps over the early peak, coarser over the exponential tail. */
function timeIntegral(m: Interval, fn: (t: number) => number): number {
  const t1 = (0.5 * m.L * m.L) / m.D;
  return simpson(fn, 0, t1, 100_000) + simpson(fn, t1, 45 / m.lambda1, 20_000);
}

function relErr(a: number, b: number): number {
  return Math.abs(a - b) / Math.abs(b);
}

describe('special functions', () => {
  it('matches reference values of erf and erfc', () => {
    expect(erf(0.1)).toBeCloseTo(0.1124629160182849, 15);
    expect(erf(0.5)).toBeCloseTo(0.5204998778130465, 15);
    expect(erf(1)).toBeCloseTo(0.8427007929497149, 15);
    expect(erf(2)).toBeCloseTo(0.9953222650189527, 15);
    expect(erf(-1)).toBeCloseTo(-0.8427007929497149, 15);
    expect(relErr(erfc(0.1), 0.8875370839817152)).toBeLessThan(1e-14);
    expect(relErr(erfc(2), 0.004677734981047266)).toBeLessThan(1e-13);
    expect(relErr(erfc(3), 2.209049699858544e-5)).toBeLessThan(1e-13);
    expect(relErr(erfc(5), 1.5374597944280349e-12)).toBeLessThan(1e-13);
    expect(relErr(erfc(10), 2.0884875837625447e-45)).toBeLessThan(1e-13);
    expect(erfc(-1)).toBeCloseTo(1.8427007929497149, 15);
  });

  it('is continuous where the series hands over to the continued fraction', () => {
    expect(relErr(erfc(2 - 1e-12), erfc(2))).toBeLessThan(1e-11);
  });
});

describe('Interval', () => {
  const base: IntervalParams = { L: 1, D: 1, x0: 0.3, start: 'point' };

  it('agrees between the image and series representations', () => {
    const xs = Array.from({ length: 101 }, (_, i) => i / 100);
    for (const start of ['point', 'uniform'] as Start[]) {
      for (const x0 of [0.05, 0.3, 0.5]) {
        const m = new Interval({ ...base, x0, start });
        for (const t of [0.01, 0.03, 0.05, 0.1, 0.3]) {
          const what = `${start} x0=${x0} t=${t}`;
          expect(m['survivalImages'](t), what).toBeCloseTo(m['survivalSeries'](t), 12);
          const a = m['fluxesImages'](t);
          const b = m['fluxesSeries'](t);
          expect(a.j0, what).toBeCloseTo(b.j0, 10);
          expect(a.jL, what).toBeCloseTo(b.jL, 10);
          const pa = m['densityImages'](xs, t);
          const pb = m['densitySeries'](xs, t);
          for (let i = 0; i < xs.length; i++) expect(pa[i], `${what} x=${xs[i]}`).toBeCloseTo(pb[i], 10);
        }
      }
    }
  });

  it('normalises the exit-time density and reproduces the exit statistics', () => {
    for (const x0 of [0.1, 0.45, 0.75, 1.2]) {
      const m = new Interval({ L: 1.5, D: 0.7, x0, start: 'point' });
      const st = m.stats();
      const j0 = (t: number) => m.fluxes(t).j0;
      const jL = (t: number) => m.fluxes(t).jL;
      const f = (t: number) => j0(t) + jL(t);
      const what = `x0=${x0}`;
      expect(timeIntegral(m, f), what).toBeCloseTo(1, 9);
      expect(timeIntegral(m, jL), what).toBeCloseTo(st.piL, 9);
      const mean = timeIntegral(m, (t) => t * f(t));
      expect(relErr(mean, st.mean), what).toBeLessThan(1e-8);
      const second = timeIntegral(m, (t) => t * t * f(t));
      expect(relErr(Math.sqrt(second - mean * mean), st.sd), what).toBeLessThan(1e-7);
      expect(relErr(timeIntegral(m, (t) => t * j0(t)) / (1 - st.piL), st.mean0), what).toBeLessThan(1e-8);
      expect(relErr(timeIntegral(m, (t) => t * jL(t)) / st.piL, st.meanL), what).toBeLessThan(1e-8);
    }
  });

  it('gives S(t) as the tail integral of the exit-time density', () => {
    for (const start of ['point', 'uniform'] as Start[]) {
      const m = new Interval({ ...base, start });
      for (const t of [0.004, 0.02, 0.1, 0.4]) {
        const f = (s: number) => m.fluxes(s).j0 + m.fluxes(s).jL;
        const tail = simpson(f, t, 1, 100_000) + simpson(f, 1, 10, 10_000);
        expect(m.survival(t), `${start} t=${t}`).toBeCloseTo(tail, 9);
      }
    }
  });

  it('integrates the density to S(t)', () => {
    for (const start of ['point', 'uniform', 'qsd'] as Start[]) {
      const m = new Interval({ L: 2, D: 1.3, x0: 0.7, start });
      for (const t of [0.01, 0.1, 0.3, 1.5]) {
        const xs = Array.from({ length: 4001 }, (_, i) => (i / 4000) * m.L);
        const p = m.density(xs, t);
        const mass = simpson((x) => p[Math.round((x / m.L) * 4000)], 0, m.L, 4000);
        expect(mass, `${start} t=${t}`).toBeCloseTo(m.survival(t), 9);
      }
    }
  });

  it('has the uniform-start moments', () => {
    const m = new Interval({ L: 1.4, D: 0.8, x0: 0, start: 'uniform' });
    const st = m.stats();
    expect(relErr(timeIntegral(m, (t) => m.survival(t)), st.mean)).toBeLessThan(1e-8);
    const second = timeIntegral(m, (t) => 2 * t * m.survival(t));
    expect(relErr(Math.sqrt(second - st.mean ** 2), st.sd)).toBeLessThan(1e-7);
    expect(st.mean).toBeCloseTo((m.L * m.L) / (6 * m.D), 14);
  });

  it('decays exactly exponentially from the quasi-stationary start', () => {
    const m = new Interval({ L: 1.2, D: 0.9, x0: 0, start: 'qsd' });
    const xs = [0.1, 0.5, 0.9];
    for (const t of [0, 0.01, 0.2, 1]) {
      const e = Math.exp(-m.lambda1 * t);
      expect(relErr(m.survival(t), e)).toBeLessThan(1e-14);
      const { j0, jL } = m.fluxes(t);
      expect(relErr(j0 + jL, m.lambda1 * e)).toBeLessThan(1e-14);
      expect(j0).toBe(jL);
      const p = m.density(xs, t);
      xs.forEach((x, i) => expect(relErr(p[i], qsdDensity(x, m.L) * e)).toBeLessThan(1e-14));
    }
    expect(relErr(m.stats().median, Math.LN2 / m.lambda1)).toBeLessThan(1e-14);
  });

  it('depends on L and D only through Dt/L² and x₀/L', () => {
    const a = new Interval({ L: 1, D: 1, x0: 0.3, start: 'point' });
    const b = new Interval({ L: 2.5, D: 0.4, x0: 0.75, start: 'point' });
    const scale = (b.L * b.L) / b.D;
    for (const tau of [0.002, 0.04, 0.3, 1.2]) {
      expect(relErr(b.survival(tau * scale), a.survival(tau))).toBeLessThan(1e-12);
      expect(relErr(b.fluxes(tau * scale).jL * scale, a.fluxes(tau).jL)).toBeLessThan(1e-12);
    }
  });

  it('matches the one-wall (half-line) results at early times', () => {
    for (const start of ['point', 'uniform'] as Start[]) {
      const m = new Interval({ ...base, start });
      for (const t of [0.001, 0.004]) {
        const one = m.oneWall(t)!;
        const { j0, jL } = m.fluxes(t);
        expect(relErr(one.S, m.survival(t))).toBeLessThan(1e-12);
        expect(relErr(one.j0, j0)).toBeLessThan(1e-12);
        expect(relErr(one.jL, jL)).toBeLessThan(1e-12);
      }
      // …and departs from them once paths have felt both walls.
      expect(relErr(m.oneWall(0.3)!.S, m.survival(0.3))).toBeGreaterThan(1e-2);
    }
    expect(new Interval({ ...base, start: 'qsd' }).oneWall(0.1)).toBeNull();
  });

  it('finds the median and the slowest mode', () => {
    for (const start of ['point', 'uniform'] as Start[]) {
      const m = new Interval({ ...base, start });
      expect(m.survival(m.stats().median)).toBeCloseTo(0.5, 12);
      const t = 3 / m.lambda1;
      expect(relErr(m.survival(t), m.c1 * Math.exp(-m.lambda1 * t))).toBeLessThan(1e-9);
    }
  });
});
