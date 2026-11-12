import { describe, expect, it } from 'vitest';
import { Interval, type Start } from '../src/pages/interval/model';
import { ExitSampler, empiricalStats, exitHistogram, survivingFraction } from '../src/pages/interval/montecarlo';

const N = 20000;

function run(start: Start, x0 = 0.3, L = 1, D = 1) {
  const s = new ExitSampler({ L, D, x0, start, n: N, seed: 20261026 });
  while (!s.done) s.advance(1e9);
  const sorted = s.times.slice().sort();
  return { s, sorted, stats: empiricalStats(s.times, s.wall, sorted), exact: new Interval({ L, D, x0, start }).stats() };
}

describe('ExitSampler', () => {
  it('reproduces the exit statistics from a point', () => {
    const { stats, exact } = run('point', 0.3, 1.5, 0.6);
    expect(Math.abs(stats.mean - exact.mean)).toBeLessThan(4 * stats.seMean);
    expect(Math.abs(stats.piL - exact.piL)).toBeLessThan(4 * stats.sePiL);
    expect(Math.abs(stats.sd / exact.sd - 1)).toBeLessThan(0.03);
    expect(Math.abs(stats.mean0 / exact.mean0 - 1)).toBeLessThan(0.03);
    expect(Math.abs(stats.meanL / exact.meanL - 1)).toBeLessThan(0.05);
  });

  it('reproduces the uniform and quasi-stationary starts', () => {
    for (const start of ['uniform', 'qsd'] as Start[]) {
      const { stats, exact } = run(start);
      expect(Math.abs(stats.mean - exact.mean), start).toBeLessThan(4 * stats.seMean);
      expect(Math.abs(stats.piL - 0.5), start).toBeLessThan(4 * stats.sePiL);
      expect(Math.abs(stats.median / exact.median - 1), start).toBeLessThan(0.03);
    }
  });

  it('matches S(t) and the exit-time density', () => {
    const { s, sorted } = run('point');
    const m = new Interval({ L: 1, D: 1, x0: 0.3, start: 'point' });
    for (const t of [0.02, 0.1, 0.3, 0.6]) {
      const S = m.survival(t);
      expect(Math.abs(survivingFraction(sorted, t) - S), `t=${t}`).toBeLessThan(4 * Math.sqrt((S * (1 - S)) / N) + 1e-3);
    }
    const edges = Float64Array.from({ length: 21 }, (_, i) => (i / 20) * 0.5);
    const h = exitHistogram(s.times, s.wall, s.count, edges, false);
    let area = 0;
    for (let k = 0; k < 20; k++) {
      area += h.total[k] * 0.025;
      const { j0, jL } = m.fluxes(h.centers[k]);
      // Bin averages vs. midpoint values: loose where the density is curved.
      expect(Math.abs(h.wall0[k] - j0), `bin ${k}`).toBeLessThan(0.2 + 0.08 * j0);
      expect(Math.abs(h.wallL[k] - jL), `bin ${k}`).toBeLessThan(0.2 + 0.08 * jL);
    }
    expect(area).toBeCloseTo(1 - survivingFraction(sorted, 0.5), 12);
  });
});
