import { describe, expect, it } from 'vitest';
import { buildProcess, type BC } from '../src/model';
import { exactModel, stationaryDensity } from '../src/analytic';

function grid(domain: [number, number], n: number) {
  const h = (domain[1] - domain[0]) / n;
  const xc = new Float64Array(n);
  for (let i = 0; i < n; i++) xc[i] = domain[0] + (i + 0.5) * h;
  return { xc, h };
}

describe('analytic', () => {
  it('normalises stationary densities', () => {
    const cases: [string, string, Record<string, number>, [number, number], [BC, BC]][] = [
      ['a*x - c*x^3', 'sigma', { a: 1, c: 1, sigma: 0.7 }, [-2.5, 2.5], ['reflecting', 'reflecting']],
      ['kappa*(theta - x)', 'sigma*sqrt(x)', { kappa: 1, theta: 1, sigma: 0.5 }, [0, 4], ['reflecting', 'reflecting']],
      ['F - V*sin(x)', 'sigma', { F: 0.8, V: 1, sigma: 0.8 }, [-Math.PI, Math.PI], ['periodic', 'periodic']],
    ];
    for (const [drift, noise, params, domain, bc] of cases) {
      const proc = buildProcess({ drift, noise, params, interp: 'ito' });
      const { xc, h } = grid(domain, 400);
      const p = stationaryDensity(proc, xc, h, bc);
      expect(p, drift).not.toBeNull();
      expect(p!.reduce((s, v) => s + v, 0) * h).toBeCloseTo(1, 10);
    }
  });

  it('gives the Ornstein–Uhlenbeck stationary Gaussian', () => {
    const proc = buildProcess({ drift: '-k*x', noise: 'sigma', params: { k: 2, sigma: 1 }, interp: 'ito' });
    const { xc, h } = grid([-4, 4], 400);
    const p = stationaryDensity(proc, xc, h, ['reflecting', 'reflecting'])!;
    const v = 1 / 4;
    const at = (x: number) => Math.exp(-(x * x) / (2 * v)) / Math.sqrt(2 * Math.PI * v);
    for (const i of [100, 200, 250]) expect(p[i]).toBeCloseTo(at(xc[i]), 6);
  });

  it('has no stationary density with absorbing walls or time dependence', () => {
    const { xc, h } = grid([0, 1], 50);
    const p1 = buildProcess({ drift: '0', noise: '1', params: {}, interp: 'ito' });
    expect(stationaryDensity(p1, xc, h, ['absorbing', 'reflecting'])).toBeNull();
    const p2 = buildProcess({ drift: 'cos(t)', noise: '1', params: {}, interp: 'ito' });
    expect(stationaryDensity(p2, xc, h, ['reflecting', 'reflecting'])).toBeNull();
  });

  it('walls series reproduces the initial Gaussian and decays', () => {
    const ex = exactModel({ kind: 'walls', params: { mu: 0, sigma: 1 }, interp: 'ito', x0: 2, s0: 0.2, domain: [0, 4] })!;
    const at0 = ex.at(0);
    expect(at0.mass).toBeCloseTo(1, 4);
    expect(at0.density(2)).toBeCloseTo(1 / (0.2 * Math.sqrt(2 * Math.PI)), 3);
    expect(ex.at(1).mass).toBeLessThan(1);
    expect(ex.at(2).mass).toBeLessThan(ex.at(1).mass);
  });
});
