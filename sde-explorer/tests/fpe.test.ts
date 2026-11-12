import { describe, expect, it } from 'vitest';
import { buildProcess, type BC, type Interp } from '../src/model';
import { FokkerPlanck } from '../src/fpe';
import { exactModel, stationaryDensity } from '../src/analytic';

function solver(
  drift: string,
  noise: string,
  params: Record<string, number>,
  domain: [number, number],
  bc: [BC, BC],
  x0: number,
  sigma0: number,
  n = 400,
  interp: Interp = 'ito',
) {
  const proc = buildProcess({ drift, noise, params, interp });
  return { proc, fp: new FokkerPlanck(proc, { n, domain, bc, x0, sigma0 }) };
}

function advance(fp: FokkerPlanck, T: number, steps: number) {
  for (let i = 0; i < steps; i++) fp.step(T / steps);
}

function l1(fp: FokkerPlanck, q: (i: number) => number) {
  let e = 0;
  for (let i = 0; i < fp.n; i++) e += Math.abs(fp.p[i] - q(i)) * fp.h;
  return e;
}

describe('FokkerPlanck', () => {
  it('conserves probability with reflecting walls', () => {
    const { fp } = solver('-k*(x - m)', 'sigma', { k: 1, m: 0, sigma: 1 }, [-4, 5], ['reflecting', 'reflecting'], 3, 0.05);
    advance(fp, 3, 1500);
    expect(Math.abs(fp.stats().mass - 1)).toBeLessThan(1e-10);
  });

  it('conserves probability with periodic boundaries', () => {
    const { fp } = solver('F - V*sin(x)', 'sigma', { F: 0.8, V: 1, sigma: 0.8 }, [-Math.PI, Math.PI], ['periodic', 'periodic'], 0, 0.05);
    advance(fp, 5, 1000);
    expect(Math.abs(fp.stats().mass - 1)).toBeLessThan(1e-10);
    expect(Math.min(...fp.p)).toBeGreaterThan(-1e-6);
  });

  it('matches the exact Ornstein–Uhlenbeck density', () => {
    const params = { k: 1, m: 0, sigma: 1 };
    const s0 = 0.06;
    const { fp } = solver('-k*(x - m)', 'sigma', params, [-5, 6], ['open', 'open'], 3, s0);
    advance(fp, 2, 1000);
    const ex = exactModel({ kind: 'ou', params, interp: 'ito', x0: 3, s0, domain: [-5, 6] })!.at(2);
    expect(l1(fp, (i) => ex.density(fp.xc[i]))).toBeLessThan(2e-3);
    expect(Math.abs(fp.stats().mean - ex.mean)).toBeLessThan(1e-3);
    expect(Math.abs(fp.stats().variance - ex.variance)).toBeLessThan(1e-3);
  });

  it('matches the exact log-normal density in both interpretations', () => {
    const params = { mu: 0.1, sigma: 0.5 };
    const s0 = 0.04;
    for (const interp of ['ito', 'strat'] as const) {
      const { fp } = solver('mu*x', 'sigma*x', params, [0, 10], ['open', 'open'], 1, s0, 800, interp);
      advance(fp, 1, 1000);
      const ex = exactModel({ kind: 'gbm', params, interp, x0: 1, s0, domain: [0, 10] })!.at(1);
      expect(l1(fp, (i) => ex.density(fp.xc[i])), interp).toBeLessThan(2e-3);
      expect(Math.abs(fp.stats().mean - ex.mean), interp).toBeLessThan(5e-4);
    }
  });

  it('relaxes to the stationary density of a double well', () => {
    const bc: [BC, BC] = ['reflecting', 'reflecting'];
    const { proc, fp } = solver('a*x - c*x^3', 'sigma', { a: 1, c: 1, sigma: 0.9 }, [-2.5, 2.5], bc, -1, 0.05);
    advance(fp, 60, 3000);
    const ps = stationaryDensity(proc, fp.xc, fp.h, bc)!;
    expect(l1(fp, (i) => ps[i])).toBeLessThan(5e-4);
  });

  it('relaxes to the current-carrying stationary density on a ring', () => {
    const bc: [BC, BC] = ['periodic', 'periodic'];
    const { proc, fp } = solver('F - V*sin(x)', 'sigma', { F: 0.8, V: 1, sigma: 0.8 }, [-Math.PI, Math.PI], bc, 0, 0.05);
    advance(fp, 40, 4000);
    const ps = stationaryDensity(proc, fp.xc, fp.h, bc)!;
    expect(l1(fp, (i) => ps[i])).toBeLessThan(5e-4);
  });

  it('matches the exact survival probability between absorbing walls', () => {
    for (const mu of [0, 0.5]) {
      const params = { mu, sigma: 1 };
      const s0 = 0.1;
      const { fp } = solver('mu', 'sigma', params, [0, 4], ['absorbing', 'absorbing'], 1.5, s0);
      const ex = exactModel({ kind: 'walls', params, interp: 'ito', x0: 1.5, s0, domain: [0, 4] })!;
      for (const T of [0.5, 1, 2]) {
        const span = T - fp.t;
        advance(fp, span, Math.round(span * 1000));
        expect(Math.abs(fp.stats().mass - ex.at(T).mass), `mu=${mu} T=${T}`).toBeLessThan(1e-4);
      }
    }
  });
});
