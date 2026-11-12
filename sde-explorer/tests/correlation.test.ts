import { describe, expect, it } from 'vitest';
import { buildProcess, type BC } from '../src/model';
import { FokkerPlanck, type GridConfig } from '../src/fpe';
import { Covariance, ensembleCovariance, meanOf } from '../src/correlation';
import { exactModel, stationaryDensity } from '../src/analytic';
import { lowestEigenvalues, quasiStationary } from '../src/qsd';
import { Ensemble } from '../src/sde';
import { Recorder } from '../src/recorder';

function setup(drift: string, noise: string, params: Record<string, number>, domain: [number, number], bc: [BC, BC], x0 = 0, sigma0 = 0.05, n = 400) {
  const proc = buildProcess({ drift, noise, params, interp: 'ito' });
  const grid: GridConfig = { n, domain, bc, x0, sigma0 };
  return { proc, grid, fp: new FokkerPlanck(proc, grid) };
}

const complete = (c: Covariance) => {
  c.advance(Infinity);
  return c.values;
};

describe('Fokker–Planck autocovariance', () => {
  it('obeys the regression theorem for Ornstein–Uhlenbeck', () => {
    const params = { k: 1, m: 0, sigma: 1 };
    const { proc, grid, fp } = setup('-k*(x - m)', 'sigma', params, [-5, 6], ['open', 'open'], 3, 0.06);
    for (let i = 0; i < 500; i++) fp.step(0.002); // s = 1
    const lags = 101;
    const dtau = 0.02;
    // Nothing is absorbed, so ∫F dx = 0 and the later means drop out.
    const cov = complete(new Covariance(proc, grid, fp.p, fp.t, lags, dtau, 4, () => ({ mass: 1, mean: 0 })));
    const ex = exactModel({ kind: 'ou', params, interp: 'ito', x0: 3, s0: 0.06, domain: [-5, 6] })!;
    for (const j of [0, 25, 50, 100]) {
      expect(Math.abs(cov[j] - ex.covariance!(1, 1 + j * dtau)), `lag ${j}`).toBeLessThan(2e-3);
    }
  });

  it('gives the stationary G(τ) = σ²/2k e^{−kτ}, decaying at λ₁ = k', () => {
    const [k, sigma] = [2, 1];
    const bc: [BC, BC] = ['reflecting', 'reflecting'];
    const { proc, grid, fp } = setup('-k*x', 'sigma', { k, sigma }, [-3, 3], bc);
    const ps = stationaryDensity(proc, fp.xc, fp.h, bc)!;
    const mean = meanOf(ps, fp.xc);
    const cov = complete(new Covariance(proc, grid, ps, 0, 51, 0.04, 4, () => ({ mass: 1, mean })));
    for (const j of [0, 10, 25, 50]) {
      expect(Math.abs(cov[j] - ((sigma * sigma) / (2 * k)) * Math.exp(-k * j * 0.04)), `lag ${j}`).toBeLessThan(1e-3);
    }
    const [l0, l1] = lowestEigenvalues(fp.operator(), 2);
    expect(Math.abs(l0)).toBeLessThan(1e-9);
    expect(Math.abs(l1 / k - 1)).toBeLessThan(1e-3);
  });

  it('decays at the slowest relaxation rate λ₁ in a double well', () => {
    const bc: [BC, BC] = ['reflecting', 'reflecting'];
    const { proc, grid, fp } = setup('x - x^3', 'sigma', { sigma: 0.8 }, [-2.5, 2.5], bc);
    const ps = stationaryDensity(proc, fp.xc, fp.h, bc)!;
    const mean = meanOf(ps, fp.xc);
    const dtau = 0.1;
    const cov = complete(new Covariance(proc, grid, ps, 0, 301, dtau, 4, () => ({ mass: 1, mean })));
    const [, l1] = lowestEigenvalues(fp.operator(), 2);
    // Late-time log slope: the fast intra-well modes are long gone.
    const slope = Math.log(cov[300] / cov[200]) / (100 * dtau);
    expect(Math.abs(-slope / l1 - 1)).toBeLessThan(1e-3);
  });

  it('decays at the gap λ₁ − λ₀ in the quasi-stationary state', () => {
    const bc: [BC, BC] = ['absorbing', 'absorbing'];
    const { proc, grid, fp } = setup('mu', 'sigma', { mu: 0.5, sigma: 1 }, [0, 4], bc);
    const q = quasiStationary(fp.operator(), fp.h)!;
    const mean = meanOf(q.density, fp.xc);
    const dtau = 0.05;
    const cov = complete(new Covariance(proc, grid, q.density, 0, 81, dtau, 4, (j) => ({ mass: Math.exp(-q.lambda0 * j * dtau), mean })));
    const slope = Math.log(cov[80] / cov[60]) / (20 * dtau);
    expect(Math.abs(-slope / q.gap - 1)).toBeLessThan(2e-3);
  });
});

describe('ensemble autocovariance', () => {
  it('matches the Fokker–Planck result among the survivors between absorbing walls', () => {
    const bc: [BC, BC] = ['absorbing', 'absorbing'];
    const params = { mu: 0, sigma: 1 };
    const { proc, grid, fp } = setup('mu', 'sigma', params, [0, 4], bc, 1.5, 0.1);
    const [W, steps, N] = [40, 400, 20000];
    const rec = new Recorder(W, steps, grid.n, N, 0);
    const ens = new Ensemble(proc, { n: N, x0: 1.5, sigma0: 0.1, domain: [0, 4], bc, seed: 3 });
    const dt = 2 / steps;
    rec.snapshot(0, ens, fp, null);
    for (let i = 1; i <= steps; i++) {
      ens.step(dt);
      fp.step(dt);
      if (i % (steps / W) === 0) rec.snapshot(i * dt, ens, fp, null);
    }
    const j = 10; // s = 0.5
    const lags = W + 1 - j;
    const sampled = new Float64Array(lags);
    expect(ensembleCovariance(rec, j, sampled, 0)).toBe(lags);
    const cov = complete(
      new Covariance(proc, grid, rec.densityAt(j), rec.times[j], lags, 2 / W, 10, (i) => ({ mass: rec.fp.mass[j + i], mean: rec.fp.mean[j + i] })),
    );
    for (const lag of [0, 10, 30]) {
      // Statistical error of a covariance from ~10⁴ survivors.
      expect(Math.abs(sampled[lag] - cov[lag]), `lag ${lag}`).toBeLessThan(0.02);
    }
  });
});
