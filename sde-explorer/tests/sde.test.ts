import { describe, expect, it } from 'vitest';
import { buildProcess, type Interp } from '../src/model';
import { Ensemble, type EnsembleConfig } from '../src/sde';
import { exactModel } from '../src/analytic';

const N = 20000;

function simulate(
  drift: string,
  noise: string,
  params: Record<string, number>,
  cfg: Partial<EnsembleConfig>,
  T: number,
  steps: number,
  interp: Interp = 'ito',
) {
  const proc = buildProcess({ drift, noise, params, interp });
  const ens = new Ensemble(proc, {
    n: N,
    x0: 0,
    sigma0: 0,
    domain: [-10, 10],
    bc: ['open', 'open'],
    seed: 12345,
    ...cfg,
  });
  for (let i = 0; i < steps; i++) ens.step(T / steps);
  return ens;
}

describe('Ensemble', () => {
  it('reproduces Ornstein–Uhlenbeck mean and variance', () => {
    const s = simulate('-k*(x - m)', 'sigma', { k: 1, m: 0, sigma: 1 }, { x0: 3 }, 1, 1000).stats();
    const mean = 3 * Math.exp(-1);
    const variance = 0.5 * (1 - Math.exp(-2));
    expect(Math.abs(s.mean - mean)).toBeLessThan(4 * Math.sqrt(variance / N));
    expect(Math.abs(s.variance - variance)).toBeLessThan(4 * variance * Math.sqrt(2 / N));
  });

  it('distinguishes Itô and Stratonovich for multiplicative noise', () => {
    const params = { mu: 0.1, sigma: 0.5 };
    const cfg: Partial<EnsembleConfig> = { x0: 1, domain: [0, 20] };
    const sd = Math.sqrt(Math.exp(0.45) * (Math.exp(0.25) - 1));
    const tol = (4 * sd) / Math.sqrt(N);
    const ito = simulate('mu*x', 'sigma*x', params, cfg, 1, 1000, 'ito').stats();
    const strat = simulate('mu*x', 'sigma*x', params, cfg, 1, 1000, 'strat').stats();
    expect(Math.abs(ito.mean - Math.exp(0.1))).toBeLessThan(tol);
    expect(Math.abs(strat.mean - Math.exp(0.1 + 0.125))).toBeLessThan(tol);
  });

  it('keeps reflected paths inside the domain', () => {
    const ens = simulate('0', '2', {}, { domain: [-1, 1], bc: ['reflecting', 'reflecting'] }, 2, 400);
    for (const x of ens.x) {
      expect(x).toBeGreaterThanOrEqual(-1);
      expect(x).toBeLessThanOrEqual(1);
    }
    expect(ens.aliveCount).toBe(N);
  });

  it('matches the exact survival probability between absorbing walls', () => {
    const params = { mu: 0.3, sigma: 1 };
    const domain: [number, number] = [0, 4];
    const ens = simulate('mu', 'sigma', params, { x0: 1.5, domain, bc: ['absorbing', 'absorbing'] }, 1.5, 300);
    const S = exactModel({ kind: 'walls', params, interp: 'ito', x0: 1.5, s0: 1e-3, domain })!.at(1.5).mass;
    expect(Math.abs(ens.stats().mass - S)).toBeLessThan(4 * Math.sqrt((S * (1 - S)) / N));
  });
});
