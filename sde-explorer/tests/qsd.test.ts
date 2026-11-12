import { describe, expect, it } from 'vitest';
import { buildProcess, type BC } from '../src/model';
import { FokkerPlanck } from '../src/fpe';
import { quasiStationary } from '../src/qsd';
import { exactModel } from '../src/analytic';
import { Ensemble } from '../src/sde';

function solver(drift: string, noise: string, params: Record<string, number>, domain: [number, number], bc: [BC, BC], x0 = 0, sigma0 = 0.05, n = 400) {
  const proc = buildProcess({ drift, noise, params, interp: 'ito' });
  return { proc, fp: new FokkerPlanck(proc, { n, domain, bc, x0, sigma0 }) };
}

function advance(fp: FokkerPlanck, T: number, steps: number) {
  for (let i = 0; i < steps; i++) fp.step(T / steps);
}

/** L1 distance between p/S and q on the grid. */
function l1(fp: FokkerPlanck, q: (i: number) => number) {
  const S = fp.stats().mass;
  let e = 0;
  for (let i = 0; i < fp.n; i++) e += Math.abs(fp.p[i] / S - q(i)) * fp.h;
  return e;
}

const WALLS: [BC, BC] = ['absorbing', 'absorbing'];

describe('quasi-stationary density', () => {
  it('is the ground state of drift-diffusion between absorbing walls', () => {
    const [mu, sigma, L] = [0.5, 1, 4];
    const D = (sigma * sigma) / 2;
    const { fp } = solver('mu', 'sigma', { mu, sigma }, [0, L], WALLS);
    const q = quasiStationary(fp.operator(), fp.h)!;
    // §5.2.6a with p = e^{μx/2D} × (pure diffusion): λₙ = μ²/4D + D(nπ/L)².
    const k = Math.PI / L;
    expect(q.lambda0 / (mu * mu / (4 * D) + D * k * k) - 1).toBeLessThan(1e-4);
    expect(Math.abs(q.gap / (3 * D * k * k) - 1)).toBeLessThan(1e-3);
    const shape = (x: number) => Math.exp((mu * x) / (2 * D)) * Math.sin(k * x);
    let norm = 0;
    for (const x of fp.xc) norm += shape(x) * fp.h;
    let err = 0;
    for (let i = 0; i < fp.n; i++) err += Math.abs(q.density[i] - shape(fp.xc[i]) / norm) * fp.h;
    expect(err).toBeLessThan(1e-4);
  });

  it('is the long-time limit of p/S, with exit rate λ₀', () => {
    const { fp } = solver('mu', 'sigma', { mu: 0.5, sigma: 1 }, [0, 4], WALLS, 1, 0.05);
    const q = quasiStationary(fp.operator(), fp.h)!;
    advance(fp, 10, 4000);
    expect(l1(fp, (i) => q.density[i])).toBeLessThan(1e-3);
    expect(Math.abs(fp.stats().rate / q.lambda0 - 1)).toBeLessThan(1e-3);
  });

  it('stays put when started in, while S decays as e^{−λ₀t}', () => {
    const { fp } = solver('a - x^2', 'sigma', { a: 1, sigma: 1 }, [-2, 2.5], ['absorbing', 'reflecting']);
    const q = quasiStationary(fp.operator(), fp.h)!;
    fp.p.set(q.density);
    expect(Math.abs(fp.stats().rate / q.lambda0 - 1)).toBeLessThan(1e-9);
    advance(fp, 20, 2000);
    expect(Math.abs(fp.stats().mass / Math.exp(-q.lambda0 * 20) - 1)).toBeLessThan(1e-4);
    expect(l1(fp, (i) => q.density[i])).toBeLessThan(1e-6);
  });

  it('resolves a tiny escape rate, matching Kramers', () => {
    // U = x³/3 − x: well at 1, barrier at −1, ΔU = 4/3, |U″| = 2 at both.
    const sigma = 0.3;
    const D = (sigma * sigma) / 2;
    const { fp } = solver('1 - x^2', 'sigma', { sigma }, [-2, 2.5], ['absorbing', 'reflecting']);
    const q = quasiStationary(fp.operator(), fp.h)!;
    const kramers = (2 / (2 * Math.PI)) * Math.exp(-4 / 3 / D);
    expect(Math.abs(q.lambda0 / kramers - 1)).toBeLessThan(0.1);
  });

  it('does not exist when nothing can leave', () => {
    const { fp: closed } = solver('-x', 'sigma', { sigma: 1 }, [-3, 3], ['reflecting', 'reflecting']);
    expect(quasiStationary(closed.operator(), closed.h)).toBeNull();
    // The noise vanishes at the wall and the drift points away from it.
    const { fp: away } = solver('1', 'x', {}, [0, 2], ['absorbing', 'reflecting']);
    expect(quasiStationary(away.operator(), away.h)).toBeNull();
  });
});

describe('exit rates and the half-line', () => {
  it('matches the exact exit rate between absorbing walls', () => {
    const params = { mu: 0.5, sigma: 1 };
    const { fp } = solver('mu', 'sigma', params, [0, 4], WALLS, 1.5, 0.1);
    const ex = exactModel({ kind: 'walls', params, interp: 'ito', x0: 1.5, s0: 0.1, domain: [0, 4] })!;
    for (const T of [0.5, 1, 2]) {
      advance(fp, T - fp.t, Math.round((T - fp.t) * 1000));
      expect(Math.abs(fp.stats().rate - ex.at(T).rate), `T=${T}`).toBeLessThan(5e-4);
    }
  });

  it('matches the method-of-images solution with drift toward the wall', () => {
    const params = { mu: 1, sigma: 1 };
    const [x0, s0] = [2, 0.06];
    const { fp } = solver('-mu', 'sigma', params, [0, 12], ['absorbing', 'open'], x0, s0);
    const ex = exactModel({ kind: 'halfline', params, interp: 'ito', x0, s0, domain: [0, 12] })!;
    for (const T of [1, 3, 6]) {
      advance(fp, T - fp.t, Math.round((T - fp.t) * 500));
      const e = ex.at(T);
      const st = fp.stats();
      expect(Math.abs(st.mass / e.mass - 1), `T=${T}`).toBeLessThan(1e-3);
      expect(Math.abs(st.rate / e.rate - 1), `T=${T}`).toBeLessThan(1e-3);
      expect(l1(fp, (i) => e.density(fp.xc[i]) / e.mass), `T=${T}`).toBeLessThan(1e-3);
    }
  });

  it('half-line survival reduces to erf(x₀/√2Dt) without drift', () => {
    const ex = exactModel({ kind: 'halfline', params: { mu: 0, sigma: 1 }, interp: 'ito', x0: 1, s0: 0.01, domain: [0, 40] })!;
    // erf(1/√2) for D = σ² = 1 (the notes' convention), t = 1.
    expect(ex.at(1).mass).toBeCloseTo(0.682689, 4);
  });
});

describe('ensemble started in the quasi-stationary density', () => {
  it('samples φ₀ and survives with probability e^{−λ₀t}', () => {
    const domain: [number, number] = [0, 4];
    const { proc, fp } = solver('mu', 'sigma', { mu: 0, sigma: 1 }, domain, WALLS);
    const q = quasiStationary(fp.operator(), fp.h)!;
    const N = 20000;
    const ens = new Ensemble(proc, { n: N, x0: 0, sigma0: 0, domain, bc: WALLS, seed: 7, initial: { lo: 0, h: fp.h, p: q.density } });
    const s = ens.stats();
    expect(Math.abs(s.mean - 2)).toBeLessThan(0.02);
    // Variance of (π/2L) sin(πx/L) on [0, L]: L²(1/4 − 2/π²).
    expect(Math.abs(s.variance - 16 * (1 / 4 - 2 / Math.PI ** 2))).toBeLessThan(0.03);
    for (let i = 0; i < 2000; i++) ens.step(0.001);
    expect(Math.abs(ens.stats().mass - Math.exp(-2 * q.lambda0))).toBeLessThan(0.015);
  });
});
