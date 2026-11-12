import type { BC, Interp } from './model';

export interface ParamDef {
  name: string;
  value: number;
  min: number;
  max: number;
  step: number;
}

/** Closed-form solutions available in analytic.ts. */
export type ExactKind = 'wiener' | 'ou' | 'gbm' | 'walls' | 'halfline';

export interface Preset {
  id: string;
  name: string;
  /** Where to read about it in Gardiner's Handbook. */
  ref: string;
  blurb: string;
  drift: string;
  noise: string;
  params: ParamDef[];
  domain: [number, number];
  x0: number;
  sigma0: number;
  T: number;
  bc: [BC, BC];
  interp?: Interp;
  /** Sample paths drawn by default (long, noisy runs read better with fewer). */
  shown?: number;
  /** Exact solution, valid while the formulas are unedited and the BCs are as given. */
  exact?: { kind: ExactKind; bc: [BC, BC] };
  /** Switch on conditioning on survival (q = p/S) when loaded. */
  conditioned?: boolean;
}

const p = (name: string, value: number, min: number, max: number, step = 0.01): ParamDef => ({
  name,
  value,
  min,
  max,
  step,
});

export const PRESETS: Preset[] = [
  {
    id: 'wiener',
    name: 'Wiener process (with drift)',
    ref: '§3.8.1, §4.4.1',
    blurb: 'Free diffusion. The density stays Gaussian and spreads with variance b²t.',
    drift: 'mu',
    noise: 'sigma',
    params: [p('mu', 0, -2, 2), p('sigma', 1, 0.1, 3)],
    domain: [-8, 8],
    x0: 0,
    sigma0: 0,
    T: 4,
    bc: ['open', 'open'],
    exact: { kind: 'wiener', bc: ['open', 'open'] },
  },
  {
    id: 'ou',
    name: 'Ornstein–Uhlenbeck',
    ref: '§3.8.4, §4.4.4',
    blurb: 'Linear restoring drift. Relaxes to a Gaussian with variance σ²/2k.',
    drift: '-k*(x - m)',
    noise: 'sigma',
    params: [p('k', 1, 0.05, 5), p('m', 0, -3, 3), p('sigma', 1, 0.1, 3)],
    domain: [-4, 5],
    x0: 3,
    sigma0: 0,
    T: 5,
    bc: ['open', 'open'],
    exact: { kind: 'ou', bc: ['open', 'open'] },
  },
  {
    id: 'gbm',
    name: 'Multiplicative linear noise (GBM)',
    ref: '§4.4.2, §4.3.6',
    blurb: 'Log-normal solution. Switch Itô ↔ Stratonovich to see the ½b∂ₓb drift shift the density.',
    drift: 'mu*x',
    noise: 'sigma*x',
    params: [p('mu', 0.1, -1, 1.5), p('sigma', 0.5, 0.05, 1.2)],
    domain: [0, 8],
    x0: 1,
    sigma0: 0,
    T: 1.5,
    bc: ['open', 'open'],
    exact: { kind: 'gbm', bc: ['open', 'open'] },
  },
  {
    id: 'double-well',
    name: 'Double-well potential',
    ref: '§9.1, §9.2',
    blurb: 'Bistable drift −U′(x) with U = −ax²/2 + cx⁴/4. Probability leaks over the barrier into the other well.',
    drift: 'a*x - c*x^3',
    noise: 'sigma',
    params: [p('a', 1, -1, 3), p('c', 1, 0.1, 3), p('sigma', 0.7, 0.05, 2)],
    domain: [-2.5, 2.5],
    x0: -1,
    sigma0: 0,
    T: 30,
    bc: ['reflecting', 'reflecting'],
    shown: 10,
  },
  {
    id: 'unstable',
    name: 'Decay from an unstable state',
    ref: '§9.1.5',
    blurb: 'Start on top of the barrier: the density splits symmetrically and runs into the two wells.',
    drift: 'a*x - c*x^3',
    noise: 'sigma',
    params: [p('a', 1, 0.1, 3), p('c', 1, 0.1, 3), p('sigma', 0.3, 0.02, 2)],
    domain: [-2, 2],
    x0: 0,
    sigma0: 0,
    T: 8,
    bc: ['reflecting', 'reflecting'],
  },
  {
    id: 'walls',
    name: 'Diffusion between absorbing walls',
    ref: '§5.2.7, §5.2.8',
    blurb: 'First-passage problem: probability drains out through both ends. Condition on survival to watch q = p/S settle into the quasi-stationary density ∝ exp(μx/D) sin(πx/L), with D = σ² as in the notes.',
    drift: 'mu',
    noise: 'sigma',
    params: [p('mu', 0, -1.5, 1.5), p('sigma', 1, 0.1, 2)],
    domain: [0, 4],
    x0: 1.5,
    sigma0: 0,
    T: 4,
    bc: ['absorbing', 'absorbing'],
    exact: { kind: 'walls', bc: ['absorbing', 'absorbing'] },
  },
  {
    id: 'halfline',
    name: 'Drift toward an absorbing wall',
    ref: '§5.2.5',
    blurb:
      'Drift −μ into the wall at 0, D = σ² as in the notes. On the half-line q = p/S creeps toward the edge QSD (μ/D)² x exp(−μx/D) while κ ≈ μ²/2D + 3/2t; the far edge of the domain raises λ₀ slightly above μ²/2D.',
    drift: '-mu',
    noise: 'sigma',
    params: [p('mu', 1, -1, 2), p('sigma', 1, 0.2, 2)],
    domain: [0, 12],
    x0: 2,
    sigma0: 0,
    T: 10,
    bc: ['absorbing', 'open'],
    shown: 15,
    exact: { kind: 'halfline', bc: ['absorbing', 'open'] },
    conditioned: true,
  },
  {
    id: 'escape',
    name: 'Escape from a metastable well',
    ref: '§5.2.7c, §9.2',
    blurb:
      'Potential U = x³/3 − ax with an absorbing wall past the barrier at −√a. q = p/S settles into the well quickly, while S decays slowly at the escape rate λ₀.',
    drift: 'a - x^2',
    noise: 'sigma',
    params: [p('a', 1, 0.2, 2), p('sigma', 1, 0.1, 2)],
    domain: [-2, 2.5],
    x0: 0,
    sigma0: 0,
    T: 30,
    bc: ['absorbing', 'reflecting'],
    shown: 10,
    conditioned: true,
  },
  {
    id: 'chemical',
    name: 'Chemical reaction X + A ⇌ 2X',
    ref: '§5.2.3c, §7.6.1',
    blurb: 'Diffusion approximation of a birth–death reaction, eq. (5.2.58). x = 0 is an exit boundary; reflecting walls keep it away.',
    drift: 'k*x - x^2',
    noise: 'sqrt(k*x + x^2)',
    params: [p('k', 5, 0.5, 10, 0.1)],
    domain: [0.05, 12],
    x0: 5,
    sigma0: 0,
    T: 3,
    bc: ['reflecting', 'reflecting'],
    shown: 10,
  },
  {
    id: 'cir',
    name: 'Square-root (Feller / CIR) diffusion',
    ref: '§5.2.1 (boundaries)',
    blurb: 'Noise vanishes at x = 0. If 2κθ < σ², the boundary is reached and the density piles up there.',
    drift: 'kappa*(theta - x)',
    noise: 'sigma*sqrt(x)',
    params: [p('kappa', 1, 0.1, 4), p('theta', 1, 0.1, 3), p('sigma', 0.5, 0.05, 2)],
    domain: [0, 4],
    x0: 0.5,
    sigma0: 0,
    T: 5,
    bc: ['reflecting', 'reflecting'],
  },
  {
    id: 'periodic',
    name: 'Tilted periodic potential',
    ref: '§5.2.1 (periodic b.c.)',
    blurb: 'Overdamped particle in a washboard potential. The stationary state carries a constant probability current.',
    drift: 'F - V*sin(x)',
    noise: 'sigma',
    params: [p('F', 0.8, -2, 2), p('V', 1, 0, 3), p('sigma', 0.8, 0.1, 2)],
    domain: [-Math.PI, Math.PI],
    x0: 0,
    sigma0: 0,
    T: 10,
    bc: ['periodic', 'periodic'],
  },
  {
    id: 'resonance',
    name: 'Periodically forced double well',
    ref: 'cf. §9.1–9.2',
    blurb: 'A weak periodic tilt plus noise: hopping between wells synchronises with the forcing (stochastic resonance).',
    drift: 'x - x^3 + eps*cos(omega*t)',
    noise: 'sigma',
    params: [p('eps', 0.25, 0, 1), p('omega', 0.15, 0.01, 1, 0.005), p('sigma', 0.6, 0.05, 2)],
    domain: [-2.2, 2.2],
    x0: -1,
    sigma0: 0,
    T: 80,
    bc: ['reflecting', 'reflecting'],
    shown: 6,
  },
];

export const CUSTOM_ID = 'custom';

export function findPreset(id: string): Preset | undefined {
  return PRESETS.find((pr) => pr.id === id);
}
