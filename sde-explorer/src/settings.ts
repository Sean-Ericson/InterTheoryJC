import { formulaParams, type BC, type Interp } from './model';
import { CUSTOM_ID, PRESETS, type ParamDef, type Preset } from './presets';
import type { HeatScale } from './render/pathsPanel';

export interface ViewOptions {
  heatmap: boolean;
  paths: boolean;
  histogram: boolean;
  fp: boolean;
  exact: boolean;
  stationary: boolean;
  /** Show q = p/S, the density conditioned on survival (absorbing walls only). */
  conditioned: boolean;
  /** Two-time and stationary autocorrelation panel. */
  correlation: boolean;
  heatScale: HeatScale;
}

/**
 * Initial density: the Gaussian N(x0, σ0²), or the stationary state (the
 * stationary density, or with absorbing walls the quasi-stationary one).
 */
export type Start = 'gaussian' | 'stationary';

export interface Settings {
  /** A preset id, or CUSTOM_ID once the formulas have been edited. */
  presetId: string;
  drift: string;
  noise: string;
  params: ParamDef[];
  interp: Interp;
  start: Start;
  x0: number;
  sigma0: number;
  domain: [number, number];
  bc: [BC, BC];
  T: number;
  steps: number;
  nPaths: number;
  nShow: number;
  nGrid: number;
  seed: number;
  /** Seconds of animation for the whole run. */
  duration: number;
  view: ViewOptions;
}

/** Snapshots per run (heatmap columns). Every step count below is a multiple of it. */
export const SNAPSHOTS = 400;
export const STEP_OPTIONS = [400, 800, 2000, 4000, 8000];
export const PATH_OPTIONS = [100, 500, 1000, 2000, 5000];
export const GRID_OPTIONS = [200, 400, 800];
export const DURATION_OPTIONS = [5, 10, 20, 40, 80];
/** Traces kept for drawing; the "shown" slider goes up to this. */
export const MAX_SHOWN = 50;

export function defaultSettings(): Settings {
  const base: Settings = {
    presetId: '',
    drift: '',
    noise: '',
    params: [],
    interp: 'ito',
    start: 'gaussian',
    x0: 0,
    sigma0: 0,
    domain: [0, 1],
    bc: ['open', 'open'],
    T: 1,
    steps: 2000,
    nPaths: 1000,
    nShow: 30,
    nGrid: 400,
    seed: 1,
    duration: 10,
    view: { heatmap: true, paths: true, histogram: true, fp: true, exact: true, stationary: true, conditioned: false, correlation: true, heatScale: 'slice' },
  };
  applyPreset(base, PRESETS.find((p) => p.id === 'ou')!);
  return base;
}

/**
 * Load a preset's process (and its run length and drawn-path count); other
 * choices are kept, except that presets about survival switch conditioning on.
 */
export function applyPreset(s: Settings, p: Preset): void {
  s.presetId = p.id;
  s.drift = p.drift;
  s.noise = p.noise;
  s.params = p.params.map((d) => ({ ...d }));
  s.interp = p.interp ?? 'ito';
  s.start = 'gaussian';
  s.x0 = p.x0;
  s.sigma0 = p.sigma0;
  s.domain = [...p.domain];
  s.bc = [...p.bc];
  s.T = p.T;
  s.nShow = p.shown ?? 25;
  if (p.conditioned) s.view.conditioned = true;
}

const BCS: BC[] = ['open', 'reflecting', 'absorbing', 'periodic'];

/**
 * A process given in the URL, as Plot Lab's figure pages link to it:
 * ?a=-k*x&b=sqrt(D)&p=k:1,D:1&lo=0&hi=4&bc=reflecting,absorbing&x0=0&T=10.
 * Formulas that do not parse are ignored (and so is the whole link);
 * anything else left out keeps its current value. Returns whether it applied.
 */
export function applyProcessLink(s: Settings, q: URLSearchParams): boolean {
  if (!q.has('a') && !q.has('b')) return false;
  const drift = q.get('a') ?? '0';
  const noise = q.get('b') ?? '1';
  let names: string[];
  try {
    names = formulaParams(drift, noise).params;
  } catch {
    return false;
  }
  const given = new Map<string, number>();
  for (const pair of (q.get('p') ?? '').split(',')) {
    const [k, v] = pair.split(':');
    if (k && Number.isFinite(Number(v))) given.set(k.trim(), Number(v));
  }
  s.presetId = CUSTOM_ID;
  s.drift = drift;
  s.noise = noise;
  s.interp = 'ito';
  s.start = 'gaussian';
  s.params = names.map((name): ParamDef => {
    const value = given.get(name) ?? 1;
    const span = Math.max(1, 2 * Math.abs(value));
    return { name, value, min: Number((value - span).toPrecision(3)), max: Number((value + span).toPrecision(3)), step: 0.01 };
  });
  const n = (k: string) => (q.has(k) && Number.isFinite(Number(q.get(k))) ? Number(q.get(k)) : null);
  const lo = n('lo');
  const hi = n('hi');
  if (lo !== null && hi !== null && hi > lo) s.domain = [lo, hi];
  const bc = (q.get('bc') ?? '').split(',') as BC[];
  if (bc.length === 2 && bc.every((b) => BCS.includes(b))) s.bc = [bc[0], bc[1]];
  s.x0 = Math.min(s.domain[1], Math.max(s.domain[0], n('x0') ?? s.x0));
  s.sigma0 = n('s0') ?? 0;
  const T = n('T');
  if (T !== null && T > 0) s.T = T;
  return true;
}

export function paramValues(s: Settings): Record<string, number> {
  return Object.fromEntries(s.params.map((p) => [p.name, p.value]));
}
