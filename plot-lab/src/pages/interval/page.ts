import type { ChartSpec, Rule, Series } from '../../plot/chart';
import { Scale, decadeFloor, formatValue, niceCeil, type ScaleKind } from '../../plot/scale';
import type { Role, Theme } from '../../plot/theme';
import { seriesKey } from '../../plot/tooltip';
import { Card } from '../../ui/card';
import { button, checkbox, group, hint, num, numberField, segmented, selectField, slider } from '../../ui/controls';
import type { PageDef, PageHost, PageInstance } from '../types';
import { Interval, meanExitTimes, qsdDensity, type ExitStats, type Start } from './model';
import { ExitSampler, empiricalStats, exitHistogram, survivingFraction, type EmpiricalStats, type ExitHistogram } from './montecarlo';

interface State {
  L: number;
  D: number;
  x0: number;
  start: Start;
  tScale: ScaleKind;
  /** Left end of a log time axis (a linear one starts at 0). */
  tMin: number;
  tMax: number;
  sScale: ScaleKind;
  fScale: ScaleKind;
  split: boolean;
  slowest: boolean;
  oneWall: boolean;
  mc: boolean;
  marks: boolean;
  paths: number;
  seed: number;
  /** Pinned time t* of the density snapshot. */
  t: number;
}

const DEFAULTS: State = {
  L: 1,
  D: 1,
  x0: 0.3,
  start: 'point',
  tScale: 'linear',
  tMin: 1e-3,
  tMax: 1,
  sScale: 'linear',
  fScale: 'linear',
  split: true,
  slowest: true,
  oneWall: false,
  mc: true,
  marks: true,
  paths: 10000,
  seed: 1,
  t: 0.1,
};

const NUM_KEYS = ['L', 'D', 'x0', 'tMin', 'tMax', 'paths', 'seed', 't'] as const;
const FLAG_KEYS = ['split', 'slowest', 'oneWall', 'mc', 'marks'] as const;
const SCALE_KEYS = ['tScale', 'sScale', 'fScale'] as const;
const PATH_OPTIONS = [1000, 10000, 100000];
const START_OPTIONS: [Start, string][] = [
  ['point', 'Point x₀'],
  ['uniform', 'Uniform'],
  ['qsd', 'QSD φ₀'],
];
const LIN_LOG: [ScaleKind, string][] = [
  ['linear', 'lin'],
  ['log', 'log'],
];

const T_SAMPLES = 800;
const X_SAMPLES = 301;
const S_DOTS = 24;
const F_BINS = 36;

function readState(q: URLSearchParams): State {
  const s: State = { ...DEFAULTS };
  for (const k of NUM_KEYS) {
    const v = Number(q.get(k));
    if (q.has(k) && Number.isFinite(v) && v > 0) s[k] = v;
  }
  for (const k of FLAG_KEYS) if (q.has(k)) s[k] = q.get(k) === '1';
  for (const k of SCALE_KEYS) {
    const v = q.get(k);
    if (v === 'linear' || v === 'log') s[k] = v;
  }
  const start = q.get('start');
  if (start === 'point' || start === 'uniform' || start === 'qsd') s.start = start;
  s.x0 = clampX0(s.x0, s.L);
  if (!(s.tMin < s.tMax)) s.tMin = s.tMax / 1000;
  if (!PATH_OPTIONS.includes(s.paths)) s.paths = DEFAULTS.paths;
  s.seed = Math.max(1, Math.floor(s.seed));
  return s;
}

/** The URL parameters that differ from the defaults. */
function writeState(s: State): Record<string, string> {
  const out: Record<string, string> = {};
  for (const k of NUM_KEYS) if (s[k] !== DEFAULTS[k]) out[k] = num(s[k]);
  for (const k of FLAG_KEYS) if (s[k] !== DEFAULTS[k]) out[k] = s[k] ? '1' : '0';
  for (const k of SCALE_KEYS) if (s[k] !== DEFAULTS[k]) out[k] = s[k];
  if (s.start !== DEFAULTS.start) out.start = s.start;
  return out;
}

function clampX0(x0: number, L: number): number {
  return Math.min(0.999 * L, Math.max(0.001 * L, x0));
}

/** n samples from lo to hi, evenly spaced on the given scale. */
function grid(lo: number, hi: number, n: number, scale: ScaleKind): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    out[i] = scale === 'log' ? lo * (hi / lo) ** u : lo + (hi - lo) * u;
  }
  return out;
}

interface Curves {
  t: Float64Array;
  S: Float64Array;
  j0: Float64Array;
  jL: Float64Array;
  f: Float64Array;
  /** Exit rate f/S, and its share through each wall. */
  kappa: Float64Array;
  kappa0: Float64Array;
  kappaL: Float64Array;
  slowS: Float64Array;
  slowF: Float64Array;
  one: { S: Float64Array; j0: Float64Array; jL: Float64Array; f: Float64Array } | null;
  /** Linear-axis cap for f, where it diverges at t = 0 (uniform start). */
  fCap: number | undefined;
  /** Exit rate from the centre, x₀ = L/2, as in the notes' figure. */
  kappaMid: Float64Array;
  /** Mean exit times against the starting point, and the standard deviation of T. */
  x: Float64Array;
  T: Float64Array;
  T0: Float64Array;
  TL: Float64Array;
  sdT: Float64Array;
}

interface McView {
  stats: EmpiricalStats;
  sT: Float64Array;
  sY: Float64Array;
  hist: ExitHistogram;
}

function mount(host: PageHost): PageInstance {
  const s = readState(host.query);
  let model = new Interval(s);
  let stats: ExitStats = model.stats();
  let curves = computeCurves();
  let hoverT: number | null = null;
  let sampler: ExitSampler | null = null;
  let samplerKey = '';
  let mc: McView | null = null;
  let lastDerive = 0;
  let mcTimer = 0;
  let persistTimer = 0;
  let frame = 0;

  // ---------------------------------------------------------------- sidebar

  const intro = group(host.controls);
  const blurb = document.createElement('p');
  blurb.className = 'blurb';
  blurb.innerHTML = 'Brownian motion started inside (0, L) and absorbed at whichever wall it reaches first.';
  const source = document.createElement('p');
  source.className = 'source';
  source.innerHTML =
    'Notes: <cite>Free diffusion, absorbing walls</cite> · <code>absorbing_density.pdf</code>, <code>absorbing_exit.pdf</code>, <code>absorbing_exit_times.pdf</code> · <span class="ref">Gardiner §5.2.6a, §5.2.7–8</span>';
  intro.append(blurb, source);

  const params = group(host.controls, 'Parameters');
  const sliderL = slider({ label: 'L', min: 0.2, max: 5, step: 0.01, value: s.L, onInput: (v) => setParam(() => (s.L = v)) });
  const sliderD = slider({ label: 'D', min: 0.05, max: 5, step: 0.01, value: s.D, onInput: (v) => setParam(() => (s.D = v)) });
  const sliderX0 = slider({ label: 'x₀', min: 0.001 * s.L, max: 0.999 * s.L, step: s.L / 1000, value: s.x0, onInput: (v) => setParam(() => (s.x0 = v)) });
  params.append(sliderL.el, sliderD.el, sliderX0.el, hint('The axes stay put as L and D change, so the curves visibly stretch. Only x₀/L changes their shape.'));

  const startGroup = group(host.controls, 'Starting density');
  const startSeg = segmented<Start>({
    label: 'Starting density',
    options: START_OPTIONS,
    value: s.start,
    onChange: (v) => setParam(() => (s.start = v)),
  });
  startGroup.append(startSeg.el, hint('From the quasi-stationary density φ₀ = (π/2L) sin(πx/L), S(t) = exp(−λ₁t) exactly.'));

  const timeGroup = group(host.controls, 'Time axis');
  const tScaleSeg = segmented<ScaleKind>({
    label: 'Time axis scale',
    options: [
      ['linear', 'Linear'],
      ['log', 'Log'],
    ],
    value: s.tScale,
    onChange: (v) => setView(() => (s.tScale = v)),
  });
  const tMinField = numberField({
    label: 'From',
    value: s.tMin,
    onChange: (v) => v > 0 && v < s.tMax && (setView(() => (s.tMin = v)), true),
  });
  const tMaxField = numberField({
    label: 'To',
    value: s.tMax,
    onChange: (v) => v > 0 && (s.tScale === 'linear' || v > s.tMin) && (setView(() => (s.tMax = v)), true),
  });
  const timeRow = document.createElement('div');
  timeRow.className = 'row3';
  timeRow.append(tMinField.el, tMaxField.el, button('Fit', fit));
  timeGroup.append(tScaleSeg.el, timeRow);

  const showGroup = group(host.controls, 'Show');
  const checks = document.createElement('div');
  checks.className = 'checks';
  const flag = (label: string, key: (typeof FLAG_KEYS)[number]) =>
    checkbox({
      label,
      checked: s[key],
      onChange: (v) => {
        s[key] = v;
        if (key === 'mc') syncSampler();
        changed();
      },
    });
  const splitBox = flag('Split by exit wall', 'split');
  const slowBox = flag('Slowest mode', 'slowest');
  const oneWallBox = flag('Walls separately', 'oneWall');
  const mcBox = flag('Simulation', 'mc');
  const marksBox = flag('Mean & median', 'marks');
  checks.append(splitBox.el, slowBox.el, oneWallBox.el, mcBox.el, marksBox.el);
  showGroup.append(
    checks,
    hint('Slowest mode: the late-time asymptote c₁ exp(−λ₁t). Walls separately: each wall as if the other were absent (half-line, exact at early times).'),
  );

  const simGroup = group(host.controls, 'Simulation');
  const pathsSel = selectField({
    label: 'Paths',
    options: PATH_OPTIONS.map((n) => [String(n), n.toLocaleString()]),
    value: String(s.paths),
    onChange: (v) => {
      s.paths = Number(v);
      syncSampler();
      changed();
    },
  });
  const seedField = numberField({
    label: 'Seed',
    value: s.seed,
    onChange: (v) => {
      if (!Number.isInteger(v) || v < 1) return false;
      s.seed = v;
      syncSampler();
      changed();
      return true;
    },
  });
  const seedRow = document.createElement('div');
  seedRow.className = 'row3';
  seedRow.append(
    pathsSel.el,
    seedField.el,
    button('Resample', () => {
      s.seed = 1 + Math.floor(Math.random() * 999_999);
      seedField.set(s.seed);
      syncSampler();
      changed();
    }),
  );
  const mcStatus = hint('');
  simGroup.append(seedRow, mcStatus);

  // ---------------------------------------------------------------- main

  const head = document.createElement('header');
  head.className = 'fig-head';
  const h2 = document.createElement('h2');
  h2.className = 'fig-title';
  h2.textContent = 'Free diffusion, absorbing walls';
  const eqs = document.createElement('div');
  eqs.className = 'equations';
  head.append(h2, eqs);
  eqs.innerHTML = `
    <p class="eq">dx = √D dW on 0 &lt; x &lt; L, absorbed at both walls</p>
    <p class="eq">∂<sub>t</sub>p = ½D ∂<sub>x</sub><sup>2</sup>p,&ensp;p(0, t) = p(L, t) = 0</p>
    <p class="eq">S(t) = ∫<sub>0</sub><sup>L</sup> p dx = P(T &gt; t),&ensp;f = −dS/dt = j<sub>0</sub> + j<sub>L</sub></p>`;

  const tiles = document.createElement('div');
  tiles.className = 'tiles';
  const tile = (label: string, key?: Role) => {
    const el = document.createElement('div');
    el.className = 'tile';
    const l = document.createElement('div');
    l.className = 'tile-label';
    if (key) l.append(seriesKey(key));
    l.append(label);
    const v = document.createElement('div');
    v.className = 'tile-value';
    const sub = document.createElement('div');
    sub.className = 'tile-sub';
    el.append(l, v, sub);
    tiles.append(el);
    return { v, sub };
  };
  const tLambda = tile('Slowest rate λ₁');
  const tMean = tile('Mean exit time ⟨T⟩');
  const tSd = tile('Std. deviation of T');
  const tMedian = tile('Median exit time');
  const tPiL = tile('P(exit at L)', 'orange');
  const tMean0 = tile('⟨T | exit at 0⟩', 'blue');
  const tMeanL = tile('⟨T | exit at L⟩', 'orange');

  const cards = document.createElement('div');
  cards.className = 'cards';
  const timeEvents = {
    hover: (t: number | null) => {
      hoverT = t;
      schedule();
    },
    pick: (t: number) => {
      s.t = t;
      changed(false);
    },
  };
  const cardS = new Card({ title: 'Survival S(t) = P(T > t)', label: 'Survival probability against time', events: timeEvents, className: 'wide' });
  const cardF = new Card({ title: 'Exit-time density f(t) = −dS/dt', label: 'Exit-time density against time', events: timeEvents, className: 'wide' });
  const cardK = new Card({ title: 'Exit rate κ(t) = f/S', label: 'Exit rate of the surviving paths against time', events: timeEvents });
  const cardP = new Card({ title: 'Density', label: 'Density of the surviving paths at the selected time' });
  const cardT = new Card({
    title: 'Mean exit time vs. x₀',
    label: 'Mean exit time against starting point',
    events: { pick: (x) => pickX0(x) },
  });

  /** Dragging on a chart against x₀ moves the point start there. */
  function pickX0(x: number): void {
    if (s.start !== 'point') startSeg.set('point');
    setParam(() => {
      s.start = 'point';
      s.x0 = x;
    });
  }
  const sScaleSeg = segmented<ScaleKind>({ label: 'Survival axis scale', options: LIN_LOG, value: s.sScale, onChange: (v) => setView(() => (s.sScale = v)), small: true });
  const fScaleSeg = segmented<ScaleKind>({ label: 'Exit-time density axis scale', options: LIN_LOG, value: s.fScale, onChange: (v) => setView(() => (s.fScale = v)), small: true });
  cardS.tools.append(sScaleSeg.el);
  cardF.tools.append(fScaleSeg.el);

  // The snapshot time t*, also settable by dragging on any of the time charts.
  const tRow = document.createElement('div');
  tRow.className = 'param';
  const tLabel = document.createElement('label');
  tLabel.className = 'param-name';
  tLabel.textContent = 't*';
  tLabel.htmlFor = 'interval-t';
  const tRange = document.createElement('input');
  tRange.type = 'range';
  tRange.id = 'interval-t';
  tRange.min = '0';
  tRange.max = '1000';
  const tBox = document.createElement('input');
  tBox.type = 'number';
  tBox.step = 'any';
  tBox.setAttribute('aria-label', 't* value');
  tRange.addEventListener('input', () => {
    s.t = timeScale().atFraction(Number(tRange.value) / 1000);
    changed(false);
  });
  tBox.addEventListener('change', () => {
    const v = Number(tBox.value);
    if (Number.isFinite(v) && v > 0) s.t = v;
    changed(false);
  });
  tRow.append(tLabel, tRange, tBox);
  cardP.foot.append(tRow, hint('Drag on a time chart to move t*; hover to preview.'));
  cardT.foot.append(hint('Drag to move the starting point.'));

  // The notes' snapshots at fixed Dt/L², and the splitting probabilities.
  const cardSnapP = new Card({ title: 'Density snapshots p(x, t | x₀, 0)', label: 'Density at several times' });
  const cardSnapQ = new Card({ title: 'Conditioned snapshots q = p/S', label: 'Density of the surviving paths at several times' });
  const cardPi = new Card({
    title: 'Splitting probabilities',
    label: 'Probability of leaving through each wall against starting point',
    events: { pick: (x) => pickX0(x) },
  });
  cardSnapP.foot.append(hint('From x₀ at the notes’ times Dt/L²: p loses mass through both walls.'));
  cardSnapQ.foot.append(hint('The density conditioned on survival converges to the QSD (π/2L) sin(πx/L), dashed, by Dt/L² ≈ 0.3.'));
  cardPi.foot.append(hint('π₀ = 1 − x₀/L and π_L = x₀/L; the dots are the simulated values at x₀. Drag to move x₀.'));
  cards.append(cardS.el, cardF.el, cardK.el, cardP.el, cardT.el, cardSnapP.el, cardSnapQ.el, cardPi.el);

  const formulas = document.createElement('details');
  formulas.className = 'formulas';
  formulas.innerHTML = `
    <summary>Formulas</summary>
    <ul>
      <li><b>Modes.</b> p(x, t) = (2/L) Σ<sub>n</sub> b<sub>n</sub> sin(nπx/L) e<sup>−λ<sub>n</sub>t</sup> with λ<sub>n</sub> = ½D(nπ/L)<sup>2</sup> (Gardiner 5.2.100). The start enters through b<sub>n</sub> = ∫ p<sub>0</sub> sin(nπx/L) dx: sin(nπx<sub>0</sub>/L) from a point, 2/nπ (odd n) from the uniform density, π/4 for n = 1 only from φ<sub>0</sub>.</li>
      <li><b>Survival and exit fluxes.</b> S = Σ<sub>n odd</sub> (4/nπ) b<sub>n</sub> e<sup>−λ<sub>n</sub>t</sup>, j<sub>0</sub> = (πD/L<sup>2</sup>) Σ n b<sub>n</sub> e<sup>−λ<sub>n</sub>t</sup>, j<sub>L</sub> = (πD/L<sup>2</sup>) Σ (−1)<sup>n+1</sup> n b<sub>n</sub> e<sup>−λ<sub>n</sub>t</sup>. Late times: S ≈ c<sub>1</sub>e<sup>−λ<sub>1</sub>t</sup>, c<sub>1</sub> = (4/π) b<sub>1</sub>.</li>
      <li><b>Images,</b> used for Dt/L<sup>2</sup> &lt; 0.05 where the series converges slowly: j<sub>0</sub> = Σ<sub>m</sub> y<sub>m</sub> e<sup>−y<sub>m</sub><sup>2</sup>/2Dt</sup> / √(2πDt<sup>3</sup>), y<sub>m</sub> = x<sub>0</sub> + 2mL, and j<sub>L</sub> likewise with x<sub>0</sub> → L − x<sub>0</sub>. The m = 0 terms alone are the walls taken separately: j<sub>0</sub> ≈ x<sub>0</sub> e<sup>−x<sub>0</sub><sup>2</sup>/2Dt</sup> / √(2πDt<sup>3</sup>) and S ≈ 1 − erfc(x<sub>0</sub>/√2Dt) − erfc((L−x<sub>0</sub>)/√2Dt).</li>
      <li><b>Exit statistics</b> (§5.2.7–8): ⟨T⟩ = x<sub>0</sub>(L − x<sub>0</sub>)/D, σ<sub>T</sub><sup>2</sup> = x<sub>0</sub>(L − x<sub>0</sub>)[x<sub>0</sub><sup>2</sup> + (L − x<sub>0</sub>)<sup>2</sup>]/3D<sup>2</sup>, P(exit at L) = x<sub>0</sub>/L, ⟨T | 0⟩ = x<sub>0</sub>(2L − x<sub>0</sub>)/3D, ⟨T | L⟩ = (L<sup>2</sup> − x<sub>0</sub><sup>2</sup>)/3D. Uniform start: ⟨T⟩ = L<sup>2</sup>/6D, σ<sub>T</sub><sup>2</sup> = 7L<sup>4</sup>/180D<sup>2</sup>. From φ<sub>0</sub>: T is exponential, ⟨T⟩ = σ<sub>T</sub> = 1/λ<sub>1</sub>.</li>
      <li><b>Exit rate.</b> κ = f/S is the hazard of the surviving paths. It tends to λ<sub>1</sub> as their density q = p/S settles into φ<sub>0</sub> = (π/2L) sin(πx/L), at rate λ<sub>2</sub> − λ<sub>1</sub> (λ<sub>3</sub> − λ<sub>1</sub> from symmetric starts).</li>
      <li><b>Simulation.</b> Exact Gaussian steps, with a Brownian-bridge test for crossings between steps and steps that shrink near the walls. Dots on S are the fraction still inside; dots on f are a histogram of exit times.</li>
    </ul>`;

  host.main.append(head, tiles, cards, formulas);

  // ---------------------------------------------------------------- state changes

  /** A model parameter changed: recompute everything. */
  function setParam(apply: () => void): void {
    apply();
    s.x0 = clampX0(s.x0, s.L);
    sliderX0.setRange(0.001 * s.L, 0.999 * s.L, s.L / 1000);
    sliderX0.set(s.x0);
    model = new Interval(s);
    stats = model.stats();
    curves = computeCurves();
    syncSampler();
    changed();
  }

  /** The time window or an axis scale changed: resample the curves and re-bin the simulation. */
  function setView(apply: () => void): void {
    apply();
    curves = computeCurves();
    if (sampler) deriveMc();
    changed();
  }

  /** Redraw and remember the view in the URL; `sync` also refreshes the sidebar's availability. */
  function changed(sync = true): void {
    if (sync) syncControls();
    clearTimeout(persistTimer);
    persistTimer = window.setTimeout(() => host.setQuery(writeState(s)), 250);
    schedule();
  }

  function syncControls(): void {
    const point = s.start === 'point';
    sliderX0.el.querySelectorAll('input').forEach((i) => (i.disabled = !point));
    sliderX0.el.title = point ? '' : 'Only the point start uses x₀. Drag on the mean-exit-time chart to switch to it.';
    oneWallBox.setAvailable(s.start !== 'qsd', 'Starting from φ₀ there is no early-time regime: S is exactly exponential.');
    tMinField.input.disabled = s.tScale === 'linear';
    tMinField.set(s.tMin);
    tMaxField.set(s.tMax);
  }

  function fit(): void {
    s.tMax = niceCeil(model.timeAtSurvival(0.01, stats.mean));
    const early =
      s.start === 'point' ? (0.05 * Math.min(s.x0, s.L - s.x0) ** 2) / s.D : s.start === 'uniform' ? (1e-4 * s.L * s.L) / s.D : s.tMax * 1e-3;
    s.tMin = Math.min(decadeFloor(early), decadeFloor(s.tMax / 100));
    setView(() => {});
  }

  function timeDomain(): [number, number] {
    return s.tScale === 'log' ? [s.tMin, s.tMax] : [0, s.tMax];
  }

  function timeScale(): Scale {
    const [lo, hi] = timeDomain();
    return new Scale(s.tScale, lo, hi, 0, 1);
  }

  // ---------------------------------------------------------------- curves

  function computeCurves(): Curves {
    const [lo, hi] = timeDomain();
    const t = grid(lo, hi, T_SAMPLES, s.tScale);
    const n = t.length;
    const S = new Float64Array(n);
    const j0 = new Float64Array(n);
    const jL = new Float64Array(n);
    const f = new Float64Array(n);
    const kappa = new Float64Array(n);
    const kappa0 = new Float64Array(n);
    const kappaL = new Float64Array(n);
    const slowS = new Float64Array(n);
    const slowF = new Float64Array(n);
    const kappaMid = new Float64Array(n);
    const mid = new Interval({ L: s.L, D: s.D, x0: s.L / 2, start: 'point' });
    const oneWall = model.start !== 'qsd';
    const one = oneWall ? { S: new Float64Array(n), j0: new Float64Array(n), jL: new Float64Array(n), f: new Float64Array(n) } : null;
    for (let i = 0; i < n; i++) {
      S[i] = model.survival(t[i]);
      const fl = model.fluxes(t[i]);
      j0[i] = fl.j0;
      jL[i] = fl.jL;
      f[i] = fl.j0 + fl.jL;
      kappa[i] = S[i] > 0 ? f[i] / S[i] : NaN;
      kappa0[i] = S[i] > 0 ? j0[i] / S[i] : NaN;
      kappaL[i] = S[i] > 0 ? jL[i] / S[i] : NaN;
      slowS[i] = model.c1 * Math.exp(-model.lambda1 * t[i]);
      slowF[i] = model.lambda1 * slowS[i];
      const midS = mid.survival(t[i]);
      const midF = mid.fluxes(t[i]);
      kappaMid[i] = midS > 0 ? (midF.j0 + midF.jL) / midS : NaN;
      if (one) {
        const o = model.oneWall(t[i])!;
        one.S[i] = o.S;
        one.j0[i] = o.j0;
        one.jL[i] = o.jL;
        one.f[i] = o.j0 + o.jL;
      }
    }
    // From the uniform start f ~ t^{−1/2} at t → 0: keep the linear axis on the rest of the curve.
    const early = model.fluxes(lo + 0.03 * (hi - lo));
    const fCap = model.start === 'uniform' ? 1.3 * (early.j0 + early.jL) : undefined;

    const x = grid(0, s.L, 201, 'linear');
    const T = new Float64Array(x.length);
    const T0 = new Float64Array(x.length);
    const TL = new Float64Array(x.length);
    const sdT = new Float64Array(x.length);
    x.forEach((xi, i) => {
      const m = meanExitTimes(xi, s.L, s.D);
      T[i] = m.mean;
      T0[i] = m.mean0;
      TL[i] = m.meanL;
      // σ_T² = x₀(L − x₀)[x₀² + (L − x₀)²]/3D² (§5.2.8)
      sdT[i] = Math.sqrt((xi * (s.L - xi) * (xi * xi + (s.L - xi) ** 2)) / (3 * s.D * s.D));
    });
    return { t, S, j0, jL, f, kappa, kappa0, kappaL, slowS, slowF, kappaMid, one, fCap, x, T, T0, TL, sdT };
  }

  // ---------------------------------------------------------------- simulation

  function syncSampler(): void {
    const key = [s.L, s.D, s.start === 'point' ? s.x0 : 0, s.start, s.paths, s.seed].join();
    if (key === samplerKey && (sampler || !s.mc)) return;
    clearTimeout(mcTimer);
    if (key !== samplerKey) {
      sampler = null;
      mc = null;
    }
    samplerKey = key;
    if (!s.mc || sampler) return;
    // Wait for slider drags to settle before starting a fresh run.
    mcTimer = window.setTimeout(() => {
      sampler = new ExitSampler({ L: s.L, D: s.D, x0: s.x0, start: s.start, n: s.paths, seed: s.seed });
      schedule();
    }, 150);
  }

  function deriveMc(): void {
    if (!sampler || sampler.count < 2) {
      mc = null;
      return;
    }
    const n = sampler.count;
    const sorted = sampler.times.slice(0, n).sort();
    const ax = timeScale();
    const sT = new Float64Array(S_DOTS);
    const sY = new Float64Array(S_DOTS);
    for (let k = 0; k < S_DOTS; k++) {
      sT[k] = ax.atFraction((k + 0.5) / S_DOTS);
      sY[k] = survivingFraction(sorted, sT[k]);
    }
    const edges = new Float64Array(F_BINS + 1);
    for (let k = 0; k <= F_BINS; k++) edges[k] = ax.atFraction(k / F_BINS);
    mc = {
      stats: empiricalStats(sampler.times, sampler.wall, sorted),
      sT,
      sY,
      hist: exitHistogram(sampler.times, sampler.wall, n, edges, s.tScale === 'log'),
    };
  }

  // ---------------------------------------------------------------- drawing

  function schedule(): void {
    if (!frame) frame = requestAnimationFrame(tick);
  }

  function tick(): void {
    frame = 0;
    if (sampler && s.mc && !sampler.done) {
      sampler.advance(12);
      const now = performance.now();
      if (sampler.done || now - lastDerive > 150) {
        deriveMc();
        lastDerive = now;
      }
      schedule();
    }
    draw();
  }

  function draw(): void {
    const theme = host.theme();
    const c = curves;
    const [lo, hi] = timeDomain();
    // At t = 0 the point start is a δ function: keep the snapshot just after it.
    const tFloor = s.tScale === 'log' ? lo : hi / 2000;
    s.t = Math.min(hi, Math.max(tFloor, s.t));
    const xAxis: ChartSpec['x'] = { domain: [lo, hi], scale: s.tScale, label: 't' };
    const heading = (t: number) => `t = ${formatValue(t)}`;
    const showMc = s.mc && mc !== null;
    const marks: Rule[] = s.marks
      ? [
          { axis: 'x', value: stats.mean, color: 'muted', label: '⟨T⟩' },
          { axis: 'x', value: stats.median, color: 'muted', dash: 'dotted', label: 'median' },
        ]
      : [];
    const one = s.oneWall ? c.one : null;
    const common = { cursor: s.t, cursorLabel: 't*', hover: hoverT, heading };

    // Survival
    const sSeries: Series[] = [{ kind: 'line', label: 'S(t)', color: 'ink', x: c.t, y: c.S, z: 1 }];
    if (s.slowest) sSeries.push({ kind: 'line', label: 'slowest mode c₁ exp(−λ₁t)', color: 'aqua', dash: 'dashed', x: c.t, y: c.slowS, noScale: true });
    if (one) sSeries.push({ kind: 'line', label: 'walls separately', color: 'ink2', dash: 'dotted', x: c.t, y: one.S, noScale: true });
    if (showMc) sSeries.push({ kind: 'dots', label: 'simulated', color: 'ink', x: mc!.sT, y: mc!.sY });
    cardS.render(
      {
        x: xAxis,
        y: s.sScale === 'log' ? { scale: 'log', decades: 6 } : { scale: 'linear', domain: [0, 1.04] },
        series: sSeries,
        rules: marks,
        ...common,
      },
      theme,
    );

    // Exit-time density
    const fSeries: Series[] = [{ kind: 'line', label: 'f(t)', color: 'ink', x: c.t, y: c.f, z: 2 }];
    if (s.split) {
      fSeries.push({ kind: 'line', label: 'exit at 0', color: 'blue', x: c.t, y: c.j0, z: 1 });
      fSeries.push({ kind: 'line', label: 'exit at L', color: 'orange', x: c.t, y: c.jL, z: 1 });
    }
    if (s.slowest) fSeries.push({ kind: 'line', label: 'slowest mode λ₁c₁ exp(−λ₁t)', color: 'aqua', dash: 'dashed', x: c.t, y: c.slowF, noScale: true });
    if (one) {
      if (s.split) {
        fSeries.push({ kind: 'line', label: 'wall at 0 alone', color: 'blue', dash: 'dotted', x: c.t, y: one.j0, noScale: true });
        fSeries.push({ kind: 'line', label: 'wall at L alone', color: 'orange', dash: 'dotted', x: c.t, y: one.jL, noScale: true });
      } else fSeries.push({ kind: 'line', label: 'walls separately', color: 'ink2', dash: 'dotted', x: c.t, y: one.f, noScale: true });
    }
    if (showMc) {
      const h = mc!.hist;
      if (s.split) {
        fSeries.push({ kind: 'dots', label: 'simulated, exit at 0', color: 'blue', x: h.centers, y: h.wall0 });
        fSeries.push({ kind: 'dots', label: 'simulated, exit at L', color: 'orange', x: h.centers, y: h.wallL });
      } else fSeries.push({ kind: 'dots', label: 'simulated', color: 'ink', x: h.centers, y: h.total });
    }
    cardF.render(
      {
        x: xAxis,
        y: s.fScale === 'log' ? { scale: 'log', decades: 5 } : { scale: 'linear', max: c.fCap },
        series: fSeries,
        rules: marks.slice(0, 1),
        ...common,
      },
      theme,
    );

    // Exit rate
    const kSeries: Series[] = [{ kind: 'line', label: 'κ(t)', color: 'ink', x: c.t, y: c.kappa, z: 1 }];
    if (s.split) {
      kSeries.push({ kind: 'line', label: 'through 0', color: 'blue', x: c.t, y: c.kappa0 });
      kSeries.push({ kind: 'line', label: 'through L', color: 'orange', x: c.t, y: c.kappaL });
    }
    kSeries.push({ kind: 'line', label: 'from x₀ = L/2', color: 'muted', dash: 'dashed', width: 1.5, x: c.t, y: c.kappaMid });
    cardK.render(
      {
        x: xAxis,
        y: { scale: 'linear', max: 3 * model.lambda1 },
        series: kSeries,
        rules: [{ axis: 'y', value: model.lambda1, color: 'aqua', label: 'λ₁' }],
        ...common,
      },
      theme,
    );

    // Density snapshot
    const tSnap = hoverT === null ? s.t : Math.max(tFloor, hoverT);
    const xs = grid(0, s.L, X_SAMPLES, 'linear');
    const p = model.density(xs, tSnap);
    const S = model.survival(tSnap);
    const q = p.map((v) => (S > 0 ? v / S : NaN));
    const phi = xs.map((x) => qsdDensity(x, s.L));
    cardP.setTitle(`${hoverT === null ? 'Density at t*' : 'Density at t'} = ${formatValue(tSnap)}`);
    cardP.render(
      {
        x: { domain: [0, s.L], scale: 'linear', label: 'x' },
        y: { scale: 'linear' },
        series: [
          { kind: 'line', label: 'p(x, t)', color: 'ink', x: xs, y: p, z: 1 },
          { kind: 'line', label: 'survivors q = p/S', color: 'aqua', x: xs, y: q },
          { kind: 'line', label: 'QSD φ₀(x)', color: 'aqua', dash: 'dashed', x: xs, y: phi },
        ],
        heading: (x) => `x = ${formatValue(x)}`,
      },
      theme,
    );
    tRange.value = String(Math.round(timeScale().fraction(s.t) * 1000));
    if (document.activeElement !== tBox) tBox.value = num(s.t);

    // Mean exit time against x₀
    const tSeries: Series[] = [
      { kind: 'line', label: '⟨T⟩', color: 'ink', x: c.x, y: c.T, z: 1 },
      { kind: 'line', label: 'σ_T', color: 'ink', dash: 'dashed', width: 1.5, x: c.x, y: c.sdT },
    ];
    if (s.split) {
      tSeries.push({ kind: 'line', label: '⟨T | exit at 0⟩', color: 'blue', x: c.x, y: c.T0 });
      tSeries.push({ kind: 'line', label: '⟨T | exit at L⟩', color: 'orange', x: c.x, y: c.TL });
    }
    // A constant: the mean exit time from φ₀, i.e. ⟨T⟩(x₀) averaged over the QSD.
    tSeries.push({ kind: 'line', label: '1/λ₁, from φ₀', color: 'aqua', dash: 'dashed', width: 1.5, x: [0, s.L], y: [1 / model.lambda1, 1 / model.lambda1] });
    cardT.render(
      {
        x: { domain: [0, s.L], scale: 'linear', label: 'x₀' },
        y: { scale: 'linear' },
        series: tSeries,
        cursor: s.start === 'point' ? s.x0 : null,
        cursorLabel: 'x₀',
        heading: (x) => `x₀ = ${formatValue(x)}`,
      },
      theme,
    );

    drawSnapshots(theme);
    drawSplitting(theme);
    drawTiles();
  }

  /** The notes' density snapshots at Dt/L² = 0.01 … 1. */
  function drawSnapshots(theme: Theme): void {
    const xs = grid(0, s.L, X_SAMPLES, 'linear');
    const roles: Role[] = ['seq1', 'seq2', 'seq3', 'seq4', 'seq5'];
    const pSeries: Series[] = [];
    const qSeries: Series[] = [];
    [0.01, 0.03, 0.1, 0.3, 1].forEach((tau, i) => {
      const t = (tau * s.L * s.L) / s.D;
      const p = model.density(xs, t);
      const S = model.survival(t);
      pSeries.push({ kind: 'line', label: num(tau), color: roles[i], width: 1.6, x: xs, y: p });
      qSeries.push({ kind: 'line', label: num(tau), color: roles[i], width: 1.6, x: xs, y: p.map((v) => v / S) });
    });
    qSeries.push({ kind: 'line', label: 'QSD', color: 'ink', dash: 'dashed', width: 1.25, x: xs, y: xs.map((x) => qsdDensity(x, s.L)) });
    const x: ChartSpec['x'] = { domain: [0, s.L], scale: 'linear', label: 'x' };
    const heading = (v: number) => `x = ${formatValue(v)}`;
    cardSnapP.render({ x, y: { scale: 'linear', max: 4.2 / s.L }, legendTitle: 'Dt/L²', series: pSeries, heading }, theme);
    cardSnapQ.render({ x, y: { scale: 'linear', max: 4.2 / s.L }, legendTitle: 'Dt/L²', series: qSeries, heading }, theme);
  }

  function drawSplitting(theme: Theme): void {
    const xs = [0, s.L];
    const series: Series[] = [
      { kind: 'line', label: 'π₀ = 1 − x₀/L (exit at 0)', color: 'blue', x: xs, y: [1, 0] },
      { kind: 'line', label: 'π_L = x₀/L (exit at L)', color: 'orange', x: xs, y: [0, 1] },
    ];
    const e = s.mc && s.start === 'point' ? mc?.stats : undefined;
    if (e && Number.isFinite(e.piL)) {
      series.push({ kind: 'dots', label: 'simulated, exit at L', color: 'orange', x: [s.x0], y: [e.piL] });
      series.push({ kind: 'dots', label: 'simulated, exit at 0', color: 'blue', x: [s.x0], y: [1 - e.piL] });
    }
    cardPi.render(
      {
        x: { domain: [0, s.L], scale: 'linear', label: 'x₀' },
        y: { scale: 'linear', domain: [0, 1.02] },
        series,
        cursor: s.start === 'point' ? s.x0 : null,
        cursorLabel: 'x₀',
        heading: (v) => `x₀ = ${formatValue(v)}`,
      },
      theme,
    );
  }

  function drawTiles(): void {
    const e = s.mc ? mc?.stats : undefined;
    const sim = (v: number | undefined, se?: number) =>
      v === undefined || !Number.isFinite(v) ? '' : `sim. ${formatValue(v)}${se !== undefined ? ` ± ${Number(se.toPrecision(2))}` : ''}`;
    tLambda.v.textContent = formatValue(model.lambda1);
    tLambda.sub.textContent = `π²D/2L², 1/λ₁ = ${formatValue(1 / model.lambda1)}`;
    tMean.v.textContent = formatValue(stats.mean);
    tMean.sub.textContent = sim(e?.mean, e?.seMean);
    tSd.v.textContent = formatValue(stats.sd);
    tSd.sub.textContent = sim(e?.sd);
    tMedian.v.textContent = formatValue(stats.median);
    tMedian.sub.textContent = sim(e?.median);
    tPiL.v.textContent = formatValue(stats.piL);
    tPiL.sub.textContent = sim(e?.piL, e?.sePiL);
    tMean0.v.textContent = formatValue(stats.mean0);
    tMean0.sub.textContent = sim(e?.mean0);
    tMeanL.v.textContent = formatValue(stats.meanL);
    tMeanL.sub.textContent = sim(e?.meanL);

    if (!s.mc) mcStatus.textContent = 'Off.';
    else if (!sampler) mcStatus.textContent = 'Starting…';
    else if (!sampler.done) mcStatus.textContent = `Simulating: ${sampler.count.toLocaleString()} of ${s.paths.toLocaleString()} paths.`;
    else mcStatus.textContent = `${s.paths.toLocaleString()} paths simulated. Simulated values are shown under each statistic.`;
  }

  // ---------------------------------------------------------------- go

  const observer = new ResizeObserver(() => schedule());
  observer.observe(cards);
  syncControls();
  syncSampler();
  schedule();

  return {
    redraw: schedule,
    destroy: () => {
      cancelAnimationFrame(frame);
      clearTimeout(mcTimer);
      clearTimeout(persistTimer);
      observer.disconnect();
    },
  };
}

export const intervalPage: PageDef = {
  id: 'interval',
  title: 'Free diffusion, absorbing walls',
  group: 'Part I · Free diffusion',
  mount,
};
