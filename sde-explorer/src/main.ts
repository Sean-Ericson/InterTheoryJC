import './style.css';
import { exactModel, stationaryDensity, type ExactModel } from './analytic';
import { Covariance, ensembleCovariance, meanOf } from './correlation';
import { FokkerPlanck, type GridConfig } from './fpe';
import { buildProcess, FormulaError, type BC, type Interp, type Process } from './model';
import { CUSTOM_ID, findPreset } from './presets';
import { lowestEigenvalues, quasiStationary, type QuasiStationary } from './qsd';
import { Recorder } from './recorder';
import { CorrelationPanel } from './render/correlationPanel';
import { DensityPanel, histogram, type Histogram } from './render/densityPanel';
import { MomentsPanel } from './render/momentsPanel';
import { Heatmap, PathsPanel } from './render/pathsPanel';
import { formatValue } from './render/scale';
import { readTheme } from './render/theme';
import { Ensemble } from './sde';
import { MAX_SHOWN, SNAPSHOTS, applyPreset, applyProcessLink, defaultSettings, paramValues, type Start } from './settings';
import { Controls, type ChangeKind } from './ui/controls';

interface Run {
  proc: Process;
  ens: Ensemble;
  fp: FokkerPlanck;
  rec: Recorder;
  exact: ExactModel | null;
  stationary: Float64Array | null;
  qsd: QuasiStationary | null;
  T: number;
  dt: number;
  nSteps: number;
  stepsPerSnap: number;
  step: number;
  domain: [number, number];
  bc: [BC, BC];
  absorbing: boolean;
  grid: GridConfig;
  /** Solver steps per snapshot interval for the autocovariances. */
  covSubsteps: number;
  /** Stationary (or quasi-stationary) autocovariance G(τ), filled as the frames allow. */
  statCov: Covariance | null;
  /** Why there is none. */
  statCovReason: string;
  /** Slowest relaxation rate of the stationary state: λ₁, or λ₁ − λ₀ for φ₀. */
  relax: number | null;
  /** Two-time covariance from the snapshot at the playback time. */
  twoTime: TwoTime | null;
  /** Messages that hold for the whole run. */
  notes: string[];
  exactCache: { k: number; cond: boolean; values: Float64Array } | null;
  histCache: { k: number; cond: boolean; hist: Histogram } | null;
}

/** ⟨X(s + τ), X(s)⟩ from snapshot j (s = t_j), by each method. */
interface TwoTime {
  j: number;
  fp: Covariance;
  ens: Float64Array;
  /** Lags of `ens` filled so far. */
  ensCount: number;
  exact: Float64Array | null;
}

function $<T extends HTMLElement = HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

// Deep links: ?preset=double-well&interp=strat&t=12&theme=light opens that
// preset paused at t; start=stationary and cond=0|1 pick the initial density
// and conditioning (the rest is applied once the controls exist, below).
const query = new URLSearchParams(location.search);
const themeParam = query.get('theme');
if (themeParam === 'light' || themeParam === 'dark') document.documentElement.dataset.theme = themeParam;

const settings = defaultSettings();
const controls = new Controls(settings, { onChange });
const pathsPanel = new PathsPanel($<HTMLCanvasElement>('cv-paths'));
const densityPanel = new DensityPanel($<HTMLCanvasElement>('cv-density'));
const momentsPanel = new MomentsPanel($<HTMLCanvasElement>('cv-moments'));
const corrPanel = new CorrelationPanel($<HTMLCanvasElement>('cv-corr'));
const heat = new Heatmap();
const tooltip = $('tooltip');
const playButton = $<HTMLButtonElement>('play');
const scrub = $<HTMLInputElement>('scrub');

let theme = readTheme();
let run: Run | null = null;
let playT = 0;
let playing = true;
let dirty = true;
let lastFrame = performance.now();
let rebuildTimer = 0;
let hoverT: number | null = null;
let hoverX: number | null = null;
let hoverTau: number | null = null;

// ---------------------------------------------------------------- runs

function onChange(kind: ChangeKind): void {
  if (kind === 'sim') {
    clearTimeout(rebuildTimer);
    rebuildTimer = window.setTimeout(startRun, 120);
  } else if (kind === 'view') {
    if (run) updateHeat(run);
    updateLabels();
  }
  dirty = true;
}

/** Whether the views show q = p/S: asked for, and there is a wall to be absorbed at. */
function conditioned(r: Run | null): boolean {
  return !!r && r.absorbing && settings.view.conditioned;
}

function updateHeat(r: Run): void {
  heat.update(r.rec, settings.view.heatScale, theme.lut, conditioned(r));
}

function startRun(): void {
  const s = settings;
  let proc: Process;
  try {
    proc = buildProcess({ drift: s.drift, noise: s.noise, params: paramValues(s), interp: s.interp });
  } catch (err) {
    if (err instanceof FormulaError) controls.setFormulaError(err);
    return;
  }
  controls.setFormulaError(null);

  const notes: string[] = [];
  const [lo, hi] = s.domain;
  const h = (hi - lo) / s.nGrid;
  // Both the paths and the solver start from N(x0, σ0²); the solver needs it
  // at least two cells wide to resolve it.
  const sigmaEff = Math.max(s.sigma0, 2 * h);
  const grid: GridConfig = { n: s.nGrid, domain: [lo, hi], bc: [...s.bc], x0: s.x0, sigma0: sigmaEff };
  const fp = new FokkerPlanck(proc, grid);

  // With an absorbing wall the stationary state of q = p/S is the
  // quasi-stationary density φ₀, the ground state of the discretised operator.
  const absorbing = s.bc.includes('absorbing');
  let qsd: QuasiStationary | null = null;
  let stationary: Float64Array | null = null;
  let statReason: string;
  if (absorbing) {
    statReason = 'No quasi-stationary density: probability never reaches the absorbing wall.';
    if (proc.timeDependent) statReason = 'No quasi-stationary density: the coefficients depend on t.';
    else qsd = quasiStationary(fp.operator(), fp.h);
    if (qsd && leaksAtOpenEdge(qsd.density, s.bc, hi - lo)) {
      qsd = null;
      statReason = 'No quasi-stationary density: φ₀ reaches the open edge, where the solver puts a wall. Widen the domain, or make that edge reflecting.';
    }
  } else {
    stationary = stationaryDensity(proc, fp.xc, fp.h, s.bc);
    statReason = 'No stationary density: b vanishes inside the domain.';
    if (proc.timeDependent) statReason = 'No stationary density: the coefficients depend on t.';
    else if (stationary && leaksAtOpenEdge(stationary, s.bc, hi - lo)) {
      stationary = null;
      statReason = 'No normalisable stationary density on the real line.';
    }
  }
  const statState = qsd?.density ?? stationary;
  controls.setStationaryKind(absorbing ? 'qsd' : 'stationary');
  controls.setAvailable('stationary', statState !== null, statReason);
  controls.setAvailable('conditioned', absorbing, 'Conditioning on survival needs an absorbing wall.');
  controls.setStartAvailable(statState !== null, statReason);

  // Started in the stationary state, the density stays put; started in φ₀,
  // q stays put and S(t) = e^{−λ₀t} exactly.
  const start = s.start === 'stationary' ? statState : null;
  if (start) fp.p.set(start);
  else if (s.sigma0 < 2 * h) notes.push(`Initial spread σ₀ raised to ${formatValue(sigmaEff)} (two grid cells) so the solver can resolve it.`);
  if (proc.numericDb) notes.push('∂ₓb is computed numerically: math.js could not differentiate b symbolically.');

  const ens = new Ensemble(proc, {
    n: s.nPaths,
    x0: s.x0,
    sigma0: sigmaEff,
    domain: [lo, hi],
    bc: [...s.bc],
    seed: s.seed,
    initial: start ? { lo, h: fp.h, p: start } : undefined,
  });
  const rec = new Recorder(SNAPSHOTS, s.steps, s.nGrid, s.nPaths, Math.min(MAX_SHOWN, s.nPaths));

  // Exact solution: only for an unedited preset with the boundaries it was derived for.
  let exact: ExactModel | null = null;
  let exactReason = 'No closed-form solution for this process.';
  const preset = s.presetId !== CUSTOM_ID ? findPreset(s.presetId) : undefined;
  if (preset?.exact) {
    const want = preset.exact.bc;
    if (start) {
      exactReason = 'The exact solution assumes the Gaussian start.';
    } else if (want[0] === s.bc[0] && want[1] === s.bc[1]) {
      exact = exactModel({ kind: preset.exact.kind, params: paramValues(s), interp: s.interp, x0: s.x0, s0: sigmaEff, domain: [lo, hi] });
      if (!exact) exactReason = 'The exact series is numerically unstable at these parameters.';
    } else {
      exactReason = `The exact solution assumes ${BC_TEXT[want[0]]} / ${BC_TEXT[want[1]]} boundaries.`;
    }
  }
  controls.setAvailable('exact', exact !== null, exactReason);

  // Stationary autocovariance G(τ): the Fokker–Planck equation again, started
  // from the stationary state. Its slowest mode sets the long-time decay.
  // Two steps per snapshot keep the curves within ~1e-3 of G(0) at a few ms each.
  const covSubsteps = Math.min(s.steps / SNAPSHOTS, 2);
  const dtau = s.T / SNAPSHOTS;
  let statCov: Covariance | null = null;
  let relax: number | null = null;
  if (statState && s.bc[0] !== 'periodic') {
    const mean = meanOf(statState, fp.xc);
    const lambda0 = qsd?.lambda0 ?? 0;
    statCov = new Covariance(proc, grid, statState, 0, SNAPSHOTS + 1, dtau, covSubsteps, (j) => ({ mass: Math.exp(-lambda0 * j * dtau), mean }));
    if (qsd) relax = qsd.gap;
    else {
      const [l0, l1] = lowestEigenvalues(fp.operator(), 2);
      relax = l1 - l0;
    }
  }

  const nSteps = s.steps;
  run = {
    proc,
    ens,
    fp,
    rec,
    exact,
    stationary,
    qsd,
    T: s.T,
    dt: s.T / nSteps,
    nSteps,
    stepsPerSnap: nSteps / SNAPSHOTS,
    step: 0,
    domain: [lo, hi],
    bc: [...s.bc],
    absorbing,
    grid,
    covSubsteps,
    statCov,
    statCovReason: statReason,
    relax,
    twoTime: null,
    notes,
    exactCache: null,
    histCache: null,
  };
  rec.recordTraces(ens);
  rec.snapshot(0, ens, fp, exact);
  heat.reset(rec.snaps, s.nGrid);
  updateHeat(run);
  playT = 0;
  setPlaying(true);
  updateEquation();
  updateLabels();
  dirty = true;
}

const BC_TEXT: Record<BC, string> = { open: 'open', reflecting: 'reflecting', absorbing: 'absorbing', periodic: 'periodic' };

/** True if an open edge carries noticeable density (edge density × L > 1%). */
function leaksAtOpenEdge(p: ArrayLike<number>, bc: [BC, BC], L: number): boolean {
  return (bc[0] === 'open' && p[0] * L > 0.01) || (bc[1] === 'open' && p[p.length - 1] * L > 0.01);
}

/** Advance the simulation for up to `budget` ms. */
function compute(r: Run, budget: number): void {
  const t0 = performance.now();
  while (r.step < r.nSteps) {
    r.ens.step(r.dt);
    r.fp.step(r.dt);
    r.step++;
    r.rec.recordTraces(r.ens);
    if (r.step % r.stepsPerSnap === 0) r.rec.snapshot(r.step * r.dt, r.ens, r.fp, r.exact);
    if ((r.step & 7) === 0 && performance.now() - t0 > budget) break;
  }
  updateHeat(r);
}

// ---------------------------------------------------------------- playback

function frontier(r: Run): number {
  return r.rec.times[r.rec.count - 1];
}

function setPlaying(on: boolean): void {
  playing = on;
  playButton.textContent = on ? 'Pause' : 'Play';
  playButton.setAttribute('aria-pressed', String(on));
}

function seek(t: number): void {
  if (!run) return;
  playT = Math.max(0, Math.min(t, frontier(run)));
  dirty = true;
}

function frame(now: number): void {
  const dtReal = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  if (run) {
    if (run.step < run.nSteps) {
      compute(run, 10);
      dirty = true;
    }
    if (playing) {
      playT = Math.min(playT + (dtReal * run.T) / settings.duration, frontier(run));
      if (playT >= run.T * (1 - 1e-9)) setPlaying(false);
      dirty = true;
    }
    if (settings.view.correlation && advanceCorrelation(run, 6)) dirty = true;
  }
  if (dirty) {
    dirty = false;
    render();
  }
  requestAnimationFrame(frame);
}

/**
 * Extend the autocovariances for up to `budget` ms: the stationary one, and
 * the two-time one from the snapshot at the playback time. Returns whether
 * anything changed.
 */
function advanceCorrelation(r: Run, budget: number): boolean {
  if (r.bc[0] === 'periodic') return false;
  const deadline = performance.now() + budget;
  let changed = false;
  if (r.statCov && !r.statCov.done) {
    r.statCov.advance(deadline);
    changed = true;
  }
  // Follow the playback time, but while playing let each curve finish (or
  // catch up with the main run) first, so it is never left half drawn.
  const j = snapshotIndex(r, playT);
  const prev = r.twoTime;
  const settled = !prev || prev.fp.done || prev.fp.count >= r.rec.count - prev.j;
  if (prev?.j !== j && (!playing || settled)) {
    r.twoTime = twoTimeFrom(r, j);
    changed = true;
  }
  const tt = r.twoTime!;
  const before = tt.fp.count;
  tt.fp.advance(deadline);
  const ensCount = ensembleCovariance(r.rec, j, tt.ens, tt.ensCount);
  changed ||= tt.fp.count !== before || ensCount !== tt.ensCount;
  tt.ensCount = ensCount;
  return changed;
}

function twoTimeFrom(r: Run, j: number): TwoTime {
  const { rec } = r;
  const lags = rec.W + 1 - j;
  const s = rec.times[j];
  const dtau = r.T / rec.W;
  // Mean and survivors at later times come from the main run as it reaches them.
  const fp = new Covariance(r.proc, r.grid, rec.densityAt(j), s, lags, dtau, r.covSubsteps, (i) =>
    j + i < rec.count ? { mass: rec.fp.mass[j + i], mean: rec.fp.mean[j + i] } : null,
  );
  const cov = r.exact?.covariance;
  const exact = cov ? Float64Array.from({ length: lags }, (_, i) => cov(s, s + i * dtau)) : null;
  return { j, fp, ens: new Float64Array(lags).fill(NaN), ensCount: 0, exact };
}

// ---------------------------------------------------------------- drawing

function snapshotIndex(r: Run, t: number): number {
  return Math.max(0, Math.min(r.rec.count - 1, Math.round((t / r.T) * r.rec.W)));
}

/** Fokker–Planck p at snapshot k, or q = p/S when conditioned. */
function fpAt(r: Run, k: number): Float64Array {
  const p = r.rec.densityAt(k);
  return conditioned(r) ? p.map((v) => v / r.rec.fp.mass[k]) : p;
}

function exactAt(r: Run, k: number): Float64Array | null {
  if (!r.exact) return null;
  const cond = conditioned(r);
  if (r.exactCache?.k !== k || r.exactCache.cond !== cond) {
    const ex = r.exact.at(r.rec.times[k]);
    const scale = cond ? 1 / ex.mass : 1;
    r.exactCache = { k, cond, values: r.fp.xc.map((x) => ex.density(x) * scale) };
  }
  return r.exactCache.values;
}

function histAt(r: Run, k: number): Histogram {
  const cond = conditioned(r);
  if (r.histCache?.k !== k || r.histCache.cond !== cond) {
    const { x, alive } = r.rec.positionsAt(k);
    const bins = Math.round(Math.min(80, Math.max(20, 1.2 * Math.sqrt(r.rec.nPaths))));
    r.histCache = { k, cond, hist: histogram(x, alive, r.domain, bins, cond) };
  }
  return r.histCache.hist;
}

/**
 * The stationary density, or the quasi-stationary φ₀. Unconditioned, φ₀ is
 * scaled by the surviving probability, since p → S(t) φ₀ at long times.
 */
function statAt(r: Run, k: number): Float64Array | null {
  if (!r.qsd) return r.stationary;
  const mass = r.rec.fp.mass[k];
  return conditioned(r) ? r.qsd.density : r.qsd.density.map((v) => v * mass);
}

function render(): void {
  const v = settings.view;
  const r = run;
  const k = r ? snapshotIndex(r, playT) : 0;
  const cond = conditioned(r);
  pathsPanel.draw({
    theme,
    rec: r?.rec ?? null,
    T: r?.T ?? settings.T,
    domain: r?.domain ?? settings.domain,
    periodic: r?.bc[0] === 'periodic',
    t: playT,
    k,
    step: r ? Math.min(r.rec.traceCount - 1, Math.round(playT / r.dt)) : 0,
    showHeat: v.heatmap,
    name: cond ? 'q' : 'p',
    showPaths: v.paths,
    nShow: settings.nShow,
    heat,
  });
  densityPanel.draw({
    theme,
    domain: r?.domain ?? settings.domain,
    t: r ? r.rec.times[k] : 0,
    name: cond ? 'q' : 'p',
    xc: r?.fp.xc ?? new Float64Array(0),
    fp: r && v.fp ? fpAt(r, k) : null,
    hist: r && v.histogram ? histAt(r, k) : null,
    exact: r && v.exact ? exactAt(r, k) : null,
    stationary: r && v.stationary ? statAt(r, k) : null,
    hoverX,
  });
  momentsPanel.draw({
    theme,
    rec: r?.rec ?? null,
    T: r?.T ?? settings.T,
    k,
    t: playT,
    right: !r?.absorbing ? 'variance' : cond ? 'rate' : 'mass',
    showFp: v.fp,
    showEns: v.paths || v.histogram,
    showExact: v.exact && !!r?.exact,
    lambda0: r?.qsd && v.stationary ? r.qsd.lambda0 : null,
    hoverT,
  });
  if (v.correlation) drawCorrelation(r);

  const T = r?.T ?? settings.T;
  $('time').textContent = `t = ${playT.toFixed(T < 10 ? 2 : 1)} / ${formatValue(T)}`;
  scrub.value = String(Math.round((playT / T) * 1000));
  updateStatus(k);
}

function drawCorrelation(r: Run | null): void {
  const v = settings.view;
  const tt = r?.twoTime ?? null;
  corrPanel.draw({
    theme,
    T: r?.T ?? settings.T,
    dtau: r ? r.T / r.rec.W : 1,
    s: r && tt ? r.rec.times[tt.j] : 0,
    fp: tt && v.fp ? tt.fp : null,
    ens: tt && (v.paths || v.histogram) ? { values: tt.ens, count: tt.ensCount } : null,
    exact: tt?.exact && v.exact ? { values: tt.exact, count: tt.exact.length } : null,
    stationary: r?.statCov ?? null,
    overlayStationary: v.stationary,
    qsd: !!r?.qsd,
    relax: r?.relax ?? null,
    message: r?.bc[0] === 'periodic' ? 'x wraps around a ring, so it has no autocorrelation.' : null,
    statMessage: r?.statCovReason ?? null,
    hoverTau,
  });
}

function updateStatus(k: number): void {
  const r = run;
  const items: string[] = [];
  if (r) {
    if (r.step < r.nSteps) items.push(`Computing… ${Math.round((100 * r.step) / r.nSteps)}%`);
    items.push(...r.notes);
    if (r.bc.includes('open')) {
      const p = r.rec.densityAt(k);
      const L = r.domain[1] - r.domain[0];
      const sides = [r.bc[0] === 'open' && p[0] * L > 0.01 ? 'left' : '', r.bc[1] === 'open' && p[p.length - 1] * L > 0.01 ? 'right' : ''].filter(Boolean);
      if (sides.length) {
        items.push(`The density has reached the ${sides.join(' and ')} edge. The solver treats an open edge as a wall, so widen the domain for an accurate comparison.`);
      }
    }
    if (conditioned(r) && (settings.view.paths || settings.view.histogram)) {
      const alive = Math.round(r.rec.ens.mass[k] * r.rec.nPaths);
      if (alive < 50) items.push(`Only ${alive} of ${r.rec.nPaths.toLocaleString()} sample paths survive to this time, so their histogram and exit rate are noisy.`);
    }
  }
  const box = $('status');
  box.replaceChildren(
    ...items.map((text) => {
      const li = document.createElement('li');
      li.textContent = text;
      return li;
    }),
  );
}

function updateEquation(): void {
  const s = settings;
  $('eq-sde').textContent = `dx = [ ${s.drift} ] dt + [ ${s.noise} ] dW(t)` + (s.interp === 'strat' ? '   (Stratonovich)' : '   (Itô)');
  $('eq-fpe-a').textContent = s.interp === 'strat' ? 'A = a + ½ b ∂ₓb' : 'A = a';
  const qsd = run?.qsd;
  $('eq-qsd').hidden = !qsd;
  if (qsd) {
    $('eq-qsd').textContent = `Quasi-stationary φ₀: λ₀ = ${formatValue(qsd.lambda0)} (decay rate of S), λ₁ − λ₀ = ${formatValue(qsd.gap)} (rate of q → φ₀)`;
  }
}

/** Legend entries and the conditioned equation, which follow the view options. */
function updateLabels(): void {
  const v = settings.view;
  const cond = conditioned(run);
  const show = (id: string, on: boolean) => {
    $(id).hidden = !on;
  };
  show('lg-fp', v.fp);
  show('lg-ens', v.paths || v.histogram);
  show('lg-exact', v.exact && !!run?.exact);
  show('lg-stat', v.stationary && !!(run?.stationary || run?.qsd));
  show('eq-cond', cond);
  show('corr-cell', v.correlation);
  $('lg-fp-text').textContent = `Fokker–Planck ${cond ? 'q' : 'p'}(x,t)`;
  $('lg-stat-text').textContent = !run?.qsd ? 'Stationary pₛ(x)' : cond ? 'Quasi-stationary φ₀(x)' : 'Quasi-stationary S(t) φ₀(x)';
}

// ---------------------------------------------------------------- interaction

function showTooltip(e: PointerEvent, rows: [string, string, string?][]): void {
  tooltip.replaceChildren(
    ...rows.map(([label, value, color]) => {
      const row = document.createElement('div');
      row.className = 'tt-row';
      const key = document.createElement('span');
      key.className = 'tt-key';
      if (color) {
        const sw = document.createElement('i');
        sw.className = 'tt-swatch';
        sw.style.background = color;
        key.append(sw);
      }
      key.append(label);
      const val = document.createElement('span');
      val.className = 'tt-val';
      val.textContent = value;
      row.append(key, val);
      return row;
    }),
  );
  tooltip.hidden = false;
  const pad = 14;
  const { innerWidth, innerHeight } = window;
  const w = tooltip.offsetWidth;
  const h = tooltip.offsetHeight;
  let x = e.clientX + pad;
  let y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - pad - w;
  if (y + h > innerHeight - 8) y = e.clientY - pad - h;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

function hideTooltip(): void {
  tooltip.hidden = true;
}

function local(e: PointerEvent): { px: number; py: number } {
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
  return { px: e.clientX - rect.left, py: e.clientY - rect.top };
}

/**
 * Pointer drag on a time axis scrubs playback. `timeAt` returns null outside
 * the plot unless `dragging`, when it should clamp instead.
 */
function scrubbable(canvas: HTMLCanvasElement, timeAt: (px: number, py: number, dragging: boolean) => number | null): void {
  let dragging = false;
  canvas.addEventListener('pointerdown', (e) => {
    const { px, py } = local(e);
    const t = timeAt(px, py, false);
    if (t === null || !run) return;
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    setPlaying(false);
    seek(t);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const { px, py } = local(e);
    const t = timeAt(px, py, true);
    if (t !== null) seek(t);
  });
  const end = () => {
    dragging = false;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
}

{
  const cv = pathsPanel.canvas;
  const inside = (px: number, py: number) => {
    const p = pathsPanel.plot;
    return px >= p.x && px <= p.x + p.w && py >= p.y && py <= p.y + p.h;
  };
  scrubbable(cv, (px, py, dragging) => (dragging || inside(px, py) ? Math.max(0, pathsPanel.tScale.invert(px)) : null));
  cv.addEventListener('pointermove', (e) => {
    const { px, py } = local(e);
    if (!run || !inside(px, py)) {
      hideTooltip();
      return;
    }
    const t = pathsPanel.tScale.invert(px);
    const x = pathsPanel.xScale.invert(py);
    const rows: [string, string, string?][] = [
      ['t', formatValue(t)],
      ['x', formatValue(x)],
    ];
    if (t <= frontier(run)) {
      const k = snapshotIndex(run, t);
      const i = Math.floor((x - run.domain[0]) / run.fp.h);
      if (i >= 0 && i < run.fp.n) {
        const cond = conditioned(run);
        const p = run.rec.densityAt(k)[i] / (cond ? run.rec.fp.mass[k] : 1);
        rows.push([cond ? 'q(x, t)' : 'p(x, t)', formatValue(Math.max(0, p)), theme.fp]);
      }
    } else {
      rows.push(['', 'not computed yet']);
    }
    showTooltip(e, rows);
  });
  cv.addEventListener('pointerleave', hideTooltip);
}

{
  const cv = densityPanel.canvas;
  cv.addEventListener('pointermove', (e) => {
    const { px, py } = local(e);
    const p = densityPanel.plot;
    const r = run;
    if (!r || px < p.x || px > p.x + p.w || py < p.y || py > p.y + p.h) {
      hideTooltip();
      if (hoverX !== null) {
        hoverX = null;
        dirty = true;
      }
      return;
    }
    hoverX = densityPanel.xScale.invert(py);
    dirty = true;
    const k = snapshotIndex(r, playT);
    const i = Math.min(r.fp.n - 1, Math.max(0, Math.floor((hoverX - r.domain[0]) / r.fp.h)));
    const v = settings.view;
    const rows: [string, string, string?][] = [['x', formatValue(hoverX)]];
    if (v.fp) rows.push(['Fokker–Planck', formatValue(Math.max(0, fpAt(r, k)[i])), theme.fp]);
    if (v.histogram) {
      const hst = histAt(r, k);
      const b = Math.floor((hoverX - hst.lo) / hst.width);
      if (b >= 0 && b < hst.values.length) rows.push(['Paths histogram', formatValue(hst.values[b]), theme.ens]);
    }
    const ex = v.exact ? exactAt(r, k) : null;
    if (ex) rows.push(['Exact', formatValue(ex[i]), theme.exact]);
    const st = v.stationary ? statAt(r, k) : null;
    if (st) rows.push([r.qsd ? 'Quasi-stationary' : 'Stationary', formatValue(st[i]), theme.stat]);
    showTooltip(e, rows);
  });
  cv.addEventListener('pointerleave', () => {
    hideTooltip();
    hoverX = null;
    dirty = true;
  });
}

{
  const cv = momentsPanel.canvas;
  scrubbable(cv, (px, py, dragging) => momentsPanel.timeAt(px, py, dragging));
  cv.addEventListener('pointermove', (e) => {
    const { px, py } = local(e);
    const t = momentsPanel.timeAt(px, py);
    const r = run;
    if (t === null || !r) {
      hideTooltip();
      if (hoverT !== null) {
        hoverT = null;
        dirty = true;
      }
      return;
    }
    hoverT = t;
    dirty = true;
    if (t > frontier(r)) {
      showTooltip(e, [['t', formatValue(t)], ['', 'not computed yet']]);
      return;
    }
    const k = snapshotIndex(r, t);
    const rec = r.rec;
    const cond = conditioned(r);
    const ms = (m: typeof rec.fp) => `${formatValue(m.mean[k])} ± ${formatValue(Math.sqrt(Math.max(0, m.variance[k])))}`;
    const extra = (m: typeof rec.fp) =>
      (r.absorbing ? `, alive ${formatValue(m.mass[k])}` : '') + (cond ? `, κ ${formatValue(m.rate[k])}` : '');
    const rows: [string, string, string?][] = [['t', formatValue(rec.times[k])]];
    if (settings.view.fp) rows.push(['Fokker–Planck', ms(rec.fp) + extra(rec.fp), theme.fp]);
    rows.push(['Paths', ms(rec.ens) + extra(rec.ens), theme.ens]);
    if (r.exact && settings.view.exact) rows.push(['Exact', ms(rec.exact) + extra(rec.exact), theme.exact]);
    if (cond && r.qsd && settings.view.stationary) rows.push(['λ₀', formatValue(r.qsd.lambda0), theme.stat]);
    showTooltip(e, rows);
  });
  cv.addEventListener('pointerleave', () => {
    hideTooltip();
    hoverT = null;
    dirty = true;
  });
}

{
  const cv = corrPanel.canvas;
  cv.addEventListener('pointermove', (e) => {
    const { px, py } = local(e);
    const tau = corrPanel.tauAt(px, py);
    const r = run;
    if (tau === null || !r || r.bc[0] === 'periodic') {
      hideTooltip();
      if (hoverTau !== null) {
        hoverTau = null;
        dirty = true;
      }
      return;
    }
    hoverTau = tau;
    dirty = true;
    const i = Math.round((tau / r.T) * r.rec.W);
    const v = settings.view;
    const val = (values: Float64Array, count: number) => (i < count ? formatValue(values[i]) : '–');
    const rows: [string, string, string?][] = [['τ', formatValue((i * r.T) / r.rec.W)]];
    const tt = r.twoTime;
    if (tt) {
      rows.push(['s', formatValue(r.rec.times[tt.j])]);
      if (v.fp) rows.push(['Fokker–Planck', val(tt.fp.values, tt.fp.count), theme.fp]);
      if (v.paths || v.histogram) rows.push(['Paths', val(tt.ens, tt.ensCount), theme.ens]);
      if (tt.exact && v.exact) rows.push(['Exact', val(tt.exact, tt.exact.length), theme.exact]);
    }
    const g = r.statCov;
    if (g) {
      rows.push(['G(τ)', val(g.values, g.count), theme.stat]);
      rows.push(['ρ(τ)', i < g.count ? formatValue(g.values[i] / g.values[0]) : '–', theme.stat]);
    }
    showTooltip(e, rows);
  });
  cv.addEventListener('pointerleave', () => {
    hideTooltip();
    hoverTau = null;
    dirty = true;
  });
}

playButton.addEventListener('click', () => {
  if (!run) return;
  if (!playing && playT >= run.T * (1 - 1e-9)) playT = 0;
  setPlaying(!playing);
});
$('restart').addEventListener('click', () => {
  playT = 0;
  setPlaying(true);
  dirty = true;
});
scrub.addEventListener('input', () => {
  if (!run) return;
  setPlaying(false);
  seek((Number(scrub.value) / 1000) * run.T);
});

window.addEventListener('keydown', (e) => {
  const target = e.target as HTMLElement;
  if (target.closest('input, select, textarea, button')) return;
  if (!run) return;
  if (e.key === ' ') {
    e.preventDefault();
    playButton.click();
  } else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    e.preventDefault();
    setPlaying(false);
    const dir = e.key === 'ArrowRight' ? 1 : -1;
    seek(playT + (dir * run.T) / run.rec.W);
  }
});

// Redraw on resize and theme change; settings changes that only affect the view redraw too.
new ResizeObserver(() => {
  dirty = true;
}).observe($('plots'));
// Plot Lab, the interactive figures of the notes: deployed alongside, or its dev server.
const PLOT_LAB_URL: string = import.meta.env.VITE_PLOT_LAB_URL ?? (import.meta.env.DEV ? 'http://localhost:5174/' : '../plot-lab/');
$<HTMLAnchorElement>('plot-lab-link').href = PLOT_LAB_URL;

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  theme = readTheme();
  if (run) updateHeat(run);
  dirty = true;
});

const linked = findPreset(query.get('preset') ?? '');
if (linked) applyPreset(settings, linked);
// A process from Plot Lab (or any link with a=…&b=…), applied over a preset.
const custom = applyProcessLink(settings, query);
if (query.get('interp') === 'strat' || query.get('interp') === 'ito') settings.interp = query.get('interp') as Interp;
const startParam = query.get('start') === 'qsd' ? 'stationary' : query.get('start');
if (startParam === 'stationary' || startParam === 'gaussian') settings.start = startParam as Start;
if (query.has('cond')) settings.view.conditioned = query.get('cond') === '1';
controls.sync();
if (custom) $('blurb').textContent = 'Opened from a link: the formulas, parameters, domain and walls below were set by it. Any symbol other than x and t is a parameter.';
startRun();
const linkedT = Number(query.get('t'));
if (run && query.has('t') && Number.isFinite(linkedT)) {
  compute(run, Infinity);
  seek(linkedT);
  setPlaying(false);
}
requestAnimationFrame(frame);
