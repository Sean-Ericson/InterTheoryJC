import type { Series } from '../plot/chart';
import { DIV, SEQ } from '../plot/theme';
import * as R from '../math/drift';
import * as F from '../math/free';
import { geomspace, linspace, memo } from '../math/num';
import { figurePage } from './framework';
import { clamp, family, fmt, heading, line, live, ref, sample, xRule } from './kit';

/**
 * Part II of the notes: constant drift, A = v, B = D. On the interval x is in
 * units of L, t in L²/D and v stands for vL/D; on the half-line, fixed-v
 * panels use the drift's own units ℓ = D/|v| and τ = D/v², and sweeps through
 * v = 0 use x₀ and x₀²/D.
 */
const GROUP = 'Part II · Constant drift';
const PE = [-4, -2, 0, 2, 4];
const V_HALF = [-1, -0.5, 0, 0.5, 1];
const SNAP_T = [0.01, 0.03, 0.1, 0.3, 1];

const drift = (v: number) => ({ a: 'v', b: 'sqrt(D)', params: { v, D: 1 } });

// ---------------------------------------------------------------- §12 half-line, reflecting

export const driftHlReflecting = figurePage({
  id: 'drift-hl-reflecting',
  group: GROUP,
  title: 'Constant drift on the half-line, reflecting at 0',
  files: ['drift_halfline_reflecting.pdf'],
  gardiner: '§5.2.1, §5.2.5',
  intro: 'Drift v with a reflecting wall at 0. For v < 0 the density settles into the barometric profile; for v ≥ 0 it never settles.',
  equations: [
    'p = g(x − x₀ − vt) + e<sup>−2αx₀</sup> g(x + x₀ − vt) − α e<sup>2αx</sup> erfc((x + x₀ + vt)/√(2Dt)),&ensp;α = v/D',
    'pₛ = (2/ℓ) e<sup>−2x/ℓ</sup> for v &lt; 0,&ensp;ℓ = D/|v|',
  ],
  defaults: { t: 1, x0: 1, w: -0.25, tm: 10 },
  controls: [
    {
      title: '(a) Fixed v < 0, drift units',
      controls: [
        { key: 't', label: 't', min: 0.01, max: 30, log: true, play: true },
        { key: 'x0', label: 'x₀', min: 0.1, max: 4, step: 0.01 },
      ],
      hint: 't/τ and x₀/ℓ, with ℓ = D/|v| and τ = D/v². The notes start at x₀ = ℓ.',
    },
    {
      title: '(b) Sweep through v = 0',
      controls: [
        { key: 'w', label: 'v', min: -2, max: 2, step: 0.01 },
        { key: 'tm', label: 't', min: 0.01, max: 100, log: true },
      ],
      hint: 'vx₀/D for the heavy curve, and a time Dt/x₀² to read it at.',
    },
  ],
  model: () => ({ xi: linspace(0, 5, 401), ts: geomspace(1e-2, 1e2, 160) }),
  panels: [
    {
      title: '(a) Relaxation to the barometric profile',
      wide: true,
      caption: 'Relaxation to the barometric profile for v < 0, in drift units ℓ = D/|v|, τ = D/v².',
      spec: (m, v) => ({
        x: { domain: [0, 5], scale: 'linear', label: 'x/ℓ' },
        y: { scale: 'linear', domain: [0, 2.1], label: 'ℓ p(x, t | x₀, 0)' },
        legendTitle: 't/τ',
        series: [
          ...family([0.05, 0.3, 1, 3, 10], m.xi, (x, t) => R.dhrDensity(x, t, v.x0, -1)),
          ref('pₛ', m.xi, sample(m.xi, (x) => 2 * Math.exp(-2 * x))),
          live(`t = ${fmt(v.t)}`, m.xi, sample(m.xi, (x) => R.dhrDensity(x, v.t, v.x0, -1))),
        ],
        heading: heading('x/ℓ'),
      }),
    },
    {
      title: '(b) Mean position',
      wide: true,
      caption: 'Mean position for several drifts, in units of x₀ because the sweep passes through v = 0: saturation at ℓ/2 for v < 0, √t for v = 0, vt for v > 0.',
      interaction: 'Drag to move t.',
      pick: (x) => ({ tm: clamp(x, 0.01, 100) }),
      spec: (m, v) => ({
        x: { domain: [1e-2, 1e2], scale: 'log', label: 'Dt/x₀²' },
        y: { scale: 'log', domain: [0.3, 200], label: '⟨x(t)⟩/x₀' },
        legendTitle: 'vx₀/D',
        series: [
          ...family(V_HALF, m.ts, (t, w) => R.dhrMeanAt(t, 1, w, 100), DIV),
          live(`v = ${fmt(v.w)}`, m.ts, sample(m.ts, (t) => R.dhrMeanAt(t, 1, v.w, 100))),
        ],
        cursor: v.tm,
        cursorLabel: 't',
        heading: heading('Dt/x₀²'),
      }),
    },
  ],
  results: [
    'The Robin image: the gauge turns the reflecting wall into u′ = αu, whose image is a point charge plus an exponential line charge.',
    '⟨x(t)⟩ = x₀ + vt + ½D ∫₀ᵗ p(0, s) ds, since J(0) = 0 makes d⟨x⟩/dt = ∫J dx = v + ½D p(0, t).',
  ],
  simulate: (v) => ({ ...drift(-1), domain: [0, 8], bc: ['reflecting', 'open'], x0: v.x0, T: 10 }),
});

// ---------------------------------------------------------------- §13 half-line, absorbing

export const driftHlAbsorbing = figurePage({
  id: 'drift-hl-absorbing',
  group: GROUP,
  title: 'Constant drift on the half-line, absorbing at 0',
  files: ['drift_halfline_absorbing.pdf'],
  gardiner: '§5.2.7',
  intro: 'Drift v with an absorbing wall at 0. Toward the wall (v < 0) there is a continuum of QSDs, λ ∈ (0, v²/2D]; from a point start q creeps toward the edge QSD.',
  equations: [
    'p = g(x − x₀ − vt) − e<sup>−2vx₀/D</sup> g(x + x₀ − vt),&ensp;f(t) = x₀ e<sup>−(x₀+vt)²/2Dt</sup>/√(2πDt³)',
    'φ<sub>λ</sub> ∝ e<sup>r₊x</sup> − e<sup>r₋x</sup>,&ensp;r<sub>±</sub> = (v ± √(v² − 2Dλ))/D',
  ],
  defaults: { t: 3, x0: 1, w: -0.25, lam: 0.35 },
  controls: [
    {
      title: '(a) Fixed v < 0, drift units',
      controls: [
        { key: 't', label: 't', min: 0.1, max: 300, log: true, play: true },
        { key: 'x0', label: 'x₀', min: 0.1, max: 4, step: 0.01 },
      ],
      hint: 't/τ and x₀/ℓ. The notes start at x₀ = ℓ.',
    },
    { title: '(b) Exit-time density', controls: [{ key: 'w', label: 'v', min: -2, max: 2, step: 0.01 }], hint: 'vx₀/D for the heavy curve.' },
    { title: '(c) QSD family', controls: [{ key: 'lam', label: 'λ', min: 0.02, max: 1, step: 0.01 }], hint: 'λ in units of the edge v²/2D (2λτ in the notes).' },
  ],
  model: () => ({ xi: linspace(0, 8, 401), ts: geomspace(1e-3, 1e2, 400) }),
  panels: [
    {
      title: '(a) Conditioned density, creeping to the edge QSD',
      caption: 'Conditioned density for v < 0, creeping toward the edge QSD x e<sup>−x/ℓ</sup>/ℓ².',
      spec: (m, v) => {
        const q = (x: number, t: number) => R.dhaDensity(x, t, v.x0, -1) / R.dhaSurvival(t, v.x0, -1);
        return {
          x: { domain: [0, 8], scale: 'linear', label: 'x/ℓ' },
          y: { scale: 'linear', domain: [0, 1.5], label: 'ℓ q(x, t)' },
          legendTitle: 't/τ',
          series: [...family([0.3, 1, 3, 10, 30], m.xi, q), ref('edge QSD', m.xi, sample(m.xi, (x) => x * Math.exp(-x))), live(`t = ${fmt(v.t)}`, m.xi, sample(m.xi, (x) => q(x, v.t)))],
          heading: heading('x/ℓ'),
        };
      },
    },
    {
      title: '(b) Exit-time densities',
      caption: 'Exit-time densities, in units of x₀ because the sweep passes through v = 0. For v > 0 the area is e<sup>−2vx₀/D</sup> < 1.',
      spec: (m, v) => ({
        x: { domain: [1e-3, 1e2], scale: 'log', label: 'Dt/x₀²' },
        y: { scale: 'log', domain: [1e-4, 1], label: 'f(t | x₀) x₀²/D' },
        legendTitle: 'vx₀/D',
        series: [...family(V_HALF, m.ts, (t, w) => R.inverseGaussian(t, 1, w), DIV), live(`v = ${fmt(v.w)}`, m.ts, sample(m.ts, (t) => R.inverseGaussian(t, 1, v.w)))],
        heading: heading('Dt/x₀²'),
      }),
    },
    {
      title: '(c) The QSD family',
      caption: 'The QSD family, in drift units: smaller λ means a heavier tail.',
      spec: (m, v) => {
        const phi = (x: number, f: number) => R.dhaQsd(x, f * 0.5, -1);
        return {
          x: { domain: [0, 8], scale: 'linear', label: 'x/ℓ' },
          y: { scale: 'linear', domain: [0, 0.5], label: 'ℓ φ_λ(x)' },
          legendTitle: '2λτ',
          series: [
            ...family([0.2, 0.5, 0.8], m.xi, phi, [SEQ[0], SEQ[2], SEQ[4]]),
            ref('1 (edge)', m.xi, sample(m.xi, (x) => phi(x, 1))),
            live(`λ = ${fmt(v.lam)}`, m.xi, sample(m.xi, (x) => phi(x, v.lam)), { color: 'aqua' }),
          ],
          heading: heading('x/ℓ'),
        };
      },
    },
  ],
  results: [
    'A continuum of QSDs, λ ∈ (0, v²/2D], all positive and integrable; a light-tailed start converges to the edge λ = v²/2D, algebraically.',
    'Away from the wall (v > 0) there is none: S(∞) = 1 − e<sup>−2vx₀/D</sup> > 0.',
  ],
  simulate: (v) => ({ ...drift(-1), domain: [0, 12], bc: ['absorbing', 'open'], x0: v.x0, T: 30, cond: true }),
});

// ---------------------------------------------------------------- §14 reflecting walls

export const driftReflecting = figurePage({
  id: 'drift-reflecting',
  group: GROUP,
  title: 'Constant drift, reflecting walls',
  files: ['drift_reflecting.pdf'],
  gardiner: '§5.2.2, §5.2.5',
  intro: 'Drift v on (0, L) between reflecting walls. The stationary state is a truncated exponential; the gap is v²/2D + π²D/2L².',
  equations: ['pₛ = 2α e<sup>2αx</sup>/(e<sup>2αL</sup> − 1),&ensp;α = v/D', 'p = pₛ + (2/L) e<sup>α(x−x₀) − v²t/2D</sup> Σ φₖ(x)φₖ(x₀) e<sup>−Dk²t/2</sup>/(1 + α²/k²),&ensp;φₖ = cos kx + (α/k) sin kx'],
  defaults: { v: 4, x0: 0.3, t: 0.05 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'v', label: 'v', min: -8, max: 8, step: 0.1 },
        { key: 'x0', label: 'x₀', min: 0, max: 1, step: 0.01 },
        { key: 't', label: 't', min: 0.001, max: 1, log: true, play: true },
      ],
      hint: 'vL/D, x₀/L and Dt/L². The notes use vL/D = 4 in (b) and x₀ = 0.3L.',
    },
  ],
  model: () => ({ x: linspace(0, 1, 401), ts: linspace(0, 0.6, 200) }),
  panels: [
    {
      title: '(a) Stationary densities',
      caption: 'Stationary densities for several vL/D.',
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x/L' },
        y: { scale: 'linear', max: 8.5, label: 'L pₛ(x)' },
        legendTitle: 'vL/D',
        series: [...family(PE, m.x, (x, w) => R.drStationary(x, w), DIV), live(`v = ${fmt(v.v)}`, m.x, sample(m.x, (x) => R.drStationary(x, v.v)))],
        heading: heading('x/L'),
      }),
    },
    {
      title: (_, v) => `(b) Relaxation at vL/D = ${fmt(v.v)}`,
      caption: 'Relaxation from x₀: the peak is carried toward the wall the drift points at while it spreads.',
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x/L' },
        y: { scale: 'linear', max: 8.5, label: 'L p(x, t | x₀, 0)' },
        legendTitle: 'Dt/L²',
        series: [
          ...family(SNAP_T, m.x, (x, t) => R.drDensity(x, t, v.x0, v.v)),
          ref('pₛ', m.x, sample(m.x, (x) => R.drStationary(x, v.v))),
          live(`t = ${fmt(v.t)}`, m.x, sample(m.x, (x) => R.drDensity(x, v.t, v.x0, v.v))),
        ],
        heading: heading('x/L'),
      }),
    },
    {
      title: '(c) Distance of the mean from its stationary value',
      caption: 'Distance of the mean from its stationary value, from x₀; colours as in (a). All decay at the gap v²/2D + π²D/2L².',
      interaction: 'Drag to move t.',
      pick: (x) => ({ t: clamp(x, 0.001, 1) }),
      spec: (m, v) => {
        const d = (t: number, w: number) => Math.abs(R.drMean(t, v.x0, w) - R.drMeanS(w));
        return {
          x: { domain: [0, 0.6], scale: 'linear', label: 'Dt/L²' },
          y: { scale: 'log', domain: [1e-5, 1], label: '|⟨x(t)⟩ − ⟨x⟩ₛ|/L' },
          legendTitle: 'vL/D',
          series: [...family(PE, m.ts, (t, w) => d(t, w), DIV), live(`v = ${fmt(v.v)}`, m.ts, sample(m.ts, (t) => d(t, v.v)))],
          cursor: v.t,
          cursorLabel: 't',
          heading: heading('Dt/L²'),
        };
      },
    },
  ],
  results: ['Gap λ₁ = v²/2D + π²D/2L²: the drift only adds the constant v²/2D, the free spectrum shifted.', '⟨x⟩ₛ = L/(1 − e<sup>−2αL</sup>) − 1/2α.'],
  simulate: (v) => ({ ...drift(v.v), domain: [0, 1], bc: ['reflecting', 'reflecting'], x0: v.x0, T: 0.6 }),
});

// ---------------------------------------------------------------- §15 absorbing walls

export const driftAbsorbing = figurePage({
  id: 'drift-absorbing',
  group: GROUP,
  title: 'Constant drift, absorbing walls',
  files: ['drift_absorbing.pdf'],
  gardiner: '§5.2.7–8',
  intro: 'Drift v on (0, L), absorbed at both walls. The gauge maps the density onto the free absorbing interval, so every exit statistic is closed form.',
  equations: ['q<sub>∞</sub> ∝ e<sup>αx</sup> sin(πx/L),&ensp;λ₁ = v²/2D + π²D/2L²', 'π<sub>L</sub> = (1 − e<sup>−2αx₀</sup>)/(1 − e<sup>−2αL</sup>),&ensp;T = (Lπ<sub>L</sub> − x₀)/v'],
  defaults: { v: 1, x0: 0.3 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'v', label: 'v', min: -8, max: 8, step: 0.1 },
        { key: 'x0', label: 'x₀', min: 0, max: 1, step: 0.01 },
      ],
      hint: 'vL/D for the heavy curves, and x₀/L for the cursor and the simulation.',
    },
  ],
  model: () => ({ x: linspace(1e-6, 1 - 1e-6, 301) }),
  panels: [
    {
      title: '(a) Quasi-stationary densities',
      caption: 'Quasi-stationary densities.',
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x/L' },
        y: { scale: 'linear', max: 4.5, label: 'L q∞(x)' },
        legendTitle: 'vL/D',
        series: [...family(PE, m.x, (x, w) => R.daQsd(x, w), DIV), live(`v = ${fmt(v.v)}`, m.x, sample(m.x, (x) => R.daQsd(x, v.v)))],
        heading: heading('x/L'),
      }),
    },
    ...(
      [
        ['(b) Probability of leaving through L', 'Probability of leaving through L.', 'π_L(x₀)', R.daPiL, [0, 1.02]],
        ['(c) Mean exit time', 'Mean exit time.', 'T(x₀) D/L²', R.daT, [0, 0.3]],
      ] as const
    ).map(([title, caption, label, f, domain]) => ({
      title,
      caption,
      interaction: 'Drag to move x₀.',
      pick: (x: number) => ({ x0: clamp(Math.round(x * 100) / 100, 0, 1) }),
      spec: (m: { x: Float64Array }, v: { v: number; x0: number }) => ({
        x: { domain: [0, 1] as [number, number], scale: 'linear' as const, label: 'x₀/L' },
        y: { scale: 'linear' as const, domain: [domain[0], domain[1]] as [number, number], label },
        legendTitle: 'vL/D',
        series: [...family(PE, m.x, (x, w) => f(x, w), DIV), live(`v = ${fmt(v.v)}`, m.x, sample(m.x, (x) => f(x, v.v)))],
        cursor: v.x0,
        cursorLabel: 'x₀',
        heading: heading('x₀/L'),
      }),
    })),
    {
      title: '(d) Mean exit time given exit through L',
      caption: 'Mean exit time given exit through L. Dashed curves (v < 0) lie exactly on the solid ones with the opposite drift.',
      interaction: 'Drag to move x₀.',
      pick: (x) => ({ x0: clamp(Math.round(x * 100) / 100, 0, 1) }),
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x₀/L' },
        y: { scale: 'linear', domain: [0, 0.36], label: 'T_L(x₀) D/L²' },
        legendTitle: 'vL/D',
        series: [
          ...PE.map((w, i) => line(fmt(w), DIV[i], m.x, sample(m.x, (x) => R.daTL(x, w)), w < 0 ? { dash: 'dashed', width: 1.6, z: 1 } : { width: 1.6 })),
          live(`v = ${fmt(v.v)}`, m.x, sample(m.x, (x) => R.daTL(x, v.v))),
        ],
        cursor: v.x0,
        cursorLabel: 'x₀',
        heading: heading('x₀/L'),
      }),
    },
  ],
  results: ['T<sub>L</sub>(x₀) is even in v: conditioning on the exit wall removes the drift’s direction (Doob’s h-transform).', 'λ₁ = v²/2D + π²D/2L², the reflecting gap.'],
  simulate: (v) => ({ ...drift(v.v), domain: [0, 1], bc: ['absorbing', 'absorbing'], x0: v.x0, T: 0.5, cond: true }),
});

// ---------------------------------------------------------------- §16 reflecting at 0, absorbing at L

const mixedRates = memo(() => {
  const pe = linspace(-6, 6, 241);
  return {
    pe,
    lam0: sample(pe, (w) => R.dmModes(w, 1)[0].lambda),
    invT0: sample(pe, (w) => 1 / R.dmT(0, w)),
    same: sample(pe, R.daLam1),
  };
});

export const driftMixed = figurePage({
  id: 'drift-mixed',
  group: GROUP,
  title: 'Constant drift, reflecting at 0, absorbing at L',
  files: ['drift_mixed.pdf'],
  gardiner: '§5.2.5, §5.2.7',
  intro: 'Drift v on (0, L), reflected at 0 and absorbed at L. For drift toward the mirror stronger than vL/D = −1 a bound state appears and escape becomes exponentially slow.',
  equations: ['k cos kL + α sin kL = 0,&ensp;λ = v²/2D + Dk²/2', 'αL &lt; −1: q cosh qL + α sinh qL = 0,&ensp;λ₀ = v²/2D − Dq²/2'],
  defaults: { v: 1, x0: 0.3 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'v', label: 'v', min: -6, max: 6, step: 0.05 },
        { key: 'x0', label: 'x₀', min: 0, max: 1, step: 0.01 },
      ],
      hint: 'vL/D for the heavy curves, and x₀/L.',
    },
  ],
  model: () => ({ x: linspace(0, 1, 301), rates: mixedRates() }),
  panels: [
    {
      title: '(a) Quasi-stationary densities',
      caption: 'Quasi-stationary densities.',
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x/L' },
        y: { scale: 'linear', max: 4.5, label: 'L q∞(x)' },
        legendTitle: 'vL/D',
        series: [...family(PE, m.x, (x, w) => R.dmQsd(w)(x), DIV), live(`v = ${fmt(v.v)}`, m.x, sample(m.x, R.dmQsd(v.v)))],
        heading: heading('x/L'),
      }),
    },
    {
      title: '(b) Slowest decay rate',
      caption: 'Slowest decay rate against vL/D; the grey line marks L = ℓ with v < 0, where the bound state appears. The orange curve is the gap 1/2τ + π²D/2L² of the two same-type intervals; the dotted curve is 1/T(0).',
      interaction: 'Drag to move v.',
      pick: (x) => ({ v: clamp(Math.round(x * 20) / 20, -6, 6) }),
      spec: (m, v) => ({
        x: { domain: [-6, 6], scale: 'linear', label: 'vL/D' },
        y: { scale: 'log', domain: [1e-4, 50], label: 'slowest rate × L²/D' },
        series: [
          line('refl. 0, abs. L', 'ink', m.rates.pe, m.rates.lam0),
          line('1/T(0)', 'ink', m.rates.pe, m.rates.invT0, { dash: 'dotted', width: 1.5 }),
          line('same-type walls', 'orange', m.rates.pe, m.rates.same),
        ],
        rules: [xRule(-1, 'L = ℓ', 'muted', 'solid')],
        cursor: v.v,
        cursorLabel: 'v',
        heading: heading('vL/D'),
      }),
    },
    {
      title: '(c) Mean exit time',
      caption: 'Mean exit time (log scale); colours as in (a).',
      interaction: 'Drag to move x₀.',
      pick: (x) => ({ x0: clamp(Math.round(x * 100) / 100, 0, 1) }),
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x₀/L' },
        y: { scale: 'log', domain: [1e-2, 2e2], label: 'T(x₀) D/L²' },
        legendTitle: 'vL/D',
        series: [...family(PE, m.x, (x, w) => R.dmT(x, w), DIV), live(`v = ${fmt(v.v)}`, m.x, sample(m.x, (x) => R.dmT(x, v.v)))],
        cursor: v.x0,
        cursorLabel: 'x₀',
        heading: heading('x₀/L'),
      }),
    },
  ],
  results: ['T(x₀) = [(L − x₀) + (e<sup>−2αL</sup> − e<sup>−2αx₀</sup>)/2α]/v.', 'For strong drift toward the mirror, λ₀ ≈ 1/T(0): escape is a rare event and the exit time is exponential.'],
  simulate: (v) => ({ ...drift(v.v), domain: [0, 1], bc: ['reflecting', 'absorbing'], x0: v.x0, T: 2, cond: true }),
});

// ---------------------------------------------------------------- §17 circle

const helix = memo((pe: number) => {
  const xs = linspace(0, 1, 181);
  const ts = linspace(0.0015, 0.12, 240);
  const z = new Float64Array(ts.length * xs.length);
  ts.forEach((t, i) => xs.forEach((x, j) => (z[i * xs.length + j] = F.circle(x, t, 0.5, pe))));
  return { xs, ts, z };
}, 8);

export const driftCircle = figurePage({
  id: 'drift-circle',
  group: GROUP,
  title: 'Constant drift on the circle',
  files: ['drift_circle.pdf'],
  gardiner: '§5.2.2',
  intro: 'Constant drift on a circle of length L: the gauge fails on a circle, but a boost does. The steady state is uniform and carries a current J = v/L.',
  equations: ['p = Σ<sub>n</sub> g(x − x₀ − vt + nL, t),&ensp;λ<sub>m</sub> = Dq²/2 + ivq,&ensp;q = 2πm/L'],
  defaults: { pe: 40, v: 3 },
  controls: [
    { title: '(a) Helix', controls: [{ key: 'pe', label: 'v', min: 0, max: 80, step: 1 }], hint: 'vL/D of the density in (a); the notes use 40.' },
    { title: '(b, c) Heavy curves', controls: [{ key: 'v', label: 'v', min: -6, max: 6, step: 0.1 }], hint: 'vL/D.' },
  ],
  model: (v) => ({ helix: helix(v.pe), d: linspace(1e-3, 1 - 1e-3, 300) }),
  panels: [
    {
      title: (_, v) => `(a) Density against time, vL/D = ${fmt(v.pe)}`,
      caption: 'Density against time from x₀ = L/2: the peak winds around while it spreads, a helix on the cylinder S¹ × ℝ<sub>t</sub>.',
      spec: (m) => ({
        x: { domain: [0, 0.12], scale: 'linear', label: 'Dt/L²', grid: false },
        y: { scale: 'linear', domain: [0, 1], label: 'x/L', grid: false },
        series: [{ kind: 'heat', label: 'L p', x: m.helix.ts, y: m.helix.xs, z: m.helix.z, zMax: 3 }],
        heading: (t, x) => `t = ${fmt(t)}, x = ${fmt(x)}`,
      }),
    },
    {
      title: '(b) Eigenvalues in the complex plane',
      caption: 'Eigenvalues λ<sub>m</sub>, |m| ≤ 3. Drift moves them vertically but not horizontally, so the gap (dotted) does not depend on v. Reversing v gives the same set.',
      spec: (_, v) => {
        const q = [-3, -2, -1, 0, 1, 2, 3].map((m) => 2 * Math.PI * m);
        const series: Series[] = [0, 2, 4].map((w, i) => ({
          kind: 'dots' as const,
          label: String(w),
          color: DIV[2 + i],
          x: q.map((k) => 0.5 * k * k),
          y: q.map((k) => w * k),
        }));
        series.push({ kind: 'dots', label: `v = ${fmt(v.v)}`, color: 'ink', x: q.map((k) => 0.5 * k * k), y: q.map((k) => v.v * k), z: 2 });
        return {
          x: { domain: [-10, 190], scale: 'linear', label: 'Re λ L²/D' },
          y: { scale: 'linear', domain: [-130, 130], label: 'Im λ L²/D' },
          legendTitle: 'vL/D',
          series,
          rules: [xRule(2 * Math.PI ** 2, 'gap', 'ink2', 'dotted')],
          heading: (x, y) => `Re λ = ${fmt(x)}, Im λ = ${fmt(y)}`,
        };
      },
    },
    {
      title: '(c) Mean time to reach a target',
      caption: 'Mean time to reach a target from a distance d above it, measured in the +x direction: the absorbing interval of the cover.',
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'd/L' },
        y: { scale: 'linear', domain: [0, 0.42], label: 'T(d) D/L²' },
        legendTitle: 'vL/D',
        series: [...family(PE, m.d, (d, w) => R.daT(d, w), DIV), live(`v = ${fmt(v.v)}`, m.d, sample(m.d, (d) => R.daT(d, v.v)))],
        heading: heading('d/L'),
      }),
    },
  ],
  results: ['The gap 2π²D/L² is that of the free circle for every v.', 'pₛ = 1/L with current J = v/L; the winding velocity is v.'],
  simulate: (v) => ({ ...drift(v.pe), domain: [0, 1], bc: ['periodic', 'periodic'], x0: 0.5, T: 0.12 }),
});

export const PART2 = [driftHlReflecting, driftHlAbsorbing, driftReflecting, driftAbsorbing, driftMixed, driftCircle];
