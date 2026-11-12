import type { Line, Series } from '../plot/chart';
import * as F from '../math/free';
import { geomspace, linspace } from '../math/num';
import { figurePage, type Panel } from './framework';
import { clamp, family, fmt, heading, hline, line, live, ref, sample, xRule, yRule } from './kit';

/** Part I of the notes: free diffusion, A = 0, B = D. Units L (or x₀) and L²/D (or x₀²/D). */
const GROUP = 'Part I · Free diffusion';
const SNAP_T = [0.01, 0.03, 0.1, 0.3, 1];
const SNAP_TH = [0.05, 0.2, 1, 5, 20];
const tLabel = (t: number) => `t = ${fmt(t)}`;

// ---------------------------------------------------------------- §3 half-line, reflecting

export const hlReflecting = figurePage({
  id: 'hl-reflecting',
  group: GROUP,
  title: 'Free diffusion on the half-line, reflecting at 0',
  files: ['halfline_reflecting.pdf'],
  gardiner: '§5.2.1',
  intro: 'Brownian motion on (0, ∞), reflected at 0: the free kernel plus its mirror image. Lengths in units of x₀, times in x₀²/D.',
  equations: ['p(x, t | x₀) = g(x − x₀, t) + g(x + x₀, t),&ensp;g(y, t) = e<sup>−y²/2Dt</sup>/√(2πDt)'],
  defaults: { t: 1 },
  controls: [{ title: 'Time', controls: [{ key: 't', label: 't', min: 0.01, max: 100, log: true, play: true }], hint: 'Dt/x₀². The heavy curve follows the slider; ▶ sweeps it.' }],
  model: () => ({
    x: linspace(0, 12, 601),
    tm: geomspace(1e-3, 1e3, 300),
  }),
  panels: [
    {
      title: '(a) Density snapshots, x₀ p(x, t | x₀, 0)',
      wide: true,
      caption: 'The density spreads without limit and develops zero slope at the wall.',
      spec: (m, v) => ({
        x: { domain: [0, 12], scale: 'linear', label: 'x/x₀' },
        y: { scale: 'linear', domain: [0, 1.9] },
        legendTitle: 'Dt/x₀²',
        series: [...family(SNAP_TH, m.x, (x, t) => F.halfLine(x, t, 1, 1)), live(tLabel(v.t), m.x, sample(m.x, (x) => F.halfLine(x, v.t, 1, 1)))],
        heading: heading('x/x₀'),
      }),
    },
    {
      title: '(b) Mean and standard deviation',
      wide: true,
      caption: 'Mean and standard deviation, with their short- and long-time asymptotes (dotted).',
      interaction: 'Drag to move t.',
      pick: (x) => ({ t: clamp(x, 0.01, 100) }),
      spec: (m, v) => {
        const mean = sample(m.tm, (t) => F.hlMean(t, 1));
        const sd = sample(m.tm, (t) => Math.sqrt(1 + t - F.hlMean(t, 1) ** 2));
        const tl = geomspace(1, 1e3, 40);
        const ts = geomspace(1e-3, 0.3, 40);
        return {
          x: { domain: [1e-3, 1e3], scale: 'log', label: 'Dt/x₀²' },
          y: { scale: 'log', domain: [0.02, 50] },
          series: [
            line('⟨x⟩/x₀', 'blue', m.tm, mean),
            line('σₓ/x₀', 'orange', m.tm, sd),
            ref('√(2Dt/π)', tl, sample(tl, (t) => Math.sqrt((2 * t) / Math.PI)), { color: 'muted', dash: 'dotted' }),
            ref('√((1 − 2/π)Dt)', tl, sample(tl, (t) => Math.sqrt((1 - 2 / Math.PI) * t)), { color: 'muted', dash: 'dotted' }),
            ref('√(Dt)', ts, sample(ts, Math.sqrt), { color: 'muted', dash: 'dotted' }),
          ],
          cursor: v.t,
          cursorLabel: 't',
          heading: heading('Dt/x₀²'),
        };
      },
    },
  ],
  results: [
    '⟨x⟩ = x₀ erf(x₀/√2Dt) + √(2Dt/π) e<sup>−x₀²/2Dt</sup>, the mean of the folded normal |x₀ + √(Dt) Z|.',
    'No stationary state: the spectrum is continuous down to 0, so everything is algebraic in t.',
  ],
  simulate: () => ({ a: '0', b: 'sqrt(D)', params: { D: 1 }, domain: [0, 12], bc: ['reflecting', 'open'], x0: 1, T: 20 }),
});

// ---------------------------------------------------------------- §4 half-line, absorbing

export const hlAbsorbing = figurePage({
  id: 'hl-absorbing',
  group: GROUP,
  title: 'Free diffusion on the half-line, absorbing at 0',
  files: ['halfline_absorbing.pdf'],
  gardiner: '§5.2.7',
  intro: 'Brownian motion killed on first reaching 0; equivalently, first passage of free Brownian motion to a level x₀ away. Lengths in units of x₀.',
  equations: ['p = g(x − x₀, t) − g(x + x₀, t),&ensp;S(t) = erf(x₀/√2Dt),&ensp;q = p/S'],
  defaults: { t: 1 },
  controls: [{ title: 'Time', controls: [{ key: 't', label: 't', min: 0.01, max: 1000, log: true, play: true }], hint: 'Dt/x₀². The heavy curves follow the slider.' }],
  model: () => ({ x: linspace(0, 8, 401), u: linspace(0, 4.5, 301), ts: geomspace(1e-2, 1e4, 300) }),
  panels: [
    {
      title: '(a) Density snapshots',
      caption: 'Mass leaks out at the wall and the remainder spreads.',
      spec: (m, v) => ({
        x: { domain: [0, 8], scale: 'linear', label: 'x/x₀' },
        y: { scale: 'linear', domain: [0, 1.9], label: 'x₀ p(x, t | x₀, 0)' },
        legendTitle: 'Dt/x₀²',
        series: [...family(SNAP_TH, m.x, (x, t) => F.halfLine(x, t, 1, -1)), live(tLabel(v.t), m.x, sample(m.x, (x) => F.halfLine(x, v.t, 1, -1)))],
        heading: heading('x/x₀'),
      }),
    },
    {
      title: '(b) Conditioned density, rescaled',
      caption: 'The conditioned density, rescaled by the diffusion length √(Dt), collapses onto the Rayleigh profile u e<sup>−u²/2</sup> for Dt ≫ x₀²; it does not settle in x.',
      spec: (m, v) => {
        const q = (u: number, t: number) => (Math.sqrt(t) * F.halfLine(u * Math.sqrt(t), t, 1, -1)) / F.hlSurvival(t, 1);
        return {
          x: { domain: [0, 4.5], scale: 'linear', label: 'x/√(Dt)' },
          y: { scale: 'linear', domain: [0, 1.9], label: '√(Dt) q(x, t)' },
          legendTitle: 'Dt/x₀²',
          series: [
            ...family(SNAP_TH, m.u, (u, t) => q(u, t)),
            ref('Rayleigh', m.u, sample(m.u, (u) => u * Math.exp(-(u * u) / 2))),
            live(tLabel(v.t), m.u, sample(m.u, (u) => q(u, v.t))),
          ],
          heading: heading('x/√(Dt)'),
        };
      },
    },
    {
      title: '(c) Survival',
      caption: 'Survival decays algebraically, S ~ x₀√(2/πDt).',
      interaction: 'Drag to move t.',
      pick: (x) => ({ t: clamp(x, 0.01, 1000) }),
      spec: (m, v) => {
        const tl = geomspace(1, 1e4, 50);
        return {
          x: { domain: [1e-2, 1e4], scale: 'log', label: 'Dt/x₀²' },
          y: { scale: 'log', domain: [5e-3, 1.5], label: 'S(t | x₀)' },
          series: [line('exact', 'ink', m.ts, sample(m.ts, (t) => F.hlSurvival(t, 1))), ref('x₀√(2/πDt)', tl, sample(tl, (t) => Math.sqrt(2 / (Math.PI * t))), { color: 'muted' })],
          cursor: v.t,
          cursorLabel: 't',
          heading: heading('Dt/x₀²'),
        };
      },
    },
  ],
  results: [
    'Exit-time density: Lévy–Smirnov, f = x₀ e<sup>−x₀²/2Dt</sup>/√(2πDt³), with infinite mean.',
    'No QSD: the conditioned density q spreads as √(Dt) with the Rayleigh shape instead of settling.',
  ],
  simulate: () => ({ a: '0', b: 'sqrt(D)', params: { D: 1 }, domain: [0, 12], bc: ['absorbing', 'open'], x0: 1, T: 20, cond: true }),
});

// ---------------------------------------------------------------- §5 reflecting walls

export const reflecting = figurePage({
  id: 'reflecting',
  group: GROUP,
  title: 'Free diffusion, reflecting walls',
  files: ['reflecting.pdf'],
  gardiner: '§5.2.5–6',
  intro: 'Brownian motion on (0, L) between two reflecting walls. Lengths in units of L, times in L²/D.',
  equations: ['p = 1/L + (2/L) Σ cos(nπx/L) cos(nπx₀/L) e<sup>−λₙt</sup>,&ensp;λₙ = ½D(nπ/L)²'],
  defaults: { x0: 0.3, t: 0.05 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'x0', label: 'x₀', min: 0, max: 1, step: 0.01 },
        { key: 't', label: 't', min: 0.001, max: 3, log: true, play: true },
      ],
      hint: 'x₀/L and Dt/L². The notes use x₀ = 0.3L.',
    },
  ],
  model: () => ({ x: linspace(0, 1, 401), ts: geomspace(1e-3, 10 ** 0.5, 300), tau: linspace(0, 1, 200) }),
  panels: [
    {
      title: '(a) Density snapshots',
      caption: 'Density snapshots relaxing to pₛ = 1/L.',
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x/L' },
        y: { scale: 'linear', max: 4.2, label: 'L p(x, t | x₀, 0)' },
        legendTitle: 'Dt/L²',
        series: [
          ...family(SNAP_T, m.x, (x, t) => F.interval(x, t, v.x0, 'reflecting')),
          hline('pₛ', 1, 0, 1),
          live(tLabel(v.t), m.x, sample(m.x, (x) => F.interval(x, v.t, v.x0, 'reflecting'))),
        ],
        heading: heading('x/L'),
      }),
    },
    {
      title: '(b) Mean and standard deviation',
      caption: 'At short times σₓ ≈ √(Dt) (dotted); both saturate on the time scale 1/λ₁ ≈ 0.2 L²/D.',
      interaction: 'Drag to move t.',
      pick: (x) => ({ t: clamp(x, 0.001, 3) }),
      spec: (m, v) => {
        const mean = sample(m.ts, (t) => F.reflMean(t, v.x0));
        const sd = sample(m.ts, (t, i) => Math.sqrt(Math.max(0, F.reflSecond(t, v.x0) - mean[i] ** 2)));
        return {
          x: { domain: [1e-3, 10 ** 0.5], scale: 'log', label: 'Dt/L²' },
          y: { scale: 'linear', domain: [0, 0.65] },
          series: [line('⟨x⟩/L', 'blue', m.ts, mean), line('σₓ/L', 'orange', m.ts, sd), ref('√(Dt)', m.ts, sample(m.ts, Math.sqrt), { color: 'muted', dash: 'dotted' })],
          rules: [yRule(0.5, 'L/2'), yRule(1 / Math.sqrt(12), 'L/√12')],
          cursor: v.t,
          cursorLabel: 't',
          heading: heading('Dt/L²'),
        };
      },
    },
    {
      title: '(c) Stationary autocorrelation',
      caption: 'Normalised stationary autocorrelation, G 5.2.110. The exact series and the single slowest mode are indistinguishable on this scale: it carries 96/π⁴ = 98.6 % of the variance.',
      spec: (m) => {
        const C0 = F.reflAutocov(0);
        return {
          x: { domain: [0, 1], scale: 'linear', label: 'Dτ/L²' },
          y: { scale: 'log', domain: [5e-3, 1.1], label: 'C(τ)/C(0)' },
          series: [
            line('exact (all odd modes)', 'blue', m.tau, sample(m.tau, (t) => F.reflAutocov(t) / C0)),
            ref('(96/π⁴) exp(−λ₁τ)', m.tau, sample(m.tau, (t) => (96 / Math.PI ** 4) * Math.exp(-F.lam(1) * t))),
          ],
          heading: heading('Dτ/L²'),
        };
      },
    },
  ],
  results: ['Gap λ₁ = π²D/2L². Only odd n contribute to ⟨x⟩ and to the stationary autocorrelation.', '⟨x⟩<sub>s</sub> = L/2 and var<sub>s</sub> x = L²/12.'],
  simulate: (v) => ({ a: '0', b: 'sqrt(D)', params: { D: 1 }, domain: [0, 1], bc: ['reflecting', 'reflecting'], x0: v.x0, T: 1 }),
});

// ---------------------------------------------------------------- §7 reflecting at 0, absorbing at L

export const mixed = figurePage({
  id: 'mixed',
  group: GROUP,
  title: 'Free diffusion, reflecting at 0, absorbing at L',
  files: ['mixed.pdf'],
  gardiner: '§5.2.7',
  intro: 'Brownian motion on (0, L), reflected at 0 and absorbed at L. Unfolding at the mirror gives the absorbing interval of length 2L.',
  equations: ['p = (2/L) Σ cos kₙx cos kₙx₀ e<sup>−λₙt</sup>,&ensp;kₙ = (n + ½)π/L', 'T(x₀) = (L² − x₀²)/D,&ensp;q<sub>∞</sub> = (π/2L) cos(πx/2L)'],
  defaults: { x0: 0.3, t: 0.05 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'x0', label: 'x₀', min: 0, max: 0.99, step: 0.01 },
        { key: 't', label: 't', min: 0.001, max: 2.5, log: true, play: true },
      ],
      hint: 'x₀/L and Dt/L².',
    },
  ],
  model: () => ({ x: linspace(0, 1, 401), ts: linspace(1e-4, 2.5, 500), xx: linspace(0, 1, 201) }),
  panels: [
    {
      title: '(a) Conditioned density q = p/S',
      caption: 'q = p/S converges to the QSD (π/2L) cos(πx/2L) (dashed).',
      spec: (m, v) => {
        const q = (x: number, t: number) => F.interval(x, t, v.x0, 'mixed') / F.mixSurvival(t, v.x0);
        return {
          x: { domain: [0, 1], scale: 'linear', label: 'x/L' },
          y: { scale: 'linear', max: 4.2, label: 'L q(x, t)' },
          legendTitle: 'Dt/L²',
          series: [...family(SNAP_T, m.x, q), ref('QSD', m.x, sample(m.x, F.mixQsd)), live(tLabel(v.t), m.x, sample(m.x, (x) => q(x, v.t)))],
          heading: heading('x/L'),
        };
      },
    },
    {
      title: '(b) Exit-time density',
      caption: 'The exit-time density from x₀, against the slowest mode alone.',
      interaction: 'Drag to move t.',
      pick: (x) => ({ t: clamp(x, 0.001, 2.5) }),
      spec: (m, v) => {
        const c0 = (4 / Math.PI) * Math.cos((Math.PI * v.x0) / 2);
        return {
          x: { domain: [0, 2.5], scale: 'linear', label: 'Dt/L²' },
          y: { scale: 'linear', max: 3, label: 'f(t | x₀) L²/D' },
          series: [
            line('exact', 'ink', m.ts, sample(m.ts, (t) => F.mixFlux(t, v.x0))),
            ref('slowest mode', m.ts, sample(m.ts, (t) => F.lamMix(0) * c0 * Math.exp(-F.lamMix(0) * t)), { color: 'muted' }),
          ],
          cursor: v.t,
          cursorLabel: 't',
          heading: heading('Dt/L²'),
        };
      },
    },
    {
      title: '(c) Exit time against the start',
      caption: 'Mean and standard deviation of the exit time. The dotted curve is T with both walls absorbing, for comparison.',
      interaction: 'Drag to move x₀.',
      pick: (x) => ({ x0: clamp(Math.round(x * 100) / 100, 0, 0.99) }),
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x₀/L' },
        y: { scale: 'linear', domain: [0, 1.15], label: 'time × D/L²' },
        series: [
          line('T', 'ink', m.xx, sample(m.xx, F.mixT)),
          line('σ_T', 'ink', m.xx, sample(m.xx, (x) => Math.sqrt(F.mixTVar(x))), { dash: 'dashed', width: 1.5 }),
          line('T, both absorbing', 'muted', m.xx, sample(m.xx, F.T_mean), { dash: 'dotted', width: 1.5 }),
        ],
        cursor: v.x0,
        cursorLabel: 'x₀',
        heading: heading('x₀/L'),
      }),
    },
  ],
  results: ['λ₀ = π²D/8L²: the mirror doubles the effective length, so the rate drops by 4 from the absorbing interval.', 'σ<sub>T</sub>² = 2(L⁴ − x₀⁴)/3D².'],
  simulate: (v) => ({ a: '0', b: 'sqrt(D)', params: { D: 1 }, domain: [0, 1], bc: ['reflecting', 'absorbing'], x0: v.x0, T: 2, cond: true }),
});

// ---------------------------------------------------------------- §8 circle

const SNAP_C = [0.002, 0.005, 0.01, 0.02, 0.05];

export const circle = figurePage({
  id: 'circle',
  group: GROUP,
  title: 'Free diffusion on the circle',
  files: ['circle.pdf'],
  gardiner: '§5.2.2',
  intro: 'Brownian motion on a circle of length L, centred on the starting point, the only distinguished point. Nothing is imposed anywhere.',
  equations: ['p = Σ<sub>n</sub> g(x − x₀ + nL, t) = 1/L + (2/L) Σ cos q(x − x₀) e<sup>−Dq²t/2</sup>,&ensp;q = 2πm/L'],
  defaults: { t: 0.01, tc: 0.15 },
  controls: [
    {
      title: 'Time',
      controls: [
        { key: 't', label: 't', min: 0.0005, max: 0.2, log: true, play: true },
        { key: 'tc', label: 't′', min: 0.005, max: 1, log: true, play: true },
      ],
      hint: 't moves the heavy curve in (a); t′ is the time of the universal cover in (b). Both in L²/D.',
    },
  ],
  model: () => ({ y: linspace(-0.5, 0.5, 401), yy: linspace(-1.75, 1.75, 701) }),
  panels: [
    {
      title: '(a) Density snapshots',
      caption: 'Density snapshots relaxing to 1/L by Dt/L² ≈ 0.05.',
      spec: (m, v) => ({
        x: { domain: [-0.5, 0.5], scale: 'linear', label: '(x − x₀)/L' },
        y: { scale: 'linear', max: 9.5, label: 'L p(x, t | x₀, 0)' },
        legendTitle: 'Dt/L²',
        series: [...family(SNAP_C, m.y, (y, t) => F.circle(y, t, 0)), hline('pₛ', 1, -0.5, 0.5), live(tLabel(v.t), m.y, sample(m.y, (y) => F.circle(y, v.t, 0)))],
        heading: heading('(x − x₀)/L'),
      }),
    },
    {
      title: (_, v) => `(b) The universal cover at Dt/L² = ${fmt(v.tc)}`,
      caption: 'The free Gaussian on the line (grey), its translates by ±L folded into one period (dashed), and their sum, the density on the circle (blue). Shading marks the neighbouring copies of the circle.',
      spec: (m, v) => {
        const series: Series[] = [
          { kind: 'band', label: 'neighbouring copies', color: 'muted', x: [-1.5, -0.5], alpha: 0.12 },
          { kind: 'band', label: 'neighbouring copies', color: 'muted', x: [0.5, 1.5], alpha: 0.12, legend: false },
          line('line: g', 'muted', m.yy, sample(m.yy, (y) => F.heat(y, v.tc)), { width: 1.5 }),
          line('n = ±1 terms', 'orange', m.y, sample(m.y, (y) => F.heat(y + 1, v.tc)), { dash: 'dashed', width: 1.5 }),
          line('n = ±1 terms', 'orange', m.y, sample(m.y, (y) => F.heat(y - 1, v.tc)), { dash: 'dashed', width: 1.5, legend: false, tooltip: false }),
          line('circle: sum of all', 'blue', m.y, sample(m.y, (y) => F.circle(y, v.tc, 0)), { z: 2 }),
        ];
        return {
          x: { domain: [-1.75, 1.75], scale: 'linear', label: 'lift (x̃ − x₀)/L' },
          y: { scale: 'linear', max: 6, label: 'L p' },
          series,
          heading: heading('(x̃ − x₀)/L'),
        };
      },
    },
    {
      title: '(c) Spectrum',
      caption: 'Spectrum in units of 2π²D/L². Each nonzero level of the circle is a doublet: one level from the reflecting interval of length L/2 (even about x₀) and one from the absorbing one (odd).',
      spec: () => {
        const seg = (x0: number, x1: number, y: number) => [x0, x1, NaN, y, y, NaN];
        const xs: Record<string, number[]> = { even: [], odd: [] };
        const ys: Record<string, number[]> = { even: [], odd: [] };
        const add = (k: 'even' | 'odd', x0: number, x1: number, y: number) => {
          const s = seg(x0, x1, y);
          xs[k].push(...s.slice(0, 3));
          ys[k].push(...s.slice(3));
        };
        for (let n = 0; n < 4; n++) {
          const lv = n * n;
          if (n === 0) add('even', 0.15, 0.85, lv);
          else {
            add('even', 0.15, 0.49, lv);
            add('odd', 0.51, 0.85, lv);
            add('odd', 2.15, 2.85, lv);
          }
          add('even', 1.15, 1.85, lv);
        }
        return {
          x: {
            domain: [0, 3],
            scale: 'linear',
            label: '',
            grid: false,
            ticks: [
              { value: 0.5, label: 'circle, L' },
              { value: 1.5, label: 'refl. (0, L/2)' },
              { value: 2.5, label: 'abs. (0, L/2)' },
            ],
          },
          y: { scale: 'linear', domain: [-0.4, 10.5], label: 'λ / (2π²D/L²)' },
          series: [
            line('even about x₀', 'blue', xs.even, ys.even, { width: 3, tooltip: false }),
            line('odd about x₀', 'orange', xs.odd, ys.odd, { width: 3, tooltip: false }),
          ],
          heading: () => 'Levels n², units 2π²D/L²',
          rows: () => [0, 1, 2, 3].map((n) => ({ label: n === 0 ? 'single (pₛ)' : 'doublet: even + odd', value: String(n * n), color: n === 0 ? 'blue' : 'orange', key: 'solid' })),
        };
      },
    },
  ],
  results: ['Gap 2π²D/L², the same as the reflecting interval of length L/2.', 'The circle is two intervals: even functions about x₀ see reflecting walls at x₀ and its antipode, odd ones absorbing walls.'],
  simulate: () => ({ a: '0', b: 'sqrt(D)', params: { D: 1 }, domain: [0, 1], bc: ['periodic', 'periodic'], x0: 0.5, T: 0.2 }),
});

// ---------------------------------------------------------------- §9 summary

interface SummaryModel {
  xf: Float64Array;
  xh: Float64Array;
  xi: Float64Array;
  ts: Float64Array;
}

/** Densities at t with walls of one type: none, at 0, at 0 and L (and, absorbing, the mixed interval). */
function wallsPanel(kind: 'reflecting' | 'absorbing', letter: string): Panel<{ x0: number; t: number }, SummaryModel> {
  return {
    title: `(${letter}) ${kind === 'reflecting' ? 'Reflecting' : 'Absorbing'} walls`,
    caption: 'Densities at the slider time; the dashed curve is the free Gaussian.',
    spec: (m, v) => {
      const sign = kind === 'reflecting' ? 1 : -1;
      const series: Line[] = [
        ref('none', m.xf, sample(m.xf, (x) => F.heat(x - v.x0, v.t))),
        line('0', 'blue', m.xh, sample(m.xh, (x) => F.halfLine(x, v.t, v.x0, sign))),
        line('0 and L', 'orange', m.xi, sample(m.xi, (x) => F.interval(x, v.t, v.x0, kind))),
      ];
      if (kind === 'absorbing') series.push(line('refl. 0, abs. L', 'aqua', m.xi, sample(m.xi, (x) => F.interval(x, v.t, v.x0, 'mixed'))));
      return {
        x: { domain: [-0.8, 1.6], scale: 'linear', label: 'x/L' },
        y: { scale: 'linear', max: 6, label: 'L p(x, t | x₀, 0)' },
        series,
        rules: [xRule(0, undefined, 'muted', 'solid'), xRule(1, undefined, 'muted', 'solid')],
        heading: heading('x/L'),
      };
    },
  };
}

export const freeSummary = figurePage({
  id: 'free-summary',
  group: GROUP,
  title: 'Summary of free diffusion',
  files: ['free_summary.pdf'],
  intro: 'Free diffusion in every geometry from the same start, so the walls can be compared directly. Grey lines mark x = 0 and x = L; a wall “at 0” alone is the half-line.',
  equations: ['∂<sub>t</sub>p = ½D ∂<sub>x</sub>²p'],
  defaults: { x0: 0.3, t: 0.1 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'x0', label: 'x₀', min: 0.01, max: 0.99, step: 0.01 },
        { key: 't', label: 't', min: 0.002, max: 2, log: true, play: true },
      ],
      hint: 'x₀/L and Dt/L² of the densities in (a) and (b).',
    },
  ],
  model: () => ({ xf: linspace(-0.8, 1.6, 601), xh: linspace(0, 1.6, 401), xi: linspace(0, 1, 251), ts: linspace(1e-3, 2, 400) }),
  panels: [
    wallsPanel('reflecting', 'a'),
    wallsPanel('absorbing', 'b'),
    {
      title: '(c) Survival in the three absorbing geometries',
      caption: 'e<sup>−π²Dt/2L²</sup>, e<sup>−π²Dt/8L²</sup>, and t<sup>−1/2</sup> for the half-line, which overtakes the mixed interval near Dt/L² ≈ 1.4 (from x₀ = 0.3L).',
      interaction: 'Drag to move t.',
      pick: (x) => ({ t: clamp(x, 0.002, 2) }),
      spec: (m, v) => ({
        x: { domain: [0, 2], scale: 'linear', label: 'Dt/L²' },
        y: { scale: 'log', domain: [1e-4, 1.5], label: 'S(t | x₀)' },
        series: [
          line('wall at 0', 'blue', m.ts, sample(m.ts, (t) => F.hlSurvival(t, v.x0))),
          line('walls at 0, L', 'orange', m.ts, sample(m.ts, (t) => F.survival(t, v.x0))),
          line('refl. 0, abs. L', 'aqua', m.ts, sample(m.ts, (t) => F.mixSurvival(t, v.x0))),
        ],
        cursor: v.t,
        cursorLabel: 't',
        heading: heading('Dt/L²'),
      }),
    },
  ],
  results: [
    'A bounded domain has a discrete spectrum with a gap, so relaxation and decay are exponential; an unbounded one has a continuous spectrum reaching 0, so everything is algebraic in t.',
    'A reflecting wall is a mirror and doubles the effective length, so the rate drops by 4.',
  ],
});

export const PART1 = [hlReflecting, hlAbsorbing, reflecting, mixed, circle, freeSummary];
