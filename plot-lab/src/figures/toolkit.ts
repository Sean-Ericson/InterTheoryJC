import type { Line, Series } from '../plot/chart';
import type { TooltipRow } from '../plot/tooltip';
import { SEQ, type Role } from '../plot/theme';
import { attractionRates, driftRate, ouRate } from '../math/attraction';
import * as C from '../math/circle';
import * as R from '../math/drift';
import { geomspace, linspace, memo } from '../math/num';
import * as O from '../math/ou';
import { figurePage } from './framework';
import { fmt, heading, levelSegment, line, live, ref, sample, xRule } from './kit';

/** §1 of the notes: the toolkit sections with figures. */
const GROUP = 'Conventions and toolkit';

// ---------------------------------------------------------------- symmetrised operator

interface Level {
  E: number;
  color: Role;
  label: string;
  region?: (x: number) => boolean;
}

/** One series per level, drawn where V < E, plus legend-only entries for the two parities. */
function levelSeries(x: Float64Array, V: Float64Array, levels: Level[]): Series[] {
  const out: Series[] = [line('V', 'ink', x, V, { width: 1.5, tooltip: false, legend: false })];
  for (const l of levels) out.push(line(l.label, l.color, x, levelSegment(x, V, l.E, l.region), { width: 2.5, tooltip: false, legend: false }));
  out.push(line('even', 'blue', [], [], { tooltip: false }), line('odd', 'orange', [], [], { tooltip: false }));
  return out;
}

function levelRows(x: number, V: (x: number) => number, levels: Level[]): TooltipRow[] {
  const rows: TooltipRow[] = [{ label: 'V(x)', value: fmt(V(x)), color: 'ink', key: 'solid' }];
  const seen = new Set<string>();
  for (const l of levels) {
    const key = `${l.E.toPrecision(6)}${l.color}`;
    if (seen.has(key) || (l.region && !l.region(x))) continue;
    seen.add(key);
    rows.push({ label: l.label, value: fmt(l.E), color: l.color, key: 'solid' });
  }
  return rows;
}

export const operators = figurePage({
  id: 'operators',
  group: GROUP,
  title: 'Which operator, on which space',
  files: ['operators.pdf'],
  gardiner: '§5.2.5',
  intro: 'p = √pₛ ψ turns the forward operator into −H, H = −(D/2)∂ₓ² + V. Levels of H are the decay rates λ. Here V and its levels for the OU examples, in units of k, drawn where V &lt; λ.',
  equations: ['V = A²/2D + A′/2,&ensp;absorbing wall: ψ = 0,&ensp;reflecting wall: ψ′ = (A/D)ψ'],
  defaults: { kappa: 16, L: 4 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'kappa', label: 'κ', min: 1, max: 40, log: true },
        { key: 'L', label: 'L', min: 1.5, max: 9, step: 0.05 },
      ],
      hint: 'κ = 2R²/ℓ² of the cosine well in (b) and the circumference L/ℓ of the sawtooth in (c). The notes use κ = 16 and L = 4ℓ.',
    },
  ],
  model: (v) => {
    const xi = linspace(-3.6, 3.6, 1201);
    const u = linspace(-0.5, 0.5, 1601);
    const Lc = 2 * Math.PI * C.cosR(v.kappa);
    const Vc = C.cosV(v.kappa);
    const ev = C.cosLevels(v.kappa, 'even', 4);
    const bottom = (x: number) => Math.abs(x) < 0.25;
    const cos: Level[] = [];
    ev.forEach((E, m) => {
      cos.push({ E, color: m % 2 === 0 ? 'blue' : 'orange', label: `λ${m}`, region: bottom });
      if (m > 0) cos.push({ E, color: m % 2 === 0 ? 'orange' : 'blue', label: `λ${m}`, region: (x) => !bottom(x) });
    });
    const saw: Level[] = [];
    O.ouLevels(v.L / 2, 'reflecting', 4).forEach((E, n) => saw.push({ E, color: 'blue', label: `even ${n}` }));
    O.ouLevels(v.L / 2, 'absorbing', 4).forEach((E, n) => saw.push({ E, color: 'orange', label: `odd ${n}` }));
    const su = linspace(-0.5, 0.5, 801);
    return {
      xi,
      Vou: sample(xi, O.ouV),
      ouLevels: [0, 1, 2, 3, 4, 5].map((n): Level => ({ E: n, color: n % 2 === 0 ? 'blue' : 'orange', label: `λ${n} = ${n}k` })),
      u,
      Vc: sample(u, (y) => Vc(y * Lc)),
      Vcf: (y: number) => Vc(y * Lc),
      cos,
      su,
      Vs: sample(su, (y) => O.ouV(y * v.L)),
      saw: saw.filter((l) => l.E < 5.6),
    };
  },
  panels: [
    {
      title: '(a) OU on the line',
      caption: 'A harmonic oscillator with levels nk. A wall at the bottom keeps one parity: reflecting the even levels, absorbing the odd ones.',
      spec: (m) => ({
        x: { domain: [-3.6, 3.6], scale: 'linear', label: 'x/ℓ' },
        y: { scale: 'linear', domain: [-0.8, 5.6], label: 'V/k, λ/k' },
        series: levelSeries(m.xi, m.Vou, m.ouLevels),
        heading: heading('x/ℓ'),
        rows: (x) => levelRows(x, O.ouV, m.ouLevels),
      }),
    },
    {
      title: (_, v) => `(b) The cosine well on a circle, κ = ${fmt(v.kappa)}`,
      caption: 'V has a well at the bottom of U (x = 0) and one at its top (x = ±L/2). Each nonzero level has a copy in each well, of opposite parity.',
      spec: (m) => ({
        x: { domain: [-0.5, 0.5], scale: 'linear', label: 'x/L' },
        y: { scale: 'linear', domain: [-0.8, 5.6], label: 'V/k, λ/k' },
        series: levelSeries(m.u, m.Vc, m.cos),
        heading: heading('x/L'),
        rows: (x) => levelRows(x, m.Vcf, m.cos),
      }),
    },
    {
      title: (_, v) => `(c) The sawtooth on a circle, L = ${fmt(v.L)}ℓ`,
      caption: 'One harmonic well and a δ-function spike at the cusp (arrows). The levels are single, in pairs split by exactly k.',
      spec: (m, v) => {
        const top = O.ouV(v.L / 2);
        const series = levelSeries(m.su, m.Vs, m.saw);
        series.push(line('δ spike at the cusp', 'ink', [-0.5, -0.5, NaN, 0.5, 0.5], [top, 5.5, NaN, top, 5.5], { width: 1.5, arrows: 'end', tooltip: false, legend: false }));
        return {
          x: { domain: [-0.5, 0.5], scale: 'linear', label: 'x/L' },
          y: { scale: 'linear', domain: [-0.8, 5.6], label: 'V/k, λ/k' },
          series,
          heading: heading('x/L'),
          rows: (x) => levelRows(x, (y) => O.ouV(y * v.L), m.saw),
        };
      },
    },
  ],
  results: [
    'The forward operator is self-adjoint in L²(1/pₛ); H is its image in plain L², so the levels are real and the ψₙ orthogonal.',
    'Gardiner’s substitution p = pₛq (5.2.71) turns the forward operator into the backward one; p = √pₛ ψ is the halfway point.',
  ],
});

// ---------------------------------------------------------------- spectral gap

export const spectra = figurePage({
  id: 'spectra',
  group: GROUP,
  title: 'The spectral gap',
  files: ['spectra.pdf'],
  intro: 'Spectra of six examples, each in its own natural rate. Ticks are discrete eigenvalues, shaded bands continua, and the arrow is the gap that controls the long-time behaviour.',
  equations: ['p(x, t) − pₛ(x) ~ e<sup>−λ₁t</sup>,&ensp;S(t) ~ e<sup>−λ₀t</sup>,&ensp;q − q<sub>∞</sub> ~ e<sup>−(λ₁−λ₀)t</sup>'],
  defaults: { L: 2.5 },
  controls: [{ title: 'Harmonic well', controls: [{ key: 'L', label: 'L', min: 1, max: 4.5, step: 0.05 }], hint: 'L/ℓ of the absorbing wall in the last column; the notes use 2.5ℓ.' }],
  model: (v) => ({ well: O.ouLevels(v.L, 'mixed', 8) }),
  panels: [
    {
      title: (m) => `Six spectra; the well’s lowest level is λ₀ ≈ ${fmt(m.well[0])} k`,
      caption: 'For the absorbing interval the lowest level is the decay rate of S, and the arrow is the rate of convergence to the QSD. The last column is the harmonic well reflecting at its bottom and absorbing at L; its lowest level is the Kramers escape rate.',
      spec: (m, v) => {
        const top = 10;
        const cols: { name: string; unit: string; levels: number[]; cont?: number; gap?: [number, number] }[] = [
          { name: 'refl. interval', unit: 'π²D/2L²', levels: [0, 1, 4, 9], gap: [0, 1] },
          { name: 'abs. interval', unit: 'π²D/2L²', levels: [1, 4, 9], gap: [1, 4] },
          { name: 'free half-line', unit: '(no gap)', levels: [], cont: 0 },
          { name: 'drift into wall', unit: '1/2τ', levels: [0], cont: 1, gap: [0, 1] },
          { name: 'OU line', unit: 'k', levels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], gap: [0, 1] },
          { name: `well, abs. at ${fmt(v.L)}ℓ`, unit: 'k', levels: m.well, gap: [m.well[0], m.well[1]] },
        ];
        const lx: number[] = [];
        const ly: number[] = [];
        const ax: number[] = [];
        const ay: number[] = [];
        const series: Series[] = [];
        cols.forEach((c, i) => {
          for (const l of c.levels) if (l <= top) lx.push(i + 0.18, i + 0.82, NaN), ly.push(l, l, NaN);
          if (c.cont !== undefined) {
            series.push({ kind: 'band', label: 'continuum', color: 'blue', x: [i + 0.18, i + 0.82], y: [c.cont, top], alpha: 0.18, legend: i === 2 });
            series.push(line('continuum edge', 'blue', [i + 0.18, i + 0.82], [c.cont, c.cont], { width: 1.5, legend: false, tooltip: false }));
          }
          if (c.gap) ax.push(i + 0.5, i + 0.5, NaN), ay.push(c.gap[0], c.gap[1], NaN);
        });
        series.push(line('levels', 'ink', lx, ly, { width: 2, tooltip: false }));
        series.push(line('gap', 'orange', ax, ay, { width: 1.5, arrows: 'both', tooltip: false }));
        return {
          x: { domain: [0, 6], scale: 'linear', label: '', grid: false, ticks: cols.map((c, i) => ({ value: i + 0.5, label: `${c.name}\n${c.unit}` })) },
          y: { scale: 'linear', domain: [-0.3, top], label: 'λ / unit' },
          series,
          heading: (x) => {
            const c = cols[Math.min(5, Math.max(0, Math.floor(x)))];
            return `${c.name}, ${c.unit === '(no gap)' ? 'no gap' : `unit ${c.unit}`}`;
          },
          rows: (x) => {
            const c = cols[Math.min(5, Math.max(0, Math.floor(x)))];
            const rows: TooltipRow[] = c.levels.slice(0, 6).map((l, n) => ({ label: `λ${n}`, value: fmt(l), color: 'ink', key: 'solid' }));
            if (c.cont !== undefined) rows.push({ label: 'continuum from', value: fmt(c.cont), color: 'blue', key: 'band' });
            if (c.gap) rows.push({ label: 'gap', value: fmt(c.gap[1] - c.gap[0]), color: 'orange', key: 'solid' });
            return rows;
          },
        };
      },
    },
  ],
  results: ['A bounded domain has a discrete spectrum with a gap: relaxation and decay are exponential.', 'An unbounded domain can have a continuum reaching 0 (free half-line: no gap, algebraic decay) or starting above 0 (drift into the wall: a continuum of QSDs).'],
});

// ---------------------------------------------------------------- which QSD a start reaches

const TAILS = [0.25, 0.5, 0.75];
const TAIL_ROLES: Role[] = [SEQ[1], SEQ[2], SEQ[3]];
/** The drift start needs a long grid to normalise q (its tail reaches far); spacing 0.01. */
const XA = linspace(0, 250, 25001);

/** The drift curves on XA, thinned to x ≤ 14 for drawing. */
const driftConditioned = memo((beta: number, t: number) => thin(R.dhaConditionedExp(XA, t, beta, -1).q), 64);
const driftPoint = memo((t: number) => {
  const p = XA.map((x) => R.dhaDensity(x, t, 1, -1));
  let Z = 0;
  for (let i = 1; i < XA.length; i++) Z += 0.5 * (p[i] + p[i - 1]) * (XA[i] - XA[i - 1]);
  return thin(p.map((y) => y / Z));
}, 32);
const XD = thin(XA);

function thin(a: Float64Array): Float64Array {
  // XA has spacing 0.01; keep every 2nd point up to x = 14.
  const n = Math.round(14 / 0.01) + 1;
  const out = new Float64Array(Math.ceil(n / 2));
  for (let i = 0; i < out.length; i++) out[i] = a[2 * i];
  return out;
}

const XO = geomspace(1e-2, 1e2, 140);
const ouConditioned = memo((nu: number, t: number) => {
  const { p, S } = O.ouHlEvolveLomax(nu, XO, t);
  return p.map((y) => y / S);
}, 64);

export const attraction = figurePage({
  id: 'attraction',
  group: GROUP,
  title: 'Stationary states and QSDs from the spectrum',
  files: ['qsd_attraction.pdf'],
  intro: 'On the half-lines with a continuum of QSDs, the initial tail selects the one reached: drift toward an absorbing wall, and OU absorbed at the bottom of the well.',
  equations: ['drift: tail e<sup>−βx</sup> → λ = βℓ(2 − βℓ)/2τ, up to the edge 1/2τ', 'OU: tail x<sup>−1−ν</sup> → λ = νk, up to k'],
  defaults: { beta: 0.4, tD: 60, nu: 0.4, tO: 12 },
  controls: [
    {
      title: '(a) Drift toward the wall',
      controls: [
        { key: 'beta', label: 'βℓ', min: 0.05, max: 1.5, step: 0.01 },
        { key: 'tD', label: 't', min: 2, max: 200, log: true, play: true },
      ],
      hint: 'Tail βℓ of the exponential start βe^(−βx), and t/τ. The notes show t = 60τ.',
    },
    {
      title: '(b) OU absorbed at the bottom',
      controls: [
        { key: 'nu', label: 'ν', min: 0.05, max: 1.5, step: 0.01 },
        { key: 'tO', label: 't', min: 1, max: 20, step: 0.1, play: true },
      ],
      hint: 'Tail ν of the Lomax start ν(1 + x/ℓ)^(−1−ν)/ℓ, and t/τ. The notes show t = 12τ.',
    },
  ],
  model: (v) => ({
    beta: v.beta,
    drift: TAILS.map((b) => driftConditioned(b, v.tD)),
    driftLive: driftConditioned(v.beta, v.tD),
    point: driftPoint(v.tD),
    ou: TAILS.map((n) => ouConditioned(n, v.tO)),
    ouLive: ouConditioned(v.nu, v.tO),
    ouPoint: sample(XO, (x) => O.ouHalf(x, v.tO, 2, -1) / O.ouHlSurvival(v.tO, 2)),
    rates: attractionRates(),
  }),
  panels: [
    {
      title: (_, v) => `(a) Drift toward an absorbing wall, t = ${fmt(v.tD)}τ`,
      caption: 'Conditioned densities from exponential starts βe<sup>−βx</sup> and from a point start at x₀ = ℓ. Dashed, the QSDs with the same tails, and the edge QSD x e<sup>−x/ℓ</sup>/ℓ², which the point start is still approaching, algebraically.',
      spec: (m, v) => {
        const qsd = (b: number) => sample(XD, (x) => R.dhaQsd(x, R.dhaLamOfTail(b, -1), -1));
        const series: Line[] = [
          ...TAILS.map((b, i) => line(fmt(b), TAIL_ROLES[i], XD, m.drift[i], { width: 1.6 })),
          line('point', 'ink', XD, m.point, { width: 1.6 }),
          ...[...TAILS, 1].map((b, i) => ref(i === 0 ? 'QSDs, same tails' : 'QSD', XD, qsd(b), { color: 'muted', width: 1, legend: i === 0, tooltip: false })),
          live(`β = ${fmt(v.beta)}`, XD, m.driftLive, { color: 'aqua' }),
          ref(`its QSD (λ = ${fmt(R.dhaLamOfTail(v.beta, -1) * 2)}/2τ)`, XD, qsd(v.beta), { color: 'aqua' }),
        ];
        return {
          x: { domain: [0, 14], scale: 'linear', label: 'x/ℓ' },
          y: { scale: 'log', domain: [1e-5, 1.5], label: 'ℓ q(x, t)' },
          legendTitle: 'tail βℓ',
          series,
          heading: heading('x/ℓ'),
        };
      },
    },
    {
      title: (_, v) => `(b) OU absorbed at the bottom, t = ${fmt(v.tO)}τ`,
      caption: 'Conditioned densities from starts with tails x<sup>−1−ν</sup>, and from a point start at x₀ = 2ℓ. Dashed, the QSDs q<sub>ν</sub> and the Rayleigh density.',
      spec: (m, v) => {
        const series: Line[] = [
          ...TAILS.map((n, i) => line(fmt(n), TAIL_ROLES[i], XO, m.ou[i], { width: 1.6 })),
          line('point', 'ink', XO, m.ouPoint, { width: 1.6 }),
          ...[...TAILS, 1].map((n, i) => ref(i === 0 ? 'QSDs q_ν' : 'q_ν', XO, sample(XO, (x) => O.ouQsdNu(x, n)), { color: 'muted', width: 1, legend: i === 0, tooltip: false })),
          live(`ν = ${fmt(v.nu)}`, XO, m.ouLive, { color: 'aqua' }),
          ref(`its QSD (λ = ${fmt(Math.min(v.nu, 1))}k)`, XO, sample(XO, (x) => O.ouQsdNu(x, Math.min(v.nu, 1))), { color: 'aqua' }),
        ];
        return {
          x: { domain: [1e-2, 1e2], scale: 'log', label: 'x/ℓ' },
          y: { scale: 'log', domain: [1e-7, 3], label: 'ℓ q(x, t)' },
          legendTitle: 'tail ν',
          series,
          heading: heading('x/ℓ'),
        };
      },
    },
    {
      title: '(c) Decay rate selected by the tail',
      caption: 'Decay rate of S selected by the initial tail, in units of λ<sub>b</sub> (1/2τ for drift, k for OU). Curves are the predictions; dots are measured from the exact evolution.',
      pick: (x, v) => (Math.abs(x - v.beta) < Math.abs(x - v.nu) ? { beta: Math.round(Math.min(1.5, Math.max(0.05, x)) * 100) / 100 } : { nu: Math.round(Math.min(1.5, Math.max(0.05, x)) * 100) / 100 }),
      interaction: 'Drag to move the nearer of β and ν.',
      spec: (m, v) => {
        const s = linspace(0, 2, 201);
        return {
          x: { domain: [0, 2], scale: 'linear', label: 'tail: βℓ (drift), ν (OU)' },
          y: { scale: 'linear', domain: [0, 1.15], label: 'λ/λ_b' },
          series: [
            line('drift', 'blue', s, sample(s, driftRate)),
            line('OU', 'orange', s, sample(s, ouRate)),
            { kind: 'dots', label: 'measured, drift', color: 'blue', x: m.rates.s, y: m.rates.drift },
            { kind: 'dots', label: 'measured, OU', color: 'orange', x: m.rates.s, y: m.rates.ou },
          ],
          rules: [xRule(v.beta, 'β', 'blue'), xRule(v.nu, 'ν', 'orange')],
          heading: heading('tail'),
        };
      },
    },
  ],
  results: [
    'Uniqueness (Cattiaux et al.): exactly one QSD, attracting every start, iff +∞ is an entrance boundary, i.e. the mean absorption time T(x₀) stays bounded as x₀ → ∞.',
    'Both half-lines fail it, so the tail of the start decides: tails lighter than every QSD reach the edge; a matching tail reaches its own QSD.',
  ],
});

export const TOOLKIT = [operators, spectra, attraction];
