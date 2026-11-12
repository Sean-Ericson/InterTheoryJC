import type { Line, Series } from '../plot/chart';
import { SEQ } from '../plot/theme';
import * as C from '../math/circle';
import { geomspace, linspace, memo } from '../math/num';
import { erf } from '../special';
import * as O from '../math/ou';
import { figurePage } from './framework';
import { clamp, family, fmt, heading, hline, line, live, ref, sample, yRule } from './kit';

/**
 * Part III of the notes: Ornstein–Uhlenbeck, A = −kx, B = D, in the process's
 * own units ℓ = √(D/k) and τ = 1/k. x = 0 is the bottom of the well.
 */
const GROUP = 'Part III · Ornstein–Uhlenbeck';
const L_OU = [0.5, 1, 2, 4];
const LEN = SEQ.slice(1);
const ou = (x0: number) => ({ a: '-k*x', b: 'sqrt(D)', params: { k: 1, D: 1 }, x0 });
const EULER_GAMMA = 0.5772156649015329;

// ---------------------------------------------------------------- §20 line

export const ouLine = figurePage({
  id: 'ou-line',
  group: GROUP,
  title: 'Ornstein–Uhlenbeck on the whole line',
  files: ['ou_line.pdf'],
  gardiner: '§3.8.4, §4.4.4',
  intro: 'The harmonic well A = −kx on the line, from a point start. Units ℓ = √(D/k), τ = 1/k.',
  equations: ['p = N(x₀e<sup>−kt</sup>, ½ℓ²(1 − e<sup>−2kt</sup>)),&ensp;pₛ = e<sup>−x²/ℓ²</sup>/(ℓ√π),&ensp;λₙ = nk'],
  defaults: { x0: 2, t: 0.5 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'x0', label: 'x₀', min: -3, max: 3, step: 0.05 },
        { key: 't', label: 't', min: 0.01, max: 5, log: true, play: true },
      ],
      hint: 'x₀/ℓ and t/τ. The notes start at x₀ = 2ℓ.',
    },
  ],
  model: () => ({ xi: linspace(-3, 3.5, 521), s: linspace(0, 4, 200) }),
  panels: [
    {
      title: '(a) The Gaussian slides to the origin',
      wide: true,
      caption: 'The Gaussian slides to the origin while widening to pₛ.',
      spec: (m, v) => ({
        x: { domain: [-3, 3.5], scale: 'linear', label: 'x/ℓ' },
        y: { scale: 'linear', domain: [0, 1.9], label: 'ℓ p(x, t | x₀, 0)' },
        legendTitle: 't/τ',
        series: [...family([0.05, 0.2, 0.5, 1, 3], m.xi, (x, t) => O.ouLine(x, t, v.x0)), ref('pₛ', m.xi, sample(m.xi, O.ouPs)), live(`t = ${fmt(v.t)}`, m.xi, sample(m.xi, (x) => O.ouLine(x, v.t, v.x0)))],
        heading: heading('x/ℓ'),
      }),
    },
    {
      title: '(b) Moments',
      wide: true,
      caption: '⟨x⟩ relaxes at the gap k; the second moment, with no overlap on mode 1, at 2k.',
      interaction: 'Drag to move t.',
      pick: (x) => ({ t: clamp(x, 0.01, 5) }),
      spec: (m, v) => ({
        x: { domain: [0, 4], scale: 'linear', label: 't/τ' },
        y: { scale: 'log', domain: [1e-3, 10], label: 'deviation from pₛ' },
        series: [
          line('|⟨x⟩|/ℓ, rate k', 'blue', m.s, sample(m.s, (s) => Math.abs(v.x0) * Math.exp(-s))),
          line('|⟨x²⟩ − ℓ²/2|/ℓ², rate 2k', 'orange', m.s, sample(m.s, (s) => Math.abs(v.x0 ** 2 - 0.5) * Math.exp(-2 * s))),
        ],
        cursor: v.t,
        cursorLabel: 't',
        heading: heading('t/τ'),
      }),
    },
  ],
  results: ['x ∝ H₁ overlaps mode 1 only, so ⟨x(t)⟩ = x₀e<sup>−kt</sup> exactly and C(t) ∝ e<sup>−k|t|</sup>.', 'x² = H₀ and H₂ only: ⟨x²⟩ − ½ℓ² = (x₀² − ½ℓ²)e<sup>−2kt</sup>, faster than the gap.'],
  simulate: (v) => ({ ...ou(v.x0), domain: [-4, 4], bc: ['open', 'open'], T: 4 }),
});

// ---------------------------------------------------------------- §21 half-line reflecting

export const ouHlReflecting = figurePage({
  id: 'ou-hl-reflecting',
  group: GROUP,
  title: 'Ornstein–Uhlenbeck on the half-line, reflecting at the bottom',
  files: ['ou_halfline_reflecting.pdf'],
  gardiner: '§4.4.4',
  intro: 'The harmonic well with a reflecting wall at its bottom. The mirror keeps the even modes only, so the gap doubles to 2k.',
  equations: ['p = N(m, σ²)(x) + N(−m, σ²)(x),&ensp;m = x₀e<sup>−kt</sup>,&ensp;pₛ = 2e<sup>−x²/ℓ²</sup>/(ℓ√π)'],
  defaults: { x0: 2, t: 0.5 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'x0', label: 'x₀', min: 0, max: 3, step: 0.05 },
        { key: 't', label: 't', min: 0.01, max: 5, log: true, play: true },
      ],
      hint: 'x₀/ℓ and t/τ.',
    },
  ],
  model: () => ({ xi: linspace(0, 3.5, 351), s: linspace(1e-3, 4, 300) }),
  panels: [
    {
      title: '(a) Relaxation to the half-Gaussian',
      wide: true,
      caption: 'Relaxation to the half-Gaussian.',
      spec: (m, v) => ({
        x: { domain: [0, 3.5], scale: 'linear', label: 'x/ℓ' },
        y: { scale: 'linear', domain: [0, 1.9], label: 'ℓ p(x, t | x₀, 0)' },
        legendTitle: 't/τ',
        series: [
          ...family([0.05, 0.2, 0.5, 1, 3], m.xi, (x, t) => O.ouHalf(x, t, v.x0, 1)),
          ref('pₛ', m.xi, sample(m.xi, (x) => 2 * O.ouPs(x))),
          live(`t = ${fmt(v.t)}`, m.xi, sample(m.xi, (x) => O.ouHalf(x, v.t, v.x0, 1))),
        ],
        heading: heading('x/ℓ'),
      }),
    },
    {
      title: '(b) Both moments relax at 2k',
      wide: true,
      caption: 'Both moments relax at 2k; the dotted line is the line’s ⟨x⟩, which relaxes at k.',
      interaction: 'Drag to move t.',
      pick: (x) => ({ t: clamp(x, 0.01, 5) }),
      spec: (m, v) => ({
        x: { domain: [0, 4], scale: 'linear', label: 't/τ' },
        y: { scale: 'log', domain: [1e-4, 5] },
        series: [
          line('|⟨x⟩ − ⟨x⟩ₛ|/ℓ', 'blue', m.s, sample(m.s, (s) => Math.abs(O.ouFoldedMean(s, v.x0) - 1 / Math.sqrt(Math.PI)))),
          line('|⟨x²⟩ − ℓ²/2|/ℓ²', 'orange', m.s, sample(m.s, (s) => Math.abs(v.x0 ** 2 - 0.5) * Math.exp(-2 * s))),
          line('line: ⟨x⟩/ℓ (rate k)', 'muted', m.s, sample(m.s, (s) => v.x0 * Math.exp(-s)), { dash: 'dotted', width: 1.5 }),
        ],
        cursor: v.t,
        cursorLabel: 't',
        heading: heading('t/τ'),
      }),
    },
  ],
  results: ['The reflecting wall removes the odd modes, including the line’s slowest, H₁: the gap is 2k.', 'x² is even, so its moment is the line’s, unchanged by the wall.'],
  simulate: (v) => ({ ...ou(v.x0), domain: [0, 4], bc: ['reflecting', 'open'], T: 4 }),
});

// ---------------------------------------------------------------- §22 half-line absorbing

const hlT = memo(() => {
  const x = geomspace(1e-2, 1e2, 80);
  return { x, T: sample(x, O.ouHlT) };
});

export const ouHlAbsorbing = figurePage({
  id: 'ou-hl-absorbing',
  group: GROUP,
  title: 'Ornstein–Uhlenbeck on the half-line, absorbing at the bottom',
  files: ['ou_halfline_absorbing.pdf'],
  gardiner: '§5.2.7',
  intro: 'The harmonic well with an absorbing wall at its bottom. From a point start the conditioned density converges to the Rayleigh QSD, unlike the free half-line, where the same shape kept spreading.',
  equations: ['S = erf(x₀/√(ℓ²(e<sup>2kt</sup> − 1))),&ensp;q<sub>∞</sub> = (2x/ℓ²) e<sup>−x²/ℓ²</sup>,&ensp;λ = k'],
  defaults: { x0: 2, t: 1 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'x0', label: 'x₀', min: 0.05, max: 4, step: 0.05 },
        { key: 't', label: 't', min: 0.02, max: 6, log: true, play: true },
      ],
      hint: 'x₀/ℓ and t/τ.',
    },
  ],
  model: (v) => ({ xi: linspace(0, 3.5, 351), s: linspace(1e-3, 6, 300), T: hlT(), x0: v.x0 }),
  panels: [
    {
      title: '(a) Conditioned density',
      caption: 'From a point start the conditioned density converges to the Rayleigh QSD; compare the free half-line, where the same shape kept spreading.',
      spec: (m, v) => {
        const q = (x: number, t: number) => O.ouHalf(x, t, v.x0, -1) / O.ouHlSurvival(t, v.x0);
        return {
          x: { domain: [0, 3.5], scale: 'linear', label: 'x/ℓ' },
          y: { scale: 'linear', domain: [0, 2.4], label: 'ℓ q(x, t)' },
          legendTitle: 't/τ',
          series: [...family([0.1, 0.3, 0.6, 1, 3], m.xi, q), ref('QSD', m.xi, sample(m.xi, O.ouHlQsd)), live(`t = ${fmt(v.t)}`, m.xi, sample(m.xi, (x) => q(x, v.t)))],
          heading: heading('x/ℓ'),
        };
      },
    },
    {
      title: '(b) Survival',
      caption: 'Survival decays at λ<sub>min</sub> = k.',
      interaction: 'Drag to move t.',
      pick: (x) => ({ t: clamp(x, 0.02, 6) }),
      spec: (m, v) => ({
        x: { domain: [0, 6], scale: 'linear', label: 't/τ' },
        y: { scale: 'log', domain: [1e-2, 1.5], label: 'S(t | x₀)' },
        series: [
          line('exact', 'ink', m.s, sample(m.s, (s) => O.ouHlSurvival(s, v.x0))),
          ref('(2/√π)(x₀/ℓ) e^(−t/τ)', m.s, sample(m.s, (s) => (2 / Math.sqrt(Math.PI)) * v.x0 * Math.exp(-s)), { color: 'muted' }),
        ],
        cursor: v.t,
        cursorLabel: 't',
        heading: heading('t/τ'),
      }),
    },
    {
      title: '(c) Mean exit time',
      caption: 'Mean exit time, with its logarithmic asymptote. Unbounded T(x₀) is what rules out a unique QSD.',
      interaction: 'Drag to move x₀.',
      pick: (x) => ({ x0: clamp(Math.round(x * 20) / 20, 0.05, 4) }),
      spec: (m, v) => {
        const big = geomspace(1, 100, 40);
        return {
          x: { domain: [1e-2, 1e2], scale: 'log', label: 'x₀/ℓ' },
          y: { scale: 'linear', domain: [0, 5.5], label: 'T(x₀)/τ' },
          series: [line('exact', 'ink', m.T.x, m.T.T), ref('ln(2x₀/ℓ) + γ/2', big, sample(big, (x) => Math.log(2 * x) + EULER_GAMMA / 2), { color: 'muted' })],
          cursor: v.x0,
          cursorLabel: 'x₀',
          heading: heading('x₀/ℓ'),
        };
      },
    },
  ],
  results: [
    'A continuum of QSDs, λ ∈ (0, k], with tails x<sup>−1−λ/k</sup>; the Rayleigh density (λ = k) is the only light-tailed one.',
    'T(x₀) = √π τ ∫₀<sup>x₀/ℓ</sup> erfcx u du ~ τ[ln(2x₀/ℓ) + γ/2]: +∞ is not an entrance boundary.',
  ],
  simulate: (v) => ({ ...ou(v.x0), domain: [0, 5], bc: ['absorbing', 'open'], T: 6, cond: true }),
});

// ---------------------------------------------------------------- §23 interval reflecting

export const ouReflecting = figurePage({
  id: 'ou-reflecting',
  group: GROUP,
  title: 'Ornstein–Uhlenbeck on an interval, both walls reflecting',
  files: ['ou_reflecting.pdf'],
  gardiner: '§5.2.5',
  intro: 'The harmonic well on (0, L), reflecting at the bottom and at L. The gap crosses over from π²D/2L² (L ≪ ℓ, Part I) to 2k (L ≫ ℓ, the half-line).',
  equations: ['pₛ ∝ e<sup>−x²/ℓ²</sup> on (0, L),&ensp;p = Σ e<sup>−λₙt</sup> φₙ(x) ψₙ(x₀)'],
  defaults: { L: 2, x0: 1.5, t: 0.3 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'L', label: 'L', min: 0.3, max: 5, step: 0.05 },
        { key: 'x0', label: 'x₀', min: 0, max: 1, step: 0.01 },
        { key: 't', label: 't', min: 0.005, max: 3, log: true, play: true },
      ],
      hint: 'L/ℓ, x₀ as a fraction of L, and t/τ. The notes use L = 2ℓ and x₀ = 1.5ℓ (0.75 L).',
    },
  ],
  model: (v) => ({ m: O.ouModes(v.L, 'reflecting', 100, 800), x: linspace(0, v.L, 401), x0: v.x0 * v.L }),
  panels: [
    {
      title: '(a) Stationary densities',
      wide: true,
      caption: 'Stationary densities, the half-Gaussian truncated at L.',
      spec: (_, v) => {
        // The half-Gaussian truncated at L: ∫₀ᴸ e^{−x²} dx = (√π/2) erf L.
        const ps = (L: number) => {
          const x = linspace(0, L, 200);
          const Z = (Math.sqrt(Math.PI) / 2) * erf(L);
          return { x, y: sample(x, (y) => Math.exp(-y * y) / Z) };
        };
        const fam: Line[] = L_OU.map((L, i) => {
          const { x, y } = ps(L);
          return line(fmt(L), LEN[i], x, y, { width: 1.6 });
        });
        const cur = ps(v.L);
        return {
          x: { domain: [0, 4], scale: 'linear', label: 'x/ℓ' },
          y: { scale: 'linear', domain: [0, 2.4], label: 'ℓ pₛ(x)' },
          legendTitle: 'L/ℓ',
          series: [...fam, live(`L = ${fmt(v.L)}`, cur.x, cur.y)],
          heading: heading('x/ℓ'),
        };
      },
    },
    {
      title: (_, v) => `(b) Relaxation for L = ${fmt(v.L)}ℓ`,
      wide: true,
      caption: 'Relaxation from x₀, by the spectral solution on (0, L).',
      spec: (m, v) => {
        const at = (t: number) => {
          const p = m.m.density(m.x0, t);
          return sample(m.x, (x) => m.m.at(p, x));
        };
        const snaps = [0.05, 0.1, 0.3, 0.6, 2];
        return {
          x: { domain: [0, v.L], scale: 'linear', label: 'x/ℓ' },
          y: { scale: 'linear', max: 4, label: 'ℓ p(x, t | x₀, 0)' },
          legendTitle: 't/τ',
          series: [
            ...snaps.map((t, i) => line(fmt(t), SEQ[i], m.x, at(t), { width: 1.6 })),
            ref('pₛ', m.x, sample(m.x, (x) => m.m.at(m.m.forward(0), x))),
            live(`t = ${fmt(v.t)}`, m.x, at(v.t)),
          ],
          heading: heading('x/ℓ'),
        };
      },
    },
  ],
  results: ['Gap λ₁: π²D/2L² for L ≪ ℓ (free reflecting interval), 2k for L ≫ ℓ (reflecting half-line).', 'The slowest function varies over whichever is shorter, L or ℓ.'],
  simulate: (v) => ({ ...ou(v.x0 * v.L), domain: [0, v.L], bc: ['reflecting', 'reflecting'], T: 3 }),
});

// ---------------------------------------------------------------- §24 interval absorbing

export const ouAbsorbing = figurePage({
  id: 'ou-absorbing',
  group: GROUP,
  title: 'Ornstein–Uhlenbeck on an interval, both walls absorbing',
  files: ['ou_absorbing.pdf'],
  gardiner: '§5.2.7–8',
  intro: 'The harmonic well on (0, L), absorbing at the bottom and at L. Climbing to L is uphill, so the process almost always falls back to the bottom wall.',
  equations: ['π<sub>L</sub> = erfi(x₀/ℓ)/erfi(L/ℓ),&ensp;T = τ[π<sub>L</sub>G(L/ℓ) − G(x₀/ℓ)],&ensp;G(ξ) = √π ∫₀^ξ e<sup>u²</sup> erf u du'],
  defaults: { L: 2 },
  controls: [{ title: 'Parameters', controls: [{ key: 'L', label: 'L', min: 0.3, max: 5, step: 0.05 }], hint: 'L/ℓ for the heavy curves.' }],
  model: () => ({ u: linspace(1e-4, 1, 300), xi: linspace(0, 4, 300) }),
  panels: [
    {
      title: '(a) Quasi-stationary densities',
      caption: 'Quasi-stationary densities, from the sine of Part I (L ≪ ℓ) to the Rayleigh QSD of the half-line (dashed).',
      spec: (m, v) => {
        const qsd = (L: number) => {
          const md = O.ouModes(L, 'absorbing', 4, 800);
          const f = md.forward(0);
          return { x: md.x, y: f };
        };
        return {
          x: { domain: [0, 4], scale: 'linear', label: 'x/ℓ' },
          y: { scale: 'linear', domain: [0, 4.2], label: 'ℓ q∞(x)' },
          legendTitle: 'L/ℓ',
          series: [
            ...L_OU.map((L, i) => {
              const q = qsd(L);
              return line(fmt(L), LEN[i], q.x, q.y, { width: 1.6 });
            }),
            ref('∞', m.xi, sample(m.xi, O.ouHlQsd)),
            (() => {
              const q = qsd(v.L);
              return live(`L = ${fmt(v.L)}`, q.x, q.y);
            })(),
          ],
          heading: heading('x/ℓ'),
        };
      },
    },
    {
      title: '(b) Probability of leaving through the upper wall',
      caption: 'Probability of leaving through the upper wall.',
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x₀/L' },
        y: { scale: 'linear', domain: [0, 1.02], label: 'π_L(x₀)' },
        legendTitle: 'L/ℓ',
        series: [...family(L_OU, m.u, (u, L) => O.ouPiL(u * L, L), LEN), live(`L = ${fmt(v.L)}`, m.u, sample(m.u, (u) => O.ouPiL(u * v.L, v.L)))],
        heading: heading('x₀/L'),
      }),
    },
    {
      title: '(c) Mean exit time',
      caption: 'Mean exit time.',
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x₀/L' },
        y: { scale: 'log', domain: [1e-3, 3], label: 'T(x₀)/τ' },
        legendTitle: 'L/ℓ',
        series: [...family(L_OU, m.u, (u, L) => O.ouT(u * L, L, 'absorbing'), LEN), live(`L = ${fmt(v.L)}`, m.u, sample(m.u, (u) => O.ouT(u * v.L, v.L, 'absorbing')))],
        heading: heading('x₀/L'),
      }),
    },
  ],
  results: ['λ₀: π²D/2L² for L ≪ ℓ, k for L ≫ ℓ (the Rayleigh QSD of the half-line).'],
  simulate: (v) => ({ ...ou(0.5 * v.L), domain: [0, v.L], bc: ['absorbing', 'absorbing'], T: 3, cond: true }),
});

// ---------------------------------------------------------------- §25 escape from a well

const arrhenius = memo(() => {
  const Ls = linspace(0.3, 4.2, 80);
  return {
    L2: sample(Ls, (L) => L * L),
    T0: sample(Ls, (L) => O.ouT(0, L, 'mixed')),
    kramers: sample(Ls, (L) => 1 / O.ouKramers(L)),
    dotsL2: [1, 4, 9, 16],
    dots: [1, 2, 3, 4].map((L) => 1 / O.ouLevels(L, 'mixed', 1)[0]),
  };
});

export const ouMixed = figurePage({
  id: 'ou-mixed',
  group: GROUP,
  title: 'Ornstein–Uhlenbeck: reflecting at the bottom, absorbing at L (escape from a well)',
  files: ['ou_mixed.pdf'],
  gardiner: '§5.2.7c, §5.2.8',
  intro: 'Escape from a harmonic well over a barrier ΔU = kL²/2: reflecting at the bottom, absorbing at L. For L ≫ ℓ the exit time becomes exponential with the Kramers rate.',
  equations: ['T(x₀) = τ[G(L/ℓ) − G(x₀/ℓ)],&ensp;λ₀ ≈ (2L/ℓ√π) k e<sup>−L²/ℓ²</sup> (Kramers)'],
  defaults: { L: 2 },
  controls: [{ title: 'Parameters', controls: [{ key: 'L', label: 'L', min: 0.3, max: 4.2, step: 0.05 }], hint: 'L/ℓ: the barrier is ΔU/(D/2) = L²/ℓ².' }],
  model: () => ({ u: linspace(0, 1 - 1e-4, 300), xi: linspace(0, 4, 300), arr: arrhenius() }),
  panels: [
    {
      title: '(a) Quasi-stationary densities',
      caption: 'Quasi-stationary densities; dashed, the half-Gaussian.',
      spec: (m, v) => {
        const qsd = (L: number) => {
          const md = O.ouModes(L, 'mixed', 4, 800);
          return { x: md.x, y: md.forward(0) };
        };
        const cur = qsd(v.L);
        return {
          x: { domain: [0, 4], scale: 'linear', label: 'x/ℓ' },
          y: { scale: 'linear', domain: [0, 3.4], label: 'ℓ q∞(x)' },
          legendTitle: 'L/ℓ',
          series: [
            ...L_OU.map((L, i) => {
              const q = qsd(L);
              return line(fmt(L), LEN[i], q.x, q.y, { width: 1.6 });
            }),
            ref('pₛ, ∞', m.xi, sample(m.xi, (x) => 2 * O.ouPs(x))),
            live(`L = ${fmt(v.L)}`, cur.x, cur.y),
          ],
          heading: heading('x/ℓ'),
        };
      },
    },
    {
      title: '(b) Mean exit time',
      caption: 'Mean exit time (log scale); colours as in (a). It is nearly flat: from anywhere in the well, the time is that of the barrier.',
      spec: (m, v) => ({
        x: { domain: [0, 1], scale: 'linear', label: 'x₀/L' },
        y: { scale: 'log', domain: [1e-3, 1e7], label: 'T(x₀)/τ' },
        legendTitle: 'L/ℓ',
        series: [...family(L_OU, m.u, (u, L) => O.ouT(u * L, L, 'mixed'), LEN), live(`L = ${fmt(v.L)}`, m.u, sample(m.u, (u) => O.ouT(u * v.L, v.L, 'mixed')))],
        heading: heading('x₀/L'),
      }),
    },
    {
      title: '(c) Arrhenius plot',
      caption: 'T(0) and 1/λ₀ against the barrier ΔU/(D/2) = L²/ℓ², with the leading Kramers asymptote.',
      interaction: 'Drag to move L.',
      pick: (x) => ({ L: clamp(Math.round(Math.sqrt(Math.max(0, x)) * 20) / 20, 0.3, 4.2) }),
      spec: (m, v) => ({
        x: { domain: [0, 18], scale: 'linear', label: 'kL²/D = ΔU/(D/2)' },
        y: { scale: 'log', domain: [0.1, 1e7], label: 'time/τ' },
        series: [
          line('T(0)', 'ink', m.arr.L2, m.arr.T0),
          ref('Kramers', m.arr.L2, m.arr.kramers, { color: 'muted' }),
          { kind: 'dots', label: '1/λ₀', color: 'orange', x: m.arr.dotsL2, y: m.arr.dots },
          { kind: 'dots', label: `1/λ₀ at L = ${fmt(v.L)}`, color: 'ink', x: [v.L * v.L], y: [1 / O.ouLevels(v.L, 'mixed', 1)[0]], z: 2 },
        ],
        cursor: v.L * v.L,
        cursorLabel: 'L²',
        heading: (x) => `L²/ℓ² = ${fmt(x)}`,
      }),
    },
  ],
  results: ['The Kramers formula with its first correction matches λ₀ to 0.1 % at L = 5ℓ, and λ₀T(0) = 1.0005 already at L = 3ℓ.'],
  simulate: (v) => ({ ...ou(0), domain: [0, v.L], bc: ['reflecting', 'absorbing'], T: Math.min(50, 2 * O.ouT(0, v.L, 'mixed')), cond: true }),
});

// ---------------------------------------------------------------- §26 sawtooth circle

const SAW_L = [1, 2, 3, 4];

const sawSpectrum = memo(() => {
  const Ls = linspace(1.2, 9, 80);
  const even = [0, 1, 2, 3].map(() => new Float64Array(Ls.length));
  const odd = [0, 1, 2, 3].map(() => new Float64Array(Ls.length));
  Ls.forEach((L, i) => {
    C.sawLevels(L, 'even', 4, 1).forEach((l, j) => (even[j][i] = l));
    C.sawLevels(L, 'odd', 4, 0).forEach((l, j) => (odd[j][i] = l));
  });
  return { Ls, even, odd };
});

export const ouCircle = figurePage({
  id: 'ou-circle',
  group: GROUP,
  title: 'Ornstein–Uhlenbeck on the circle',
  files: ['ou_circle.pdf'],
  gardiner: '§5.2.2, §5.2.7',
  intro: 'The harmonic well wrapped onto a circle of length L: A = −kx for |x| &lt; L/2, repeated, with a cusp at x = ±L/2. x is measured from the bottom of the well.',
  equations: ['pₛ ∝ e<sup>−x²/ℓ²</sup>, |x| &lt; L/2,&ensp;D<sub>eff</sub> = DL²/[ℓ²π erf(L/2ℓ) erfi(L/2ℓ)] (Lifson–Jackson)'],
  defaults: { L: 4 },
  controls: [{ title: 'Parameters', controls: [{ key: 'L', label: 'L', min: 1, max: 9, step: 0.05 }], hint: 'The circumference L/ℓ.' }],
  model: () => ({ u: linspace(-0.5, 0.5, 401), spec: sawSpectrum(), Ld: linspace(0.3, 9, 160) }),
  panels: [
    {
      title: '(a) Stationary densities',
      caption: 'Truncated Gaussians, repeated, with a kink at the cusp x = ±L/2.',
      spec: (m, v) => ({
        x: { domain: [-0.5, 0.5], scale: 'linear', label: 'x/L' },
        y: { scale: 'linear', domain: [0, 2.7], label: 'L pₛ(x)' },
        legendTitle: 'L/ℓ',
        series: [...family(SAW_L, m.u, (u, L) => L * C.sawPs(u * L, L), LEN), live(`L = ${fmt(v.L)}`, m.u, sample(m.u, (u) => v.L * C.sawPs(u * v.L, v.L)))],
        heading: heading('x/L'),
      }),
    },
    {
      title: '(b) Spectrum',
      caption: 'Even sector (blue; reflecting walls at the bottom and the cusp) and odd sector (orange; absorbing). Each pair is split by exactly k. Dotted, the free doublets 2π²n²D/L²; dashed, the OU ladder.',
      interaction: 'Drag to move L.',
      pick: (x) => ({ L: clamp(Math.round(x * 20) / 20, 1, 9) }),
      spec: (m, v) => {
        const { Ls, even, odd } = m.spec;
        const series: Series[] = [];
        even.forEach((y, j) => series.push(line('even (refl.)', 'blue', Ls, y, { legend: j === 0, tooltip: true, width: 1.8 })));
        odd.forEach((y, j) => series.push(line('odd (abs.)', 'orange', Ls, y, { legend: j === 0, width: 1.8 })));
        for (let n = 1; n <= 3; n++) {
          const Lf = linspace(1.2, Math.min(9, 5 * n), 60);
          series.push(line('free doublets', 'muted', Lf, sample(Lf, (L) => (2 * Math.PI ** 2 * n * n) / (L * L)), { dash: 'dotted', width: 1.25, legend: n === 1, tooltip: false, noScale: true }));
        }
        return {
          x: { domain: [1.2, 9], scale: 'linear', label: 'L/ℓ' },
          y: { scale: 'linear', domain: [0, 7.5], label: 'λ/k' },
          series,
          rules: [1, 2, 3, 4, 5, 6, 7].map((n) => yRule(n, undefined, 'muted', 'dashed')),
          cursor: v.L,
          cursorLabel: 'L',
          heading: heading('L/ℓ'),
          rows: (x) => {
            const ev = C.sawLevels(x, 'even', 4, 1);
            const od = C.sawLevels(x, 'odd', 4, 0);
            return [...od.map((l, j) => ({ l, c: 'orange' as const, n: `odd ${j}` })), ...ev.map((l, j) => ({ l, c: 'blue' as const, n: `even ${j + 1}` }))]
              .sort((a, b) => a.l - b.l)
              .map((r) => ({ label: r.n, value: fmt(r.l), color: r.c, key: 'solid' as const }));
          },
        };
      },
    },
    {
      title: '(c) Effective diffusion of the lift',
      caption: 'Lifson–Jackson (black) and the hopping limit ½L²λ₀<sup>mix</sup> with the leading Kramers rate (dashed). The notes add an independent ring discretisation, which the tests here check against.',
      interaction: 'Drag to move L.',
      pick: (x) => ({ L: clamp(Math.round(x * 20) / 20, 1, 9) }),
      spec: (m, v) => {
        const Lk = linspace(3.5, 9, 60);
        return {
          x: { domain: [0, 9], scale: 'linear', label: 'L/ℓ' },
          y: { scale: 'log', domain: [1e-6, 3], label: 'D_eff/D' },
          series: [line('exact', 'ink', m.Ld, sample(m.Ld, C.sawDeff)), ref('½L²λ₀ᵐⁱˣ (Kramers)', Lk, sample(Lk, (L) => 0.5 * L * L * O.ouKramers(L / 2)), { color: 'muted' })],
          cursor: v.L,
          cursorLabel: 'L',
          heading: heading('L/ℓ'),
        };
      },
    },
  ],
  results: ['Folding at the bottom: the even part of p is OU reflecting on (0, L/2), the odd part OU absorbing.', 'For winding, no point of the circle is special, not even the cusp.'],
  // One period with periodic ends: −kx on (−L/2, L/2) is the sawtooth, cusp at the seam.
  simulate: (v) => ({ ...ou(0), domain: [-v.L / 2, v.L / 2], bc: ['periodic', 'periodic'], T: 10 }),
});

// ---------------------------------------------------------------- §27 cosine well

const KAPPAS = [1, 3, 9];
const KAPPA_ROLES = [SEQ[1], SEQ[2], SEQ[4]];

const cosSweep = memo(() => {
  const ks = geomspace(0.05, 100, 60);
  const lv = [0, 1, 2, 3].map(() => new Float64Array(ks.length));
  ks.forEach((k, i) => C.cosLevels(k, 'even', 5).slice(1).forEach((l, j) => (lv[j][i] = l)));
  return { ks, lv };
});

const cosTransportCurve = memo((kappa: number) => {
  const fs = linspace(0, 3, 121);
  const tr = Array.from(fs, (f) => C.cosTransport(kappa, f));
  const R = C.cosR(kappa);
  return { fs, V: Float64Array.from(tr, (t) => t.V / R), D: Float64Array.from(tr, (t) => t.Deff) };
}, 32);

export const cosCircle = figurePage({
  id: 'cos-circle',
  group: GROUP,
  title: 'A smooth well on the circle',
  files: ['cos_circle.pdf'],
  gardiner: '§5.2.2, §5.2.7c',
  intro: 'The cosine well A = v − kR sin(x/R) on a circle of length L = 2πR, with depth κ = 2R²/ℓ² and tilt v. Without tilt it has no special point; with tilt it carries a current, depins at v = kR and shows giant diffusion there.',
  equations: ['U = −kR² cos(x/R) − vx,&ensp;κ = 2R²/ℓ²,&ensp;tilt f = v/kR'],
  defaults: { kappa: 3, f: 1 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'kappa', label: 'κ', min: 0.1, max: 30, log: true },
        { key: 'f', label: 'f', min: 0, max: 3, step: 0.01, play: true },
      ],
      hint: 'Depth κ = 2R²/ℓ² and tilt f = v/kR for the heavy curves. The notes use κ = 3 in (a).',
    },
  ],
  model: (v) => ({ u: linspace(-0.5, 0.5, 401), sweep: cosSweep(), cur: cosTransportCurve(v.kappa), fam: KAPPAS.map(cosTransportCurve) }),
  panels: [
    {
      title: (_, v) => `(a) Stationary densities at κ = ${fmt(v.kappa)}`,
      caption: 'Stationary densities for several tilts: von Mises at v = 0, a current-carrying NESS otherwise.',
      spec: (m, v) => {
        const L = 2 * Math.PI * C.cosR(v.kappa);
        const ps = (f: number) => {
          const tr = C.cosTransport(v.kappa, f);
          return sample(m.u, (u) => L * C.cosDensity(tr, u * L, v.kappa));
        };
        return {
          x: { domain: [-0.5, 0.5], scale: 'linear', label: 'x/L' },
          y: { scale: 'linear', max: 6, label: 'L pₛ(x)' },
          legendTitle: 'v/kR',
          series: [...[0, 0.5, 1, 1.5, 3].map((f, i) => line(fmt(f), SEQ[i], m.u, ps(f), { width: 1.6 })), live(`f = ${fmt(v.f)}`, m.u, ps(v.f))],
          heading: heading('x/L'),
        };
      },
    },
    {
      title: '(b) Spectrum without tilt',
      caption: 'Every level is a doublet. Dotted, the free circle m²k/κ; dashed, the OU ladder nk.',
      interaction: 'Drag to move κ.',
      pick: (x) => ({ kappa: clamp(Number(x.toPrecision(3)), 0.1, 30) }),
      spec: (m, v) => {
        const { ks, lv } = m.sweep;
        const series: Series[] = lv.map((y, j) => line('exact (doublets)', 'blue', ks, y, { legend: j === 0, width: 1.8 }));
        for (let mm = 1; mm <= 3; mm++) series.push(line('free circle', 'muted', ks, sample(ks, (k) => (mm * mm) / k), { dash: 'dotted', width: 1.25, legend: mm === 1, tooltip: false, noScale: true }));
        return {
          x: { domain: [0.05, 100], scale: 'log', label: 'κ = 2R²/ℓ²' },
          y: { scale: 'log', domain: [0.5, 40], label: 'λ/k' },
          series,
          rules: [1, 2, 3, 4].map((n) => yRule(n, n === 1 ? 'nk' : undefined, 'muted', 'dashed')),
          cursor: v.kappa,
          cursorLabel: 'κ',
          heading: (x) => `κ = ${fmt(x)}`,
          rows: (x) => C.cosLevels(x, 'even', 5).slice(1).map((l, j) => ({ label: `doublet ${j + 1}`, value: fmt(l), color: 'blue', key: 'solid' })),
        };
      },
    },
    {
      title: '(c) Winding velocity',
      caption: 'Winding velocity against tilt for three depths; dashed, the noiseless rotor; dotted, no well.',
      interaction: 'Drag to move f.',
      pick: (x) => ({ f: clamp(Math.round(x * 100) / 100, 0, 3) }),
      spec: (m, v) => {
        const fd = linspace(1, 3, 200);
        return {
          x: { domain: [0, 3], scale: 'linear', label: 'v/kR' },
          y: { scale: 'linear', domain: [0, 3], label: 'V/kR' },
          legendTitle: 'κ',
          series: [
            ...KAPPAS.map((k, i) => line(fmt(k), KAPPA_ROLES[i], m.fam[i].fs, m.fam[i].V, { width: 1.6 })),
            ref('no noise', fd, sample(fd, (f) => Math.sqrt(f * f - 1))),
            ref('no well', [0, 3], [0, 3], { color: 'muted', dash: 'dotted' }),
            live(`κ = ${fmt(v.kappa)}`, m.cur.fs, m.cur.V),
          ],
          cursor: v.f,
          cursorLabel: 'f',
          heading: heading('v/kR'),
        };
      },
    },
    {
      title: '(d) Effective diffusion of the lift',
      caption: 'Effective diffusion of the lift; dotted, the bare D. Near depinning (v ≈ kR) it is enhanced by orders of magnitude: giant diffusion.',
      interaction: 'Drag to move f.',
      pick: (x) => ({ f: clamp(Math.round(x * 100) / 100, 0, 3) }),
      spec: (m, v) => ({
        x: { domain: [0, 3], scale: 'linear', label: 'v/kR' },
        y: { scale: 'log', domain: [1e-7, 300], label: 'D_eff/D' },
        legendTitle: 'κ',
        series: [...KAPPAS.map((k, i) => line(fmt(k), KAPPA_ROLES[i], m.fam[i].fs, m.fam[i].D, { width: 1.6 })), live(`κ = ${fmt(v.kappa)}`, m.cur.fs, m.cur.D)],
        rules: [yRule(1, 'D', 'muted', 'dotted')],
        cursor: v.f,
        cursorLabel: 'f',
        heading: heading('v/kR'),
      }),
    },
  ],
  results: [
    'Without tilt the doublets stay exactly degenerate: the smooth well has no special point.',
    'The NESS density is not the Boltzmann factor of any potential; for large tilt it approaches the dwell-time density ∝ 1/|v − kR sin(x/R)|.',
  ],
  simulate: (v) => {
    const R = C.cosR(v.kappa);
    return { a: 'f*k*R - k*R*sin(x/R)', b: 'sqrt(D)', params: { f: v.f, k: 1, R, D: 1 }, x0: 0, domain: [-Math.PI * R, Math.PI * R], bc: ['periodic', 'periodic'], T: 20 };
  },
});

// ---------------------------------------------------------------- §28 summary

const summaryRates = memo(() => {
  const Ls = geomspace(0.2, 4.5, 60);
  return {
    Ls,
    refl: sample(Ls, (L) => O.ouLevels(L, 'reflecting', 1, 1)[0]),
    abs: sample(Ls, (L) => O.ouLevels(L, 'absorbing', 1)[0]),
    mixed: sample(Ls, (L) => O.ouLevels(L, 'mixed', 1)[0]),
  };
});

export const ouSummary = figurePage({
  id: 'ou-summary',
  group: GROUP,
  title: 'Summary of Ornstein–Uhlenbeck',
  files: ['ou_summary.pdf'],
  intro: 'The slowest rates of the three OU intervals against L/ℓ: the free results of Part I for L ≪ ℓ, the half-line limits for L ≫ ℓ, and Kramers escape for the mixed interval.',
  equations: ['λ ≈ π²D/2L² or π²D/8L² (L ≪ ℓ),&ensp;2k, k (L ≫ ℓ),&ensp;λ₀<sup>mix</sup> ≈ (2L/ℓ√π) k e<sup>−L²/ℓ²</sup>'],
  defaults: { L: 1.5 },
  controls: [{ title: 'Cursor', controls: [{ key: 'L', label: 'L', min: 0.2, max: 4.5, log: true }], hint: 'L/ℓ; hover the chart for every rate.' }],
  model: () => summaryRates(),
  panels: [
    {
      title: 'Slowest rates of the three OU intervals',
      caption: 'Slowest rates against L/ℓ, with their Part I asymptotes (dotted), half-line limits (dashed horizontal) and the Kramers rate (dashed).',
      interaction: 'Drag to move L.',
      pick: (x) => ({ L: clamp(Number(x.toPrecision(3)), 0.2, 4.5) }),
      spec: (m, v) => {
        const small = geomspace(0.2, 1.2, 30);
        const large = geomspace(1.5, 4.5, 40);
        return {
          x: { domain: [0.2, 4.5], scale: 'log', label: 'L/ℓ' },
          y: { scale: 'log', domain: [1e-6, 200], label: 'slowest rate × τ' },
          series: [
            line('both reflecting (gap)', 'blue', m.Ls, m.refl),
            line('both absorbing', 'orange', m.Ls, m.abs),
            line('refl. 0, abs. L', 'aqua', m.Ls, m.mixed),
            ref('π²D/2L²', small, sample(small, (L) => Math.PI ** 2 / (2 * L * L)), { color: 'muted', dash: 'dotted' }),
            ref('π²D/8L²', small, sample(small, (L) => Math.PI ** 2 / (8 * L * L)), { color: 'muted', dash: 'dotted' }),
            ref('Kramers', large, sample(large, O.ouKramers), { color: 'muted' }),
            hline('k, 2k', 1, 0.2, 4.5, { color: 'muted', width: 1 }),
            hline('k, 2k', 2, 0.2, 4.5, { color: 'muted', width: 1, legend: false, tooltip: false }),
          ],
          cursor: v.L,
          cursorLabel: 'L',
          heading: heading('L/ℓ'),
        };
      },
    },
  ],
  results: ['The prefactor of the escape rate depends on the shape of the barrier top: a wall (escape from a well), a cusp crossed with probability ½ once reached (the sawtooth circle), a smooth maximum, Gardiner 5.2.174 (the cosine well).'],
});

export const PART3 = [ouLine, ouHlReflecting, ouHlAbsorbing, ouReflecting, ouAbsorbing, ouMixed, ouCircle, cosCircle, ouSummary];
