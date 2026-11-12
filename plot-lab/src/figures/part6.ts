import type { Series } from '../plot/chart';
import type { Role } from '../plot/theme';
import { geomspace, linspace } from '../math/num';
import * as S from '../math/singular';
import { figurePage } from './framework';
import { clamp, family, fmt, heading, line, live, ref, sample, xRule, yRule } from './kit';

/**
 * Part VI of the notes: interfaces and singular boundaries. A step in D
 * (Gardiner §5.2.1c), the square-root process where B vanishes (§5.2.1e,
 * Feller's classification), and the QIF neuron's current through infinity
 * (§5.2.1f). D = k = 1 where they appear.
 */
const GROUP = 'Part VI · Interfaces and singular boundaries';

// ---------------------------------------------------------------- §36 a step in the diffusion coefficient

/** The notes' three readings of the noise, in the notes' colours. */
const NOISE: { key: string; name: string; chi: number; color: Role }[] = [
  { key: 'thermal', name: 'thermal', chi: 1, color: 'blue' },
  { key: 'strat', name: 'Stratonovich', chi: 0.5, color: 'aqua' },
  { key: 'ito', name: 'Itô', chi: 0, color: 'orange' },
];
const chiOf = (key: string) => NOISE.find((n) => n.key === key)!.chi;
const IF_DM = 0.25; // the notes' D₋/D₊

export const interfacePage = figurePage({
  id: 'interface',
  group: GROUP,
  title: 'A step in the diffusion coefficient',
  files: ['interface.pdf'],
  gardiner: '§5.2.1c',
  intro:
    'Free diffusion with B = D₋ for x &lt; 0 and D₊ for x &gt; 0. The current is continuous at the interface; the density jumps by K = (D₋/D₊)<sup>1−χ</sup>, which depends on where the noise comes from. Lengths in the half-width a of the box, D₊ = 1.',
  equations: [
    'J(0⁺) = J(0⁻),&ensp;p(0⁺) = K p(0⁻);&ensp;χ = 1 thermal (Gardiner 5.2.16), ½ Stratonovich, 0 Itô',
    'p = g₊(x − x₀) + r g₊(x + x₀) for x &gt; 0,&ensp;(1 − r) g₋(x − x₀√(D₋/D₊)) for x &lt; 0,&ensp;r = (η − 1)/(η + 1),&ensp;η = K√(D₊/D₋)',
  ],
  defaults: { t: 0.25, ratio: IF_DM, noise: 'thermal', x0: 0.5 },
  controls: [
    {
      title: 'Interface',
      controls: [
        { key: 'ratio', label: 'D₋/D₊', min: 0.02, max: 50, log: true },
        { kind: 'choice', key: 'noise', label: 'noise', options: NOISE.map((n) => [n.key, n.name] as [string, string]) },
      ],
      hint: 'The ratio of the diffusion coefficients and the reading of the noise for the heavy curves. The notes use D₋ = D₊/4.',
    },
    {
      title: 'Start and time',
      controls: [
        { key: 'x0', label: 'x₀', min: -0.95, max: 0.95, step: 0.01 },
        { key: 't', label: 't', min: 0.01, max: 2, log: true, play: true },
      ],
      hint: 'Start x₀/a and time D₊t/a² for the heavy curve in (a); (c) marks x₀.',
    },
  ],
  model: () => ({ x: linspace(-1.5, 2.5, 801), xb: linspace(-1, 1, 401) }),
  panels: [
    {
      title: (_, v) => `(a) The density across the interface at D₊t/a² = ${fmt(v.t)}`,
      caption:
        'A thermal particle from x₀ = a/2 with D₋ = D₊/4 (the notes’ snapshots). p is continuous at the interface and its slope jumps by D₊/D₋. The heavy curve uses the sliders; with Itô or Stratonovich noise p jumps by K.',
      spec: (m, v) => {
        const K = S.ifaceK(chiOf(v.noise as string), v.ratio as number);
        return {
          x: { domain: [-1.5, 2.5], scale: 'linear', label: 'x/a' },
          y: { scale: 'linear', domain: [0, 2.1], label: 'a p(x, t | x₀)' },
          legendTitle: 'D₊t/a²',
          series: [
            { kind: 'band', label: 'slow side', color: 'muted', x: [-1.5, 0], alpha: 0.12, legend: false },
            ...family([0.04, 0.1, 0.25, 0.6], m.x, (x, t) => S.ifaceP(x, t, 0.5, 1, IF_DM)),
            // A start on the slow side is the mirror image: the sides (D₊ ↔ D₋) swap and K → 1/K.
            live(`t = ${fmt(v.t)}`, m.x, sample(m.x, (x) => (v.x0 > 0 ? S.ifaceP(x, v.t, v.x0, K, v.ratio) : S.ifaceP(-x, v.t, -v.x0, 1 / K, 1, v.ratio)))),
          ],
          heading: heading('x/a'),
        };
      },
    },
    {
      title: '(b) Stationary densities in the box',
      caption: 'The box (−a, a) with reflecting walls: p<sub>s</sub> ∝ B<sup>χ−1</sup>, flat for a thermal particle, ∝ 1/√B (Stratonovich), ∝ 1/B (Itô).',
      spec: (m, v) => {
        const steps = (K: number) => sample(m.xb, (x) => (x < 0 ? 1 : K) / (1 + K));
        return {
          x: { domain: [-1, 1], scale: 'linear', label: 'x/a' },
          y: { scale: 'linear', domain: [0, 1.2], label: 'a pₛ(x)' },
          series: [
            { kind: 'band', label: 'slow side', color: 'muted', x: [-1, 0], alpha: 0.12, legend: false },
            ...NOISE.map((n) => line(n.name, n.color, m.xb, steps(S.ifaceK(n.chi, IF_DM)))),
            live(`${NOISE.find((n) => n.key === v.noise)!.name}, D₋/D₊ = ${fmt(v.ratio)}`, m.xb, steps(S.ifaceK(chiOf(v.noise as string), v.ratio as number))),
          ],
          heading: heading('x/a'),
        };
      },
    },
    {
      title: '(c) Probability of leaving through +a',
      caption: 'Both walls absorbing. π₊ is piecewise linear, with slopes in the ratio D₋ : KD₊. With Itô noise x is a martingale, so π₊ = (x₀ + a)/2a whatever B does.',
      interaction: 'Drag to move x₀.',
      pick: (x) => ({ x0: clamp(Math.round(x * 100) / 100, -0.95, 0.95) }),
      spec: (m, v) => {
        const K = S.ifaceK(chiOf(v.noise as string), v.ratio as number);
        return {
          x: { domain: [-1, 1], scale: 'linear', label: 'x₀/a' },
          y: { scale: 'linear', domain: [0, 1], label: 'π₊(x₀)' },
          series: [
            { kind: 'band', label: 'slow side', color: 'muted', x: [-1, 0], alpha: 0.12, legend: false },
            ...NOISE.map((n) => line(n.name, n.color, m.xb, sample(m.xb, (x) => S.ifaceSplit(x, S.ifaceK(n.chi, IF_DM), IF_DM)))),
            live(`sliders: T(x₀) = ${fmt(S.ifaceT(v.x0 as number, K, v.ratio as number))} a²/D₊`, m.xb, sample(m.xb, (x) => S.ifaceSplit(x, K, v.ratio as number))),
          ],
          cursor: v.x0 as number,
          cursorLabel: 'x₀',
          heading: heading('x₀/a'),
        };
      },
    },
  ],
  results: [
    'From the interface, T = a²(1 + K)/(D₋ + KD₊) = a²/⟨B⟩ₛ: a² over the arithmetic (thermal), geometric (Stratonovich) or harmonic (Itô) mean of D₊ and D₋.',
    'r = +1 is a reflecting wall, r = −1 an absorbing one. A thermal particle sees a very slow side as a hard wall and a very fast one as a cliff.',
    'A potential step ΔU at fixed D gives K = e<sup>−2ΔU/D</sup> and r = −tanh(ΔU/D).',
  ],
  simulate: (v) => ({
    // Width w, not e: SDE Explorer reads e as Euler's number.
    a: 'c*(p - m)*0.25*sech(x/w)^2/w',
    b: 'sqrt(m + (p - m)*0.5*(1 + tanh(x/w)))',
    params: { c: chiOf(v.noise as string), p: 1, m: v.ratio as number, w: 0.03 },
    domain: [-1, 1],
    bc: ['absorbing', 'absorbing'],
    x0: v.x0 as number,
    T: 3,
  }),
});

// ---------------------------------------------------------------- §37 where B vanishes: the square-root process

const SQRT_BANDS: [number, number, string, Role][] = [
  [-2, 0, 'exit', 'orange'],
  [0, 2, 'regular', 'muted'],
  [2, 4, 'entrance', 'blue'],
];

export const sqrtPage = figurePage({
  id: 'sqrt',
  group: GROUP,
  title: 'Where B vanishes: the square-root process',
  files: ['sqrt.pdf'],
  gardiner: '§5.2.1e',
  intro:
    'A = d − 2y, B = 4y on y &gt; 0, for any real d: the squared radius of the radial section, and the equation Feller studied in 1951 (CIR). The origin is an exit for d ≤ 0, regular for 0 &lt; d &lt; 2 and an entrance for d ≥ 2. Lengths in ℓ² = D/k, times in 1/k.',
  equations: [
    'p<sub>±</sub> = (1/2σ²)(y/ȳ)<sup>μ/2</sup> e<sup>−(y+ȳ)/2σ²</sup> I<sub>±μ</sub>(√(yȳ)/σ²),&ensp;μ = d/2 − 1,&ensp;ȳ = y₀e<sup>−2kt</sup>,&ensp;σ² = ½(1 − e<sup>−2kt</sup>)',
    '+: reflected (d &gt; 0), λₙ = 2nk;&ensp;−: absorbed (d &lt; 2), λₙ = (2n + 2 − d)k,&ensp;S = P(1 − d/2, z<sub>t</sub>)',
  ],
  defaults: { d: 1, t: 0.25, seed: 1 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'd', label: 'd', min: -2, max: 4, step: 0.05 },
        { key: 't', label: 't', min: 0.02, max: 2, log: true, play: true },
        { key: 'seed', label: 'paths', min: 1, max: 200, step: 1 },
      ],
      hint: 'Dimension d for the heavy curves in (c) and the cursor in (b); time kt for (c) (the notes use kt = 0.25); which draw of paths in (a).',
    },
  ],
  model: (v) => ({ paths: S.sqrtPaths(v.seed), dd: linspace(-2, 4, 601), y: geomspace(1e-5, 4, 400) }),
  panels: [
    {
      title: '(a) One path of each kind',
      caption:
        'Paths from y₀ = ℓ². d = 3: an entrance, never reached. d = 1: regular, reflected each time it arrives; absorption would stop it at the cross. d = 0: an exit, absorbed for good. The d = 3 and d = 1 paths are squared radii of OU motions; the d = 0 path is sampled exactly.',
      spec: (m) => {
        const { t, y3, y1, tHit, y0b } = m.paths;
        return {
          x: { domain: [0, 2.5], scale: 'linear', label: 'kt' },
          y: { scale: 'linear', domain: [-0.08, 3.6], label: 'y/ℓ²' },
          series: [
            line('d = 3, entrance', 'blue', t, y3, { width: 1 }),
            line('d = 1, regular', 'aqua', t, y1, { width: 1 }),
            line('d = 0, exit', 'orange', t, y0b, { width: 1 }),
            ...(Number.isFinite(tHit) ? [{ kind: 'dots' as const, label: `d = 1 absorbed at kt = ${fmt(tHit)}`, color: 'ink' as Role, x: [tHit], y: [0] }] : []),
          ],
          heading: heading('kt'),
        };
      },
    },
    {
      title: (_, v) => `(b) Rates against d (cursor at d = ${fmt(v.d)})`,
      caption: 'The reflected ladder 2nk exists for d &gt; 0, the absorbed one (2n + 2 − d)k for d &lt; 2. Only in the regular window do both exist, and there a condition must be chosen.',
      interaction: 'Drag to move d.',
      pick: (x) => ({ d: clamp(Math.round(x * 20) / 20, -2, 4) }),
      spec: (m, v) => {
        const series: Series[] = SQRT_BANDS.map(([lo, hi, label, c]) => ({ kind: 'band', label, color: c, x: [lo, hi], alpha: 0.1 }));
        for (let n = 0; n < 5; n++) series.push(line(n === 0 ? 'reflected, 2nk' : `reflected n = ${n}`, 'blue', m.dd, sample(m.dd, (d) => (d > 0 ? 2 * n : NaN)), { legend: n === 0, tooltip: true }));
        for (let n = 0; n < 4; n++) series.push(line(n === 0 ? 'absorbed, (2n + 2 − d)k' : `absorbed n = ${n}`, 'orange', m.dd, sample(m.dd, (d) => (d < 2 ? 2 * n + 2 - d : NaN)), { legend: n === 0 }));
        return {
          x: { domain: [-2, 4], scale: 'linear', label: 'd' },
          y: { scale: 'linear', domain: [-0.3, 8.2], label: 'λₙ/k' },
          series,
          cursor: v.d as number,
          cursorLabel: 'd',
          heading: (d) => `d = ${fmt(d)}`,
        };
      },
    },
    {
      title: (_, v) => `(c) Densities near the origin at kt = ${fmt(v.t)}`,
      caption: 'From y₀ = ℓ², at the notes’ kt = 0.25: the entrance density vanishes, the reflected one diverges like y<sup>−1/2</sup>, the absorbed ones tend to constants (absorbing does not mean p = 0 where B vanishes). Heavy curves: the sliders’ d and t, reflected (solid) and absorbed (dashed) where each exists.',
      spec: (m, v) => {
        const d = v.d as number;
        const series: Series[] = [
          line('d = 3', 'blue', m.y, sample(m.y, (y) => S.sqrtP(y, 0.25, 1, 3, 1))),
          line('d = 1, reflected', 'aqua', m.y, sample(m.y, (y) => S.sqrtP(y, 0.25, 1, 1, 1))),
          line('d = 1, absorbed', 'aqua', m.y, sample(m.y, (y) => S.sqrtP(y, 0.25, 1, 1, -1)), { dash: 'dashed' }),
          line('d = −1', 'orange', m.y, sample(m.y, (y) => S.sqrtP(y, 0.25, 1, -1, -1)), { dash: 'dashed' }),
        ];
        if (d > 0) series.push(live(`d = ${fmt(d)}, reflected`, m.y, sample(m.y, (y) => S.sqrtP(y, v.t as number, 1, d, 1))));
        if (d < 2) series.push(live(`d = ${fmt(d)}, absorbed`, m.y, sample(m.y, (y) => S.sqrtP(y, v.t as number, 1, d, -1)), { dash: 'dashed' }));
        return {
          x: { domain: [1e-5, 4], scale: 'log', label: 'y/ℓ²' },
          y: { scale: 'log', domain: [1e-3, 30], label: 'ℓ² p(y, t | y₀)' },
          series,
          heading: heading('y/ℓ²'),
        };
      },
    },
  ],
  results: [
    'Reflected (d &gt; 0): pₛ is the gamma density y<sup>d/2−1</sup>e<sup>−y/ℓ²</sup>/Γ(d/2)ℓ<sup>d</sup>; the Laguerre ladder 2nk.',
    'Absorbed (d &lt; 2): S = P(1 − d/2, z<sub>t</sub>), z<sub>t</sub> = (y₀/ℓ²)/(e<sup>2kt</sup> − 1); decay rate (2 − d)k; for d = 0, Pr(extinct by t) = e<sup>−z<sub>t</sub></sup>.',
    'Infinity is natural, so the absorbed process has a continuum of QSDs, M(1 + λ/2k, 2 − d/2, −y/ℓ²) for 0 &lt; λ ≤ (2 − d)k; point starts reach e<sup>−y/ℓ²</sup>.',
  ],
  simulate: (v) => ({ a: 'd - 2*x', b: '2*sqrt(x)', params: { d: v.d as number }, domain: [0, 6], bc: [(v.d as number) > 0 ? 'reflecting' : 'absorbing', 'open'], x0: 1, T: 5 }),
});

// ---------------------------------------------------------------- §38 a current through infinity

const QIF_I = [-1, 0, 1];
const QIF_ROLES: Role[] = ['blue', 'aqua', 'orange'];

export const qifPage = figurePage({
  id: 'qif',
  group: GROUP,
  title: 'A current through infinity',
  files: ['qif.pdf'],
  gardiner: '§5.2.1f',
  intro:
    'The quadratic integrate-and-fire neuron, A = x² + I, B = D. +∞ is an exit (reached in finite time) and −∞ an entrance; what leaves at +∞ is reinjected at −∞, J(+∞) = J(−∞) = J<sub>s</sub> (Gardiner 5.2.33). In θ = 2 arctan x both infinities are one point of a circle.',
  equations: [
    'pₛ(x) = (2J<sub>s</sub>/D) ∫<sub>x</sub><sup>∞</sup> e<sup>φ(x)−φ(y)</sup> dy,&ensp;φ = (2/D)(x³/3 + Ix)',
    '1/J<sub>s</sub> = √(2π/D) ∫₀<sup>∞</sup> u<sup>−1/2</sup> exp[−(2/D)(u³/12 + Iu)] du',
  ],
  defaults: { I: 0.5, D: 1 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'I', label: 'I', min: -2, max: 3, step: 0.05 },
        { key: 'D', label: 'D', min: 0.05, max: 4, log: true },
      ],
      hint: 'Input current I and noise D for the heavy curves. The notes show I = −1, 0, 1 at D = 1.',
    },
  ],
  model: () => ({ x: linspace(-5, 5, 401), Is: linspace(-2, 3, 201), th: linspace(-Math.PI + 1e-3, Math.PI - 1e-3, 401) }),
  panels: [
    {
      title: '(a) Stationary densities',
      caption: 'Stationary densities for D = 1 (the notes’ three I); dotted, the tails J<sub>s</sub>/x² of the heavy curve.',
      spec: (m, v) => {
        const I = v.I as number;
        const D = v.D as number;
        const J = 1 / S.qifT(I, D);
        const tail = sample(m.x, (x) => (Math.abs(x) > 2 ? J / (x * x) : NaN));
        return {
          x: { domain: [-5, 5], scale: 'linear', label: 'x' },
          y: { scale: 'linear', max: 1.5, label: 'pₛ(x)' },
          legendTitle: 'I (D = 1)',
          series: [
            ...family(QIF_I, m.x, (x, i) => S.qifPs(x, i, 1), QIF_ROLES),
            live(`I = ${fmt(I)}, D = ${fmt(D)}`, m.x, sample(m.x, (x) => S.qifPs(x, I, D))),
            ref('Jₛ/x²', m.x, tail, { dash: 'dotted', tooltip: false }),
          ],
          heading: heading('x'),
        };
      },
    },
    {
      title: '(b) Firing rate',
      caption: 'Firing rate against I: exact (solid), the noiseless √I/π (dashed), and Kramers’ formula (c/π)e<sup>−8c³/3D</sup> for I = −c² &lt; 0 (dotted).',
      interaction: 'Drag to move I.',
      pick: (x) => ({ I: clamp(Math.round(x * 20) / 20, -2, 3) }),
      spec: (m, v) => {
        const D = v.D as number;
        const pos = sample(m.Is, (I) => (I > 0 ? Math.sqrt(I) / Math.PI : NaN));
        const kramers = (Dq: number) => sample(m.Is, (I) => (I < -0.3 ? (Math.sqrt(-I) / Math.PI) * Math.exp((-8 * (-I) ** 1.5) / (3 * Dq)) : NaN));
        return {
          x: { domain: [-2, 3], scale: 'linear', label: 'I' },
          y: { scale: 'log', domain: [1e-4, 1], label: 'firing rate Jₛ' },
          series: [
            line('D = 1', 'blue', m.Is, sample(m.Is, (I) => 1 / S.qifT(I, 1))),
            line('D = 0.25', 'orange', m.Is, sample(m.Is, (I) => 1 / S.qifT(I, 0.25))),
            live(`D = ${fmt(D)}`, m.Is, sample(m.Is, (I) => 1 / S.qifT(I, D))),
            ref('√I/π', m.Is, pos),
            ref('Kramers, D = 1', m.Is, kramers(1), { dash: 'dotted', color: 'blue' }),
            ref('Kramers, D = 0.25', m.Is, kramers(0.25), { dash: 'dotted', color: 'orange' }),
          ],
          cursor: v.I as number,
          cursorLabel: 'I',
          heading: heading('I'),
        };
      },
    },
    {
      title: '(c) The same densities on the circle',
      caption: 'The densities of (a) in the angle θ = 2 arctan x. Both infinities are the point θ = π, where the noise vanishes, the drift is 2 and the density is J<sub>s</sub>/2.',
      spec: (m, v) => {
        const I = v.I as number;
        const D = v.D as number;
        return {
          x: { domain: [-1, 1], scale: 'linear', label: 'θ/π' },
          y: { scale: 'linear', max: 1.5, label: 'pₛ(θ)' },
          legendTitle: 'I (D = 1)',
          series: [
            ...QIF_I.map((i, j) => line(fmt(i), QIF_ROLES[j], sample(m.th, (t) => t / Math.PI), sample(m.th, (t) => S.qifPsTheta(t, i, 1)))),
            live(`I = ${fmt(I)}, D = ${fmt(D)}`, sample(m.th, (t) => t / Math.PI), sample(m.th, (t) => S.qifPsTheta(t, I, D))),
          ],
          rules: [xRule(1, 'x = ±∞'), xRule(-1), yRule(0.5 / S.qifT(I, D), 'Jₛ/2')],
          heading: (x) => `θ = ${fmt(x)}π`,
        };
      },
    },
  ],
  results: [
    '1/J<sub>s</sub> → π/√I without noise (I &gt; 0); 6.27 D<sup>−1/3</sup> at I = 0; (π/c)e<sup>8c³/3D</sup> for I = −c² as D → 0.',
    'Through x = tan(θ/2) the free circle has regular boundaries at ±∞: J(±∞) = 0 (5.2.32) is the reflecting interval (−π, π), J(+∞) = J(−∞) (5.2.33) the circle, with the Cauchy density.',
  ],
  simulate: (v) => ({
    a: '1 - cos(x) + I*(1 + cos(x)) - D/2*sin(x)*(1 + cos(x))',
    b: 'sqrt(D)*(1 + cos(x))',
    params: { I: v.I as number, D: v.D as number },
    domain: [-Math.PI, Math.PI],
    bc: ['periodic', 'periodic'],
    x0: -1,
    T: 20,
  }),
});

export const PART6 = [interfacePage, sqrtPage, qifPage];
