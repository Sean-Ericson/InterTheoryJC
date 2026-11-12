import type { Line, Series } from '../plot/chart';
import { SEQ, type Role } from '../plot/theme';
import { geomspace, linspace, memo, simpson } from '../math/num';
import { ouV } from '../math/ou';
import * as P from '../math/potentials';
import { Rng } from '../rng';
import { figurePage } from './framework';
import { clamp, family, fmt, heading, levelSegment, line, live, ref, sample, xRule, yRule } from './kit';

/**
 * The later material of the notes: where the equation comes from (toolkit),
 * Part IV (more exactly solvable potentials) and Part V (arbitrary
 * potentials, worked on the double well). D = 1 throughout.
 */
const PART4 = 'Part IV · More exactly solvable potentials';
const PART5 = 'Part V · Arbitrary potentials';
const LEN: Role[] = SEQ.slice(1);

// ---------------------------------------------------------------- toolkit: where the equation comes from

/**
 * One Brownian path from x₀ = 1 (D = 1, step 10⁻³ up to t = 4). The notes scan
 * seeds for a path that hits 0 between t = 1 and 2, dips below −0.8 and climbs
 * above 2.2; `pick` scans from the given seed the same way.
 */
const wallsPath = memo((seed: number) => {
  const n = 4000;
  const dt = 1e-3;
  const t = linspace(0, 4, n + 1);
  for (let s = seed; s < seed + 2000; s++) {
    const rng = new Rng(s);
    const x = new Float64Array(n + 1);
    x[0] = 1;
    for (let i = 1; i <= n; i++) x[i] = x[i - 1] + Math.sqrt(dt) * rng.normal();
    let hit = -1;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i <= n; i++) {
      if (hit < 0 && x[i] <= 0) hit = i;
      lo = Math.min(lo, x[i]);
      hi = Math.max(hi, x[i]);
    }
    if (hit > 0 && t[hit] > 1 && t[hit] < 2 && lo < -0.8 && hi > 2.2) return { t, x, hit, seed: s };
  }
  return { t, x: new Float64Array(n + 1).fill(1), hit: n, seed };
}, 16);

export const physical = figurePage({
  id: 'physical',
  group: 'Conventions and toolkit',
  title: 'Where the equation comes from: overdamped Brownian motion',
  files: ['walls.pdf', 'overdamped.pdf'],
  gardiner: '§6.4',
  intro:
    'The Fokker–Planck equation of the notes describes an overdamped particle: inertia gone, A = −𝒰′/γ. Walls and domains are features of the landscape: a hard wall reflects, a cliff removes, a ring wraps.',
  equations: ['m dv = −γv dt − 𝒰′(x) dt + √(2γkT) dW&ensp;→&ensp;γ dx = −𝒰′(x) dt + √(2γkT) dW&ensp;(γ² ≫ m𝒰″)'],
  defaults: { seed: 1, ring: 2, zeta: 0.5 },
  controls: [
    {
      title: '(a–c) One path, three walls',
      controls: [
        { key: 'seed', label: 'path', min: 1, max: 200, step: 1 },
        { key: 'ring', label: 'L', min: 0.5, max: 4, step: 0.05 },
      ],
      hint: 'Which path (each is the first from that seed to hit 0 between t = 1 and 2), and the ring’s circumference L/x₀. The notes use L = 2x₀.',
    },
    { title: '(d, e) Inertia', controls: [{ key: 'zeta', label: 'ζ', min: 0.1, max: 20, log: true }], hint: 'Damping ratio ζ = γ/2√(mκ) for the heavy curve.' },
  ],
  model: (v) => ({ path: wallsPath(v.seed), tc: linspace(0, 5, 501), zs: geomspace(0.2, 30, 400) }),
  panels: [
    {
      title: '(a) Hard wall: the path is turned back',
      caption: 'A hard wall turns the path back; this is |x|, the image construction of the reflecting half-line.',
      spec: (m) => ({
        x: { domain: [0, 4], scale: 'linear', label: 'Dt/x₀²' },
        y: { scale: 'linear', domain: [-1.5, 3.2], label: 'x/x₀' },
        series: [
          { kind: 'band', label: 'beyond the wall', color: 'muted', x: [0, 4], y: [-1.5, 0], alpha: 0.14 },
          line('reflected path |x|', 'blue', m.path.t, sample(m.path.x, Math.abs), { width: 1.2 }),
        ],
        heading: heading('Dt/x₀²'),
      }),
    },
    {
      title: '(b) Cliff: the path is removed',
      caption: 'At a cliff the path is removed at its first arrival (cross); the dotted continuation is what it would have done.',
      spec: (m) => {
        const { t, x, hit } = m.path;
        const before = sample(x, (y, i) => (i <= hit ? y : NaN));
        const after = sample(x, (y, i) => (i >= hit ? y : NaN));
        return {
          x: { domain: [0, 4], scale: 'linear', label: 'Dt/x₀²' },
          y: { scale: 'linear', domain: [-1.5, 3.2], label: 'x/x₀' },
          series: [
            { kind: 'band', label: 'off the cliff', color: 'muted', x: [0, 4], y: [-1.5, 0], alpha: 0.14 },
            line('continuation', 'muted', t, after, { width: 1, dash: 'dotted' }),
            line('path until removed', 'blue', t, before, { width: 1.2 }),
            { kind: 'dots', label: `removed at t = ${fmt(t[hit])}`, color: 'orange', x: [t[hit]], y: [0] },
          ],
          heading: heading('Dt/x₀²'),
        };
      },
    },
    {
      title: (_, v) => `(c) Ring of circumference ${fmt(v.ring)}x₀`,
      caption: 'On a ring the path wraps around (solid); the dotted curve is its lift, which counts the windings.',
      spec: (m, v) => {
        const L = v.ring;
        const { t, x } = m.path;
        const w = sample(x, (y) => ((y % L) + L) % L);
        // Break the wrapped curve where it jumps across the seam.
        for (let i = 1; i < w.length; i++) if (Math.abs(w[i] - w[i - 1]) > L / 2) w[i - 1] = NaN;
        return {
          x: { domain: [0, 4], scale: 'linear', label: 'Dt/x₀²' },
          y: { scale: 'linear', domain: [-1.5, 3.2], label: 'x/x₀' },
          series: [line('lift (winding)', 'muted', t, x, { width: 1, dash: 'dotted' }), line('on the ring', 'blue', t, w, { width: 1.2, tooltip: false })],
          rules: [yRule(0, undefined, 'muted', 'solid'), yRule(L, 'L', 'muted', 'solid')],
          heading: heading('Dt/x₀²'),
        };
      },
    },
    {
      title: '(d) Position correlation with inertia',
      wide: true,
      caption: 'A particle in a harmonic trap with inertia: stationary position correlation for several damping ratios; dashed, the OU (Smoluchowski) prediction e<sup>−κt/γ</sup>.',
      spec: (m, v) => ({
        x: { domain: [0, 5], scale: 'linear', label: 't κ/γ' },
        y: { scale: 'linear', domain: [-0.75, 1.05], label: 'C(t)/C(0)' },
        legendTitle: 'damping ratio ζ',
        series: [
          ...family([0.25, 1, 3, 10], m.tc, (t, z) => P.dampedC(t, z), LEN),
          ref('Smoluchowski', m.tc, sample(m.tc, (t) => Math.exp(-t))),
          live(`ζ = ${fmt(v.zeta)}`, m.tc, sample(m.tc, (t) => P.dampedC(t, v.zeta))),
        ],
        heading: heading('tκ/γ'),
      }),
    },
    {
      title: '(e) Slowest rate against damping',
      wide: true,
      caption: 'Long-time decay rate in units of the Smoluchowski rate κ/γ. It is largest, twice the Smoluchowski value, at critical damping. Dashed, Gardiner’s corrected Smoluchowski equation (6.4.108).',
      interaction: 'Drag to move ζ.',
      pick: (x) => ({ zeta: clamp(Number(x.toPrecision(3)), 0.1, 20) }),
      spec: (m, v) => {
        const zz = geomspace(1.3, 30, 100);
        return {
          x: { domain: [0.2, 30], scale: 'log', label: 'damping ratio ζ = γ/2√(mκ)' },
          y: { scale: 'linear', domain: [0, 2.2], label: 'slowest rate / (κ/γ)' },
          series: [
            line('exact', 'ink', m.zs, sample(m.zs, P.dampedRate)),
            ref('corrected, 1 + 1/4ζ²', zz, sample(zz, (z) => 1 + 1 / (4 * z * z)), { color: 'orange' }),
            ref('Smoluchowski', [0.2, 30], [1, 1], { color: 'muted', dash: 'dotted' }),
          ],
          rules: [xRule(1, 'critical', 'muted', 'solid')],
          cursor: v.zeta,
          cursorLabel: 'ζ',
          heading: (z) => `ζ = ${fmt(z)}`,
        };
      },
    },
  ],
  results: [
    'Overdamped means γ² ≫ m𝒰″: velocity relaxes (time m/γ) before the force changes. Otherwise the problem lives in (x, v) (Kramers’ equation).',
    'Gardiner’s corrected Smoluchowski equation gives the rate 1 + 1/4ζ² in units of κ/γ.',
  ],
});

// ---------------------------------------------------------------- Part IV: the gallery

interface Gallery {
  title: string;
  x: Float64Array;
  V: (x: number) => number;
  levels: number[];
  edge: number | null;
  y: [number, number];
}

function galleryModel(v: { d: number; sigma: number; sm: number; sc: number; st: number }): Gallery[] {
  const cr = (v.d - 1) / 2;
  const n = (k: number) => Array.from({ length: k }, (_, i) => i);
  return [
    { title: '(a) Harmonic: ½kx²', x: linspace(-3.4, 3.4, 1201), V: ouV, levels: n(6), edge: null, y: [-0.8, 5.6] },
    {
      title: `(b) Harmonic + log, d = ${fmt(v.d)}`,
      x: linspace(0.05, 3.6, 1201),
      V: P.vFrom((x) => x - cr / x, (x) => 1 + cr / (x * x)),
      levels: n(4).map((k) => 2 * k),
      edge: null,
      y: [-3, 5.6],
    },
    {
      title: `(c) Linear + log: vx − c ln x, σ = ${fmt(v.sigma)}`,
      x: linspace(0.1, 30, 1501),
      V: P.vFrom((x) => 1 - v.sigma / x, (x) => v.sigma / (x * x)),
      levels: n(30).map((k) => 0.5 * (1 - (v.sigma / (k + v.sigma)) ** 2)),
      edge: 0.5,
      y: [-0.6, 0.85],
    },
    {
      title: `(d) Morse: vx + U₀e^(−x/w), s = ${fmt(v.sm)}`,
      x: linspace(-3, 5, 1201),
      V: P.vFrom((x) => v.sm - Math.exp(-x), (x) => Math.exp(-x)),
      levels: n(Math.ceil(v.sm)).map((k) => 0.5 * k * (2 * v.sm - k)),
      edge: 0.5 * v.sm * v.sm,
      y: [-2.2, Math.max(8, 0.6 * v.sm * v.sm)],
    },
    {
      title: `(e) a ln cosh(x/w), s = ${fmt(v.sc)}`,
      x: linspace(-5, 5, 1201),
      V: P.vFrom((x) => v.sc * Math.tanh(x), (x) => v.sc * P.sech(x) ** 2),
      levels: P.lncoshLevels(v.sc, 1),
      edge: 0.5 * v.sc * v.sc,
      y: [-2.2, Math.max(8, 0.6 * v.sc * v.sc)],
    },
    {
      title: `(f) −a ln cos(x/w), s = ${fmt(v.st)}`,
      x: linspace(-1.5, 1.5, 1201),
      V: P.vFrom((x) => v.st * Math.tan(x), (x) => v.st / Math.cos(x) ** 2),
      levels: n(4).map((k) => P.lncosLam(k, v.st, 1)),
      edge: null,
      y: [-1.5, Math.max(13, 1.2 * P.lncosLam(3, v.st, 1))],
    },
  ];
}

function galleryPanel(i: number) {
  return {
    title: (m: Gallery[]) => m[i].title,
    caption: [
      'Grows faster than linearly: discrete levels nk only.',
      'The radius of a d-dimensional OU motion: levels 2nk.',
      'Levels off at v²/2D with infinitely many levels crowding up to it.',
      'Levels off at v²/2D, with finitely many levels below it (n < s).',
      'Levels off at v²/2D, with finitely many levels below it (n < s).',
      'Grows without bound at its own walls: levels n(n + 2s) in units of D/2w².',
    ][i],
    spec: (m: Gallery[]) => {
      const g = m[i];
      const Vx = sample(g.x, g.V);
      const series: Series[] = [];
      if (g.edge !== null) {
        series.push({ kind: 'band', label: 'continuum', color: 'blue', x: [g.x[0], g.x[g.x.length - 1]], y: [g.edge, g.y[1]], alpha: 0.12 });
        series.push(line('continuum edge', 'blue', [g.x[0], g.x[g.x.length - 1]], [g.edge, g.edge], { width: 1, legend: false, tooltip: false }));
      }
      series.push(line('V', 'ink', g.x, Vx, { width: 1.5, legend: false }));
      g.levels.forEach((E) => series.push(line('levels', 'blue', g.x, levelSegment(g.x, Vx, E), { width: 2, legend: false, tooltip: false })));
      return {
        x: { domain: [g.x[0], g.x[g.x.length - 1]] as [number, number], scale: 'linear' as const, label: 'x' },
        y: { scale: 'linear' as const, domain: g.y, label: 'V, λ' },
        series,
        heading: heading('x'),
        rows: (x: number) => [
          { label: 'V(x)', value: fmt(g.V(x)), color: 'ink' as const, key: 'solid' as const },
          ...g.levels.filter((E) => E <= g.y[1]).slice(0, 6).map((E, k) => ({ label: `λ${k}`, value: fmt(E), color: 'blue' as const, key: 'solid' as const })),
          ...(g.edge !== null ? [{ label: 'continuum from', value: fmt(g.edge), color: 'blue' as const, key: 'band' as const }] : []),
        ],
      };
    },
  };
}

export const solvable = figurePage({
  id: 'solvable',
  group: PART4,
  title: 'Factorisation, partners and shape invariance',
  files: ['solvable.pdf'],
  intro: 'The shape-invariant potentials in the symmetrised picture: V (black), the discrete rates drawn where V &lt; λ (blue), and the continuum (shaded) where there is one.',
  equations: ['V = U′²/2D − U″/2,&ensp;H = −(D/2)∂ₓ² + V = A†A + λ₀,&ensp;partner U ↔ −U'],
  defaults: { d: 5, sigma: 2, sm: 3.5, sc: 3.5, st: 2 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'd', label: 'd', min: 1, max: 9, step: 0.1 },
        { key: 'sigma', label: 'σ', min: 0.2, max: 6, step: 0.05 },
        { key: 'sm', label: 's, Morse', min: 0.3, max: 6, step: 0.05 },
        { key: 'sc', label: 's, ln cosh', min: 0.3, max: 6, step: 0.05 },
        { key: 'st', label: 's, −ln cos', min: 0.1, max: 6, step: 0.05 },
      ],
      hint: 'The dimension d in (b), σ in (c), and s = a/D in (d), (e), (f). D = 1, k = 1, v = 1, w = U₀ = 1.',
    },
  ],
  model: galleryModel,
  panels: [0, 1, 2, 3, 4, 5].map(galleryPanel),
  results: [
    'Discrete-only spectra (a, b, f) come from potentials that grow faster than linearly.',
    'The others level off at V = v²/2D, with finitely many levels below it (d, e), or infinitely many crowding up to it (c).',
  ],
});

// ---------------------------------------------------------------- Part IV: radial motion

const DIMS = [1, 2, 3, 5];

export const radial = figurePage({
  id: 'radial',
  group: PART4,
  title: 'Logarithmic potentials: radial motion in d dimensions',
  files: ['radial.pdf'],
  gardiner: '§5.2.6',
  intro: 'The radius of a d-dimensional free or OU motion is a one-dimensional diffusion with a logarithmic potential: the drift (d − 1)D/2r pushes away from the origin.',
  equations: ['U = kr²/2 − c ln r,&ensp;c = (d − 1)D/2', 'free: p = (r/Dt)(r/r₀)<sup>μ</sup> e<sup>−(r²+r₀²)/2Dt</sup> I<sub>μ</sub>(rr₀/Dt),&ensp;μ = d/2 − 1'],
  defaults: { d: 2.5, t: 1, ratio: 0.35 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'd', label: 'd', min: 1, max: 8, step: 0.05 },
        { key: 't', label: 't', min: 0.02, max: 5, log: true, play: true },
        { key: 'ratio', label: 'a/r₀', min: 0.05, max: 0.95, step: 0.01 },
      ],
      hint: 'Dimension d (any real d ≥ 1) for the heavy curves, Dt/r₀² for (a), and the target radius a/r₀ for (c). The notes show Dt = r₀².',
    },
  ],
  model: () => ({ r: linspace(1e-4, 4, 500), rs: linspace(0, 3.5, 400), dd: linspace(1, 8, 300) }),
  panels: [
    {
      title: (_, v) => `(a) Bessel process from r₀ at Dt = ${fmt(v.t)} r₀²`,
      caption: 'Free motion (Bessel process). The repulsion (d − 1)D/2r empties the region near the origin.',
      spec: (m, v) => ({
        x: { domain: [0, 4], scale: 'linear', label: 'r/r₀' },
        y: { scale: 'linear', max: 3, label: 'r₀ p(r, t | r₀)' },
        legendTitle: 'd',
        series: [...family(DIMS, m.r, (r, d) => P.radialP(r, v.t, 1, d), LEN), live(`d = ${fmt(v.d)}`, m.r, sample(m.r, (r) => P.radialP(r, v.t, 1, v.d)))],
        heading: heading('r/r₀'),
      }),
    },
    {
      title: '(b) Stationary radius of OU motion',
      caption: 'Stationary chi densities of the radius of a d-dimensional OU motion; d = 2 is the Rayleigh density.',
      spec: (m, v) => ({
        x: { domain: [0, 3.5], scale: 'linear', label: 'r/ℓ' },
        y: { scale: 'linear', domain: [0, 1.25], label: 'ℓ pₛ(r)' },
        legendTitle: 'd',
        series: [...family(DIMS, m.rs, (r, d) => P.radialPs(r, d), LEN), live(`d = ${fmt(v.d)}`, m.rs, sample(m.rs, (r) => P.radialPs(r, v.d)))],
        heading: heading('r/ℓ'),
      }),
    },
    {
      title: '(c) Does it ever come back?',
      caption: 'Probability that free motion ever comes within a of the origin. Curves are (a/r₀)<sup>d−2</sup>; dots are the splitting-probability quadrature (reach a before R = 10⁶ r₀).',
      interaction: 'Drag to move d.',
      pick: (x) => ({ d: clamp(Math.round(x * 20) / 20, 1, 8) }),
      spec: (m, v) => {
        const di = [3, 4, 5, 6];
        const series: Series[] = [];
        ([0.5, 0.2] as const).forEach((a, i) => {
          const c = i === 0 ? 'blue' : 'orange';
          series.push(line(`a/r₀ = ${a}`, c, m.dd, sample(m.dd, (d) => P.radialReach(a, 1, d))));
          series.push({ kind: 'dots', label: `quadrature, a/r₀ = ${a}`, color: c, x: di, y: di.map((d) => P.radialReachBefore(a, 1, d)), legend: false });
        });
        series.push(live(`a/r₀ = ${fmt(v.ratio)}`, m.dd, sample(m.dd, (d) => P.radialReach(v.ratio, 1, d))));
        return {
          x: { domain: [1, 8], scale: 'linear', label: 'dimension d' },
          y: { scale: 'linear', domain: [0, 1.05], label: 'P(ever reach r = a)' },
          series,
          cursor: v.d,
          cursorLabel: 'd',
          heading: (d) => `d = ${fmt(d)}`,
        };
      },
    },
  ],
  results: ['Recurrent for d ≤ 2 (the origin is approached with certainty), transient for d > 2 (Pólya).', 'With k > 0 the radius has the chi density; the spectrum is 2nk (Gardiner 5.2.135, Laguerre polynomials).'],
  simulate: (v) => ({ a: 'c/x', b: 'sqrt(D)', params: { c: (v.d - 1) / 2, D: 1 }, domain: [0, 6], bc: ['reflecting', 'open'], x0: 1, T: 5 }),
});

// ---------------------------------------------------------------- Part IV: the soft V

/** Bound states n = 0…5: the notes' ramp for the first four, then ink. */
const BOUND_ROLES: Role[] = ['seq2', 'seq3', 'seq4', 'seq5', 'ink2', 'ink'];

const lncoshLevelsCurve = memo(() => {
  const s = linspace(1e-3, 6, 600);
  const lv = [0, 1, 2, 3, 4, 5].map((n) => sample(s, (x) => (x > n ? 1 - (1 - n / x) ** 2 : NaN)));
  return { s, lv };
});

export const lncosh = figurePage({
  id: 'lncosh',
  group: PART4,
  title: 'A soft V: U = a ln cosh(x/w)',
  files: ['lncosh.pdf'],
  intro: 'Between a V and a parabola: U ≈ a|x|/w far out (constant drift v = a/w toward the centre) and a parabola near the bottom. Lengths in ℓ = D/v, s = a/D = w/ℓ.',
  equations: ['pₛ ∝ sech<sup>2s</sup>(x/w),&ensp;λₙ = (D/2w²) n(2s − n), n &lt; s,&ensp;continuum from v²/2D = 1/2τ'],
  defaults: { s: 3.5 },
  controls: [{ title: 'Parameters', controls: [{ key: 's', label: 's', min: 0.1, max: 6, step: 0.05 }], hint: 's = a/D = w/ℓ for the heavy curve in (a), the cursor in (b), and the bound states in (c). The notes use s = 3.5 in (c).' }],
  model: () => ({ x: linspace(-6, 6, 801), xc: linspace(-12, 12, 801), lv: lncoshLevelsCurve() }),
  panels: [
    {
      title: '(a) Stationary densities',
      caption: 'Stationary densities; dashed, the sharp V (s → 0).',
      spec: (m, v) => ({
        x: { domain: [-6, 6], scale: 'linear', label: 'x/ℓ' },
        y: { scale: 'linear', domain: [0, 1.1], label: 'ℓ pₛ(x)' },
        legendTitle: 's = a/D',
        series: [
          ...family([0.25, 1, 4], m.x, (x, s) => P.lncoshPs(x, s, s), LEN),
          ref('v|x|', m.x, sample(m.x, (x) => Math.exp(-2 * Math.abs(x)))),
          live(`s = ${fmt(v.s)}`, m.x, sample(m.x, (x) => P.lncoshPs(x, v.s, v.s))),
        ],
        heading: heading('x/ℓ'),
      }),
    },
    {
      title: '(b) Discrete rates',
      caption: 'Discrete rates in units of the continuum edge 1/2τ against s = w/ℓ. A new level leaves the edge at each integer s; dotted, the OU values nk = 2n/s.',
      interaction: 'Drag to move s.',
      pick: (x) => ({ s: clamp(Math.round(x * 20) / 20, 0.1, 6) }),
      spec: (m, v) => {
        const series: Series[] = [{ kind: 'band', label: 'continuum', color: 'blue', x: [0, 6], y: [1, 1.2], alpha: 0.12 }];
        m.lv.lv.forEach((y, n) => series.push(line('levels λₙ', 'ink', m.lv.s, y, { width: 1.6, legend: n === 0, tooltip: false })));
        for (const n of [1, 2]) {
          const s = linspace(2 * n + 1e-3, 6, 80);
          series.push(line('OU, 2n/s', 'muted', s, sample(s, (x) => (2 * n) / x), { dash: 'dotted', width: 1.25, legend: n === 1, tooltip: false }));
        }
        return {
          x: { domain: [0, 6], scale: 'linear', label: 's = w/ℓ' },
          y: { scale: 'linear', domain: [-0.05, 1.2], label: 'λ · 2τ' },
          series,
          rules: [yRule(1, 'continuum edge', 'blue', 'solid')],
          cursor: v.s,
          cursorLabel: 's',
          heading: (x) => `s = ${fmt(x)}`,
          // With w = sℓ and ℓ = τ = 1, λₙ·2τ = n(2s − n)/s².
          rows: (x) => P.lncoshLevels(x, x).map((l, n) => ({ label: `λ${n} · 2τ`, value: fmt(2 * l), color: 'ink', key: 'solid' })),
        };
      },
    },
    {
      title: (_, v) => `(c) The bound states at s = ${fmt(v.s)}`,
      caption: 'The bound states n &lt; s, with alternating parity (each scaled to its largest value).',
      spec: (m, v) => {
        const N = Math.ceil(v.s);
        const series: Line[] = Array.from({ length: Math.min(N, 6) }, (_, n) => {
          const y = sample(m.xc, (x) => P.lncoshP(x, n, v.s, v.s));
          let big = 0;
          for (const yi of y) big = Math.max(big, Math.abs(yi));
          return line(String(n), BOUND_ROLES[n], m.xc, y.map((yi) => yi / big), { width: 1.8 });
        });
        return {
          x: { domain: [-12, 12], scale: 'linear', label: 'x/ℓ' },
          y: { scale: 'linear', domain: [-1.1, 1.1], label: 'Pₙ(x) (scaled)' },
          legendTitle: 'n',
          series,
          heading: heading('x/ℓ'),
        };
      },
    },
  ],
  results: ['For s → 0 the sharp V: drift ±v, the continuum alone, with pₛ = e<sup>−2|x|/ℓ</sup>/ℓ. For s → ∞ the parabola: OU with k = a/w².'],
  simulate: (v) => ({ a: '-s*tanh(x/s)', b: 'sqrt(D)', params: { s: v.s, D: 1 }, domain: [-10, 10], bc: ['reflecting', 'reflecting'], x0: 3, T: 20 }),
});

// ---------------------------------------------------------------- Part IV: a well with its own walls

const LNCOS_W = 1 / Math.PI;

export const lncos = figurePage({
  id: 'lncos',
  group: PART4,
  title: 'A well with its own walls: U = −a ln cos(x/w)',
  files: ['lncos.pdf'],
  intro: 'Between a box and a parabola: the potential diverges at x = ±πw/2, so the walls come with the well. Width L = πw, s = a/D.',
  equations: ['pₛ ∝ cos<sup>2s</sup>(x/w),&ensp;λₙ = (π²D/2L²)(n² + 2sn),&ensp;p = pₛ Σ C<sub>n</sub><sup>(s)</sup>(sin x/w) C<sub>n</sub><sup>(s)</sup>(sin x₀/w) e<sup>−λₙt</sup>/hₙ'],
  defaults: { s: 1, x0: 0.35, t: 0.01 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 's', label: 's', min: 0.05, max: 12, log: true },
        { key: 'x0', label: 'x₀', min: -0.45, max: 0.45, step: 0.01 },
        { key: 't', label: 't', min: 0.0005, max: 0.3, log: true, play: true },
      ],
      hint: 's = a/D, x₀/L and Dt/L². The notes use s = 1 and x₀ = 0.35L in (c).',
    },
  ],
  model: () => ({ x: linspace(-0.4995, 0.4995, 601), s: geomspace(1e-2, 1e2, 300) }),
  panels: [
    {
      title: '(a) Stationary densities',
      caption: 'Stationary densities, from nearly uniform (s → 0, a reflecting box) to Gaussian.',
      spec: (m, v) => ({
        x: { domain: [-0.5, 0.5], scale: 'linear', label: 'x/L' },
        y: { scale: 'linear', max: 8, label: 'L pₛ(x)' },
        legendTitle: 's = a/D',
        series: [...family([0.1, 0.5, 2, 8], m.x, (x, s) => P.lncosPs(x, s, LNCOS_W), LEN), live(`s = ${fmt(v.s)}`, m.x, sample(m.x, (x) => P.lncosPs(x, v.s, LNCOS_W)))],
        heading: heading('x/L'),
      }),
    },
    {
      title: '(b) Rates: the box plus OU',
      caption: 'Rates in units of the box rate π²D/2L²: exactly n² + 2sn, the box (dashed) plus OU (dotted).',
      interaction: 'Drag to move s.',
      pick: (x) => ({ s: clamp(Number(x.toPrecision(3)), 0.05, 12) }),
      spec: (m, v) => {
        const roles: Role[] = ['blue', 'orange', 'aqua'];
        const series: Series[] = [];
        [1, 2, 3].forEach((n, i) => {
          series.push(line(`n = ${n}`, roles[i], m.s, sample(m.s, (s) => n * n + 2 * s * n)));
          series.push(line(`OU, 2sn`, roles[i], m.s, sample(m.s, (s) => 2 * s * n), { dash: 'dotted', width: 1.25, legend: false, tooltip: false, noScale: true }));
        });
        return {
          x: { domain: [1e-2, 1e2], scale: 'log', label: 's = a/D' },
          y: { scale: 'log', domain: [0.5, 1e3], label: 'λₙ / (π²D/2L²)' },
          series,
          rules: [1, 4, 9].map((y, i) => yRule(y, i === 0 ? 'box n²' : undefined, roles[i], 'dashed')),
          cursor: v.s,
          cursorLabel: 's',
          heading: (s) => `s = ${fmt(s)}`,
        };
      },
    },
    {
      title: (_, v) => `(c) The Gegenbauer propagator at s = ${fmt(v.s)}`,
      caption: 'Relaxation from x₀ by the Gegenbauer expansion, to pₛ (dashed).',
      spec: (m, v) => ({
        x: { domain: [-0.5, 0.5], scale: 'linear', label: 'x/L' },
        y: { scale: 'linear', max: 9.5, label: 'L p(x, t | x₀)' },
        legendTitle: 'Dt/L²',
        series: [
          ...[0.002, 0.006, 0.02, 0.05].map((t, i) => line(fmt(t), LEN[i], m.x, P.lncosP(m.x, t, v.x0, v.s, LNCOS_W), { width: 1.6 })),
          ref('pₛ', m.x, sample(m.x, (x) => P.lncosPs(x, v.s, LNCOS_W))),
          live(`t = ${fmt(v.t)}`, m.x, P.lncosP(m.x, v.t, v.x0, v.s, LNCOS_W)),
        ],
        heading: heading('x/L'),
      }),
    },
  ],
  results: ['The rates add exactly: the box’s n² and OU’s 2sn. For s → 0 the reflecting box of Part I; for s → ∞ OU with k = 2a/w².'],
  simulate: (v) => ({ a: '-(s/w)*tan(x/w)', b: 'sqrt(D)', params: { s: v.s, w: LNCOS_W, D: 1 }, domain: [-0.5, 0.5], bc: ['reflecting', 'reflecting'], x0: v.x0, T: 0.3 }),
});

// ---------------------------------------------------------------- Part V: the double well

const BETAS = [2, 4, 8];

const dwSweep = memo(() => {
  const bs = linspace(1, 16, 31);
  const lv = Array.from(bs, (b) => P.dwLevels(b, 4));
  const ka = sample(bs, (b) => P.dwCurvatures(b)[0]);
  return {
    bs,
    l1: sample(bs, (_, i) => lv[i][1] / ka[i]),
    l2: sample(bs, (_, i) => lv[i][2] / ka[i]),
    l3: sample(bs, (_, i) => lv[i][3] / ka[i]),
    kramers: sample(bs, (b, i) => P.dwKramers(b) / ka[i]),
    dots: [2, 6, 10, 14].map((b) => P.dwGardiner9218(b) / P.dwCurvatures(b)[0]),
  };
});

export const doubleWell = figurePage({
  id: 'double-well',
  group: PART5,
  title: 'Worked example: the double well',
  files: ['double_well.pdf'],
  gardiner: '§5.2.7c, §9.1–9.2',
  intro: 'The quartic double well with minima at ±a and barrier ΔU, solved numerically: a jump-process discretisation keeps detailed balance exact, so exponentially small rates come out with relative accuracy.',
  equations: ['U = ΔU((x/a)² − 1)²,&ensp;β = 2ΔU/D,&ensp;λ₁ ≈ (√(U″(a)|U″(0)|)/π) e<sup>−β</sup> (Kramers)'],
  defaults: { beta: 8, t: 30 },
  controls: [
    {
      title: 'Parameters',
      controls: [
        { key: 'beta', label: 'β', min: 1, max: 16, step: 0.1 },
        { key: 't', label: 't', min: 0.001, max: 10000, log: true, play: true },
      ],
      hint: 'Barrier β = 2ΔU/D, and a time Dt/a² for the cursor in (c). The notes use β = 8 in (c).',
    },
  ],
  model: (v) => ({ x: linspace(-2, 2, 601), sweep: dwSweep(), ts: geomspace(1e-3, 1e4, 240), pop: P.dwPopulation(v.beta, geomspace(1e-3, 1e4, 240)) }),
  panels: [
    {
      title: '(a) Stationary densities',
      wide: true,
      caption: 'Stationary densities e<sup>−2U/D</sup>, normalised.',
      spec: (m, v) => {
        const ps = (b: number) => {
          const U = P.dwU(b);
          const Z = simpson((x) => Math.exp(-2 * U(x)), -3, 3, 3000);
          return sample(m.x, (x) => Math.exp(-2 * U(x)) / Z);
        };
        return {
          x: { domain: [-2, 2], scale: 'linear', label: 'x/a' },
          y: { scale: 'linear', max: 3, label: 'a pₛ(x)' },
          legendTitle: '2ΔU/D',
          series: [...BETAS.map((b, i) => line(fmt(b), LEN[i], m.x, ps(b), { width: 1.6 })), live(`β = ${fmt(v.beta)}`, m.x, ps(v.beta))],
          heading: heading('x/a'),
        };
      },
    },
    {
      title: '(b) The three slowest rates',
      wide: true,
      caption: 'In units of U″(a): λ₁ (exact; dashed, Kramers; dots, Gardiner’s 9.2.18) falls exponentially. λ₂ tends to the hilltop value |U″(0)|/U″(a) = ½, and λ₃ to the intrawell value 1 (dotted).',
      interaction: 'Drag to move β.',
      pick: (x) => ({ beta: clamp(Math.round(x * 10) / 10, 1, 16) }),
      spec: (m, v) => ({
        x: { domain: [1, 16], scale: 'linear', label: '2ΔU/D' },
        y: { scale: 'log', domain: [1e-8, 3], label: 'λ / U″(a)' },
        series: [
          line('λ₁', 'blue', m.sweep.bs, m.sweep.l1),
          line('λ₂', 'orange', m.sweep.bs, m.sweep.l2),
          line('λ₃', 'aqua', m.sweep.bs, m.sweep.l3),
          ref('Kramers', m.sweep.bs, m.sweep.kramers, { color: 'muted' }),
          { kind: 'dots', label: '(9.2.18)', color: 'blue', x: [2, 6, 10, 14], y: m.sweep.dots },
        ],
        rules: [yRule(0.5, undefined, 'muted', 'dotted'), yRule(1, undefined, 'muted', 'dotted')],
        cursor: v.beta,
        cursorLabel: 'β',
        heading: (b) => `2ΔU/D = ${fmt(b)}`,
      }),
    },
    {
      title: (_, v) => `(c) Hopping from the left well, β = ${fmt(v.beta)}`,
      wide: true,
      caption: 'From the left minimum: probability of being on the right, exact against the two-state form ½(1 − e<sup>−λ₁t</sup>).',
      interaction: 'Drag to move t.',
      pick: (x) => ({ t: clamp(Number(x.toPrecision(3)), 0.001, 10000) }),
      spec: (m, v) => {
        const l1 = P.dwLevels(v.beta, 2)[1];
        return {
          x: { domain: [1e-3, 1e4], scale: 'log', label: 'Dt/a²' },
          y: { scale: 'linear', domain: [0, 0.55], label: 'P(x(t) > 0)' },
          series: [line('exact', 'ink', m.ts, m.pop), ref('½(1 − e^(−λ₁t))', m.ts, sample(m.ts, (t) => -0.5 * Math.expm1(-l1 * t)), { color: 'orange' })],
          cursor: v.t,
          cursorLabel: 't',
          heading: heading('Dt/a²'),
        };
      },
    },
    {
      title: '(d) The slow mode is the committor',
      wide: true,
      caption: 'The slowest backward eigenfunction Q₁ (solid) and 2π<sub>c</sub> − 1 (dashed), π<sub>c</sub> the probability of reaching +a before −a, at β = 2 and at the slider’s β.',
      spec: (m, v) => {
        const curves = (b: number, role: Role, heavy: boolean): Line[] => {
          const q = P.dwQ1(b);
          const keep = sample(q.x, (x) => (Math.abs(x) < 2 ? 1 : 0));
          const xs = q.x.filter((_, i) => keep[i]);
          const Qs = q.Q.filter((_, i) => keep[i]);
          const pc = P.dwCommittor(b);
          return [
            line(`Q₁, β = ${fmt(b)}`, role, xs, Qs, { width: heavy ? 2.75 : 1.8, z: heavy ? 5 : 0 }),
            line(`2π_c − 1, β = ${fmt(b)}`, role, m.x, sample(m.x, (x) => 2 * pc(x) - 1), { dash: 'dashed', width: 1.25 }),
          ];
        };
        return {
          x: { domain: [-2, 2], scale: 'linear', label: 'x/a' },
          y: { scale: 'linear', domain: [-1.6, 1.6], label: 'Q₁, 2π_c − 1' },
          series: [...curves(2, 'seq2', false), ...curves(v.beta, 'ink', true)],
          heading: heading('x/a'),
        };
      },
    },
  ],
  results: [
    'Three time scales: the intrawell rate U″(a), the hilltop |U″(0)|, and the hopping rate λ₁ ~ e<sup>−2ΔU/D</sup>.',
    'For a high barrier the slow mode is the committor, so λ₁ is the rate of the two-state reduction between the wells.',
  ],
  simulate: (v) => ({ a: '-2*b*x*(x^2 - 1)', b: 'sqrt(D)', params: { b: v.beta, D: 1 }, domain: [-2.5, 2.5], bc: ['reflecting', 'reflecting'], x0: -1, T: 200 }),
});

export const LATER = { physical, solvable, radial, lncosh, lncos, doubleWell };
