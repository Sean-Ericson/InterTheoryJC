import type { PageDef, PageHost, PageInstance } from '../pages/types';
import { group, hint } from '../ui/controls';

/**
 * Part VI of the notes: how the problems reduce to each other (reductions.pdf),
 * as a map of this app. Each box opens the page for that kind of problem;
 * each arrow is labelled by the move that turns one problem into the simpler
 * one. Coordinates are the figure's (x in [0, 12], y in [0, 7.4]).
 */

interface Node {
  x: number;
  y: number;
  lines: string[];
  page: string;
  hub?: boolean;
}

const NODES: Record<string, Node> = {
  kramers: { x: 6, y: 7.0, lines: ['underdamped particle in a potential', '(Kramers’ equation, x and v)'], page: 'physical' },
  any: { x: 6, y: 5.85, lines: ['overdamped: any A(x), B(x)'], page: 'physical' },
  sym: { x: 6, y: 4.55, lines: ['B = D, A = −U′:  H = −½D∂ₓ² + V', 'V = U′²/2D − U″/2'], page: 'operators' },
  si: { x: 1.55, y: 3.05, lines: ['shape-invariant V', '(Part IV: log, ln cosh,', '−ln cos, Morse, …)'], page: 'solvable' },
  ou: { x: 4.55, y: 3.05, lines: ['harmonic V:', 'OU (Part III)'], page: 'ou-line' },
  drift: { x: 7.45, y: 3.05, lines: ['constant V:', 'drift v (Part II)'], page: 'drift-hl-reflecting' },
  gen: { x: 10.45, y: 3.05, lines: ['any other V (Part V):', 'quadratures, bounds,', 'Kramers, numerics'], page: 'double-well' },
  free: { x: 6, y: 1.45, lines: ['V = 0: free diffusion, any walls,', 'circle (Part I)'], page: 'free-summary' },
  gauss: { x: 6, y: 0.3, lines: ['the Gaussian on the line'], page: 'hl-reflecting', hub: true },
};

const EDGES: [string, string][] = [
  ['kramers', 'any'],
  ['any', 'sym'],
  ['sym', 'si'],
  ['sym', 'ou'],
  ['sym', 'drift'],
  ['sym', 'gen'],
  ['si', 'ou'],
  ['ou', 'free'],
  ['drift', 'free'],
  ['free', 'gauss'],
];

/** Labels on the arrows (the move), and two notes in secondary ink. */
const LABELS: { x: number; y: number; lines: string[]; anchor: 'start' | 'middle' | 'end'; note?: boolean }[] = [
  { x: 6.15, y: 6.42, lines: ['eliminate v (γ² ≫ m𝒰″)'], anchor: 'start' },
  { x: 6.15, y: 5.27, lines: ['Lamperti y = ∫dx/√B, then symmetrise p = √pₛ ψ'], anchor: 'start' },
  { x: 3.2, y: 4.05, lines: ['read off V'], anchor: 'end' },
  { x: 3.05, y: 2.45, lines: ['radius of', 'd-dim OU'], anchor: 'middle' },
  { x: 4.95, y: 1.98, lines: ['y = x e^(kt), clock θ(t)', '(walls at the bottom)'], anchor: 'end' },
  { x: 7.05, y: 2.2, lines: ['gauge e^(α(x−x₀) − t/2τ)', '(boost on a circle)'], anchor: 'start' },
  { x: 6.15, y: 0.95, lines: ['images, cover, fold, cut'], anchor: 'start' },
  { x: 10.45, y: 1.75, lines: ['symmetry still helps:', 'fold, cover, cut;', 'Doob, partner U ↔ −U'], anchor: 'middle', note: true },
  { x: 1.55, y: 1.75, lines: ['partner U ↔ −U', '(ladders, doublets);', 'Doob: conditioning'], anchor: 'middle', note: true },
];

const W = 1200;
const H = 780;
const px = (x: number) => (x / 12) * W;
const py = (y: number) => 30 + ((7.4 - y) / 7.4) * (H - 60);
const LINE = 18;
const CHAR = 7.6;

function box(n: Node): { cx: number; cy: number; w: number; h: number } {
  const w = Math.max(...n.lines.map((l) => l.length)) * CHAR + 28;
  const h = n.lines.length * LINE + 16;
  return { cx: px(n.x), cy: py(n.y), w, h };
}

/** Where the segment from the box centre toward (tx, ty) leaves the box, plus a gap. */
function exit(b: ReturnType<typeof box>, tx: number, ty: number, gap: number): [number, number] {
  const dx = tx - b.cx;
  const dy = ty - b.cy;
  const s = Math.min(Math.abs(b.w / 2 / (dx || 1e-9)), Math.abs(b.h / 2 / (dy || 1e-9)));
  const len = Math.hypot(dx, dy);
  return [b.cx + dx * s + (dx / len) * gap, b.cy + dy * s + (dy / len) * gap];
}

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');

function svg(): string {
  const tspans = (lines: string[], x: number, y: number) => {
    const y0 = y - ((lines.length - 1) * LINE) / 2;
    return lines.map((l, i) => `<tspan x="${x}" y="${y0 + i * LINE}">${esc(l)}</tspan>`).join('');
  };
  const edges = EDGES.map(([a, b]) => {
    const A = box(NODES[a]);
    const B = box(NODES[b]);
    const [x0, y0] = exit(A, B.cx, B.cy, 4);
    const [x1, y1] = exit(B, A.cx, A.cy, 6);
    return `<line class="red-edge" x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" marker-end="url(#red-arrow)" />`;
  }).join('');
  const nodes = Object.values(NODES)
    .map((n) => {
      const b = box(n);
      return `<a href="?page=${n.page}" class="red-node${n.hub ? ' hub' : ''}"><title>Open: ${esc(n.lines.join(' '))}</title><rect x="${b.cx - b.w / 2}" y="${b.cy - b.h / 2}" width="${b.w}" height="${b.h}" rx="8" /><text text-anchor="middle" dominant-baseline="central">${tspans(n.lines, b.cx, b.cy)}</text></a>`;
    })
    .join('');
  const labels = LABELS.map(
    (l) => `<text class="${l.note ? 'red-note' : 'red-label'}" text-anchor="${l.anchor}" dominant-baseline="central">${tspans(l.lines, px(l.x), py(l.y))}</text>`,
  ).join('');
  return `<svg class="reductions" viewBox="0 0 ${W} ${H}" role="img" aria-label="Map of how the problems reduce to each other; each box links to its page">
    <defs><marker id="red-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="red-head" /></marker></defs>
    ${edges}${labels}${nodes}</svg>`;
}

function mount(host: PageHost): PageInstance {
  const intro = group(host.controls);
  const blurb = document.createElement('p');
  blurb.className = 'blurb';
  blurb.textContent =
    'Change variables until the problem is one you know. Each arrow points to a simpler problem and is labelled by the move that gets there; everything solvable in closed form ends at the Gaussian.';
  const source = document.createElement('p');
  source.className = 'source';
  source.innerHTML = 'Notes: <cite>Reductions: a summary</cite> · <code>reductions.pdf</code>';
  intro.append(blurb, source, hint('Click a box to open the page for that kind of problem.'));

  const head = document.createElement('header');
  head.className = 'fig-head';
  const h2 = document.createElement('h2');
  h2.className = 'fig-title';
  h2.textContent = 'Reductions: a summary';
  head.append(h2);
  const card = document.createElement('section');
  card.className = 'card reductions-card';
  const scroll = document.createElement('div');
  scroll.className = 'reductions-scroll';
  scroll.innerHTML = svg();
  const cap = document.createElement('p');
  cap.className = 'caption';
  cap.textContent = 'How the problems reduce. Each arrow points to a simpler problem and is labelled by the move that gets there. Everything that is solvable in closed form ends at the Gaussian.';
  card.append(scroll, cap);
  host.main.append(head, card);
  return { redraw: () => {}, destroy: () => {} };
}

export const reductions: PageDef = {
  id: 'reductions',
  title: 'Reductions: a summary',
  group: 'Part VII · Reductions',
  mount,
};
