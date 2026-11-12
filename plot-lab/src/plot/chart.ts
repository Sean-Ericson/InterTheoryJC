import { Scale, crisp, decadeFloor, fitCanvas, formatValue, type Rect, type ScaleKind, type Tick } from './scale';
import { labelFont, type Role, type Theme } from './theme';
import { hideTooltip, showTooltip, type TooltipRow } from './tooltip';

export type Dash = 'solid' | 'dashed' | 'dotted';

const DASHES: Record<Dash, number[]> = { solid: [], dashed: [6, 4], dotted: [0.5, 4] };

export interface Line {
  kind: 'line';
  label: string;
  color: Role;
  /** Ascending, except where NaN gaps separate segments of a series kept out of the tooltip. */
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  dash?: Dash;
  width?: number;
  /** Left out of the automatic y range: reference curves that may run off the chart. */
  noScale?: boolean;
  /** Drawing order, low first (default 0); dots always go over lines. */
  z?: number;
  /** Arrowheads at the ends of each segment. */
  arrows?: 'end' | 'both';
  /** Left out of the hover tooltip (and the legend if `legend` is false too). */
  tooltip?: boolean;
  legend?: boolean;
}

export interface Dots {
  kind: 'dots';
  label: string;
  color: Role;
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  z?: number;
  tooltip?: boolean;
  legend?: boolean;
}

/** A filled rectangle behind the data: a band, a continuum, a shaded region. */
export interface Band {
  kind: 'band';
  label: string;
  color: Role;
  x: [number, number];
  /** Defaults to the whole y range. */
  y?: [number, number];
  alpha?: number;
  legend?: boolean;
}

/** A field z(x, y) drawn as an image with the sequential ramp, 0 to zMax. */
export interface Heat {
  kind: 'heat';
  label: string;
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  /** z[i·ny + j] at (x[i], y[j]). */
  z: ArrayLike<number>;
  zMax: number;
}

export type Series = Line | Dots | Band | Heat;

/** A reference line across the plot at a fixed x or y. */
export interface Rule {
  axis: 'x' | 'y';
  value: number;
  color: Role;
  label?: string;
  dash?: Dash;
}

export interface ChartSpec {
  x: {
    domain: [number, number];
    scale: ScaleKind;
    label: string;
    /** Replace the automatic ticks (e.g. category names under a level diagram). */
    ticks?: Tick[];
    grid?: boolean;
  };
  y: {
    scale: ScaleKind;
    /** Fixed range; otherwise fitted to the line series. */
    domain?: [number, number];
    /** Log axis: at most this many decades below the top. */
    decades?: number;
    /** Linear axis: never fit the top above this. */
    max?: number;
    /** Quantity on the y axis, written above it. */
    label?: string;
    grid?: boolean;
  };
  /** In legend and tooltip order, main curve first; `z` sets the drawing order. */
  series: Series[];
  /** Shown before the legend entries, e.g. the parameter a family of curves is labelled by. */
  legendTitle?: string;
  rules?: Rule[];
  /** Pinned position on the x axis, drawn solid. */
  cursor?: number | null;
  cursorLabel?: string;
  /** Hovered position on the x axis, maybe from a linked chart, drawn dashed. */
  hover?: number | null;
  /** Tooltip heading for a hovered point. */
  heading(x: number, y: number): string;
  /** Tooltip rows in place of the series readout at (x, y). */
  rows?(x: number, y: number): TooltipRow[];
  /** Shown in place of the data, e.g. why there is none. */
  note?: string;
}

export interface ChartEvents {
  /** Pointer moved to x over the plot, or left it (null). */
  hover?(x: number | null): void;
  /** Pointer pressed or dragged at x. */
  pick?(x: number): void;
}

const MARGIN = { top: 14, right: 16, bottom: 36, left: 58 };
const DOT_RADIUS = 4;

/**
 * A canvas chart drawn from a declarative spec: linear or log axes, lines,
 * dots, bands, heatmaps and reference rules, with a crosshair tooltip that
 * lists every series at the pointer.
 */
export class Chart {
  private spec: ChartSpec | null = null;
  private plot: Rect = { x: 0, y: 0, w: 1, h: 1 };
  private xs = new Scale('linear', 0, 1, 0, 1);
  private ys = new Scale('linear', 0, 1, 1, 0);
  private dragging = false;
  private heatCache: { key: unknown; theme: Theme; image: HTMLCanvasElement } | null = null;

  constructor(
    readonly canvas: HTMLCanvasElement,
    private readonly events: ChartEvents = {},
  ) {
    canvas.addEventListener('pointermove', (e) => this.onMove(e));
    canvas.addEventListener('pointerleave', () => {
      if (this.dragging) return;
      hideTooltip();
      this.events.hover?.(null);
    });
    canvas.addEventListener('pointerdown', (e) => {
      const x = this.dataX(e, false);
      if (x === null || !this.events.pick) return;
      this.dragging = true;
      canvas.setPointerCapture(e.pointerId);
      this.events.pick(x);
    });
    const end = () => {
      this.dragging = false;
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
  }

  set(spec: ChartSpec): void {
    this.spec = spec;
  }

  draw(theme: Theme): void {
    const { ctx, width, height } = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, width, height);
    const spec = this.spec;
    if (!spec) return;
    const top = MARGIN.top + (spec.y.label ? 12 : 0);
    // Category ticks may run to a second line (e.g. a unit under a name).
    const bottom = MARGIN.bottom + (spec.x.ticks?.some((t) => t.label.includes('\n')) ? 12 : 0);
    const plot: Rect = {
      x: MARGIN.left,
      y: top,
      w: Math.max(10, width - MARGIN.left - MARGIN.right),
      h: Math.max(10, height - top - bottom),
    };
    this.plot = plot;
    this.xs = new Scale(spec.x.scale, spec.x.domain[0], spec.x.domain[1], plot.x, plot.x + plot.w);
    const [y0, y1] = this.yDomain(spec);
    this.ys = new Scale(spec.y.scale, y0, y1, plot.y + plot.h, plot.y);

    ctx.fillStyle = theme.surface;
    ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
    if (!spec.note) for (const s of spec.series) if (s.kind === 'heat') this.heat(ctx, s, theme);
    this.axes(ctx, spec, theme);

    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.w, plot.h);
    ctx.clip();
    if (!spec.note) {
      for (const s of spec.series) if (s.kind === 'band') this.band(ctx, s, theme);
      for (const r of spec.rules ?? []) this.rule(ctx, r, theme);
      const ordered = spec.series.map((s, i) => ({ s, i })).sort((a, b) => z(a.s) - z(b.s) || a.i - b.i);
      for (const { s } of ordered) if (s.kind === 'line') this.line(ctx, s, theme);
      for (const { s } of ordered) if (s.kind === 'dots') this.dots(ctx, s, theme);
    }
    this.marker(ctx, spec.cursor, [], theme, spec.cursorLabel);
    this.marker(ctx, spec.hover, [3, 3], theme);
    ctx.restore();

    if (spec.note) {
      ctx.save();
      ctx.font = labelFont(theme, 13);
      ctx.fillStyle = theme.muted;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(spec.note, plot.x + plot.w / 2, plot.y + plot.h / 2);
      ctx.restore();
    } else this.ruleLabels(ctx, spec.rules ?? [], theme);
  }

  // ------------------------------------------------------------ layout

  private yDomain(spec: ChartSpec): [number, number] {
    if (spec.y.domain) return spec.y.domain;
    const [a, b] = spec.x.domain;
    let lo = Infinity;
    let hi = -Infinity;
    let minPos = Infinity;
    for (const s of spec.series) {
      if (s.kind !== 'line' || s.noScale) continue;
      for (let i = 0; i < s.x.length; i++) {
        const xv = s.x[i];
        const yv = s.y[i];
        if (xv < a || xv > b || !Number.isFinite(yv)) continue;
        if (yv < lo) lo = yv;
        if (yv > hi) hi = yv;
        if (yv > 0 && yv < minPos) minPos = yv;
      }
    }
    if (spec.y.scale === 'log') {
      if (!(hi > 0)) return [0.1, 1];
      const top = hi * 1.6;
      const floor = top * 10 ** -(spec.y.decades ?? 6);
      return [Math.min(decadeFloor(Math.max(minPos, floor)), decadeFloor(top / 10)), top];
    }
    if (!(hi >= lo)) return [0, 1];
    lo = Math.min(0, lo);
    if (spec.y.max !== undefined) hi = Math.min(hi, spec.y.max);
    if (!(hi > lo)) hi = lo + 1;
    return [lo, hi + (hi - lo) * 0.06];
  }

  private axes(ctx: CanvasRenderingContext2D, spec: ChartSpec, theme: Theme): void {
    const { plot, xs, ys } = this;
    ctx.save();
    ctx.font = labelFont(theme);
    ctx.lineWidth = 1;

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const t of ys.ticks(Math.max(3, Math.round(plot.h / 50)))) {
      const py = crisp(ys.map(t.value));
      if (!(py >= plot.y - 1 && py <= plot.y + plot.h + 1)) continue;
      if (spec.y.grid !== false) {
        ctx.strokeStyle = theme.grid;
        ctx.beginPath();
        ctx.moveTo(plot.x, py);
        ctx.lineTo(plot.x + plot.w, py);
        ctx.stroke();
      }
      ctx.fillStyle = theme.muted;
      ctx.fillText(t.label, plot.x - 6, py);
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const t of spec.x.ticks ?? xs.ticks(Math.max(3, Math.round(plot.w / 90)))) {
      const px = crisp(xs.map(t.value));
      if (!(px >= plot.x - 1 && px <= plot.x + plot.w + 1)) continue;
      if (spec.x.grid !== false) {
        ctx.strokeStyle = theme.grid;
        ctx.beginPath();
        ctx.moveTo(px, plot.y);
        ctx.lineTo(px, plot.y + plot.h);
        ctx.stroke();
      }
      ctx.fillStyle = theme.muted;
      t.label.split('\n').forEach((part, k) => ctx.fillText(part, px, plot.y + plot.h + 6 + 13 * k));
    }

    ctx.strokeStyle = theme.axis;
    ctx.beginPath();
    ctx.moveTo(plot.x, crisp(plot.y + plot.h));
    ctx.lineTo(plot.x + plot.w, crisp(plot.y + plot.h));
    ctx.stroke();

    ctx.font = `italic ${labelFont(theme, 12)}`;
    ctx.fillStyle = theme.muted;
    ctx.textAlign = 'right';
    ctx.fillText(spec.x.label, plot.x + plot.w, plot.y + plot.h + 21);
    if (spec.y.label) {
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(spec.y.label, 8, plot.y - 6);
    }
    ctx.restore();
  }

  // ------------------------------------------------------------ marks

  private line(ctx: CanvasRenderingContext2D, s: Line, theme: Theme): void {
    const { xs, ys } = this;
    ctx.strokeStyle = theme.color(s.color);
    ctx.fillStyle = theme.color(s.color);
    ctx.lineWidth = s.width ?? 2;
    ctx.setLineDash(DASHES[s.dash ?? 'solid']);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    // Split at non-finite points into segments of screen points.
    const segments: [number, number][][] = [];
    let seg: [number, number][] = [];
    for (let i = 0; i < s.x.length; i++) {
      const X = xs.map(s.x[i]);
      const Y = ys.map(s.y[i]);
      if (!Number.isFinite(X) || !Number.isFinite(Y)) {
        if (seg.length) segments.push(seg);
        seg = [];
      } else seg.push([X, Math.max(-1e4, Math.min(1e4, Y))]);
    }
    if (seg.length) segments.push(seg);
    ctx.beginPath();
    for (const pts of segments) pts.forEach(([X, Y], i) => (i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y)));
    ctx.stroke();
    ctx.setLineDash([]);
    // Arrowheads: [from, tip] pairs at the end (and start) of each segment.
    const ends: [number, number, number, number][] = [];
    for (const pts of segments) {
      const n = pts.length;
      if (!s.arrows || n < 2) continue;
      ends.push([...pts[n - 2], ...pts[n - 1]]);
      if (s.arrows === 'both') ends.push([...pts[1], ...pts[0]]);
    }
    for (const [x0, y0, x1, y1] of ends) {
      const a = Math.atan2(y1 - y0, x1 - x0);
      const size = 7;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - size * Math.cos(a - 0.4), y1 - size * Math.sin(a - 0.4));
      ctx.lineTo(x1 - size * Math.cos(a + 0.4), y1 - size * Math.sin(a + 0.4));
      ctx.closePath();
      ctx.fill();
    }
  }

  /** Filled dots with a surface ring, so lines beneath stay visible. */
  private dots(ctx: CanvasRenderingContext2D, s: Dots, theme: Theme): void {
    const { xs, ys } = this;
    ctx.fillStyle = theme.color(s.color);
    ctx.strokeStyle = theme.surface;
    ctx.lineWidth = 1.5;
    for (let i = 0; i < s.x.length; i++) {
      const X = xs.map(s.x[i]);
      const Y = ys.map(s.y[i]);
      if (!Number.isFinite(X) || !Number.isFinite(Y)) continue;
      ctx.beginPath();
      ctx.arc(X, Y, DOT_RADIUS, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }

  private band(ctx: CanvasRenderingContext2D, s: Band, theme: Theme): void {
    const { plot, xs, ys } = this;
    const xa = xs.map(s.x[0]);
    const xb = xs.map(s.x[1]);
    const ya = s.y ? ys.map(s.y[0]) : plot.y + plot.h;
    const yb = s.y ? ys.map(s.y[1]) : plot.y;
    if (![xa, xb, ya, yb].every(Number.isFinite)) return;
    ctx.globalAlpha = s.alpha ?? 0.16;
    ctx.fillStyle = theme.color(s.color);
    ctx.fillRect(Math.min(xa, xb), Math.min(ya, yb), Math.abs(xb - xa), Math.abs(yb - ya));
    ctx.globalAlpha = 1;
  }

  /** The field as an image, one pixel per sample, stretched over the plot. */
  private heat(ctx: CanvasRenderingContext2D, s: Heat, theme: Theme): void {
    const nx = s.x.length;
    const ny = s.y.length;
    if (!this.heatCache || this.heatCache.key !== s.z || this.heatCache.theme !== theme) {
      const image = document.createElement('canvas');
      image.width = nx;
      image.height = ny;
      const ictx = image.getContext('2d')!;
      const data = ictx.createImageData(nx, ny);
      const stops = theme.heat.map(hexRgb);
      for (let i = 0; i < nx; i++) {
        for (let j = 0; j < ny; j++) {
          const u = Math.max(0, Math.min(1, s.z[i * ny + j] / s.zMax)) * (stops.length - 1);
          const k = Math.min(stops.length - 2, Math.floor(u));
          const f = u - k;
          const o = ((ny - 1 - j) * nx + i) * 4;
          for (let c = 0; c < 3; c++) data.data[o + c] = stops[k][c] * (1 - f) + stops[k + 1][c] * f;
          data.data[o + 3] = 255;
        }
      }
      ictx.putImageData(data, 0, 0);
      this.heatCache = { key: s.z, theme, image };
    }
    const { xs, ys } = this;
    const x0 = xs.map(s.x[0]);
    const x1 = xs.map(s.x[nx - 1]);
    const y0 = ys.map(s.y[0]);
    const y1 = ys.map(s.y[ny - 1]);
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.heatCache.image, x0, y1, x1 - x0, y0 - y1);
    ctx.restore();
  }

  private rule(ctx: CanvasRenderingContext2D, r: Rule, theme: Theme): void {
    const { plot } = this;
    ctx.strokeStyle = theme.color(r.color);
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';
    ctx.setLineDash(DASHES[r.dash ?? 'dashed']);
    ctx.beginPath();
    if (r.axis === 'x') {
      const px = crisp(this.xs.map(r.value));
      if (!Number.isFinite(px)) return;
      ctx.moveTo(px, plot.y);
      ctx.lineTo(px, plot.y + plot.h);
    } else {
      const py = crisp(this.ys.map(r.value));
      if (!Number.isFinite(py)) return;
      ctx.moveTo(plot.x, py);
      ctx.lineTo(plot.x + plot.w, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * Rule labels in secondary ink. Each x rule is labelled along the top, at
   * the first of right/left of its line, then the same a row lower, that stays
   * inside the plot and clear of other labels and rule lines. y rules are
   * labelled just above the line at either end.
   */
  private ruleLabels(ctx: CanvasRenderingContext2D, rules: Rule[], theme: Theme): void {
    const { plot } = this;
    ctx.save();
    ctx.font = labelFont(theme);
    ctx.fillStyle = theme.ink2;
    const xr = rules
      .filter((r) => r.axis === 'x' && r.label)
      .map((r) => ({ label: r.label!, px: this.xs.map(r.value) }))
      .filter((o) => o.px >= plot.x && o.px <= plot.x + plot.w)
      .sort((a, b) => a.px - b.px);
    const placed: { row: number; a: number; b: number }[] = [];
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    for (const { label, px } of xr) {
      const w = ctx.measureText(label).width;
      const spots = [0, 1].flatMap((row) => [
        { row, a: px + 5, b: px + 5 + w },
        { row, a: px - 5 - w, b: px - 5 },
      ]);
      const clear = (s: { row: number; a: number; b: number }) =>
        s.a >= plot.x &&
        s.b <= plot.x + plot.w &&
        !xr.some((o) => o.px !== px && o.px > s.a - 2 && o.px < s.b + 2) &&
        !placed.some((p) => p.row === s.row && p.a < s.b + 4 && s.a < p.b + 4);
      const spot = spots.find(clear) ?? spots[0];
      placed.push(spot);
      ctx.fillText(label, spot.a, plot.y + 4 + 15 * spot.row);
    }
    for (const r of rules) {
      if (r.axis !== 'y' || !r.label) continue;
      const py = this.ys.map(r.value);
      if (!(py >= plot.y && py <= plot.y + plot.h)) continue;
      const below = py < plot.y + 16;
      ctx.textAlign = 'right';
      ctx.textBaseline = below ? 'top' : 'bottom';
      ctx.fillText(r.label, plot.x + plot.w - 6, py + (below ? 4 : -4));
    }
    ctx.restore();
  }

  private marker(ctx: CanvasRenderingContext2D, x: number | null | undefined, dash: number[], theme: Theme, label?: string): void {
    if (x === null || x === undefined) return;
    const px = crisp(this.xs.map(x));
    if (!Number.isFinite(px)) return;
    ctx.strokeStyle = theme.ink2;
    ctx.globalAlpha = 0.75;
    ctx.lineWidth = 1;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(px, this.plot.y);
    ctx.lineTo(px, this.plot.y + this.plot.h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    if (label) {
      ctx.font = `italic ${labelFont(theme)}`;
      ctx.fillStyle = theme.ink2;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText(label, px + 4, this.plot.y + this.plot.h - 4);
    }
  }

  // ------------------------------------------------------------ pointer

  /** Data x under the pointer; null outside the plot unless `loose` (mid-drag), which clamps. */
  private dataX(e: PointerEvent, loose: boolean): number | null {
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const p = this.plot;
    if (!loose && (px < p.x - 6 || px > p.x + p.w + 6 || py < p.y - 6 || py > p.y + p.h + 6)) return null;
    return this.xs.invert(Math.max(p.x, Math.min(p.x + p.w, px)));
  }

  private dataY(e: PointerEvent): number {
    const rect = this.canvas.getBoundingClientRect();
    const p = this.plot;
    return this.ys.invert(Math.max(p.y, Math.min(p.y + p.h, e.clientY - rect.top)));
  }

  private onMove(e: PointerEvent): void {
    const x = this.dataX(e, this.dragging);
    if (x === null) {
      hideTooltip();
      this.events.hover?.(null);
      return;
    }
    if (this.dragging) this.events.pick?.(x);
    this.events.hover?.(x);
    this.tooltip(e, x, this.dataY(e));
  }

  private tooltip(e: PointerEvent, x: number, y: number): void {
    const spec = this.spec;
    if (!spec || spec.note) return hideTooltip();
    if (spec.rows) {
      const rows = spec.rows(x, y);
      return rows.length ? showTooltip(e.clientX, e.clientY, spec.heading(x, y), rows) : hideTooltip();
    }
    const rows: TooltipRow[] = [];
    const seen = new Set<string>();
    for (const s of spec.series) {
      if (seen.has(s.label) || s.kind === 'band' || (s.kind !== 'heat' && s.tooltip === false)) continue;
      const v = s.kind === 'line' ? interpolate(s.x, s.y, x) : s.kind === 'dots' ? this.nearestDot(s, x) : heatAt(s, x, y);
      if (v === null) continue;
      seen.add(s.label);
      const color: Role = s.kind === 'heat' ? 'seq4' : s.color;
      rows.push({ label: s.label, value: formatValue(v), color, key: s.kind === 'line' ? (s.dash ?? 'solid') : 'dot' });
    }
    showTooltip(e.clientX, e.clientY, spec.heading(x, y), rows);
  }

  /** y of the dot nearest x on screen, if one is within a few pixels. */
  private nearestDot(s: Dots, x: number): number | null {
    const px = this.xs.map(x);
    let best = -1;
    let dist = 12;
    for (let i = 0; i < s.x.length; i++) {
      const d = Math.abs(this.xs.map(s.x[i]) - px);
      if (d < dist && Number.isFinite(s.y[i])) {
        dist = d;
        best = i;
      }
    }
    return best < 0 ? null : s.y[best];
  }
}

function z(s: Series): number {
  return s.kind === 'line' || s.kind === 'dots' ? (s.z ?? 0) : -1;
}

function hexRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? [...h].map((c) => c + c).join('') : h;
  const n = parseInt(full.slice(0, 6), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function nearestIndex(xs: ArrayLike<number>, x: number): number {
  let best = 0;
  for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - x) < Math.abs(xs[best] - x)) best = i;
  return best;
}

function heatAt(s: Heat, x: number, y: number): number | null {
  const ny = s.y.length;
  if (x < s.x[0] || x > s.x[s.x.length - 1] || y < s.y[0] || y > s.y[ny - 1]) return null;
  return s.z[nearestIndex(s.x, x) * ny + nearestIndex(s.y, y)];
}

/** Linear interpolation in ascending samples; null outside them or between non-finite ones. */
function interpolate(xs: ArrayLike<number>, ys: ArrayLike<number>, x: number): number | null {
  const n = xs.length;
  if (n === 0 || x < xs[0] || x > xs[n - 1]) return null;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const a = ys[lo];
  const b = ys[hi];
  if (Number.isFinite(a) && Number.isFinite(b)) {
    const u = xs[hi] > xs[lo] ? (x - xs[lo]) / (xs[hi] - xs[lo]) : 0;
    return a + u * (b - a);
  }
  const near = x - xs[lo] < xs[hi] - x ? a : b;
  return Number.isFinite(near) || near === Infinity ? near : null;
}
