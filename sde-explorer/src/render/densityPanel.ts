import { ROW_MARGIN, axisName, bottomAxis, caption, leftAxis } from './axes';
import { Linear, crisp, fitCanvas, type Rect } from './scale';
import type { Theme } from './theme';

export interface Histogram {
  lo: number;
  width: number;
  /** Normalised so the bars' total area is the surviving fraction (1 if conditioned). */
  values: Float64Array;
}

/**
 * Histogram of the surviving paths, normalised by the full ensemble size or,
 * conditioned on survival, by the number of survivors.
 */
export function histogram(x: Float32Array, alive: Uint8Array, domain: [number, number], bins: number, conditioned = false): Histogram {
  const [lo, hi] = domain;
  const width = (hi - lo) / bins;
  const values = new Float64Array(bins);
  let survivors = 0;
  for (let i = 0; i < x.length; i++) {
    if (!alive[i]) continue;
    survivors++;
    const b = Math.floor((x[i] - lo) / width);
    if (b >= 0 && b < bins) values[b]++;
    else if (x[i] === hi) values[bins - 1]++;
  }
  const total = conditioned ? survivors : x.length;
  if (total > 0) for (let b = 0; b < bins; b++) values[b] /= total * width;
  return { lo, width, values };
}

export interface DensityView {
  theme: Theme;
  domain: [number, number];
  t: number;
  /** Name of the plotted density: p, or q = p/S when conditioned on survival. */
  name: 'p' | 'q';
  xc: Float64Array;
  fp: Float64Array | null;
  hist: Histogram | null;
  /** Exact density at the cell centres. */
  exact: Float64Array | null;
  stationary: Float64Array | null;
  /** Hovered x, for the crosshair. */
  hoverX: number | null;
}

/**
 * The density at the playback time, turned sideways so its x axis lines up
 * with the paths panel on its left.
 */
export class DensityPanel {
  plot: Rect = { x: 0, y: 0, w: 1, h: 1 };
  xScale = new Linear(0, 1, 0, 1);
  pScale = new Linear(0, 1, 0, 1);

  constructor(readonly canvas: HTMLCanvasElement) {}

  draw(v: DensityView): void {
    const { ctx, width, height } = fitCanvas(this.canvas);
    const { theme } = v;
    ctx.clearRect(0, 0, width, height);
    const plot: Rect = { x: 8, y: ROW_MARGIN.top, w: Math.max(10, width - 8 - 14), h: Math.max(10, height - ROW_MARGIN.top - ROW_MARGIN.bottom) };
    this.plot = plot;
    const [lo, hi] = v.domain;
    const xy = (this.xScale = new Linear(lo, hi, plot.y + plot.h, plot.y));

    // Scale to the smooth curves; histogram noise shouldn't make the axis jitter.
    let pmax = 0;
    const consider = (q: number) => {
      if (Number.isFinite(q) && q > pmax) pmax = q;
    };
    if (v.fp) for (const q of v.fp) consider(q);
    if (v.stationary) for (const q of v.stationary) consider(q);
    if (v.exact) for (const q of v.exact) consider(q);
    if (pmax === 0 && v.hist) for (const q of v.hist.values) consider(q);
    if (pmax === 0) pmax = 1;
    const px = (this.pScale = new Linear(0, pmax * 1.08, plot.x, plot.x + plot.w));

    ctx.fillStyle = theme.surface;
    ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
    leftAxis(ctx, plot, xy, theme, { labels: false });
    bottomAxis(ctx, plot, px, theme, { ticks: 3 });
    caption(ctx, plot, `${v.name}(x, t = ${v.t.toFixed(2)})`, theme);
    axisName(ctx, v.name, plot.x + plot.w, plot.y + plot.h + 20, theme);

    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.w, plot.h);
    ctx.clip();

    if (v.hist) {
      // Horizontal bars growing from the p = 0 baseline, 1px surface gap between.
      const { values, width: bw } = v.hist;
      ctx.fillStyle = theme.ens;
      ctx.globalAlpha = 0.35;
      for (let b = 0; b < values.length; b++) {
        if (values[b] <= 0) continue;
        const y0 = xy.map(v.hist.lo + (b + 1) * bw);
        const y1 = xy.map(v.hist.lo + b * bw);
        const gap = y1 - y0 > 4 ? 1 : 0;
        ctx.fillRect(plot.x, y0 + gap / 2, px.map(values[b]) - plot.x, y1 - y0 - gap);
      }
      ctx.globalAlpha = 1;
    }

    const curve = (val: (i: number) => number, color: string, dash: number[], lw: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.setLineDash(dash);
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      for (let i = 0; i < v.xc.length; i++) {
        const X = px.map(Math.max(0, val(i)));
        const Y = xy.map(v.xc[i]);
        if (i === 0) ctx.moveTo(X, Y);
        else ctx.lineTo(X, Y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };

    if (v.stationary) {
      const ps = v.stationary;
      curve((i) => ps[i], theme.stat, [2, 4], 2);
    }
    if (v.fp) {
      const fp = v.fp;
      curve((i) => fp[i], theme.fp, [], 2);
    }
    if (v.exact) {
      const ex = v.exact;
      curve((i) => ex[i], theme.exact, [6, 4], 1.5);
    }

    if (v.hoverX !== null) {
      ctx.strokeStyle = theme.ink2;
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1;
      const y = crisp(xy.map(v.hoverX));
      ctx.beginPath();
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.w, y);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }
}
