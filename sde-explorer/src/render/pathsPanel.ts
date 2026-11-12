import type { Recorder } from '../recorder';
import { ROW_MARGIN, axisName, bottomAxis, caption, leftAxis } from './axes';
import { Linear, crisp, fitCanvas, type Rect } from './scale';
import type { Theme } from './theme';

export type HeatScale = 'slice' | 'log';

/**
 * p(x,t), or q = p/S when conditioned on survival, as an image: one column per
 * snapshot, one row per grid cell (top row = right end of the domain).
 * Columns are coloured incrementally as the run records them.
 */
export class Heatmap {
  readonly canvas = document.createElement('canvas');
  private img: ImageData | null = null;
  private filled = 0;
  private gmax = 0;
  private mode: HeatScale = 'slice';
  private lut: Uint8ClampedArray | null = null;
  private conditioned = false;

  reset(snaps: number, nGrid: number): void {
    this.canvas.width = snaps;
    this.canvas.height = nGrid;
    this.img = new ImageData(snaps, nGrid);
    this.filled = 0;
    this.gmax = 0;
  }

  /** Colour any newly recorded columns (or everything, if the scale changed). */
  update(rec: Recorder, mode: HeatScale, lut: Uint8ClampedArray, conditioned: boolean): void {
    if (!this.img) return;
    if (conditioned !== this.conditioned) {
      this.conditioned = conditioned;
      this.gmax = 0;
      this.filled = 0;
    }
    let from = this.filled;
    if (mode !== this.mode || lut !== this.lut) {
      this.mode = mode;
      this.lut = lut;
      from = 0;
    }
    // Dividing by S only matters on the global log scale; slices are normalised anyway.
    const scale = (k: number) => {
      const s = conditioned ? rec.fp.mass[k] : 1;
      return s > 0 ? 1 / s : 0;
    };
    // Track the global maximum in either mode, so it is ready on a switch to log.
    let gmax = this.gmax;
    for (let k = this.filled; k < rec.count; k++) {
      const col = rec.densityAt(k);
      const sk = scale(k);
      for (let i = 0; i < col.length; i++) if (col[i] * sk > gmax) gmax = col[i] * sk;
    }
    if (gmax > this.gmax) {
      this.gmax = gmax;
      if (mode === 'log') from = 0;
    }
    if (from >= rec.count) return;
    const { data, width } = this.img;
    const n = rec.nGrid;
    const logRange = 4; // decades shown in log mode
    for (let k = from; k < rec.count; k++) {
      const col = rec.densityAt(k);
      const sk = scale(k);
      let cmax = 0;
      if (mode === 'slice') for (let i = 0; i < n; i++) if (col[i] > cmax) cmax = col[i];
      for (let i = 0; i < n; i++) {
        const p = Math.max(mode === 'log' ? col[i] * sk : col[i], 0);
        let u: number;
        if (mode === 'slice') u = cmax > 0 ? p / cmax : 0;
        else u = p > 0 && this.gmax > 0 ? 1 + Math.log10(p / this.gmax) / logRange : 0;
        const c = Math.round(Math.min(1, Math.max(0, u)) * 255) * 4;
        const o = ((n - 1 - i) * width + k) * 4;
        data[o] = lut[c];
        data[o + 1] = lut[c + 1];
        data[o + 2] = lut[c + 2];
        data[o + 3] = 255;
      }
    }
    this.canvas.getContext('2d')!.putImageData(this.img, 0, 0, from, 0, rec.count - from, n);
    this.filled = rec.count;
  }
}

export interface PathsView {
  theme: Theme;
  rec: Recorder | null;
  T: number;
  domain: [number, number];
  periodic: boolean;
  /** Playback time and the matching snapshot / trace step. */
  t: number;
  k: number;
  step: number;
  showHeat: boolean;
  /** Name of the density in the heatmap: p, or q = p/S when conditioned on survival. */
  name: 'p' | 'q';
  showPaths: boolean;
  nShow: number;
  heat: Heatmap;
}

/** Sample paths over the p(x,t) heatmap; time runs left→right, x bottom→top. */
export class PathsPanel {
  plot: Rect = { x: 0, y: 0, w: 1, h: 1 };
  tScale = new Linear(0, 1, 0, 1);
  xScale = new Linear(0, 1, 0, 1);

  constructor(readonly canvas: HTMLCanvasElement) {}

  draw(v: PathsView): void {
    const { ctx, width, height } = fitCanvas(this.canvas);
    const { theme } = v;
    ctx.clearRect(0, 0, width, height);
    const plot: Rect = { x: 48, y: ROW_MARGIN.top, w: Math.max(10, width - 48 - 10), h: Math.max(10, height - ROW_MARGIN.top - ROW_MARGIN.bottom) };
    this.plot = plot;
    const [lo, hi] = v.domain;
    const tx = (this.tScale = new Linear(0, v.T, plot.x, plot.x + plot.w));
    const xy = (this.xScale = new Linear(lo, hi, plot.y + plot.h, plot.y));

    ctx.fillStyle = theme.surface;
    ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
    leftAxis(ctx, plot, xy, theme);
    bottomAxis(ctx, plot, tx, theme);
    caption(ctx, plot, v.showHeat ? `Sample paths over ${v.name}(x, t)` : 'Sample paths', theme);
    axisName(ctx, 't', plot.x + plot.w, plot.y + plot.h + 20, theme);
    axisName(ctx, 'x', plot.x - 34, plot.y - 18, theme, 'left');

    const rec = v.rec;
    if (!rec || rec.count === 0) return;

    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.w, plot.h);
    ctx.clip();

    if (v.showHeat) {
      const half = v.T / rec.W / 2;
      const x0 = tx.map(-half);
      const x1 = tx.map(rec.times[v.k] + half);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(v.heat.canvas, 0, 0, v.k + 1, rec.nGrid, x0, plot.y, x1 - x0, plot.h);
    }

    if (v.showPaths) this.drawPaths(ctx, v, rec, tx, xy);

    // Playback cursor.
    ctx.strokeStyle = theme.ink2;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1;
    const cx = crisp(tx.map(v.t));
    ctx.beginPath();
    ctx.moveTo(cx, plot.y);
    ctx.lineTo(cx, plot.y + plot.h);
    ctx.stroke();
    ctx.restore();
  }

  private drawPaths(ctx: CanvasRenderingContext2D, v: PathsView, rec: Recorder, tx: Linear, xy: Linear): void {
    const { theme } = v;
    const nShow = Math.min(v.nShow, rec.nShow);
    const dtStep = v.T / rec.nSteps;
    const L = v.domain[1] - v.domain[0];
    const stride = Math.max(1, Math.floor(rec.nSteps / (this.plot.w * 1.5)));
    const last = Math.min(v.step, rec.traceCount - 1);
    if (last < 0) return;

    ctx.lineWidth = 1;
    ctx.lineJoin = 'round';
    ctx.strokeStyle = theme.ens;
    // Fade as more paths overlap so the heatmap underneath stays readable.
    ctx.globalAlpha = Math.min(0.85, 2.6 / Math.sqrt(Math.max(1, nShow)));
    ctx.beginPath();
    for (let j = 0; j < nShow; j++) {
      const tr = rec.trace(j);
      const exit = rec.traceExit[j];
      const end = exit >= 0 ? Math.min(last, exit) : last;
      ctx.moveTo(tx.map(0), xy.map(tr[0]));
      let prev = tr[0];
      let s = 0;
      while (s < end) {
        s = Math.min(s + stride, end);
        const xv = tr[s];
        // Wrapping round a periodic domain: lift the pen rather than draw across.
        if (v.periodic && Math.abs(xv - prev) > L / 2) ctx.moveTo(tx.map(s * dtStep), xy.map(xv));
        else ctx.lineTo(tx.map(s * dtStep), xy.map(xv));
        prev = xv;
      }
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Path heads (or absorption points), ringed in the surface colour.
    for (let j = 0; j < nShow; j++) {
      const tr = rec.trace(j);
      const exit = rec.traceExit[j];
      const absorbed = exit >= 0 && exit <= last;
      const s = absorbed ? exit : last;
      const px = tx.map(s * dtStep);
      const py = xy.map(tr[s]);
      ctx.beginPath();
      ctx.arc(px, py, absorbed ? 3.5 : 2.5, 0, 2 * Math.PI);
      ctx.fillStyle = absorbed ? theme.surface : theme.ens;
      ctx.strokeStyle = absorbed ? theme.ens : theme.surface;
      ctx.lineWidth = absorbed ? 1.5 : 1;
      ctx.fill();
      ctx.stroke();
    }
  }
}
