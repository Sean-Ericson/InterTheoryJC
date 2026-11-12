import { crisp, formatTick, niceTicks, type Linear, type Rect } from './scale';
import { labelFont, type Theme } from './theme';

/** Top/bottom margins shared by the paths and density panels so their x axes line up. */
export const ROW_MARGIN = { top: 22, bottom: 34 };

/** Two charts side by side across a full-width row, the left one wider. */
export function chartPair(width: number, height: number): [Rect, Rect] {
  const top = 22;
  const bottom = 34;
  const gutter = 64;
  const left = 48;
  const wl = Math.max(10, (width - left - gutter - 10) * 0.58);
  const wr = Math.max(10, width - left - gutter - 10 - wl);
  const h = Math.max(10, height - top - bottom);
  return [
    { x: left, y: top, w: wl, h },
    { x: left + wl + gutter, y: top, w: wr, h },
  ];
}

interface AxisOpts {
  ticks?: number;
  /** Draw gridlines across the plot as well as tick labels. */
  grid?: boolean;
  labels?: boolean;
  /** Tick label text, in place of the default decimal formatting. */
  format?: (v: number) => string;
}

/** Horizontal gridlines with tick labels on the left edge of the plot. */
export function leftAxis(ctx: CanvasRenderingContext2D, plot: Rect, y: Linear, theme: Theme, opts: AxisOpts = {}): void {
  const lo = Math.min(y.d0, y.d1);
  const hi = Math.max(y.d0, y.d1);
  const { ticks, step } = niceTicks(lo, hi, opts.ticks ?? 5);
  ctx.save();
  ctx.font = labelFont(theme);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 1;
  for (const v of ticks) {
    const py = crisp(y.map(v));
    if (opts.grid !== false) {
      ctx.strokeStyle = theme.grid;
      ctx.beginPath();
      ctx.moveTo(plot.x, py);
      ctx.lineTo(plot.x + plot.w, py);
      ctx.stroke();
    }
    if (opts.labels !== false) {
      ctx.fillStyle = theme.muted;
      ctx.fillText(opts.format ? opts.format(v) : formatTick(v, step), plot.x - 6, py);
    }
  }
  ctx.restore();
}

/** Vertical gridlines with tick labels under the plot, plus the baseline. */
export function bottomAxis(ctx: CanvasRenderingContext2D, plot: Rect, x: Linear, theme: Theme, opts: AxisOpts = {}): void {
  const { ticks, step } = niceTicks(x.d0, x.d1, opts.ticks ?? 6);
  ctx.save();
  ctx.font = labelFont(theme);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.lineWidth = 1;
  for (const v of ticks) {
    const px = crisp(x.map(v));
    if (opts.grid !== false) {
      ctx.strokeStyle = theme.grid;
      ctx.beginPath();
      ctx.moveTo(px, plot.y);
      ctx.lineTo(px, plot.y + plot.h);
      ctx.stroke();
    }
    if (opts.labels !== false) {
      ctx.fillStyle = theme.muted;
      ctx.fillText(formatTick(v, step), px, plot.y + plot.h + 6);
    }
  }
  ctx.strokeStyle = theme.axis;
  ctx.beginPath();
  ctx.moveTo(plot.x, crisp(plot.y + plot.h));
  ctx.lineTo(plot.x + plot.w, crisp(plot.y + plot.h));
  ctx.stroke();
  ctx.restore();
}

/** Small caption in the top-left corner above a plot. */
export function caption(ctx: CanvasRenderingContext2D, plot: Rect, text: string, theme: Theme): void {
  ctx.save();
  ctx.font = labelFont(theme, 12);
  ctx.fillStyle = theme.ink2;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText(text, plot.x, plot.y - 6);
  ctx.restore();
}

/** Axis name at the end of an axis, in muted ink. */
export function axisName(ctx: CanvasRenderingContext2D, text: string, px: number, py: number, theme: Theme, align: CanvasTextAlign = 'right'): void {
  ctx.save();
  ctx.font = `italic ${labelFont(theme, 12)}`;
  ctx.fillStyle = theme.muted;
  ctx.textAlign = align;
  ctx.textBaseline = 'top';
  ctx.fillText(text, px, py);
  ctx.restore();
}
