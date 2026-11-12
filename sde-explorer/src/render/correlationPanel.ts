import { axisName, bottomAxis, caption, chartPair, leftAxis } from './axes';
import { Linear, crisp, fitCanvas, type Rect } from './scale';
import { labelFont, type Theme } from './theme';

/** Values at lags τ_j = j Δτ, valid below `count`. */
export interface Lagged {
  values: Float64Array;
  count: number;
}

export interface CorrelationView {
  theme: Theme;
  T: number;
  dtau: number;
  /** Reference time s of the two-time covariance. */
  s: number;
  fp: Lagged | null;
  ens: Lagged | null;
  exact: Lagged | null;
  /** Stationary covariance G(τ), drawn on its own on the right. */
  stationary: Lagged | null;
  /** Also overlay G(τ) on the two-time covariance. */
  overlayStationary: boolean;
  /** The stationary state is the quasi-stationary one. */
  qsd: boolean;
  /** Slowest relaxation rate γ of the stationary state: ρ(τ) ∼ e^{−γτ}. */
  relax: number | null;
  /** Why the whole panel is empty. */
  message: string | null;
  /** Why the stationary chart is empty. */
  statMessage: string | null;
  hoverTau: number | null;
}

interface Sub {
  plot: Rect;
  t: Linear;
}

/** Decades shown by the log-scale correlation coefficient. */
const DECADES = 3;
const DASH = [6, 4];

/**
 * Two-time covariance ⟨X(s + τ), X(s)⟩ at the playback time s, beside the
 * stationary correlation coefficient ρ(τ) = G(τ)/G(0) on a log scale.
 */
export class CorrelationPanel {
  private subs: Sub[] = [];

  constructor(readonly canvas: HTMLCanvasElement) {}

  /** Lag under a canvas point, if it is inside either chart. */
  tauAt(px: number, py: number): number | null {
    for (const s of this.subs) {
      const p = s.plot;
      if (px >= p.x && px <= p.x + p.w && py >= p.y - 4 && py <= p.y + p.h + 4) return Math.min(s.t.d1, Math.max(0, s.t.invert(px)));
    }
    return null;
  }

  draw(v: CorrelationView): void {
    const { ctx, width, height } = fitCanvas(this.canvas);
    const { theme } = v;
    ctx.clearRect(0, 0, width, height);
    const [pl, pr] = chartPair(width, height);
    // Lags out to where exp(−γτ) leaves the log chart, so fast decays aren't squashed against τ = 0.
    const tauMax = v.relax && v.relax > 0 ? Math.min(v.T, (DECADES * Math.LN10) / v.relax) : v.T;
    const subL: Sub = { plot: pl, t: new Linear(0, tauMax, pl.x, pl.x + pl.w) };
    const subR: Sub = { plot: pr, t: new Linear(0, tauMax, pr.x, pr.x + pr.w) };
    this.subs = [subL, subR];

    // Left: scale to the smooth curves in view; the sampled one only if it is all there is.
    let lo = 0;
    let hi = 0;
    const inView = Math.floor(tauMax / v.dtau + 1e-9) + 1;
    const extend = (c: Lagged | null) => {
      if (!c) return;
      for (let i = 0; i < Math.min(c.count, inView); i++) {
        if (!Number.isFinite(c.values[i])) continue;
        lo = Math.min(lo, c.values[i]);
        hi = Math.max(hi, c.values[i]);
      }
    };
    extend(v.fp);
    extend(v.exact);
    if (v.overlayStationary) extend(v.stationary);
    if (hi === 0 && lo === 0) extend(v.ens);
    if (!(hi > lo)) hi = lo + 1;
    const pad = (hi - lo) * 0.06;
    const yl = new Linear(lo - pad, hi + pad, pl.y + pl.h, pl.y);
    const yr = new Linear(-DECADES, 0, pr.y + pr.h, pr.y);

    for (const s of this.subs) {
      ctx.fillStyle = theme.surface;
      ctx.fillRect(s.plot.x, s.plot.y, s.plot.w, s.plot.h);
      bottomAxis(ctx, s.plot, s.t, theme, { ticks: 5, grid: false });
      axisName(ctx, 'τ', s.plot.x + s.plot.w, s.plot.y + s.plot.h + 20, theme);
    }
    leftAxis(ctx, pl, yl, theme, { ticks: 4 });
    leftAxis(ctx, pr, yr, theme, { ticks: DECADES, format: (d) => String(10 ** d) });
    caption(ctx, pl, `⟨X(s + τ), X(s)⟩ at s = ${v.s.toFixed(2)}`, theme);
    caption(ctx, pr, `${v.qsd ? 'Quasi-stationary' : 'Stationary'} ρ(τ) = G(τ)/G(0)`, theme);

    if (v.message) {
      for (const s of this.subs) note(ctx, s.plot, v.message, theme);
      return;
    }

    const line = (sub: Sub, y: Linear, c: Lagged, f: (x: number) => number, color: string, lw: number, dash: number[] = [], alpha = 1) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.globalAlpha = alpha;
      ctx.setLineDash(dash);
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < c.count; i++) {
        const val = f(c.values[i]);
        if (!Number.isFinite(val)) {
          pen = false;
          continue;
        }
        const X = sub.t.map(i * v.dtau);
        const Y = y.map(val);
        if (pen) ctx.lineTo(X, Y);
        else ctx.moveTo(X, Y);
        pen = true;
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    };
    const id = (x: number) => x;

    ctx.save();
    clip(ctx, pl);
    if (v.overlayStationary && v.stationary) line(subL, yl, v.stationary, id, theme.stat, 2, [2, 4]);
    if (v.ens) line(subL, yl, v.ens, id, theme.ens, 1.25, [], 0.8);
    if (v.fp) line(subL, yl, v.fp, id, theme.fp, 2);
    if (v.exact) line(subL, yl, v.exact, id, theme.exact, 1.5, [6, 4]);
    this.hover(ctx, subL, v, theme);
    ctx.restore();

    if (!v.stationary) {
      note(ctx, pr, v.statMessage ?? '', theme);
      return;
    }
    ctx.save();
    clip(ctx, pr);
    const relax = v.relax !== null && v.relax > 0 ? v.relax : null;
    if (relax) this.relaxLine(ctx, subR, yr, relax, tauMax, theme);
    const g0 = v.stationary.values[0];
    line(subR, yr, v.stationary, (g) => Math.log10(g / g0), theme.stat, 2);
    if (relax) this.relaxKey(ctx, pr, relax, v.qsd, theme);
    this.hover(ctx, subR, v, theme);
    ctx.restore();
  }

  /** The single-mode decay e^{−γτ}, a straight line on the log scale. */
  private relaxLine(ctx: CanvasRenderingContext2D, s: Sub, y: Linear, gamma: number, T: number, theme: Theme): void {
    ctx.strokeStyle = theme.muted;
    ctx.lineWidth = 1.25;
    ctx.setLineDash(DASH);
    ctx.beginPath();
    ctx.moveTo(s.t.map(0), y.map(0));
    ctx.lineTo(s.t.map(T), y.map(-gamma * Math.LOG10E * T));
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /** Key for the decay line in the top-right corner, which a decaying curve leaves empty. */
  private relaxKey(ctx: CanvasRenderingContext2D, p: Rect, gamma: number, qsd: boolean, theme: Theme): void {
    const text = `exp(−γτ), γ = ${qsd ? 'λ₁ − λ₀' : 'λ₁'} = ${gamma.toPrecision(3)}`;
    ctx.font = labelFont(theme);
    const right = p.x + p.w - 8;
    const y = crisp(p.y + 12);
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = theme.muted;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, right, y);
    ctx.strokeStyle = theme.muted;
    ctx.lineWidth = 1.25;
    ctx.setLineDash(DASH);
    ctx.beginPath();
    ctx.moveTo(right - tw - 30, y);
    ctx.lineTo(right - tw - 8, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  private hover(ctx: CanvasRenderingContext2D, s: Sub, v: CorrelationView, theme: Theme): void {
    if (v.hoverTau === null) return;
    ctx.strokeStyle = theme.ink2;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    const hx = crisp(s.t.map(v.hoverTau));
    ctx.beginPath();
    ctx.moveTo(hx, s.plot.y);
    ctx.lineTo(hx, s.plot.y + s.plot.h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }
}

function clip(ctx: CanvasRenderingContext2D, p: Rect): void {
  ctx.beginPath();
  ctx.rect(p.x, p.y, p.w, p.h);
  ctx.clip();
}

/** A short explanation centred in an empty chart, wrapped to its width. */
function note(ctx: CanvasRenderingContext2D, p: Rect, text: string, theme: Theme): void {
  ctx.save();
  ctx.font = labelFont(theme, 12);
  ctx.fillStyle = theme.muted;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const lines: string[] = [];
  let current = '';
  for (const word of text.split(' ')) {
    const next = current ? `${current} ${word}` : word;
    if (current && ctx.measureText(next).width > p.w - 24) {
      lines.push(current);
      current = word;
    } else current = next;
  }
  if (current) lines.push(current);
  const lh = 16;
  const y0 = p.y + p.h / 2 - ((lines.length - 1) * lh) / 2;
  lines.forEach((l, i) => ctx.fillText(l, p.x + p.w / 2, y0 + i * lh));
  ctx.restore();
}
