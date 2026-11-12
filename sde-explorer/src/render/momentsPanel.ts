import type { Moments, Recorder } from '../recorder';
import { axisName, bottomAxis, caption, chartPair, leftAxis } from './axes';
import { Linear, crisp, fitCanvas, type Rect } from './scale';
import { labelFont, type Theme } from './theme';

export interface MomentsView {
  theme: Theme;
  rec: Recorder | null;
  T: number;
  k: number;
  t: number;
  /** What the right-hand chart shows: surviving probability, exit rate or variance. */
  right: 'mass' | 'rate' | 'variance';
  showFp: boolean;
  showEns: boolean;
  showExact: boolean;
  /** Quasi-stationary decay rate: the exit rate's limit, drawn as a reference line. */
  lambda0: number | null;
  hoverT: number | null;
}

interface Sub {
  plot: Rect;
  t: Linear;
  y: Linear;
}

const RIGHT_CAPTIONS: Record<MomentsView['right'], string> = {
  mass: 'Probability not yet absorbed',
  rate: 'Exit rate κ = −Ṡ/S',
  variance: 'Variance',
};

/** Mean ± s.d. over time, beside surviving probability or exit rate (absorbing walls) or variance. */
export class MomentsPanel {
  subs: Sub[] = [];

  constructor(readonly canvas: HTMLCanvasElement) {}

  /**
   * Time under a canvas point, if it is inside either chart. With `loose`
   * (mid-drag) the vertical position is ignored and the time is clamped.
   */
  timeAt(px: number, py: number, loose = false): number | null {
    if (this.subs.length < 2) return null;
    const clampT = (s: Sub) => Math.min(s.t.d1, Math.max(0, s.t.invert(px)));
    if (loose) {
      const [l, r] = this.subs;
      return clampT(px < (l.plot.x + l.plot.w + r.plot.x) / 2 ? l : r);
    }
    for (const s of this.subs) {
      const p = s.plot;
      if (px >= p.x && px <= p.x + p.w && py >= p.y - 4 && py <= p.y + p.h + 4) return clampT(s);
    }
    return null;
  }

  /** Dotted reference line at λ₀, styled like the quasi-stationary density it belongs to. */
  private lambdaLine(ctx: CanvasRenderingContext2D, s: Sub, lambda0: number, theme: Theme): void {
    const y = crisp(s.y.map(lambda0));
    ctx.save();
    ctx.strokeStyle = theme.stat;
    ctx.lineWidth = 2;
    ctx.setLineDash([2, 4]);
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(s.plot.x, y);
    ctx.lineTo(s.plot.x + s.plot.w, y);
    ctx.stroke();
    ctx.font = labelFont(theme);
    ctx.fillStyle = theme.stat;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText('λ₀', s.plot.x + s.plot.w - 4, y - 3);
    ctx.restore();
  }

  draw(v: MomentsView): void {
    const { ctx, width, height } = fitCanvas(this.canvas);
    const { theme, rec } = v;
    ctx.clearRect(0, 0, width, height);
    const [pl, pr] = chartPair(width, height);

    const count = rec?.count ?? 0;
    const k = Math.min(v.k, count - 1);

    // Left: mean ± s.d.
    let lo = Infinity;
    let hi = -Infinity;
    const extend = (m: Moments | null) => {
      if (!m) return;
      for (let i = 0; i < count; i++) {
        const sd = Math.sqrt(Math.max(0, m.variance[i]));
        if (!Number.isFinite(m.mean[i])) continue;
        lo = Math.min(lo, m.mean[i] - (Number.isFinite(sd) ? sd : 0));
        hi = Math.max(hi, m.mean[i] + (Number.isFinite(sd) ? sd : 0));
      }
    };
    if (rec) {
      if (v.showFp) extend(rec.fp);
      if (v.showEns) extend(rec.ens);
      if (v.showExact) extend(rec.exact);
    }
    if (!(hi > lo)) {
      lo = Number.isFinite(lo) ? lo - 1 : 0;
      hi = Number.isFinite(hi) ? hi + 1 : 1;
    }
    const pad = (hi - lo) * 0.06;
    const subL: Sub = { plot: pl, t: new Linear(0, v.T, pl.x, pl.x + pl.w), y: new Linear(lo - pad, hi + pad, pl.y + pl.h, pl.y) };

    // Right: surviving probability, exit rate or variance.
    let rlo = 0;
    let rhi = 1.02;
    if (v.right !== 'mass') {
      const key = v.right;
      rhi = 0;
      const ext = (m: Moments) => {
        for (let i = 0; i < count; i++) if (Number.isFinite(m[key][i])) rhi = Math.max(rhi, m[key][i]);
      };
      if (rec) {
        if (v.showFp) ext(rec.fp);
        if (v.showExact) ext(rec.exact);
        // The ensemble's exit rate is noisy: let it set the scale only when it is all there is.
        if (v.showEns && (key === 'variance' || rhi === 0)) ext(rec.ens);
      }
      if (key === 'rate' && v.lambda0 !== null) {
        // Keep the approach to λ₀ readable even after a large early spike.
        rhi = Math.min(Math.max(rhi, 1.25 * v.lambda0), 3 * v.lambda0);
      }
      rhi = rhi > 0 ? rhi * 1.08 : 1;
    }
    const subR: Sub = { plot: pr, t: new Linear(0, v.T, pr.x, pr.x + pr.w), y: new Linear(rlo, rhi, pr.y + pr.h, pr.y) };
    this.subs = [subL, subR];

    for (const s of this.subs) {
      ctx.fillStyle = theme.surface;
      ctx.fillRect(s.plot.x, s.plot.y, s.plot.w, s.plot.h);
      leftAxis(ctx, s.plot, s.y, theme, { ticks: 4 });
      bottomAxis(ctx, s.plot, s.t, theme, { ticks: 5, grid: false });
      axisName(ctx, 't', s.plot.x + s.plot.w, s.plot.y + s.plot.h + 20, theme);
    }
    caption(ctx, pl, 'Mean ± 1 s.d.', theme);
    caption(ctx, pr, RIGHT_CAPTIONS[v.right], theme);

    if (!rec || k < 0) return;
    const times = rec.times;

    const line = (sub: Sub, vals: (i: number) => number, color: string, lw: number, dash: number[] = [], alpha = 1) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.globalAlpha = alpha;
      ctx.setLineDash(dash);
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i <= k; i++) {
        const y = vals(i);
        if (!Number.isFinite(y)) {
          pen = false;
          continue;
        }
        const X = sub.t.map(times[i]);
        const Y = sub.y.map(y);
        if (pen) ctx.lineTo(X, Y);
        else ctx.moveTo(X, Y);
        pen = true;
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    };
    const sd = (m: Moments, i: number) => Math.sqrt(Math.max(0, m.variance[i]));

    for (const s of this.subs) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(s.plot.x, s.plot.y, s.plot.w, s.plot.h);
      ctx.clip();
      if (s === subL) {
        if (v.showFp) {
          // ±1 s.d. band as a 12% wash of the series hue.
          ctx.fillStyle = theme.fp;
          ctx.globalAlpha = 0.12;
          ctx.beginPath();
          for (let i = 0; i <= k; i++) ctx.lineTo(s.t.map(times[i]), s.y.map(rec.fp.mean[i] + sd(rec.fp, i)));
          for (let i = k; i >= 0; i--) ctx.lineTo(s.t.map(times[i]), s.y.map(rec.fp.mean[i] - sd(rec.fp, i)));
          ctx.closePath();
          ctx.fill();
          ctx.globalAlpha = 1;
        }
        if (v.showEns) {
          line(s, (i) => rec.ens.mean[i] + sd(rec.ens, i), theme.ens, 1, [], 0.6);
          line(s, (i) => rec.ens.mean[i] - sd(rec.ens, i), theme.ens, 1, [], 0.6);
          line(s, (i) => rec.ens.mean[i], theme.ens, 2);
        }
        if (v.showFp) line(s, (i) => rec.fp.mean[i], theme.fp, 2);
        if (v.showExact) {
          line(s, (i) => rec.exact.mean[i], theme.exact, 1.5, [6, 4]);
          line(s, (i) => rec.exact.mean[i] + sd(rec.exact, i), theme.exact, 1, [6, 4], 0.7);
          line(s, (i) => rec.exact.mean[i] - sd(rec.exact, i), theme.exact, 1, [6, 4], 0.7);
        }
      } else {
        const key = v.right;
        if (key === 'rate' && v.lambda0 !== null) this.lambdaLine(ctx, s, v.lambda0, theme);
        if (v.showEns) line(s, (i) => rec.ens[key][i], theme.ens, key === 'rate' ? 1.25 : 2, [], key === 'rate' ? 0.8 : 1);
        if (v.showFp) line(s, (i) => rec.fp[key][i], theme.fp, 2);
        if (v.showExact) line(s, (i) => rec.exact[key][i], theme.exact, 1.5, [6, 4]);
      }

      // Playback cursor, and the hover crosshair.
      ctx.strokeStyle = theme.ink2;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.7;
      const cx = crisp(s.t.map(v.t));
      ctx.beginPath();
      ctx.moveTo(cx, s.plot.y);
      ctx.lineTo(cx, s.plot.y + s.plot.h);
      ctx.stroke();
      if (v.hoverT !== null) {
        ctx.setLineDash([3, 3]);
        const hx = crisp(s.t.map(v.hoverT));
        ctx.beginPath();
        ctx.moveTo(hx, s.plot.y);
        ctx.lineTo(hx, s.plot.y + s.plot.h);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.globalAlpha = 1;
      ctx.restore();
    }
  }
}
