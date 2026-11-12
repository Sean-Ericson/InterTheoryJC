export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Linear map from data [d0, d1] to pixels [r0, r1]. */
export class Linear {
  constructor(
    public d0: number,
    public d1: number,
    public r0: number,
    public r1: number,
  ) {}

  map(v: number): number {
    return this.r0 + ((v - this.d0) / (this.d1 - this.d0)) * (this.r1 - this.r0);
  }

  invert(px: number): number {
    return this.d0 + ((px - this.r0) / (this.r1 - this.r0)) * (this.d1 - this.d0);
  }
}

/** Round tick values covering [lo, hi], roughly `count` of them. */
export function niceTicks(lo: number, hi: number, count: number): { ticks: number[]; step: number } {
  if (!(hi > lo)) return { ticks: [lo], step: 1 };
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return { ticks, step };
}

export function formatTick(v: number, step: number): string {
  const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return v.toFixed(Math.min(digits, 6));
}

/** Format a value for readouts: ~4 significant digits, no exponent unless needed. */
export function formatValue(v: number): string {
  if (!Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toPrecision(4).replace(/\.?0+$/, '');
}

/** Size a canvas's backing store to its CSS box at device resolution. */
export function fitCanvas(canvas: HTMLCanvasElement): { ctx: CanvasRenderingContext2D; width: number; height: number } {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const bw = Math.max(1, Math.round(width * dpr));
  const bh = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

/** Snap a coordinate to the pixel centre so 1px hairlines stay crisp. */
export function crisp(v: number): number {
  return Math.round(v) + 0.5;
}
