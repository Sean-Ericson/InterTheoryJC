export type ScaleKind = 'linear' | 'log';

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Tick {
  value: number;
  label: string;
}

/** Map from data [d0, d1] to pixels [r0, r1], linear or logarithmic. */
export class Scale {
  private readonly f0: number;
  private readonly f1: number;

  constructor(
    readonly kind: ScaleKind,
    readonly d0: number,
    readonly d1: number,
    readonly r0: number,
    readonly r1: number,
  ) {
    this.f0 = this.fwd(d0);
    this.f1 = this.fwd(d1);
  }

  private fwd(v: number): number {
    return this.kind === 'log' ? Math.log10(v) : v;
  }

  /** Pixel for a data value; NaN for v ≤ 0 on a log scale. */
  map(v: number): number {
    if (this.kind === 'log' && !(v > 0)) return NaN;
    return this.r0 + ((this.fwd(v) - this.f0) / (this.f1 - this.f0)) * (this.r1 - this.r0);
  }

  invert(px: number): number {
    const u = this.f0 + ((px - this.r0) / (this.r1 - this.r0)) * (this.f1 - this.f0);
    return this.kind === 'log' ? 10 ** u : u;
  }

  /** Position in [0, 1] along the axis, and back: for sliders that follow the axis. */
  fraction(v: number): number {
    return (this.fwd(v) - this.f0) / (this.f1 - this.f0);
  }

  atFraction(u: number): number {
    const f = this.f0 + u * (this.f1 - this.f0);
    return this.kind === 'log' ? 10 ** f : f;
  }

  ticks(count: number): Tick[] {
    const lo = Math.min(this.d0, this.d1);
    const hi = Math.max(this.d0, this.d1);
    return this.kind === 'log' ? logTicks(lo, hi) : linearTicks(lo, hi, count);
  }
}

function linearTicks(lo: number, hi: number, count: number): Tick[] {
  const { ticks, step } = niceTicks(lo, hi, count);
  return ticks.map((value) => ({ value, label: formatTick(value, step) }));
}

/** Decades, with 2× and 5× in between when fewer than two decades show; thinned when many do. */
function logTicks(lo: number, hi: number): Tick[] {
  const k0 = Math.ceil(Math.log10(lo) - 1e-9);
  const k1 = Math.floor(Math.log10(hi) + 1e-9);
  const span = Math.log10(hi / lo);
  const every = span > 14 ? 4 : span > 7 ? 2 : 1;
  const out: Tick[] = [];
  for (let k = k0; k <= k1; k++) if (k % every === 0) out.push({ value: 10 ** k, label: pow10(k) });
  if (span < 2) {
    for (let k = k0 - 1; k <= k1; k++) {
      for (const m of [2, 5]) {
        const v = m * 10 ** k;
        if (v >= lo && v <= hi) out.push({ value: v, label: formatValue(v) });
      }
    }
    out.sort((a, b) => a.value - b.value);
  }
  return out;
}

const SUPERSCRIPT: Record<string, string> = {
  '-': '⁻',
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
};

/** 10^k as a plain number near 1, as 10 with a superscript otherwise. */
export function pow10(k: number): string {
  if (k >= -2 && k <= 3) return (10 ** k).toFixed(Math.max(0, -k));
  return `10${[...String(k)].map((c) => SUPERSCRIPT[c]).join('')}`;
}

/** Round tick values covering [lo, hi], roughly `count` of them. */
export function niceTicks(lo: number, hi: number, count: number): { ticks: number[]; step: number } {
  if (!(hi > lo)) return { ticks: [lo], step: 1 };
  const step = niceStep((hi - lo) / Math.max(1, count));
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return { ticks, step };
}

function niceStep(raw: number): number {
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
}

/** The smallest of 1, 2, 2.5, 5 × 10ᵏ that is at least v: for axis limits. */
export function niceCeil(v: number): number {
  const mag = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * mag >= v * (1 - 1e-12)) return m * mag;
  return 10 * mag;
}

/** The largest power of ten at most v. */
export function decadeFloor(v: number): number {
  return 10 ** Math.floor(Math.log10(v) + 1e-12);
}

export function formatTick(v: number, step: number): string {
  const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return v.toFixed(Math.min(digits, 6));
}

/** Format a value for readouts: ~4 significant digits, no exponent unless needed. */
export function formatValue(v: number): string {
  if (v === Infinity) return '∞';
  if (!Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) {
    const [mantissa, exp] = v.toExponential(2).split('e');
    return `${mantissa}×10${[...String(Number(exp))].map((c) => SUPERSCRIPT[c]).join('')}`;
  }
  return v.toPrecision(4).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
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
