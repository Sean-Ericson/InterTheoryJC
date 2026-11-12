import type { Line, Rule } from '../plot/chart';
import { formatValue } from '../plot/scale';
import { SEQ, type Role } from '../plot/theme';
import { num } from '../ui/controls';

/** Small builders shared by the figure pages. */

export const fmt = formatValue;

export function sample(x: ArrayLike<number>, f: (x: number, i: number) => number): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = f(x[i], i);
  return out;
}

export function line(label: string, color: Role, x: ArrayLike<number>, y: ArrayLike<number>, opts: Partial<Line> = {}): Line {
  return { kind: 'line', label, color, x, y, ...opts };
}

/** A reference curve: thin, dashed, kept out of the y fit. */
export function ref(label: string, x: ArrayLike<number>, y: ArrayLike<number>, opts: Partial<Line> = {}): Line {
  return line(label, 'ink', x, y, { dash: 'dashed', width: 1.25, noScale: true, ...opts });
}

/** The slider's curve: ink, heavier, drawn on top of the notes' family. */
export function live(label: string, x: ArrayLike<number>, y: ArrayLike<number>, opts: Partial<Line> = {}): Line {
  return line(label, 'ink', x, y, { width: 2.75, z: 5, ...opts });
}

/** A family of curves at the notes' parameter values, labelled by the value, in a ramp of roles. */
export function family(values: number[], x: ArrayLike<number>, f: (x: number, p: number) => number, roles: Role[] = SEQ, label = (p: number) => num(p)): Line[] {
  const start = Math.max(0, roles.length - values.length);
  return values.map((p, i) => line(label(p), roles[Math.min(roles.length - 1, start + i)], x, sample(x, (y) => f(y, p)), { width: 1.6 }));
}

/** A horizontal reference across [x0, x1], as a line (so it appears in the legend). */
export function hline(label: string, y: number, x0: number, x1: number, opts: Partial<Line> = {}): Line {
  return ref(label, [x0, x1], [y, y], opts);
}

/** y where V(x) < E, NaN elsewhere: a level drawn across the classically allowed region. */
export function levelSegment(x: ArrayLike<number>, V: ArrayLike<number>, E: number, region?: (x: number) => boolean): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = V[i] < E && (!region || region(x[i])) ? E : NaN;
  return out;
}

export const heading = (name: string) => (x: number) => `${name} = ${fmt(x)}`;

/** A pinned value on the x axis, labelled. */
export function xRule(value: number, label?: string, color: Role = 'muted', dash: Rule['dash'] = 'dashed'): Rule {
  return { axis: 'x', value, color, label, dash };
}

export function yRule(value: number, label?: string, color: Role = 'muted', dash: Rule['dash'] = 'dashed'): Rule {
  return { axis: 'y', value, color, label, dash };
}

/** Clamp x into [lo, hi]. */
export const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
