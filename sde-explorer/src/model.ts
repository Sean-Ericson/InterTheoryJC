import { analyze, compile, differentiate, numericDerivative, ExprError, type RawFn } from './expr';

export type Interp = 'ito' | 'strat';
export type BC = 'open' | 'reflecting' | 'absorbing' | 'periodic';
export type Fn = (x: number, t: number) => number;

/**
 * A compiled 1D diffusion in Gardiner's notation: dx = a(x,t) dt + b(x,t) dW,
 * with Fokker–Planck equation ∂t p = −∂x[a p] + ½ ∂x²[b² p]  (eq. 4.3.20).
 *
 * `a` is always the Itô drift: for a Stratonovich SDE it already includes the
 * correction ½ b ∂x b  (eq. 4.3.40).
 */
export interface Process {
  a: Fn;
  b: Fn;
  /** ∂b/∂x */
  db: Fn;
  timeDependent: boolean;
  /** True when ∂b/∂x had to be approximated by finite differences. */
  numericDb: boolean;
}

export interface ProcessSpec {
  drift: string;
  noise: string;
  params: Record<string, number>;
  interp: Interp;
}

export interface FormulaInfo {
  params: string[];
  usesT: boolean;
}

/** Parameter symbols used by a drift/noise pair, in first-appearance order. */
export function formulaParams(drift: string, noise: string): FormulaInfo {
  const a = tagged('a', () => analyze(drift));
  const b = tagged('b', () => analyze(noise));
  const params = [...a.params];
  for (const p of b.params) if (!params.includes(p)) params.push(p);
  return { params, usesT: a.usesT || b.usesT };
}

/** Build a Process. Throws FormulaError tagged with which formula failed. */
export function buildProcess(spec: ProcessSpec): Process {
  const a = tagged('a', () => analyze(spec.drift));
  const b = tagged('b', () => analyze(spec.noise));
  const names = [...a.params];
  for (const p of b.params) if (!names.includes(p)) names.push(p);
  const P = new Float64Array(names.map((n) => spec.params[n] ?? 1));

  const rawA = compile(a.node, names);
  const rawB = compile(b.node, names);
  const dbNode = differentiate(b.node);
  let rawDb: RawFn;
  let numericDb = false;
  if (dbNode) {
    rawDb = compile(dbNode, names);
  } else {
    rawDb = numericDerivative(rawB);
    numericDb = true;
  }

  // Non-finite values (e.g. sqrt of a negative x) are treated as zero so a
  // single bad evaluation can't poison the whole ensemble or grid.
  const bFn: Fn = (x, t) => finite(rawB(x, t, P));
  const db: Fn = (x, t) => finite(rawDb(x, t, P));
  const aFn: Fn =
    spec.interp === 'strat'
      ? (x, t) => finite(rawA(x, t, P)) + 0.5 * finite(rawB(x, t, P) * rawDb(x, t, P))
      : (x, t) => finite(rawA(x, t, P));

  return { a: aFn, b: bFn, db, timeDependent: a.usesT || b.usesT, numericDb };
}

function finite(v: number): number {
  return Number.isFinite(v) ? v : 0;
}

export class FormulaError extends ExprError {
  constructor(
    readonly which: 'a' | 'b',
    message: string,
  ) {
    super(message);
  }
}

function tagged<T>(which: 'a' | 'b', f: () => T): T {
  try {
    return f();
  } catch (err) {
    throw new FormulaError(which, (err as Error).message);
  }
}
