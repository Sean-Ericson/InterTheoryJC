import { create, parseDependencies, derivativeDependencies } from 'mathjs';
import type { MathNode } from 'mathjs';

// Only the parser and symbolic differentiation are needed; this keeps the bundle small.
const { parse, derivative } = create({ parseDependencies, derivativeDependencies });

/**
 * User formulas are parsed by math.js, checked against a whitelist, and
 * translated into a native JS function. Only whitelisted constructs are ever
 * emitted (parameters become `P[i]`), so the generated code cannot do anything
 * other than arithmetic.
 */

export type RawFn = (x: number, t: number, P: Float64Array) => number;

export class ExprError extends Error {}

const VARS = new Set(['x', 't']);
const CONSTS: Record<string, string> = { pi: 'Math.PI', e: 'Math.E' };

type Emitter = (args: string[]) => string;
const call = (name: string): Emitter => (a) => `${name}(${a.join(',')})`;
const recip = (name: string): Emitter => (a) => `(1/${name}(${a[0]}))`;

/** Allowed functions: name -> [arities, emitter]. */
const FUNCS: Record<string, [number[], Emitter]> = {
  sin: [[1], call('Math.sin')],
  cos: [[1], call('Math.cos')],
  tan: [[1], call('Math.tan')],
  sec: [[1], recip('Math.cos')],
  csc: [[1], recip('Math.sin')],
  cot: [[1], recip('Math.tan')],
  asin: [[1], call('Math.asin')],
  acos: [[1], call('Math.acos')],
  atan: [[1], call('Math.atan')],
  sinh: [[1], call('Math.sinh')],
  cosh: [[1], call('Math.cosh')],
  tanh: [[1], call('Math.tanh')],
  sech: [[1], recip('Math.cosh')],
  csch: [[1], recip('Math.sinh')],
  coth: [[1], recip('Math.tanh')],
  asinh: [[1], call('Math.asinh')],
  acosh: [[1], call('Math.acosh')],
  atanh: [[1], call('Math.atanh')],
  exp: [[1], call('Math.exp')],
  log: [[1, 2], (a) => (a.length === 1 ? `Math.log(${a[0]})` : `(Math.log(${a[0]})/Math.log(${a[1]}))`)],
  log10: [[1], call('Math.log10')],
  sqrt: [[1], call('Math.sqrt')],
  cbrt: [[1], call('Math.cbrt')],
  abs: [[1], call('Math.abs')],
  sign: [[1], call('Math.sign')],
  min: [[2], call('Math.min')],
  max: [[2], call('Math.max')],
};

const BINARY: Record<string, string> = { add: '+', subtract: '-', multiply: '*', divide: '/' };

export const ALLOWED_FUNCTIONS = Object.keys(FUNCS);

export interface Analysis {
  node: MathNode;
  /** Free symbols other than x, t and constants, in order of first appearance. */
  params: string[];
  usesT: boolean;
}

/** Parse and validate a formula. Throws ExprError with a readable message. */
export function analyze(src: string): Analysis {
  if (!src.trim()) throw new ExprError('Formula is empty');
  let node: MathNode;
  try {
    node = parse(src);
  } catch (err) {
    throw new ExprError(`Syntax error: ${(err as Error).message}`);
  }
  const params: string[] = [];
  let usesT = false;
  walk(node, (name) => {
    if (name === 't') usesT = true;
    else if (!VARS.has(name) && !(name in CONSTS) && !params.includes(name)) params.push(name);
  });
  return { node, params, usesT };
}

/** Compile a validated node into a fast native function of (x, t, P). */
export function compile(node: MathNode, paramNames: string[]): RawFn {
  const body = emit(node, paramNames);
  return new Function('x', 't', 'P', `"use strict"; return ${body};`) as RawFn;
}

/** Symbolic d/dx, or null if math.js can't differentiate it (e.g. min/max). */
export function differentiate(node: MathNode): MathNode | null {
  try {
    return derivative(node, 'x');
  } catch {
    return null;
  }
}

/** Central-difference fallback derivative of a compiled function in x. */
export function numericDerivative(f: RawFn): RawFn {
  return (x, t, P) => {
    const h = 1e-5 * (1 + Math.abs(x));
    return (f(x + h, t, P) - f(x - h, t, P)) / (2 * h);
  };
}

function walk(node: MathNode, onSymbol: (name: string) => void): void {
  emit(node, null, onSymbol);
}

/**
 * Emit JS source for a node. With `params === null` this is a validation pass
 * that reports symbols instead of resolving them.
 */
function emit(node: MathNode, params: string[] | null, onSymbol?: (name: string) => void): string {
  const rec = (n: MathNode) => emit(n, params, onSymbol);
  const n = node as any;
  switch (node.type) {
    case 'ConstantNode': {
      const v = Number(n.value);
      if (typeof n.value !== 'number' || !Number.isFinite(v)) throw new ExprError(`Unsupported constant "${String(n.value)}"`);
      return `(${v})`;
    }
    case 'SymbolNode': {
      const name: string = n.name;
      onSymbol?.(name);
      if (VARS.has(name)) return name;
      if (name in CONSTS) return CONSTS[name];
      if (params === null) return '0';
      const i = params.indexOf(name);
      if (i < 0) throw new ExprError(`Unknown symbol "${name}"`);
      return `P[${i}]`;
    }
    case 'ParenthesisNode':
      return `(${rec(n.content)})`;
    case 'OperatorNode': {
      const args: MathNode[] = n.args;
      if (n.fn === 'unaryMinus') return `(-${rec(args[0])})`;
      if (n.fn === 'unaryPlus') return `(+${rec(args[0])})`;
      if (n.fn === 'pow') return `Math.pow(${rec(args[0])},${rec(args[1])})`;
      const op = BINARY[n.fn];
      if (op && args.length === 2) return `(${rec(args[0])}${op}${rec(args[1])})`;
      throw new ExprError(`Operator "${n.op}" is not supported`);
    }
    case 'FunctionNode': {
      const fname: string | undefined = n.fn?.type === 'SymbolNode' ? n.fn.name : undefined;
      const spec = fname !== undefined ? FUNCS[fname] : undefined;
      if (!fname || !spec) throw new ExprError(`Unknown function "${fname ?? '?'}"`);
      const [arities, emitFn] = spec;
      if (!arities.includes(n.args.length)) {
        throw new ExprError(`${fname}() takes ${arities.join(' or ')} argument(s)`);
      }
      return emitFn((n.args as MathNode[]).map(rec));
    }
    default:
      throw new ExprError(`"${node.toString()}" is not allowed in a formula`);
  }
}
