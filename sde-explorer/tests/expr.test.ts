import { describe, expect, it } from 'vitest';
import { evaluate } from 'mathjs';
import { analyze, compile, differentiate, ExprError } from '../src/expr';

describe('expr', () => {
  const cases = [
    '-k*(x - m)',
    'a*x - c*x^3 + eps*cos(omega*t)',
    'sigma*sqrt(x)',
    'sqrt(k*x + x^2)',
    '2x + 3 x^2 - -x',
    'exp(-x^2/2) / (1 + tanh(x)) + log(2 + x, 3) + abs(x) + max(x, 0.5)',
    'pi * e * sec(x) + cot(x + 1) + sign(x - 0.2)',
  ];

  it('compiles to the same values math.js evaluates', () => {
    const values: Record<string, number> = { k: 1.3, m: -0.4, a: 0.7, c: 1.1, eps: 0.25, omega: 2, sigma: 0.6 };
    for (const src of cases) {
      const { node, params } = analyze(src);
      const fn = compile(node, params);
      const P = new Float64Array(params.map((p) => values[p]));
      for (const [x, t] of [
        [0.3, 0],
        [1.7, 2.5],
        [0.9, -1],
      ]) {
        const expected = evaluate(src, { ...values, x, t }) as number;
        expect(fn(x, t, P)).toBeCloseTo(expected, 10);
      }
    }
  });

  it('differentiates symbolically', () => {
    const { node, params } = analyze('sigma*sqrt(x)');
    const d = compile(differentiate(node)!, params);
    const P = new Float64Array([0.8]);
    expect(d(2, 0, P)).toBeCloseTo(0.8 / (2 * Math.sqrt(2)), 12);
  });

  it('returns null when it cannot differentiate', () => {
    expect(differentiate(analyze('max(x, 1)').node)).toBeNull();
  });

  it('detects parameters and time dependence', () => {
    const a = analyze('-k*(x - m) + eps*cos(omega*t) + pi');
    expect(a.params).toEqual(['k', 'm', 'eps', 'omega']);
    expect(a.usesT).toBe(true);
    expect(analyze('x^2').usesT).toBe(false);
  });

  it('rejects anything outside the whitelist', () => {
    const bad = ['x = 1', 'f(x) = x', 'foo(x)', 'import("fs")', 'x > 0 ? 1 : 2', '[1, 2]', '"abc"', 'x % 2', '', 'x +'];
    for (const src of bad) expect(() => analyze(src), src).toThrow(ExprError);
  });
});
