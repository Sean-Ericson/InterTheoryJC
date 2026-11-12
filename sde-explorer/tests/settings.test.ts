import { describe, expect, it } from 'vitest';
import { applyProcessLink, defaultSettings } from '../src/settings';

describe('process links (from Plot Lab)', () => {
  it('sets formulas, parameters, domain, walls, start and run length', () => {
    const s = defaultSettings();
    const q = new URLSearchParams('a=-k*x&b=sqrt(D)&p=k:2,D:0.5&lo=0&hi=4&bc=reflecting,absorbing&x0=1.5&T=12');
    expect(applyProcessLink(s, q)).toBe(true);
    expect(s.presetId).toBe('custom');
    expect([s.drift, s.noise]).toEqual(['-k*x', 'sqrt(D)']);
    expect(s.params.map((p) => [p.name, p.value])).toEqual([
      ['k', 2],
      ['D', 0.5],
    ]);
    for (const p of s.params) expect(p.min < p.value && p.value < p.max).toBe(true);
    expect(s.domain).toEqual([0, 4]);
    expect(s.bc).toEqual(['reflecting', 'absorbing']);
    expect([s.x0, s.sigma0, s.T]).toEqual([1.5, 0, 12]);
  });

  it('gives unnamed parameters the value 1 and clamps x0 into the domain', () => {
    const s = defaultSettings();
    expect(applyProcessLink(s, new URLSearchParams('a=v&b=sqrt(D)&lo=0&hi=1&x0=3'))).toBe(true);
    expect(s.params.map((p) => p.value)).toEqual([1, 1]);
    expect(s.x0).toBe(1);
  });

  it('ignores links without formulas, or with formulas that do not parse', () => {
    const s = defaultSettings();
    const before = JSON.stringify(s);
    expect(applyProcessLink(s, new URLSearchParams('preset=ou&t=2'))).toBe(false);
    expect(applyProcessLink(s, new URLSearchParams('a=-k*(x&b=1'))).toBe(false);
    expect(JSON.stringify(s)).toBe(before);
  });
});
