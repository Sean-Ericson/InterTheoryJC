/** Seeded xoshiro128** generator with Box–Muller normals. Reproducible across runs. */
export class Rng {
  private a: number;
  private b: number;
  private c: number;
  private d: number;
  private spare = 0;
  private hasSpare = false;

  constructor(seed: number) {
    // splitmix32 expands the seed into four non-zero state words.
    let s = seed >>> 0;
    const next = () => {
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.a = next();
    this.b = next();
    this.c = next();
    this.d = next();
    if ((this.a | this.b | this.c | this.d) === 0) this.a = 1;
  }

  /** Uniform on [0, 1). */
  next(): number {
    const r = Math.imul(rotl(Math.imul(this.b, 5), 7), 9) >>> 0;
    const t = this.b << 9;
    this.c ^= this.a;
    this.d ^= this.b;
    this.b ^= this.c;
    this.a ^= this.d;
    this.c ^= t;
    this.d = rotl(this.d, 11);
    return r / 4294967296;
  }

  /** Standard normal. */
  normal(): number {
    if (this.hasSpare) {
      this.hasSpare = false;
      return this.spare;
    }
    let u = 0;
    while (u === 0) u = this.next();
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    const th = 2 * Math.PI * v;
    this.spare = r * Math.sin(th);
    this.hasSpare = true;
    return r * Math.cos(th);
  }
}

function rotl(x: number, k: number): number {
  return (x << k) | (x >>> (32 - k));
}
