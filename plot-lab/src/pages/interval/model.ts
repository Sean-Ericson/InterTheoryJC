import { erfc, normalCdf } from '../../special';

/**
 * Brownian motion dx = √D dW on (0, L), absorbed at both walls. In Gardiner's
 * notation A = 0, B = D, so
 *
 *   ∂ₜp = ½D ∂ₓ²p,   p(0, t) = p(L, t) = 0.
 *
 * Everything has two exact representations, each used where it converges
 * fastest: the sine series in the eigenfunctions sin(nπx/L), with
 * λₙ = ½D(nπ/L)², at late times, and the method of images at early times.
 * The switch is at diffusive time τ = Dt/L² = TAU_SWITCH, where both need only
 * a handful of terms.
 */

/** Starting density: δ(x − x₀), uniform on (0, L), or the quasi-stationary φ₀. */
export type Start = 'point' | 'uniform' | 'qsd';

export interface IntervalParams {
  L: number;
  D: number;
  /** Starting point, used by the point start. */
  x0: number;
  start: Start;
}

export interface Fluxes {
  /** Exit-time density through x = 0: j₀ = ½D ∂ₓp(0, t). */
  j0: number;
  /** Exit-time density through x = L: j_L = −½D ∂ₓp(L, t). */
  jL: number;
}

export interface ExitStats {
  mean: number;
  sd: number;
  median: number;
  /** Probability of leaving through x = L. */
  piL: number;
  /** Mean exit time of the paths that leave through 0, and through L. */
  mean0: number;
  meanL: number;
}

const TAU_SWITCH = 0.05;
/** Series terms stop once e^{−λₙt} is this small relative to e^{−λ₁t}. */
const CUTOFF = 1e-17;
const MAX_MODES = 100_000;
const SQRT_2PI = Math.sqrt(2 * Math.PI);

export class Interval {
  readonly L: number;
  readonly D: number;
  readonly x0: number;
  readonly start: Start;
  /** Slowest decay rate λ₁ = ½D(π/L)²: the exit rate of the quasi-stationary state. */
  readonly lambda1: number;

  constructor(p: IntervalParams) {
    this.L = p.L;
    this.D = p.D;
    this.x0 = Math.min(p.L, Math.max(0, p.x0));
    this.start = p.start;
    this.lambda1 = 0.5 * p.D * (Math.PI / p.L) ** 2;
  }

  /** Diffusive time τ = Dt/L². */
  tau(t: number): number {
    return (this.D * t) / (this.L * this.L);
  }

  /** Sine coefficient bₙ = ∫ p₀(x) sin(nπx/L) dx of the starting density. */
  coef(n: number): number {
    switch (this.start) {
      case 'point':
        return Math.sin((n * Math.PI * this.x0) / this.L);
      case 'uniform':
        return n % 2 === 1 ? 2 / (n * Math.PI) : 0;
      case 'qsd':
        return n === 1 ? Math.PI / 4 : 0;
    }
  }

  /** Weight of the slowest mode: at late times S(t) ≈ c₁ e^{−λ₁t}. */
  get c1(): number {
    return (4 / Math.PI) * this.coef(1);
  }

  /** Survival S(t) = P(T > t) = ∫₀ᴸ p(x, t) dx. */
  survival(t: number): number {
    if (t <= 0) return 1;
    return this.useImages(t) ? this.survivalImages(t) : this.survivalSeries(t);
  }

  /** Exit-time density through each wall; f = −dS/dt = j₀ + j_L. */
  fluxes(t: number): Fluxes {
    if (this.start === 'qsd') {
      // A single mode, shared equally between the walls.
      const h = 0.5 * this.lambda1 * Math.exp(-this.lambda1 * Math.max(0, t));
      return { j0: h, jL: h };
    }
    if (t <= 0) {
      const h = this.start === 'uniform' ? Infinity : 0;
      return { j0: h, jL: h };
    }
    return this.useImages(t) ? this.fluxesImages(t) : this.fluxesSeries(t);
  }

  /** p(x, t) at each x. */
  density(xs: ArrayLike<number>, t: number): Float64Array {
    const { L } = this;
    if (this.start === 'qsd') {
      const out = new Float64Array(xs.length);
      const a = (Math.PI / (2 * L)) * Math.exp(-this.lambda1 * Math.max(0, t));
      for (let i = 0; i < xs.length; i++) out[i] = a * Math.sin((Math.PI * xs[i]) / L);
      return out;
    }
    if (t <= 0) return new Float64Array(xs.length).fill(this.start === 'uniform' ? 1 / L : NaN);
    return this.useImages(t) ? this.densityImages(xs, t) : this.densitySeries(xs, t);
  }

  /** Late times: S = Σ_{n odd} (4/nπ) bₙ e^{−λₙt}. */
  private survivalSeries(t: number): number {
    let sum = 0;
    this.modes(t, (n, e) => {
      if (n % 2 === 1) sum += (4 / (n * Math.PI)) * this.coef(n) * e;
    });
    return sum;
  }

  /** Late times: j₀ = (πD/L²) Σ n bₙ e^{−λₙt},  j_L = (πD/L²) Σ (−1)ⁿ⁺¹ n bₙ e^{−λₙt}. */
  private fluxesSeries(t: number): Fluxes {
    let s0 = 0;
    let sL = 0;
    this.modes(t, (n, e) => {
      const a = n * this.coef(n) * e;
      s0 += a;
      sL += n % 2 === 1 ? a : -a;
    });
    const k = (Math.PI * this.D) / (this.L * this.L);
    return { j0: k * s0, jL: k * sL };
  }

  private fluxesImages(t: number): Fluxes {
    if (this.start === 'point') return { j0: this.pointImageFlux(this.x0, t), jL: this.pointImageFlux(this.L - this.x0, t) };
    const h = this.uniformImageFlux(t);
    return { j0: h, jL: h };
  }

  /** Early times: the image sum of Gaussians for a point start, of their integrals for a uniform one. */
  private densityImages(xs: ArrayLike<number>, t: number): Float64Array {
    const { L } = this;
    const out = new Float64Array(xs.length);
    const s = Math.sqrt(this.D * t);
    const M = this.imageCount(t);
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i];
      let sum = 0;
      for (let m = -M; m <= M; m++) {
        const c = 2 * m * L;
        if (this.start === 'point') {
          sum += gauss(x - this.x0 - c, s) - gauss(x + this.x0 - c, s);
        } else {
          sum += (2 * normalCdf((x - c) / s) - normalCdf((x - L - c) / s) - normalCdf((x + L - c) / s)) / L;
        }
      }
      out[i] = sum;
    }
    return out;
  }

  /** Late times: p = (2/L) Σ bₙ sin(nπx/L) e^{−λₙt}, with sin(nθ) by the Chebyshev recurrence. */
  private densitySeries(xs: ArrayLike<number>, t: number): Float64Array {
    const { L } = this;
    const out = new Float64Array(xs.length);
    const amp: number[] = [];
    this.modes(t, (n, e) => amp.push((2 / L) * this.coef(n) * e));
    for (let i = 0; i < xs.length; i++) {
      const th = (Math.PI * xs[i]) / L;
      const c2 = 2 * Math.cos(th);
      let prev = 0;
      let cur = Math.sin(th);
      let sum = 0;
      for (let n = 0; n < amp.length; n++) {
        sum += amp[n] * cur;
        const next = c2 * cur - prev;
        prev = cur;
        cur = next;
      }
      out[i] = sum;
    }
    return out;
  }

  /**
   * Each wall on its own, as if the other were absent: the half-line results,
   * which are the nearest image terms and exact at early times. Null for the
   * quasi-stationary start, which has no early-time regime.
   */
  oneWall(t: number): { S: number; j0: number; jL: number } | null {
    if (this.start === 'qsd') return null;
    const { L, D } = this;
    if (t <= 0) return { S: 1, ...this.fluxes(0) };
    if (this.start === 'point') {
      const r = Math.sqrt(2 * D * t);
      return {
        S: 1 - erfc(this.x0 / r) - erfc((L - this.x0) / r),
        j0: this.halfLineFlux(this.x0, t),
        jL: this.halfLineFlux(L - this.x0, t),
      };
    }
    const j = (Math.sqrt(D / (2 * Math.PI * t)) * (1 - Math.exp((-L * L) / (2 * D * t)))) / L;
    const absorbed = (Math.sqrt(D / (2 * Math.PI)) * (this.absorbedK(0, t) - this.absorbedK(L, t))) / L;
    return { S: 1 - 2 * absorbed, j0: j, jL: j };
  }

  stats(): ExitStats {
    const { L, D } = this;
    let mean: number;
    let sd: number;
    let piL: number;
    let mean0: number;
    let meanL: number;
    switch (this.start) {
      case 'point': {
        const x = this.x0;
        const y = L - x;
        ({ mean, mean0, meanL } = meanExitTimes(x, L, D));
        sd = Math.sqrt((x * y * (x * x + y * y)) / 3) / D;
        piL = x / L;
        break;
      }
      case 'uniform':
        mean = mean0 = meanL = (L * L) / (6 * D);
        sd = (Math.sqrt(7 / 180) * L * L) / D;
        piL = 0.5;
        break;
      case 'qsd':
        // Exponential with rate λ₁, symmetric between the walls.
        mean = sd = mean0 = meanL = 1 / this.lambda1;
        piL = 0.5;
        break;
    }
    return { mean, sd, median: this.median(mean), piL, mean0, meanL };
  }

  /** The time t at which S(t) = target, by bisection (S decreases). */
  timeAtSurvival(target: number, guess = 1 / this.lambda1): number {
    let lo = 0;
    let hi = Math.max(guess, 1e-12 * this.L * this.L / this.D);
    while (this.survival(hi) > target && hi < 1e6 / this.lambda1) hi *= 2;
    for (let i = 0; i < 200 && hi - lo > 1e-13 * hi; i++) {
      const mid = 0.5 * (lo + hi);
      if (this.survival(mid) > target) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  private median(mean: number): number {
    if (this.start === 'qsd') return Math.LN2 / this.lambda1;
    return this.timeAtSurvival(0.5, mean);
  }

  private useImages(t: number): boolean {
    return this.start !== 'qsd' && this.tau(t) < TAU_SWITCH;
  }

  /** Image pairs m = −M…M needed for e^{−(2mL)²/2Dt} to be negligible. */
  private imageCount(t: number): number {
    return 2 + Math.ceil(Math.sqrt(80 * this.tau(t)) / 2);
  }

  /** Calls fn(n, e^{−λₙt}) for n = 1, 2, … until the terms are negligible. */
  private modes(t: number, fn: (n: number, e: number) => void): void {
    // e^{−λₙt} = qⁿ², stepped by qⁿ²·q²ⁿ⁺¹ = q⁽ⁿ⁺¹⁾².
    const q = Math.exp(-this.lambda1 * t);
    const q2 = q * q;
    let e = q;
    let r = q2 * q;
    for (let n = 1; n <= MAX_MODES; n++) {
      fn(n, e);
      e *= r;
      r *= q2;
      if (e < CUTOFF * q || e === 0) break;
    }
  }

  /** Σₘ yₘ e^{−yₘ²/2Dt} / √(2πDt³), yₘ = a + 2mL: flux through the wall a distance a from the start. */
  private pointImageFlux(a: number, t: number): number {
    const v = this.D * t;
    const M = this.imageCount(t);
    let sum = 0;
    for (let m = -M; m <= M; m++) {
      const y = a + 2 * m * this.L;
      sum += y * Math.exp((-y * y) / (2 * v));
    }
    return sum / (SQRT_2PI * Math.sqrt(v) * t);
  }

  /** Point fluxes averaged over a uniform start: (1/L)√(D/2πt) Σₘ [e^{−(2mL)²/2Dt} − e^{−((2m+1)L)²/2Dt}]. */
  private uniformImageFlux(t: number): number {
    const { L } = this;
    const v2 = 2 * this.D * t;
    const M = this.imageCount(t);
    let sum = 0;
    for (let m = -M; m <= M; m++) {
      const a = 2 * m * L;
      const b = (2 * m + 1) * L;
      sum += Math.exp((-a * a) / v2) - Math.exp((-b * b) / v2);
    }
    return (Math.sqrt(this.D / (2 * Math.PI * t)) * sum) / L;
  }

  private survivalImages(t: number): number {
    const { L } = this;
    const s = Math.sqrt(this.D * t);
    const M = this.imageCount(t);
    if (this.start === 'point') {
      // ∫₀ᴸ of the image sum Σₘ [G(x − x₀ − 2mL) − G(x + x₀ − 2mL)]
      const x0 = this.x0;
      let sum = 0;
      for (let m = -M; m <= M; m++) {
        const c = 2 * m * L;
        sum += normalCdf((L - x0 - c) / s) - normalCdf((-x0 - c) / s) - normalCdf((L + x0 - c) / s) + normalCdf((x0 - c) / s);
      }
      return sum;
    }
    // Uniform: 1 − 2 × (mass absorbed through 0), integrating the image flux in time.
    let k = 0;
    for (let m = -M; m <= M; m++) k += this.absorbedK(2 * m * L, t) - this.absorbedK((2 * m + 1) * L, t);
    return 1 - (2 / L) * Math.sqrt(this.D / (2 * Math.PI)) * k;
  }

  /** ∫₀ᵗ s^{−1/2} e^{−a²/2Ds} ds = 2√t e^{−a²/2Dt} − |a|√(2π/D) erfc(|a|/√(2Dt)). */
  private absorbedK(a: number, t: number): number {
    const b = Math.abs(a);
    const r = Math.sqrt(2 * this.D * t);
    return 2 * Math.sqrt(t) * Math.exp(-(b * b) / (r * r)) - b * Math.sqrt((2 * Math.PI) / this.D) * erfc(b / r);
  }

  /** First-passage density to a wall a distance a away on the half-line (Lévy–Smirnov). */
  private halfLineFlux(a: number, t: number): number {
    return (a * Math.exp((-a * a) / (2 * this.D * t))) / (SQRT_2PI * Math.sqrt(this.D * t) * t);
  }
}

/** Mean exit time from x, overall and conditioned on the exit wall: solutions of ½D T'' = −1 and relatives. */
export function meanExitTimes(x: number, L: number, D: number): { mean: number; mean0: number; meanL: number } {
  return {
    mean: (x * (L - x)) / D,
    mean0: (x * (2 * L - x)) / (3 * D),
    meanL: (L * L - x * x) / (3 * D),
  };
}

/** Quasi-stationary density φ₀(x) = (π/2L) sin(πx/L). */
export function qsdDensity(x: number, L: number): number {
  return (Math.PI / (2 * L)) * Math.sin((Math.PI * x) / L);
}

function gauss(y: number, s: number): number {
  return Math.exp((-y * y) / (2 * s * s)) / (SQRT_2PI * s);
}
