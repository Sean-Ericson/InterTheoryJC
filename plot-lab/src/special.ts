const SQRT_PI = Math.sqrt(Math.PI);

/** Error function, accurate to a few ulp. */
export function erf(z: number): number {
  if (z < 0) return -erf(-z);
  return z < 2 ? erfSeries(z) : 1 - erfcFraction(z);
}

/** Complementary error function, accurate in relative terms far into the tail. */
export function erfc(z: number): number {
  if (Number.isNaN(z)) return NaN;
  if (z < 0) return 2 - erfc(-z);
  return z < 2 ? 1 - erfSeries(z) : erfcFraction(z);
}

/** Standard normal CDF. */
export function normalCdf(z: number): number {
  return 0.5 * erfc(-z / Math.SQRT2);
}

/** erf z = (2/√π) e^{−z²} Σ 2ⁿ z^{2n+1} / (1·3···(2n+1)): every term positive, so no cancellation. */
function erfSeries(z: number): number {
  const z2 = z * z;
  let term = z;
  let sum = z;
  for (let n = 1; n < 200 && term > sum * 1e-17; n++) {
    term *= (2 * z2) / (2 * n + 1);
    sum += term;
  }
  return (2 / SQRT_PI) * Math.exp(-z2) * sum;
}

/**
 * erfc z = (e^{−z²}/√π) / (z + ½/(z + 1/(z + (3/2)/(z + …)))), by the modified
 * Lentz method. Converges quickly for z ≥ 2.
 */
function erfcFraction(z: number): number {
  return Math.exp(-z * z) * erfcxFraction(z);
}

/** Scaled complementary error function e^{z²} erfc z: no underflow far into the tail (≈ 1/(z√π)). */
export function erfcx(z: number): number {
  if (Number.isNaN(z)) return NaN;
  if (z >= 2) return erfcxFraction(z);
  if (z >= 0) return Math.exp(z * z) * (1 - erfSeries(z));
  return 2 * Math.exp(z * z) - erfcx(-z);
}

/** log Φ(z) of the standard normal CDF, accurate for very negative z. */
export function logNormalCdf(z: number): number {
  if (z < 0) return Math.log(0.5 * erfcx(-z / Math.SQRT2)) - 0.5 * z * z;
  return Math.log1p(-0.5 * erfc(z / Math.SQRT2));
}

/** Imaginary error function erfi z = (2/√π) ∫₀ᶻ e^{u²} du. */
export function erfi(z: number): number {
  if (z < 0) return -erfi(-z);
  if (z > 7) {
    // e^{z²}/(√π z) (1 + 1/2z² + 3/4z⁴ + 15/8z⁶ + …)
    const w = 1 / (2 * z * z);
    let term = 1;
    let sum = 1;
    for (let k = 1; k < 30; k++) {
      term *= (2 * k - 1) * w;
      sum += term;
      if (term < 1e-17 * sum) break;
    }
    return (Math.exp(z * z) / (SQRT_PI * z)) * sum;
  }
  // Σ z^{2n+1}/(n!(2n+1)): every term positive.
  const z2 = z * z;
  let pow = z;
  let sum = z;
  for (let n = 1; n < 400; n++) {
    pow *= z2 / n;
    const term = pow / (2 * n + 1);
    sum += term;
    if (term < 1e-17 * sum) break;
  }
  return (2 / SQRT_PI) * sum;
}

/** Natural log of Γ(x), x > 0 (Lanczos, g = 7). */
export function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  const y = x - 1;
  let a = c[0];
  const t = y + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (y + i);
  return 0.5 * Math.log(2 * Math.PI) + (y + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Kummer's M(a, b, −z) for z ≥ 0 and b ≥ a (b − a ≥ 0). Small z: Kummer's
 * transformation e^{−z} M(b − a, b, z), a series of positive terms. Large z:
 * the algebraic asymptote Γ(b)/Γ(b−a) z^{−a} Σ (a)ₛ(a−b+1)ₛ/s! z^{−s}.
 */
export function kummerNeg(a: number, b: number, z: number): number {
  const c = b - a;
  if (z <= 40) {
    let term = 1;
    let sum = 1;
    for (let n = 0; n < 2000; n++) {
      term *= ((c + n) / (b + n)) * (z / (n + 1));
      sum += term;
      if (Math.abs(term) < 1e-17 * sum) break;
    }
    return Math.exp(-z) * sum;
  }
  if (c <= 0) return Math.exp(-z);
  let term = 1;
  let sum = 1;
  for (let s = 0; s < 60; s++) {
    const next = term * (((a + s) * (a - b + 1 + s)) / ((s + 1) * z));
    if (Math.abs(next) > Math.abs(term)) break;
    term = next;
    sum += term;
    if (Math.abs(term) < 1e-17 * Math.abs(sum)) break;
  }
  return Math.exp(lgamma(b) - lgamma(c) - a * Math.log(z)) * sum;
}

function erfcxFraction(z: number): number {
  const tiny = 1e-300;
  let f = z;
  let c = z;
  let d = 0;
  for (let k = 1; k < 500; k++) {
    const a = k / 2;
    d = z + a * d;
    if (d === 0) d = tiny;
    d = 1 / d;
    c = z + a / c;
    if (c === 0) c = tiny;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return 1 / (SQRT_PI * f);
}
