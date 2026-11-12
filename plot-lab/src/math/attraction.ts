import { dhaConditionedExp } from './drift';
import { linspace, memo } from './num';
import { ouHlEvolveLomax } from './ou';

/**
 * §1.6 of the notes: on the half-lines with a continuum of QSDs, the initial
 * tail selects the one reached. Measured long-time decay rates of S against
 * the tail parameter (βℓ for drift, ν for OU), in units of the bottom of the
 * L² spectrum (1/2τ for drift, k for OU), as in attraction_rates().
 */
export const TAIL_SAMPLES = [0.25, 0.5, 0.75, 1.5];

export const attractionRates = memo((): { s: number[]; drift: number[]; ou: number[] } => {
  const v = -1;
  const x = linspace(0, 250, 50001);
  const drift = TAIL_SAMPLES.map((s) => {
    const [t1, t2] = s < 1 ? [40, 60] : [300, 400];
    const l1 = dhaConditionedExp(x, t1, s, v).logS;
    const l2 = dhaConditionedExp(x, t2, s, v).logS;
    return (l1 - l2) / (t2 - t1) / ((v * v) / 2);
  });
  const ou = TAIL_SAMPLES.map((s) => Math.log(ouHlEvolveLomax(s, [1], 15).S / ouHlEvolveLomax(s, [1], 20).S) / 5);
  return { s: TAIL_SAMPLES, drift, ou };
});

/** Predicted rates: βℓ(2 − βℓ) below the edge for drift, ν for OU, both capped at 1. */
export const driftRate = (s: number) => (s < 1 ? s * (2 - s) : 1);
export const ouRate = (s: number) => Math.min(s, 1);
