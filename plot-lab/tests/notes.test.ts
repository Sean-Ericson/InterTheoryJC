import { describe, expect, it } from 'vitest';
import * as C from '../src/math/circle';
import * as R from '../src/math/drift';
import * as F from '../src/math/free';
import { linspace } from '../src/math/num';
import * as O from '../src/math/ou';
import * as P from '../src/math/potentials';
import { attractionRates } from '../src/math/attraction';
import * as S from '../src/math/singular';
import data from './fixtures/notes-reference.json';

/**
 * The figure models against make_figures.py, the script behind the notes'
 * figures. tests/fixtures/notes-reference.json holds its values at sample
 * points; regenerate it with the script in the README when the notes change.
 */
const ref = data as unknown as Record<string, number[] & number[][] & number[][][]>;

/** |got − want| ≤ tol · max(1, |want|), elementwise. */
function close(got: number[], want: number[], tol: number, name = ''): void {
  expect(got.length, name).toBe(want.length);
  got.forEach((g, i) => {
    const w = want[i];
    expect(Math.abs(g - w), `${name}[${i}]: ${g} vs ${w}`).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(w)));
  });
}

/** Relative error, for values spanning many decades (an exact zero must come out as zero). */
function rel(got: number[], want: number[], tol: number, name = ''): void {
  got.forEach((g, i) => {
    const w = want[i];
    const err = w === 0 ? Math.abs(g) : Math.abs(g - w) / Math.abs(w);
    expect(err, `${name}[${i}]: ${g} vs ${w}`).toBeLessThanOrEqual(tol);
  });
}

const xs = [0.1, 0.5, 0.9];
const ts = [0.01, 0.1, 1];
const X0 = 0.3;
const PE = [-4, -2, 0, 2, 4];

describe('Part I: free diffusion', () => {
  it('interval densities by images and modes', () => {
    ts.forEach((t, j) => {
      close(xs.map((x) => F.interval(x, t, X0, 'reflecting')), ref.images_refl[0][j], 1e-10, `refl t=${t}`);
      close(xs.map((x) => F.interval(x, t, X0, 'absorbing')), ref.images_abs[0][j], 1e-10, `abs t=${t}`);
      close(xs.map((x) => F.interval(x, t, X0, 'mixed')), ref.mix_images[j], 1e-10, `mixed t=${t}`);
    });
  });
  it('moments, autocorrelation, survival and fluxes', () => {
    close(ts.map((t) => F.reflMean(t, X0)), ref.refl_mean, 1e-10);
    close(ts.map((t) => F.reflSecond(t, X0)), ref.refl_second, 1e-10);
    close([0.1, 0.5].map((t) => F.reflAutocov(t) / F.reflAutocov(0)), ref.refl_autocov_ratio, 1e-12);
    close(ts.map((t) => F.survival(t, X0)), ref.survival, 1e-10);
    close(ts.map((t) => F.flux0(t, X0)), ref.flux0, 1e-10);
    close(ts.map((t) => F.fluxL(t, X0)), ref.fluxL, 1e-10);
    close(ts.map((t) => F.mixSurvival(t, X0)), ref.mix_survival, 1e-10);
    close(ts.map((t) => F.mixFlux(t, X0)), ref.mix_flux, 1e-10);
  });
  it('half-line', () => {
    [0.05, 1, 20].forEach((t, j) => close([0.5, 1, 3].map((x) => F.halfLine(x, t, 1, 1)), ref.half_refl[j], 1e-12));
    close([0.05, 1, 20].map((t) => F.hlMean(t, 1)), ref.hl_mean, 1e-12);
    close([0.05, 1, 20].map((t) => F.hlSurvival(t, 1)), ref.hl_survival, 1e-12);
  });
});

describe('Part II: constant drift', () => {
  it('reflecting interval', () => {
    PE.forEach((v, j) => close(xs.map((x) => R.drStationary(x, v)), ref.dr_stationary[j], 1e-12));
    ts.forEach((t, j) => close(xs.map((x) => R.drDensity(x, t, X0, 4)), ref.dr_modes[j], 1e-8, `t=${t}`));
    PE.forEach((v, j) => close([0.05, 0.2].map((t) => R.drMean(t, X0, v)), ref.dr_mean[j], 1e-6, `v=${v}`));
  });
  it('absorbing interval', () => {
    const x = [0.2, 0.5, 0.8];
    PE.forEach((v, j) => {
      close(x.map((y) => R.daQsd(y, v)), ref.da_qsd[j], 1e-12);
      close(x.map((y) => R.daPiL(y, v)), ref.da_piL[j], 1e-12);
      close(x.map((y) => R.daT(y, v)), ref.da_T[j], 1e-12);
      close(x.map((y) => R.daTL(y, v)), ref.da_TL[j], 1e-12);
      close([0.02, 0.1].map((t) => R.daSurvival(t, X0, v)), ref.da_survival[j], 1e-9, `S v=${v}`);
      close([0.02, 0.1].map((t) => R.daFlux0(t, X0, v)), ref.da_flux0[j], 1e-10);
    });
  });
  it('reflecting at 0, absorbing at L', () => {
    [-6, -2, -1, 0, 2, 6].forEach((v, j) =>
      close(
        R.dmModes(v, 3).map((m) => m.lambda),
        ref.dm_lambda[j],
        1e-9,
        `v=${v}`,
      ),
    );
    PE.forEach((v, j) => close([0.2, 0.5, 0.8].map(R.dmQsd(v)), ref.dm_qsd[j], 1e-6, `qsd v=${v}`));
    PE.forEach((v, j) => close([0, 0.5, 0.9].map((x) => R.dmT(x, v)), ref.dm_T[j], 1e-12));
  });
  it('half-lines', () => {
    [0.05, 1, 10].forEach((t, j) => close([0.2, 1, 3].map((x) => R.dhrDensity(x, t, 1, -1)), ref.dhr_p[j], 1e-12));
    [-1, -0.5, 0, 0.5, 1].forEach((v, j) =>
      close([0.1, 1, 10, 100].map((t) => R.dhrMeanAt(t, 1, v, 100)), ref.dhr_mean[j], 2e-5, `mean v=${v}`),
    );
    [0.3, 3, 30].forEach((t, j) => close([0.2, 1, 3].map((x) => R.dhaDensity(x, t, 1, -1)), ref.dha_p[j], 1e-12));
    [-1, 0, 1].forEach((v, j) => close([0.3, 3, 30].map((t) => R.dhaSurvival(t, 1, v)), ref.dha_survival[j], 1e-12));
    [-1, 0, 1].forEach((v, j) => close([0.01, 1, 50].map((t) => R.inverseGaussian(t, 1, v)), ref.inverse_gaussian[j], 1e-12));
    [0.2, 0.5, 0.8, 1].forEach((f, j) => close([0.5, 2, 6].map((x) => R.dhaQsd(x, f * 0.5, -1)), ref.dha_qsd[j], 1e-12));
  });
  it('conditioned density from an exponential start', () => {
    const grid = linspace(0, 250, 50001);
    [0.25, 0.5, 0.75].forEach((beta, j) => {
      const { q, logS } = R.dhaConditionedExp(grid, 60, beta, -1);
      const at = (x: number) => q[Math.round(x / 0.005)];
      close([at(1), at(5), at(12), logS], ref.dha_conditioned_exp[j], 1e-9, `beta=${beta}`);
    });
  });
});

describe('Part III: Ornstein–Uhlenbeck', () => {
  it('line and half-line', () => {
    [0.05, 1].forEach((t, j) => close([-1, 0.5, 2].map((x) => O.ouLine(x, t, 2)), ref.ou_line[j], 1e-12));
    let j = 0;
    for (const t of [0.1, 1]) for (const s of [1, -1]) close([0.2, 1, 2].map((x) => O.ouHalf(x, t, 2, s)), ref.ou_half[j++], 1e-12);
    close([0.1, 1, 5].map((t) => O.ouHlSurvival(t, 2)), ref.ou_hl_survival, 1e-12);
    close([0.01, 0.5, 2, 30].map(O.ouHlT), ref.ou_hl_T, 1e-9);
  });
  it('the QSD family q_ν', () => {
    [0.25, 0.5, 0.9, 1].forEach((nu, j) => rel([0.3, 2, 7, 40].map((x) => O.ouQsdNu(x, nu)), ref.ou_qsd_nu[j], 1e-9, `nu=${nu}`));
  });
  it('evolution from a Lomax start', () => {
    const { p, S } = O.ouHlEvolveLomax(0.5, [0.3, 2, 30], 12);
    rel(Array.from(p), ref.ou_hl_evolve_p, 2e-4, 'p');
    rel([S], ref.ou_hl_evolve_S, 1e-5, 'S');
  });
  it('interval spectra', () => {
    for (const walls of ['reflecting', 'absorbing', 'mixed'] as const) {
      [0.5, 1, 2, 2.5, 4].forEach((L, j) => {
        const want = ref[`ou_nu_${walls}`][j];
        rel(O.ouLevels(L, walls, 4), want, 2e-4, `${walls} L=${L}`);
      });
    }
  });
  it('interval QSDs, densities and exit times', () => {
    for (const walls of ['absorbing', 'mixed', 'reflecting'] as const) {
      [0.5, 1, 2, 4].forEach((L, j) => {
        const m = O.ouModes(L, walls);
        const f = m.forward(0);
        rel([0.1, 0.4, 0.8].map((u) => m.at(f, u * L)), ref[`ou_qsd_${walls}`][j], 2e-4, `${walls} L=${L}`);
      });
    }
    const m = O.ouModes(2, 'reflecting');
    [0.05, 0.3, 2].forEach((t, j) => {
      const p = m.density(1.5, t);
      // Modes near the grid scale carry an O(h²) error: absolute, against a peak of order 1.
      close([0.2, 1, 1.8].map((x) => m.at(p, x)), ref.ou_p_refl[j], 1e-4, `p t=${t}`);
    });
    // The script integrates e^{u²} erf u by trapezoids (≈ 2e−7 relative at ξ = 4); ouG sums the exact series.
    rel([0.5, 2, 4].map(O.ouG), ref.ou_G, 1e-6);
    [1, 2].forEach((L, j) => close([0.5, 1].map((x) => O.ouPiL(x, L)), ref.ou_piL[j], 1e-12));
    // T inherits the reference's trapezoid error in G(L) (≈ 2e−7 of G(4) ≈ 2×10⁶).
    [1, 2, 4].forEach((L, j) => close([0.25, 0.5].map((u) => O.ouT(u * L, L, 'absorbing')), ref.ou_T_abs[j], 1e-5));
    [1, 2, 4].forEach((L, j) => rel([0, 0.5].map((u) => O.ouT(u * L, L, 'mixed')), ref.ou_T_mixed[j], 1e-6));
    close([1, 2, 4].map(O.ouKramers), ref.ou_kramers, 1e-14);
  });
});

describe('circles', () => {
  it('free circle with drift', () => {
    let j = 0;
    for (const t of [0.005, 0.05]) for (const v of [0, 40]) close([0.1, 0.5, 0.8].map((x) => F.circle(x, t, 0.5, v)), ref.circ_images[j++], 1e-10);
  });
  it('sawtooth', () => {
    [1, 4].forEach((L, j) => close([0, 0.3, 0.9].map((x) => C.sawPs(x, L)), ref.saw_ps[j], 1e-12));
    rel([1, 3, 7].map(C.sawDeff), ref.saw_Deff, 1e-10);
    rel([1, 3].map(C.sawDeff), ref.saw_ring_Deff, 2e-4, 'independent ring discretisation');
  });
  it('cosine well: spectrum, transport and density', () => {
    [1, 3, 16].forEach((k, j) => {
      close(C.cosLevels(k, 'even', 5), ref.cos_spectrum_even[j], 2e-4, `even kappa=${k}`);
      close(C.cosLevels(k, 'odd', 4), ref.cos_spectrum_odd[j], 2e-4, `odd kappa=${k}`);
    });
    let j = 0;
    for (const k of [1, 3, 9])
      for (const f of [0, 0.5, 1, 2]) {
        const tr = C.cosTransport(k, f);
        close([tr.V, tr.Deff], ref.cos_transport[j++], 1e-8, `kappa=${k} f=${f}`);
      }
    const tr = C.cosTransport(3, 1);
    const L = 2 * Math.PI * C.cosR(3);
    close([0, 100, 200, 300].map((i) => C.cosDensity(tr, (i * L) / 400, 3)), ref.circle_ps_cos.slice(0, 4), 1e-6, 'ps');
  });
});

describe('which QSD a start reaches', () => {
  it('measured decay rates', () => {
    const r = attractionRates();
    close(r.drift, ref.attraction_drift, 1e-6, 'drift');
    close(r.ou, ref.attraction_ou, 2e-4, 'ou');
  });
});

describe('the overdamped limit; Parts IV and V', () => {
  it('damped oscillator against Smoluchowski', () => {
    [0.25, 1, 3, 10].forEach((z, j) => close([0.3, 1, 3].map((t) => P.dampedC(t, z)), ref.damped_C[j], 1e-12, `zeta=${z}`));
    close([0.3, 1, 2, 10].map(P.dampedRate), ref.damped_rate, 1e-12);
  });
  it('radial motion', () => {
    [1, 2, 3, 5, 2.5].forEach((d, j) => close([0.2, 1, 2.5].map((r) => P.radialP(r, 1, 1, d)), ref.radial_p[j], 1e-10, `d=${d}`));
    [1, 3].forEach((d, j) => rel([0.2, 1, 2.5].map((r) => P.radialP(r, 0.02, 1, d)), ref.radial_p_short[j], 1e-9, `short d=${d}`));
    [1, 2, 3, 5].forEach((d, j) => close([0.2, 1, 2.5].map((r) => P.radialPs(r, d)), ref.radial_ps[j], 1e-12, `ps d=${d}`));
    // Reaching a before R = 10⁶ (the notes' dots) differs from ever reaching a by O(R^(2−d)).
    close([3, 5].map((d) => P.radialReachBefore(0.5, 1, d)), [3, 5].map((d) => P.radialReach(0.5, 1, d)), 1e-6);
  });
  it('the soft V and the well with its own walls', () => {
    [0.25, 1, 4].forEach((s, j) => close([-1, 0.3, 2.5].map((x) => P.lncoshPs(x, s, s)), ref.lncosh_ps[j], 1e-12, `s=${s}`));
    [0, 1, 2, 3].forEach((n) => close([-1, 0.3, 2.5].map((x) => P.lncoshP(x, n, 3.5, 3.5)), ref.lncosh_P[n], 1e-12, `n=${n}`));
    [0.1, 0.5, 2, 8].forEach((s, j) => close([-0.4, 0, 0.3].map((x) => P.lncosPs(x, s, 1 / Math.PI)), ref.lncos_ps[j], 1e-12, `s=${s}`));
    [0.002, 0.02, 0.05].forEach((t, j) => close(Array.from(P.lncosP([-0.3, 0.1, 0.4], t, 0.35, 1, 1 / Math.PI)), ref.lncos_p[j], 1e-6, `t=${t}`));
  });
  it('the double well', () => {
    [2, 8, 16].forEach((b, j) => {
      const got = P.dwLevels(b, 4);
      // λ₀ = 0 is exact up to roundoff, ~ε‖T‖ ≈ 1e−10, which also bounds the absolute error of λ₁ (1.6e−6 at β = 16).
      expect(Math.abs(got[0])).toBeLessThan(1e-9);
      rel(got.slice(1), ref.dw_eigen[j].slice(1), 1e-4, `beta=${b}`);
    });
    rel([2, 8, 16].map((b) => P.dwKramers(b)), ref.dw_kramers, 1e-12);
    rel([2, 8, 14].map((b) => P.dwGardiner9218(b)), ref.dw_9218, 1e-6);
    // The time evolution uses a coarser grid (400 cells against the notes' 1500).
    close(Array.from(P.dwPopulation(8, [0.1, 10, 100, 1000])), ref.dw_population, 3e-3);
    [2, 8].forEach((b, j) => close([-0.5, 0, 0.3].map(P.dwCommittor(b)), ref.dw_committor[j], 1e-6, `beta=${b}`));
  });
});

describe('Part VI: interfaces and singular boundaries', () => {
  const CHIS = [1, 0.5, 0];
  it('a step in the diffusion coefficient', () => {
    [0.05, 0.3].forEach((t, i) =>
      CHIS.forEach((chi, j) => close([-0.8, -0.1, 0.3, 1.2].map((x) => S.ifaceP(x, t, 0.5, S.ifaceK(chi, 0.25), 0.25)), ref.iface_p[3 * i + j], 1e-12, `t=${t} chi=${chi}`)),
    );
    CHIS.forEach((chi, j) => {
      close([-0.7, 0, 0.4].map((x) => S.ifaceSplit(x, S.ifaceK(chi, 0.25), 0.25)), ref.iface_split[j], 1e-14, `split chi=${chi}`);
      close([-0.7, 0, 0.4].map((x) => S.ifaceT(x, S.ifaceK(chi, 0.25), 0.25)), ref.iface_T[j], 1e-14, `T chi=${chi}`);
    });
  });
  it('the square-root process', () => {
    const ys = [1e-4, 0.05, 0.6, 2.5];
    [0.25, 1].forEach((t, i) => {
      [3, 1.3, 0.5].forEach((d, j) => rel(ys.map((y) => S.sqrtP(y, t, 1, d, 1)), ref.sqrt_p_plus[3 * i + j], 1e-10, `p+ t=${t} d=${d}`));
      [1.3, 0, -1].forEach((d, j) => rel(ys.map((y) => S.sqrtP(y, t, 1, d, -1)), ref.sqrt_p_minus[3 * i + j], 1e-10, `p- t=${t} d=${d}`));
    });
    [1.5, 1, 0, -1].forEach((d, j) => rel([0.1, 0.5, 2].map((t) => S.sqrtS(t, 1, d)), ref.sqrt_S[j], 1e-12, `S d=${d}`));
    [1, 0].forEach((d, i) => [0.3, 1].forEach((f, j) => rel([0.1, 1, 5, 60].map((y) => S.sqrtQsd(y, f * (2 - d), d)), ref.sqrt_qsd[2 * i + j], 1e-10, `qsd d=${d}`)));
    rel([-1, 0, 1, 1.5].map((d) => S.sqrtT(1, d)), ref.sqrt_T, 1e-6);
  });
  it('a current through infinity', () => {
    rel([-1, 0, 1, 2.5].flatMap((I) => [1, 0.25].map((D) => S.qifT(I, D))), ref.qif_T, 1e-9);
    // p_s on a grid of spacing 0.002 with linear interpolation; beyond |x| = 30 the far field J/(x² + I).
    [-1, 0, 1].forEach((I, j) => rel([-4, -1, 0, 0.7, 3, 50].map((x) => S.qifPs(x, I, 1)), ref.qif_ps[j], 1e-4, `ps I=${I}`));
  });
});
