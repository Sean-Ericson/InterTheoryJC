"""Reference values from fpe-examples/scripts/make_figures.py for the Plot Lab tests.

    python -B scripts/notes_reference.py ../fpe-examples/scripts tests/fixtures/notes-reference.json

from plot-lab/ in the conda env `research`. -B keeps __pycache__ out of the notes' folder.
"""
import json
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
import make_figures as M  # noqa: E402

out = {}


def put(name, value):
    out[name] = np.asarray(value, float).tolist()


xs = np.array([0.1, 0.5, 0.9])
ts = np.array([0.01, 0.1, 1.0])
X0 = 0.3

# ---- free diffusion
put("images_refl", [[M.images(xs, t, X0, +1) for t in ts]])
put("images_abs", [[M.images(xs, t, X0, -1) for t in ts]])
put("refl_mean", M.refl_mean(ts, X0))
put("refl_second", M.refl_second(ts, X0))
put("refl_autocov_ratio", M.refl_autocov(np.array([0.1, 0.5])) / M.refl_autocov(np.array([0.0]))[0])
put("survival", M.survival(ts, X0))
put("flux0", M.exit_flux0(ts, X0))
put("fluxL", M.exit_fluxL(ts, X0))
put("mix_images", [M.mix_images(xs, t, X0) for t in ts])
put("mix_survival", M.mix_survival(ts, X0))
put("mix_flux", M.mix_flux(ts, X0))
th = np.array([0.05, 1.0, 20.0])
put("half_refl", [M.half_line(np.array([0.5, 1.0, 3.0]), t, 1.0, +1) for t in th])
put("hl_mean", M.hl_mean(th, 1.0))
put("hl_survival", M.hl_survival(th, 1.0))

# ---- drift
PE = [-4.0, -2.0, 0.0, 2.0, 4.0]
put("dr_stationary", [M.dr_stationary(xs, v) for v in PE])
put("dr_modes", [M.dr_modes(xs, t, X0, 4.0) for t in ts])
put("dr_mean", [M.dr_mean(np.array([0.05, 0.2]), X0, v) for v in PE])
put("da_qsd", [M.da_qsd(np.array([0.2, 0.5, 0.8]), v) for v in PE])
put("da_piL", [M.da_piL(np.array([0.2, 0.5, 0.8]), v) for v in PE])
put("da_T", [M.da_T(np.array([0.2, 0.5, 0.8]), v) for v in PE])
put("da_TL", [M.da_TL(np.array([0.2, 0.5, 0.8]), v) for v in PE])
put("da_survival", [M.da_survival(np.array([0.02, 0.1]), X0, v) for v in PE])
put("da_flux0", [M.da_flux0(np.array([0.02, 0.1]), X0, v) for v in PE])
VM = [-6.0, -2.0, -1.0, 0.0, 2.0, 6.0]
put("dm_lambda", [[m[0] for m in M.dm_modes(v)[:3]] for v in VM])
put("dm_qsd", [M.dm_qsd(np.array([0.2, 0.5, 0.8]), v) for v in PE])
put("dm_T", [M.dm_T(np.array([0.0, 0.5, 0.9]), v) for v in PE])
put("dhr_p", [M.dhr_p(np.array([0.2, 1.0, 3.0]), t, 1.0, -1.0) for t in [0.05, 1.0, 10.0]])
xm = np.linspace(0, 260, 52001)
put("dhr_mean", [[M.trapz(xm * M.dhr_p(xm, t, 1.0, v), xm) for t in [0.1, 1.0, 10.0, 100.0]] for v in [-1.0, -0.5, 0.0, 0.5, 1.0]])
put("dha_p", [M.dha_p(np.array([0.2, 1.0, 3.0]), t, 1.0, -1.0) for t in [0.3, 3.0, 30.0]])
put("dha_survival", [M.dha_survival(np.array([0.3, 3.0, 30.0]), 1.0, v) for v in [-1.0, 0.0, 1.0]])
put("inverse_gaussian", [M.inverse_gaussian(np.array([0.01, 1.0, 50.0]), 1.0, v) for v in [-1.0, 0.0, 1.0]])
put("dha_qsd", [M.dha_qsd(np.array([0.5, 2.0, 6.0]), f * 0.5, -1.0) for f in [0.2, 0.5, 0.8, 1.0]])
xa = np.linspace(0, 250, 50001)
cond = []
for beta in [0.25, 0.5, 0.75]:
    q, logS = M.dha_conditioned_exp(xa, 60.0, beta, -1.0)
    cond.append([float(np.interp(1.0, xa, q)), float(np.interp(5.0, xa, q)), float(np.interp(12.0, xa, q)), float(logS)])
put("dha_conditioned_exp", cond)
s_, drift, ou = M.attraction_rates()
put("attraction_drift", drift)
put("attraction_ou", ou)

# ---- OU
put("ou_line", [M.ou_line(np.array([-1.0, 0.5, 2.0]), t, 2.0) for t in [0.05, 1.0]])
put("ou_half", [M.ou_half(np.array([0.2, 1.0, 2.0]), t, 2.0, s) for t in [0.1, 1.0] for s in (+1, -1)])
put("ou_hl_survival", M.ou_hl_survival(np.array([0.1, 1.0, 5.0]), 2.0))
put("ou_hl_T", [M.ou_hl_T(v) for v in [0.01, 0.5, 2.0, 30.0]])
put("ou_qsd_nu", [M.ou_qsd_nu(np.array([0.3, 2.0, 7.0, 40.0]), nu) for nu in [0.25, 0.5, 0.9, 1.0]])
p, S = M.ou_hl_evolve(M.lomax(0.5), np.array([0.3, 2.0, 30.0]), 12.0)
put("ou_hl_evolve_p", p)
put("ou_hl_evolve_S", [S])
for kind in ["reflecting", "absorbing", "mixed"]:
    put(f"ou_nu_{kind}", [[m[0] for m in M.ou_modes(Lxi, kind)[:4]] for Lxi in [0.5, 1.0, 2.0, 2.5, 4.0]])
    put(f"ou_qsd_{kind}", [M.ou_qsd(np.array([0.1, 0.4, 0.8]) * Lxi, Lxi, kind) for Lxi in [0.5, 1.0, 2.0, 4.0]])
put("ou_p_refl", [M.ou_p(np.array([0.2, 1.0, 1.8]), t, 1.5, 2.0, "reflecting") for t in [0.05, 0.3, 2.0]])
put("ou_G", M.ou_G(np.array([0.5, 2.0, 4.0])))
put("ou_piL", [M.ou_piL(np.array([0.5, 1.0]), Lxi) for Lxi in [1.0, 2.0]])
put("ou_T_abs", [M.ou_T(np.array([0.25, 0.5]) * Lxi, Lxi, "absorbing") for Lxi in [1.0, 2.0, 4.0]])
put("ou_T_mixed", [M.ou_T(np.array([0.0, 0.5]) * Lxi, Lxi, "mixed") for Lxi in [1.0, 2.0, 4.0]])
put("ou_kramers", M.ou_kramers(np.array([1.0, 2.0, 4.0])))

# ---- circle
put("circ_images", [M.circ_images(np.array([0.1, 0.5, 0.8]), t, 0.5, v) for t in [0.005, 0.05] for v in [0.0, 40.0]])
put("saw_ps", [M.saw_ps(np.array([0.0, 0.3, 0.9]), Lc) for Lc in [1.0, 4.0]])
put("saw_Deff", M.saw_Deff(np.array([1.0, 3.0, 7.0])))
put("saw_ring_Deff", [M.saw_ring(Lc)[2] for Lc in [1.0, 3.0]])
put("cos_spectrum_even", [M.cos_spectrum(k)[0][:5] for k in [1.0, 3.0, 16.0]])
put("cos_spectrum_odd", [M.cos_spectrum(k)[1][:4] for k in [1.0, 3.0, 16.0]])
put("cos_transport", [[*M.cos_transport(k, f)[:2]] for k in [1.0, 3.0, 9.0] for f in [0.0, 0.5, 1.0, 2.0]])
x, ps, J = M.circle_ps(M.cos_phi(3.0, 1.0), 2 * np.pi * M.cos_R(3.0), n=400)
put("circle_ps_cos", [ps[0], ps[100], ps[200], ps[300], J])


# ---- toolkit: the overdamped limit; Parts IV and V
tt = np.array([0.3, 1.0, 3.0])
put("damped_C", [M.damped_C(tt, z) for z in [0.25, 1.0, 3.0, 10.0]])
put("damped_rate", M.damped_rate(np.array([0.3, 1.0, 2.0, 10.0])))
rr = np.array([0.2, 1.0, 2.5])
put("radial_p", [M.radial_p(rr, 1.0, 1.0, d) for d in [1.0, 2.0, 3.0, 5.0, 2.5]])
put("radial_p_short", [M.radial_p(rr, 0.02, 1.0, d) for d in [1.0, 3.0]])
put("radial_ps", [M.radial_ps(rr, d) for d in [1.0, 2.0, 3.0, 5.0]])
xx = np.array([-1.0, 0.3, 2.5])
put("lncosh_ps", [M.lncosh_ps(xx, s, s) for s in [0.25, 1.0, 4.0]])
put("lncosh_P", [M.lncosh_P(xx, n, 3.5, 3.5) for n in range(4)])
put("lncos_ps", [M.lncos_ps(np.array([-0.4, 0.0, 0.3]), s, 1 / np.pi) for s in [0.1, 0.5, 2.0, 8.0]])
put("lncos_p", [M.lncos_p(np.array([-0.3, 0.1, 0.4]), t, 0.35, 1.0, 1 / np.pi) for t in [0.002, 0.02, 0.05]])
put("dw_eigen", [M.dw_eigen(b, n=3000, m=4)[1] for b in [2.0, 8.0, 16.0]])
put("dw_kramers", [M.dw_kramers(b) for b in [2.0, 8.0, 16.0]])
put("dw_9218", [M.dw_gardiner_9218(b) for b in [2.0, 8.0, 14.0]])
put("dw_population", M.dw_population(8.0, np.array([0.1, 10.0, 100.0, 1000.0]))[0])
put("dw_committor", [M.dw_committor(b, np.array([-0.5, 0.0, 0.3])) for b in [2.0, 8.0]])

# ---- Part VI: interfaces and singular boundaries
CHIS = [1.0, 0.5, 0.0]
xi = np.array([-0.8, -0.1, 0.3, 1.2])
put("iface_p", [M.iface_p(xi, t, 0.5, M.iface_K(chi)) for t in [0.05, 0.3] for chi in CHIS])
put("iface_split", [M.iface_split(np.array([-0.7, 0.0, 0.4]), M.iface_K(chi)) for chi in CHIS])
put("iface_T", [M.iface_T(np.array([-0.7, 0.0, 0.4]), M.iface_K(chi)) for chi in CHIS])
yy = np.array([1e-4, 0.05, 0.6, 2.5])
put("sqrt_p_plus", [M.sqrt_p(yy, t, 1.0, d, +1) for t in [0.25, 1.0] for d in [3.0, 1.3, 0.5]])
put("sqrt_p_minus", [M.sqrt_p(yy, t, 1.0, d, -1) for t in [0.25, 1.0] for d in [1.3, 0.0, -1.0]])
put("sqrt_S", [M.sqrt_S(np.array([0.1, 0.5, 2.0]), 1.0, d) for d in [1.5, 1.0, 0.0, -1.0]])
put("sqrt_qsd", [M.sqrt_qsd(np.array([0.1, 1.0, 5.0, 60.0]), f * (2 - d), d) for d in [1.0, 0.0] for f in [0.3, 1.0]])
put("sqrt_T", [M.sqrt_T(1.0, d) for d in [-1.0, 0.0, 1.0, 1.5]])
put("qif_T", [M.qif_T(I, Dq) for I in [-1.0, 0.0, 1.0, 2.5] for Dq in [1.0, 0.25]])
put("qif_ps", [M.qif_ps(np.array([-4.0, -1.0, 0.0, 0.7, 3.0, 50.0]), I, 1.0) for I in [-1.0, 0.0, 1.0]])
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"{len(out)} reference arrays written")
