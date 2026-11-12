"""Figures for fpe-examples.tex, plus numerical checks of the formulas in it.

Part I is free diffusion, A = 0 and B = D, so d_t p = (D/2) d_x^2 p; Part II
adds a constant drift A = v; Part III is Ornstein-Uhlenbeck, A = -kx. Units:
D = 1, and L = 1 on the interval (x in units of L, t in units of L^2/D,
v = vL/D); on the half-line and the line the start x0 = 1 sets the length unit
instead. Part III uses k = 1, so ell = sqrt(D/k) = 1 and tau = 1/k = 1. The
circle sections (periodic) use the same units as the rest of their part.

    python scripts/make_figures.py                 # every section
    python scripts/make_figures.py mixed halfline  # just these sections

writes PDFs to ../figures and prints each check next to its expected value.
Section names are the keys of SECTIONS at the bottom. Uses numpy, scipy and
matplotlib (the conda env `research` has all three).
"""

import sys
from functools import cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse
from scipy.integrate import cumulative_trapezoid, quad, simpson
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from scipy.sparse.linalg import spsolve, splu
from scipy.special import beta as beta_fn
from scipy.special import (erf, erfc, erfcx, erfi, erfinv, eval_gegenbauer, eval_genlaguerre, eval_jacobi, gamma, gammainc,
                           hyp1f1, i0, ive, jv, log_ndtr)

L, D = 1.0, 1.0
X0 = 0.3  # starting point used throughout the figures
FIG = Path(__file__).resolve().parent.parent / "figures"

# --- style -------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Sequential blue for time snapshots, early (light) to late (dark).
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SNAP_T = [0.01, 0.03, 0.1, 0.3, 1.0]
# Entities keep one color across figures: exit through 0, exit through L.
C_EXIT0, C_EXITL = BLUE, ORANGE

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.labelcolor": INK2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, ha="right", va="bottom", color=INK, fontsize=9)


def snapshot_legend(ax, title=r"$Dt/L^2$", **kw):
    kw = {"loc": "upper right", "bbox_to_anchor": (1.02, 1.04), **kw}
    leg = ax.legend(handlelength=1.5, title=title, title_fontsize=7.5, **kw)
    leg.get_title().set_color(INK2)


def mc_dots(ax, x, y, color, label=None, **kw):
    """Monte Carlo estimates: filled dots with a white ring so lines stay visible."""
    ax.plot(x, y, "o", ms=4, mfc=color, mec="white", mew=0.8, color=color, label=label, zorder=3, **kw)


# --- shared pieces -----------------------------------------------------------
def lam(n):
    """Eigenvalues for both cases: lambda_n = (D/2)(n pi/L)^2."""
    return 0.5 * D * (n * np.pi / L) ** 2


def heat(y, t):
    """Free kernel of d_t p = (D/2) d_x^2 p: Gaussian of variance D t."""
    return np.exp(-(y**2) / (2 * D * t)) / np.sqrt(2 * np.pi * D * t)


def images(x, t, x0, sign, M=40):
    """Method of images: sign = +1 reflecting, -1 absorbing."""
    m = np.arange(-M, M + 1)[:, None]
    return (heat(x - x0 - 2 * m * L, t) + sign * heat(x + x0 - 2 * m * L, t)).sum(0)


def modes(x, t, x0, kind, N=400):
    """Eigenfunction expansion of p(x, t | x0, 0)."""
    n = np.arange(1, N + 1)[:, None]
    k = n * np.pi / L
    if kind == "reflecting":
        return 1 / L + (2 / L) * (np.cos(k * x) * np.cos(k * x0) * np.exp(-lam(n) * t)).sum(0)
    return (2 / L) * (np.sin(k * x) * np.sin(k * x0) * np.exp(-lam(n) * t)).sum(0)


def trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def check(name, got, want, tol):
    """Pass if |got - want| <= tol, relative to |want| once that exceeds 1."""
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {got:.6g} (expected {want:.6g})")
    return ok


# --- reflecting walls --------------------------------------------------------
def refl_mean(t, x0, N=4001):
    n = np.arange(1, N + 1, 2)[:, None]
    return L / 2 - (4 * L / np.pi**2) * (np.cos(n * np.pi * x0 / L) * np.exp(-lam(n) * t) / n**2).sum(0)


def refl_second(t, x0, N=4000):
    n = np.arange(1, N + 1)[:, None]
    terms = (-1.0) ** n * np.cos(n * np.pi * x0 / L) * np.exp(-lam(n) * t) / n**2
    return L**2 / 3 + (4 * L**2 / np.pi**2) * terms.sum(0)


def refl_autocov(tau, N=401):
    """Stationary covariance <x(tau), x(0)>_s, Gardiner (5.2.110)."""
    n = np.arange(1, N + 1, 2)[:, None]
    return (8 * L**2 / np.pi**4) * (np.exp(-lam(n) * tau) / n**4).sum(0)


# --- absorbing walls ---------------------------------------------------------
def survival(t, x0, N=4001):
    n = np.arange(1, N + 1, 2)[:, None]
    return (4 / np.pi) * (np.sin(n * np.pi * x0 / L) * np.exp(-lam(n) * t) / n).sum(0)


def exit_flux0(t, x0, M=40):
    """Probability current out through x = 0, by images."""
    m = np.arange(-M, M + 1)[:, None]
    y = x0 + 2 * m * L
    return (y / np.sqrt(2 * np.pi * D * t**3) * np.exp(-(y**2) / (2 * D * t))).sum(0)


def exit_fluxL(t, x0, **kw):
    return exit_flux0(t, L - x0, **kw)


def T_mean(x0):
    return x0 * (L - x0) / D


def T_var(x0):
    return x0 * (L - x0) * (x0**2 + (L - x0) ** 2) / (3 * D**2)


def T_given0(x0):
    return x0 * (2 * L - x0) / (3 * D)


def T_givenL(x0):
    return (L**2 - x0**2) / (3 * D)


def qsd(x):
    return np.pi / (2 * L) * np.sin(np.pi * x / L)


def simulate_exits(x0s, n_paths, dt, rng, lower="absorbing"):
    """Euler steps of dx = sqrt(D) dW with a Brownian-bridge test for crossings
    between steps; a reflecting wall at 0 folds the path, x -> |x|. Returns
    start index, exit time, and wall (0 or 1 for L)."""
    x = np.repeat(np.asarray(x0s, float), n_paths)
    start = np.repeat(np.arange(len(x0s)), n_paths)
    t_exit = np.full(x.size, np.nan)
    wall = np.zeros(x.size, int)
    alive = np.arange(x.size)
    t, s = 0.0, np.sqrt(D * dt)
    while alive.size:
        t += dt
        xa = x[alive]
        xb = xa + s * rng.standard_normal(alive.size)
        if lower == "reflecting":
            xb = np.abs(xb)
        inside = (xb > 0) & (xb < L)
        p0 = np.where(inside, np.exp(-2 * xa * np.clip(xb, 0, None) / (D * dt)), 0.0)
        if lower == "reflecting":
            p0[:] = 0.0
        pL = np.where(inside, np.exp(-2 * (L - xa) * np.clip(L - xb, 0, None) / (D * dt)), 0.0)
        u = rng.random(alive.size)
        out0 = (xb <= 0) | (inside & (u < p0))
        outL = (xb >= L) | (inside & ~out0 & (u < p0 + pL))
        done = out0 | outL
        t_exit[alive[done]] = t - dt / 2
        wall[alive[done]] = outL[done]
        x[alive] = xb
        alive = alive[~done]
    return start, t_exit, wall


# --- reflecting at 0, absorbing at L ------------------------------------------
def lam_mix(n):
    """lambda_n = (D/2) k_n^2 with k_n = (n + 1/2) pi / L, n = 0, 1, ..."""
    return 0.5 * D * ((n + 0.5) * np.pi / L) ** 2


def mix_images(x, t, x0, M=40):
    m = np.arange(-M, M + 1)[:, None]
    return ((-1.0) ** m * (heat(x - x0 + 2 * m * L, t) + heat(x + x0 + 2 * m * L, t))).sum(0)


def mix_modes(x, t, x0, N=400):
    n = np.arange(N)[:, None]
    k = (n + 0.5) * np.pi / L
    return (2 / L) * (np.cos(k * x) * np.cos(k * x0) * np.exp(-lam_mix(n) * t)).sum(0)


def mix_survival(t, x0, N=4000):
    n = np.arange(N)[:, None]
    terms = (-1.0) ** n / (2 * n + 1) * np.cos((2 * n + 1) * np.pi * x0 / (2 * L)) * np.exp(-lam_mix(n) * t)
    return (4 / np.pi) * terms.sum(0)


def mix_flux(t, x0, M=40):
    """Exit-time density (every exit is at L), by images."""
    f = np.zeros_like(t)
    for m in range(-M, M + 1):
        y = (2 * m + 1) * L - x0
        f += (-1.0) ** m * y / np.sqrt(2 * np.pi * D * t**3) * np.exp(-(y**2) / (2 * D * t))
    return f


def mix_T(x0):
    return (L**2 - x0**2) / D


def mix_T_var(x0):
    return 2 * (L**4 - x0**4) / (3 * D**2)


def mix_qsd(x):
    return np.pi / (2 * L) * np.cos(np.pi * x / (2 * L))


# --- half-line (0, inf) and the whole line; x0 = 1 is the length unit ---------
X0H = 1.0
SNAP_TH = [0.05, 0.2, 1.0, 5.0, 20.0]


def half_line(x, t, x0, sign):
    """sign = +1 reflecting, -1 absorbing at x = 0."""
    return heat(x - x0, t) + sign * heat(x + x0, t)


def hl_mean(t, x0):
    """<x(t)> with reflection at 0: mean of the folded normal |x0 + sqrt(Dt) Z|."""
    s = np.sqrt(D * t)
    return x0 * erf(x0 / (np.sqrt(2) * s)) + s * np.sqrt(2 / np.pi) * np.exp(-(x0**2) / (2 * s**2))


def hl_survival(t, x0):
    return erf(x0 / np.sqrt(2 * D * t))


def levy_smirnov(t, a):
    """First-passage density to a level a distance a away (one wall, no other)."""
    return a / np.sqrt(2 * np.pi * D * t**3) * np.exp(-(a**2) / (2 * D * t))


# --- constant drift A = v, B = D ----------------------------------------------
# p = exp(a(x - x0) - v^2 t/2D) u with a = v/D, where u diffuses freely: an
# absorbing wall stays u = 0, a reflecting wall becomes u' = a u.
PE = [-4.0, -2.0, 0.0, 2.0, 4.0]  # vL/D on the interval, vx0/D on the half-line
V_HALF = [-1.0, -0.5, 0.0, 0.5, 1.0]
# Diverging: blue for drift toward x = 0 (v < 0), red for drift toward +x, grey for none.
DIVERGING = ["#1c5cab", "#6da7ec", MUTED, "#f08a85", "#b0302f"]


def coth(z):
    return 1 / np.tanh(z)


def fpe_residual(pfun, x, t, v, h=2e-4, dt=1e-5):
    """max |d_t p + v p' - (D/2) p''| / max |p| by central differences."""
    p = pfun(x, t)
    pt = (pfun(x, t + dt) - pfun(x, t - dt)) / (2 * dt)
    px = (pfun(x + h, t) - pfun(x - h, t)) / (2 * h)
    pxx = (pfun(x + h, t) - 2 * p + pfun(x - h, t)) / h**2
    return np.max(np.abs(pt + v * px - 0.5 * D * pxx)) / np.max(np.abs(p))


def current(pfun, x, t, v, h=1e-6):
    return v * pfun(x, t) - 0.5 * D * (pfun(x + h, t) - pfun(x - h, t)) / (2 * h)


# interval, both walls reflecting
def dr_stationary(x, v):
    a = v / D
    if abs(a) < 1e-12:
        return np.full_like(np.asarray(x, float), 1 / L)
    if a > 0:
        return 2 * a * np.exp(2 * a * (x - L)) / -np.expm1(-2 * a * L)
    return -2 * a * np.exp(2 * a * x) / -np.expm1(2 * a * L)


def dr_mean_s(v):
    a = v / D
    return L / 2 if abs(a) < 1e-12 else L / -np.expm1(-2 * a * L) - 1 / (2 * a)


def dr_phi(y, k, a):
    return np.cos(k * y) + (a / k) * np.sin(k * y)


def dr_modes(x, t, x0, v, N=400):
    a = v / D
    k = np.arange(1, N + 1)[:, None] * np.pi / L
    series = (dr_phi(x, k, a) * dr_phi(x0, k, a) / (1 + (a / k) ** 2) * np.exp(-0.5 * D * k**2 * t)).sum(0)
    return dr_stationary(x, v) + (2 / L) * np.exp(a * (x - x0) - v**2 * t / (2 * D)) * series


def dr_mean(t, x0, v, N=400):
    a = v / D
    xs = np.linspace(0, L, 4001)
    k = np.arange(1, N + 1)[:, None] * np.pi / L
    c = trapz_rows(xs * np.exp(a * xs) * dr_phi(xs, k, a), xs)
    w = c * dr_phi(x0, k[:, 0], a) / (1 + (a / k[:, 0]) ** 2)
    decay = np.exp(-(v**2 / (2 * D) + 0.5 * D * k**2) * t)
    return dr_mean_s(v) + (2 / L) * np.exp(-a * x0) * (w[:, None] * decay).sum(0)


def trapz_rows(y, x):
    return np.sum(0.5 * (y[:, 1:] + y[:, :-1]) * np.diff(x), axis=1)


# interval, both walls absorbing
def da_p(x, t, x0, v):
    return np.exp(v / D * (x - x0) - v**2 * t / (2 * D)) * images(x, t, x0, -1)


def da_survival(t, x0, v, N=4000):
    a = v / D
    n = np.arange(1, N + 1)[:, None]
    k = n * np.pi / L
    terms = np.sin(k * x0) * k * (1 - (-1.0) ** n * np.exp(a * L)) / (a**2 + k**2) * np.exp(-0.5 * D * k**2 * t)
    return (2 / L) * np.exp(-a * x0 - v**2 * t / (2 * D)) * terms.sum(0)


def da_flux0(t, x0, v, **kw):
    return np.exp(-v / D * x0 - v**2 * t / (2 * D)) * exit_flux0(t, x0, **kw)


def da_fluxL(t, x0, v, **kw):
    return np.exp(v / D * (L - x0) - v**2 * t / (2 * D)) * exit_fluxL(t, x0, **kw)


def da_piL(x0, v):
    a = v / D
    return x0 / L if abs(a) < 1e-12 else np.expm1(-2 * a * x0) / np.expm1(-2 * a * L)


def da_T(x0, v):
    return T_mean(x0) if abs(v) < 1e-12 else (L * da_piL(x0, v) - x0) / v


def da_TL(x0, v):
    a = v / D
    return T_givenL(x0) if abs(a) < 1e-12 else (L * coth(a * L) - x0 * coth(a * x0)) / v


def da_T0(x0, v):
    return da_TL(L - x0, -v)


def da_qsd(x, v):
    a, k = v / D, np.pi / L
    return np.exp(a * x) * np.sin(k * x) * (a**2 + k**2) / (k * (1 + np.exp(a * L)))


def da_lam1(v):
    return v**2 / (2 * D) + np.pi**2 * D / (2 * L**2)


# interval, reflecting at 0 and absorbing at L
@cache
def dm_modes(v, N=200):
    """Eigenpairs (lambda_n, u_n) in ascending order. Real k_n solve
    k cos kL + a sin kL = 0; for aL < -1 there is also a bound state
    u = cosh qx + (a/q) sinh qx with q cosh qL + a sinh qL = 0."""
    a = v / D
    g = lambda k: k * np.cos(k * L) + a * np.sin(k * L)
    grid = np.linspace(1e-9, (N + 2) * np.pi / L, 400 * (N + 2))
    vals = g(grid)
    idx = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    ks = [brentq(g, grid[i], grid[i + 1]) for i in idx][:N]
    modes = []
    if a * L < -1:
        q = brentq(lambda q: q * np.cosh(q * L) + a * np.sinh(q * L), 1e-9, abs(a))
        modes.append((v**2 / (2 * D) - 0.5 * D * q**2, lambda y, q=q: np.cosh(q * y) + (a / q) * np.sinh(q * y)))
    for k in ks:
        modes.append((v**2 / (2 * D) + 0.5 * D * k**2, lambda y, k=k: dr_phi(y, k, a)))
    xs = np.linspace(0, L, 20001)
    return [(lam_n, u, trapz(u(xs) ** 2, xs)) for lam_n, u in modes]


def dm_p(x, t, x0, v, N=60):
    a = v / D
    total = sum(u(x) * u(x0) / nrm * np.exp(-lam_n * t) for lam_n, u, nrm in dm_modes(v)[:N])
    return np.exp(a * (x - x0)) * total


def dm_survival_coeffs(x0, v, N=200):
    """S(t) = sum_n c_n exp(-lambda_n t)."""
    a = v / D
    xs = np.linspace(0, L, 20001)
    modes = dm_modes(v)[:N]
    lams = np.array([m[0] for m in modes])
    c = np.array([np.exp(-a * x0) * u(x0) * trapz(np.exp(a * xs) * u(xs), xs) / nrm for _, u, nrm in modes])
    return lams, c


def dm_T(x0, v):
    a = v / D
    return mix_T(x0) if abs(a) < 1e-12 else ((L - x0) + (np.exp(-2 * a * L) - np.exp(-2 * a * x0)) / (2 * a)) / v


def dm_qsd(x, v):
    a = v / D
    _, u, _ = dm_modes(v)[0]
    xs = np.linspace(0, L, 20001)
    return np.exp(a * x) * u(x) / trapz(np.exp(a * xs) * u(xs), xs)


# half-line, reflecting at 0
def dhr_p(x, t, x0, v):
    a, s = v / D, np.sqrt(2 * D * t)
    z = (x + x0 + v * t) / s
    with np.errstate(over="ignore", invalid="ignore"):
        third = np.where(z > 0, erfcx(np.abs(z)) * np.exp(2 * a * x - z**2), np.exp(2 * a * x) * erfc(z))
    return heat(x - x0 - v * t, t) + np.exp(-2 * a * x0) * heat(x + x0 - v * t, t) - a * third


# half-line, absorbing at 0
def dha_p(x, t, x0, v):
    return heat(x - x0 - v * t, t) - np.exp(-2 * v / D * x0) * heat(x + x0 - v * t, t)


def dha_survival(t, x0, v):
    s = np.sqrt(2 * D * t)
    return 0.5 * (erfc(-(x0 + v * t) / s) - np.exp(-2 * v / D * x0) * erfc((x0 - v * t) / s))


def inverse_gaussian(t, x0, v):
    """First-passage density to 0 from x0 > 0 with drift v (v < 0 is toward 0)."""
    return x0 / np.sqrt(2 * np.pi * D * t**3) * np.exp(-((x0 + v * t) ** 2) / (2 * D * t))


def dha_qsd(x, lam_, v):
    """Quasi-stationary density with decay rate 0 < lam_ <= v^2/2D (v < 0)."""
    disc = v**2 - 2 * D * lam_
    if disc < 1e-12:
        a = v / D
        return a**2 * x * np.exp(a * x)
    rp, rm = (v + np.sqrt(disc)) / D, (v - np.sqrt(disc)) / D
    return (np.exp(rp * x) - np.exp(rm * x)) / (1 / rm - 1 / rp)


def dha_lam_of_tail(beta, v):
    """Decay rate selected by an initial tail e^{-beta x} (v < 0): beta ell (2 - beta ell)/2 tau, up to the edge."""
    b = min(beta * D / abs(v), 1.0)
    return b * (2 - b) * v**2 / (2 * D)


def dha_log_p_exp(x, t, beta0, v):
    """log p(x,t) for v < 0 from p0 = beta0 e^{-beta0 y}: the gauge carries the free half-line over, and the
    Gaussian integrals over the gauged start w0 = beta0 e^{g y}, g = -v/D - beta0, are done in closed form."""
    a = v / D
    g = -a - beta0
    sd = np.sqrt(D * t)
    A1 = g * x + 0.5 * g**2 * D * t + log_ndtr((x + g * D * t) / sd)
    A2 = -g * x + 0.5 * g**2 * D * t + log_ndtr((g * D * t - x) / sd)
    with np.errstate(divide="ignore"):
        return a * x - v**2 * t / (2 * D) + np.log(beta0) + A1 + np.log(-np.expm1(A2 - A1))


def dha_conditioned_exp(x, t, beta0, v):
    """Conditioned density q and log S from the exponential start, on a grid x starting at 0."""
    lp = dha_log_p_exp(x, t, beta0, v)
    m = lp.max()
    Z = trapz(np.exp(lp - m), x)
    return np.exp(lp - m) / Z, np.log(Z) + m


# --- Ornstein-Uhlenbeck A = -kx, B = D -----------------------------------------
# ell = sqrt(D/k), tau = 1/k, xi = x/ell. The wall at x = 0 sits at the bottom of
# the well. Gardiner's substitution y = x e^{kt} (4.4.23) maps the process onto
# free diffusion run for the time theta(t) = (e^{2kt} - 1)/2k, so walls at 0
# carry the images of Part I over unchanged.
K = 1.0
ELL, TAU = np.sqrt(D / K), 1 / K
X0_OU = 2.0  # start for the line and half-line figures (in units of ell)
L_OU = [0.5, 1.0, 2.0, 4.0]  # interval lengths L/ell in the figures
OU_WALLS = {"reflecting": ("even", 1), "absorbing": ("odd", 0), "mixed": ("even", 0)}


def ou_theta(t):
    return np.expm1(2 * K * t) / (2 * K)


def ou_mean_var(t, x0):
    """Gardiner (4.4.27) and (4.4.29) for a sharp start."""
    return x0 * np.exp(-K * t), 0.5 * ELL**2 * -np.expm1(-2 * K * t)


def gauss(x, m, var):
    return np.exp(-((x - m) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def ou_line(x, t, x0):
    m, var = ou_mean_var(t, x0)
    return gauss(x, m, var)


def ou_half(x, t, x0, sign):
    """Half-line with the wall at the bottom of the well: +1 reflecting, -1 absorbing."""
    m, var = ou_mean_var(t, x0)
    return gauss(x, m, var) + sign * gauss(x, -m, var)


def ou_mehler(x, t, x0, N=80):
    """p_s(x) sum_n h_n(xi) h_n(xi0) e^{-nkt} with normalised Hermite polynomials h_n."""
    xi, xi0 = x / ELL, x0 / ELL
    h_prev, h = np.ones_like(xi), np.sqrt(2) * xi
    g_prev, g = 1.0, np.sqrt(2) * xi0
    total = h_prev * g_prev + h * g * np.exp(-K * t)
    for n in range(1, N):
        h_prev, h = h, np.sqrt(2 / (n + 1)) * xi * h - np.sqrt(n / (n + 1)) * h_prev
        g_prev, g = g, np.sqrt(2 / (n + 1)) * xi0 * g - np.sqrt(n / (n + 1)) * g_prev
        total = total + h * g * np.exp(-(n + 1) * K * t)
    return np.exp(-(xi**2)) / (ELL * np.sqrt(np.pi)) * total


def ou_hl_survival(t, x0):
    return erf(x0 / ELL / np.sqrt(np.expm1(2 * K * t)))


def ou_hl_T(x0):
    """Mean time to reach the bottom of the well, Gardiner (5.2.161) with b -> infinity."""
    return np.sqrt(np.pi) * TAU * quad(erfcx, 0, x0 / ELL, limit=500, epsabs=1e-13, epsrel=1e-12)[0]


def ou_hl_qsd(x):
    return 2 * x / ELL**2 * np.exp(-((x / ELL) ** 2))


def ou_qsd_nu(x, nu):
    """QSD with decay rate nu k, 0 < nu <= 1: (2 nu/ell) xi e^{-xi^2} M((1-nu)/2, 3/2, xi^2), written with
    Kummer's transformation so that it does not overflow. Tail ~ x^{-1-nu}; nu = 1 is the Rayleigh density."""
    xi = x / ELL
    return 2 * nu / ELL * xi * hyp1f1(1 + nu / 2, 1.5, -(xi**2))


def ou_hl_evolve(p0, x, t, ny=60001):
    """Density and survival on the half-line absorbed at the bottom, from an initial density p0, through the
    map onto the free half-line. The y-integral runs over log y, far enough to catch power-law tails."""
    s2 = D * ou_theta(t)
    u = np.linspace(np.log(1e-9), np.log(1e60), ny)
    y = np.exp(u)
    w = p0(y) * y
    p = [trapz(w * np.exp(-((X - y) ** 2) / (2 * s2)) * -np.expm1(-2 * X * y / s2), u) / np.sqrt(2 * np.pi * s2)
         for X in np.exp(K * t) * np.atleast_1d(x)]
    return np.exp(K * t) * np.array(p), trapz(w * erf(y / np.sqrt(2 * s2)), u)


def lomax(nu0):
    """Initial density nu0 (1 + x/ell)^{-1-nu0}/ell, tail ~ x^{-1-nu0}."""
    return lambda y: nu0 / ELL * (1 + y / ELL) ** (-1 - nu0)


def ou_Q(nu, xi, parity):
    """Backward eigenfunction of Q'' - 2 xi Q' + 2 nu Q = 0 (lambda = nu k) and dQ/dxi.
    even: Q'(0) = 0 (reflecting at the bottom); odd: Q(0) = 0 (absorbing there)."""
    z = xi**2
    if parity == "even":
        a = -nu / 2
        return hyp1f1(a, 0.5, z), 4 * a * xi * hyp1f1(a + 1, 1.5, z)
    a = (1 - nu) / 2
    return xi * hyp1f1(a, 1.5, z), hyp1f1(a, 1.5, z) + (4 * a / 3) * z * hyp1f1(a + 1, 2.5, z)


@cache
def ou_modes(Lxi, kind, N=60):
    """[(nu_n, Q_n, norm_n)] on (0, L) with L = Lxi ell, ascending; norm_n = int_0^L e^{-xi^2} Q_n^2 dx."""
    parity, which = OU_WALLS[kind]
    f = lambda sv: ou_Q(sv**2, Lxi, parity)[which]
    smax = np.sqrt((N + 2) ** 2 * np.pi**2 / (2 * Lxi**2) + 2 * (N + 2) + 10)
    grid = np.concatenate([np.logspace(-8, -1, 300), np.linspace(0.1001, smax, 200 * (N + 2))])
    vals = f(grid)
    idx = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    nus = [brentq(f, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-14) ** 2 for i in idx]
    if kind == "reflecting":
        nus = [0.0] + [nu for nu in nus if nu > 1e-6]
    nus = nus[:N]
    xi = np.linspace(0, Lxi, 8001)
    w = np.exp(-(xi**2))
    out = []
    for nu in nus:
        Qf = lambda y, nu=nu: ou_Q(nu, y, parity)[0]
        out.append((nu, Qf, ELL * trapz(w * Qf(xi) ** 2, xi)))
    return out


def ou_p(x, t, x0, Lxi, kind, N=60):
    xi, xi0 = x / ELL, x0 / ELL
    total = sum(Qf(xi) * Qf(xi0) / nrm * np.exp(-nu * K * t) for nu, Qf, nrm in ou_modes(Lxi, kind)[:N])
    return np.exp(-(xi**2)) * total


def ou_survival_coeffs(x0, Lxi, kind, N=60):
    """S(t) = sum_n c_n exp(-nu_n k t)."""
    xi = np.linspace(0, Lxi, 8001)
    w = np.exp(-(xi**2))
    modes = ou_modes(Lxi, kind)[:N]
    nus = np.array([m[0] for m in modes])
    c = np.array([Qf(x0 / ELL) * ELL * trapz(w * Qf(xi), xi) / nrm for _, Qf, nrm in modes])
    return nus, c


def ou_qsd(x, Lxi, kind):
    _, Qf, _ = ou_modes(Lxi, kind)[0]
    xi = np.linspace(0, Lxi, 8001)
    return np.exp(-((x / ELL) ** 2)) * Qf(x / ELL) / (ELL * trapz(np.exp(-(xi**2)) * Qf(xi), xi))


def ou_G(xi):
    """G(xi) = sqrt(pi) int_0^xi e^{u^2} erf(u) du, so that F(x) = tau G(x/ell)."""
    xi = np.atleast_1d(np.asarray(xi, float))
    out = np.empty_like(xi)
    for i, X in enumerate(xi):
        u = np.linspace(0, X, 20001)
        out[i] = np.sqrt(np.pi) * trapz(np.exp(u**2) * erf(u), u)
    return out


def ou_piL(x0, Lxi):
    return erfi(x0 / ELL) / erfi(Lxi)


def ou_T(x0, Lxi, kind):
    """Mean exit time: absorbing T = pi_L F(L) - F(x0); mixed T = F(L) - F(x0)."""
    GL, G0 = ou_G(Lxi)[0], ou_G(np.asarray(x0) / ELL)
    if kind == "absorbing":
        return TAU * (ou_piL(np.asarray(x0), Lxi) * GL - G0)
    return TAU * (GL - G0)


def ou_kramers(Lxi):
    """Leading Arrhenius-Kramers rate for the half-well, in units of k."""
    return 2 * Lxi / np.sqrt(np.pi) * np.exp(-(Lxi**2))


def ou_residual(pfun, x, t, h=2e-4, dt=1e-5):
    """max |d_t p - d_x(kxp) - (D/2) p''| / max |p|."""
    p_ = pfun(x, t)
    pt = (pfun(x, t + dt) - pfun(x, t - dt)) / (2 * dt)
    px = (pfun(x + h, t) - pfun(x - h, t)) / (2 * h)
    pxx = (pfun(x + h, t) - 2 * p_ + pfun(x - h, t)) / h**2
    return np.max(np.abs(pt - K * p_ - K * x * px - 0.5 * D * pxx)) / np.max(np.abs(p_))


def ou_current(pfun, x, t, h=1e-6):
    return -K * x * pfun(x, t) - 0.5 * D * (pfun(x + h, t) - pfun(x - h, t)) / (2 * h)


# --- symmetrised form: p = sqrt(p_s) psi turns -L into H = -(D/2) d^2 + V ---------
# with V = A^2/2D + A'/2; an absorbing wall becomes psi = 0, a reflecting one psi' = (A/D) psi.
def h_interval(V, a, b, n, robin=(None, None), m=6):
    """Lowest m eigenvalues of H on [a, b] by second-order finite differences. robin[i] = c imposes
    psi' = c psi at that end (ghost point), None imposes psi = 0."""
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    lo, hi = (0 if robin[0] is not None else 1), (n if robin[1] is not None else n - 1)
    d = D / h**2 + V(x[lo:hi + 1])
    e = np.full(hi - lo, -0.5 * D / h**2)
    if robin[0] is not None:
        d[0] += D * robin[0] / h
        e[0] = -D / (np.sqrt(2) * h**2)  # symmetrised with trapezoid weight 1/2 on the end node
    if robin[1] is not None:
        d[-1] -= D * robin[1] / h
        e[-1] = -D / (np.sqrt(2) * h**2)
    return eigh_tridiagonal(d, e, eigvals_only=True, select="i", select_range=(0, m - 1))


def h_circle(V, Lc, n, delta=None, m=8):
    """Lowest m eigenvalues of H on a circle of length Lc (nodes -Lc/2 + i h). delta = g adds g delta(x - Lc/2),
    a spike g/h on the node at the antipode."""
    h = Lc / n
    x = -Lc / 2 + h * np.arange(n)
    M = np.diag(D / h**2 + V(x)) - 0.5 * D / h**2 * (np.eye(n, k=1) + np.eye(n, k=-1))
    M[0, -1] = M[-1, 0] = -0.5 * D / h**2
    if delta is not None:
        M[0, 0] += delta / h
    return np.linalg.eigvalsh(M)[:m]


def V_ou(x):
    return 0.5 * K * ((x / ELL) ** 2 - 1)


def V_cos(kappa):
    R = cos_R(kappa)
    return lambda x: K * kappa / 4 * np.sin(x / R) ** 2 - 0.5 * K * np.cos(x / R)


# --- the circle ----------------------------------------------------------------
# S^1 = R/LZ has no boundary: A, B and p are L-periodic and nothing is imposed
# anywhere. The propagator is the line's, summed over the lifts x -> x + nL, all
# with a plus sign. The lift's transport (V, D_eff) comes from the tilted
# generator, d/dx -> d/dx - s on periodic functions: V = Lambda'(0), D_eff = Lambda''(0).
SNAP_C = [0.002, 0.005, 0.01, 0.02, 0.05]
SAW_L = [1.0, 2.0, 3.0, 4.0]  # circumferences L/ell of the sawtooth circle
KAPPAS = [1.0, 3.0, 9.0]  # cosine well, kappa = 2R^2/ell^2


def circ_images(x, t, x0, v=0.0, M=30):
    n = np.arange(-M, M + 1)[:, None]
    return heat(x - x0 - v * t + n * L, t).sum(0)


def circ_modes(x, t, x0, v=0.0, N=400):
    """Fourier modes e^{iqx}, lambda = D q^2/2 + i v q, q = 2 pi m/L."""
    q = 2 * np.pi * np.arange(1, N + 1)[:, None] / L
    return 1 / L + (2 / L) * (np.cos(q * (x - x0 - v * t)) * np.exp(-0.5 * D * q**2 * t)).sum(0)


def circle_ps(phi, Lc, n=400, ns=4001):
    """Cut-invariant stationary state for B = D on a circle of length Lc:
    p_s(x) = (N/D) int_x^{x+L} e^{phi(x) - phi(y)} dy and J_s = N (1 - e^{-Phi})/2, with
    phi = int 2A/D on the cover and Phi = phi(x + L) - phi(x). Returns a periodic grid, p_s, J_s."""
    x = np.arange(n) * Lc / n
    s = np.linspace(0, Lc, ns)
    I = simpson(np.exp(phi(x)[:, None] - phi(x[:, None] + s)), x=s, axis=1)
    N = D / (I.mean() * Lc)
    return x, N * I / D, -N * np.expm1(phi(0.0) - phi(Lc)) / 2


def gardiner_ps(phi, Lc, x, n=200001):
    """Gardiner (5.2.47) with the seam at a = 0 and B = D, normalised on the periodic grid x."""
    y = np.linspace(0, Lc, n)
    w = np.exp(phi(0.0) - phi(y))  # 1/psi with psi(a) = 1
    left = cumulative_trapezoid(w, y, initial=0)
    right = cumulative_trapezoid(w[::-1], dx=y[1] - y[0], initial=0)[::-1]
    p = np.exp(phi(x) - phi(0.0)) * (np.interp(x, y, left) * w[-1] + np.interp(x, y, right)) / left[-1]
    return p / (p.mean() * Lc)


def gardiner_J(phi, Lc, ps_a, n=200001):
    """(5.2.46) as printed, and as it follows from (5.2.45), for B = D."""
    y = np.linspace(0, Lc, n)
    tot = trapz(np.exp(phi(0.0) - phi(y)), y)
    inv_psi_b = np.exp(phi(0.0) - phi(Lc))
    return D * (inv_psi_b - 1) * ps_a / tot, D * (1 - inv_psi_b) * ps_a / (2 * tot)


def winding_T(U2D, Lc, x0, n=200001):
    """Mean time for the lift to first reach x0 - L or x0 + L, from Gardiner (5.2.158) on the cover:
    T = pi_b F(b) - F(x0), F(x) = (2/D) int_a^x dy/psi int_a^y psi, psi = e^{-2U/D}; U2D(y) = 2U/D."""
    y = np.linspace(x0 - Lc, x0 + Lc, n)
    psi = np.exp(-U2D(y))
    F = 2 / D * cumulative_trapezoid(cumulative_trapezoid(psi, y, initial=0) / psi, y, initial=0)
    W = cumulative_trapezoid(1 / psi, y, initial=0)
    mid = (n - 1) // 2
    return W[mid] / W[-1] * F[-1] - F[mid]


def ring_ops(A_bond, Lc):
    """Jump-process discretisation of the FPE (B = D) on a ring of n sites, spacing h = Lc/n,
    with A_bond[i] the drift on the bond from site i to i + 1. Returns h, the generator L0, and
    the first two derivatives of the tilted generator (jumps weighted by e^{+-s h})."""
    n = len(A_bond)
    h = Lc / n
    up = 0.5 * D / h**2 + A_bond / (2 * h)
    dn = 0.5 * D / h**2 - A_bond / (2 * h)
    i = np.arange(n)
    U = sparse.csr_matrix((up, ((i + 1) % n, i)), shape=(n, n))
    Dn = sparse.csr_matrix((dn, (i, (i + 1) % n)), shape=(n, n))
    L0 = (U + Dn - sparse.diags(up + np.roll(dn, 1))).tocsr()
    return h, L0, h * (U - Dn), 0.5 * h**2 * (U + Dn)


def ring_transport(A_bond, Lc):
    """Stationary density, V and D_eff of the lift, by perturbation theory in s."""
    h, L0, L1, L2 = ring_ops(A_bond, Lc)
    n = L0.shape[0]
    Lb = sparse.vstack([L0[:-1], sparse.csr_matrix(np.ones((1, n)))]).tocsc()

    def solve(rhs, total):
        rhs = rhs.copy()
        rhs[-1] = total
        return spsolve(Lb, rhs)

    p0 = solve(np.zeros(n), 1.0)
    lam1 = (L1 @ p0).sum()
    p1 = solve(lam1 * p0 - L1 @ p0, 0.0)
    return p0 / h, lam1, 2 * ((L2 @ p0).sum() + (L1 @ p1).sum())


# sawtooth: A = -kx for |x| < L/2, repeated; the cusp is at x = L/2 = -L/2
def saw_wrap(x, Lc):
    return (x + Lc / 2) % Lc - Lc / 2


def saw_ps(x, Lc):
    return np.exp(-((saw_wrap(x, Lc) / ELL) ** 2)) / (ELL * np.sqrt(np.pi) * erf(Lc / (2 * ELL)))


def saw_p(x, t, x0, Lc):
    """Folded at the bottom: the even part is OU reflecting on (0, L/2), the odd part OU absorbing."""
    a = Lc / (2 * ELL)
    xw = saw_wrap(x, Lc)
    y, s = np.abs(xw), np.sign(xw) * np.sign(x0)
    return 0.5 * (ou_p(y, t, abs(x0), a, "reflecting") + s * ou_p(y, t, abs(x0), a, "absorbing"))


def saw_Deff(Lc):
    """Lifson-Jackson for the sawtooth."""
    z = Lc / (2 * ELL)
    return D * (Lc / ELL) ** 2 / (np.pi * erf(z) * erfi(z))


def saw_ring(Lc, n=4000):
    """Ring discretisation with the cusp on a site, so that A is continuous on every bond."""
    h = Lc / n
    return ring_transport(-K * (-Lc / 2 + (np.arange(n) + 0.5) * h), Lc)


# cosine well: A = v - kR sin(x/R), R = L/2pi; kappa = 2R^2/ell^2, tilt f = v/kR
def cos_R(kappa):
    return np.sqrt(kappa / 2) * ELL


def cos_phi(kappa, f):
    """phi = int 2A/D on the cover, phi(0) = 0."""
    R = cos_R(kappa)
    return lambda y: 2 / D * (f * K * R * y + K * R**2 * (np.cos(y / R) - 1))


def cos_fourier(kappa, f, N=100):
    """Fourier matrices (m = -N..N, angle theta = x/R) of the tilted generator L0 + s L1 + s^2 D_th/2:
    dc_m/dt = -[D_th m^2/2 + i m v/R] c_m + (k m/2)(c_{m-1} - c_{m+1})."""
    R = cos_R(kappa)
    v, Dth = f * K * R, D / R**2
    m = np.arange(-N, N + 1)
    i = np.arange(2 * N + 1)
    L0 = np.zeros((2 * N + 1, 2 * N + 1), complex)
    L1 = np.zeros_like(L0)
    L0[i, i] = -1j * m * v / R - 0.5 * Dth * m**2
    L1[i, i] = v / R - 1j * m * Dth
    L0[i[1:], i[:-1]] = K * m[1:] / 2
    L0[i[:-1], i[1:]] = -K * m[:-1] / 2
    L1[i[1:], i[:-1]] = 1j * K / 2
    L1[i[:-1], i[1:]] = -1j * K / 2
    return Dth, L0, L1


def cos_transport(kappa, f, N=100):
    """V, D_eff of the lift and the Fourier coefficients of p_s(theta) (2 pi c_0 = 1)."""
    Dth, L0, L1 = cos_fourier(kappa, f, N)
    keep = np.arange(2 * N + 1) != N
    A = L0[np.ix_(keep, keep)]
    p = np.zeros(2 * N + 1, complex)
    p[N] = 1 / (2 * np.pi)
    p[keep] = np.linalg.solve(A, -L0[keep, N] * p[N])
    lam1 = 2 * np.pi * (L1 @ p)[N]
    p1 = np.zeros_like(p)
    p1[keep] = np.linalg.solve(A, (lam1 * p - L1 @ p)[keep])
    lam2 = 0.5 * Dth + 2 * np.pi * (L1 @ p1)[N]
    R = cos_R(kappa)
    return R * lam1.real, 2 * R**2 * lam2.real, p


def cos_density(c, x, kappa):
    R = cos_R(kappa)
    N = (len(c) - 1) // 2
    return (c * np.exp(1j * np.outer(x / R, np.arange(-N, N + 1)))).sum(1).real / R


def cos_spectrum(kappa, N=120):
    """Eigenvalues (units of k) without tilt, in the even (cos) and odd (sin) sectors."""
    M = cos_fourier(kappa, 0.0, N)[1].real
    Pe, Po = np.zeros((2 * N + 1, N + 1)), np.zeros((2 * N + 1, N))
    Pe[N, 0] = 1
    for j in range(1, N + 1):
        Pe[N + j, j] = Pe[N - j, j] = 1
        Po[N + j, j - 1], Po[N - j, j - 1] = 1, -1
    return [np.sort(-np.linalg.eigvals(np.linalg.pinv(P) @ M @ P).real) / K for P in (Pe, Po)]


# --- where the equation comes from: overdamped Brownian motion ------------------------
# m dv = -gamma v dt - Phi'(x) dt + sqrt(2 gamma kT) dW. For Phi = kappa x^2/2, in units where the Smoluchowski
# (OU) time gamma/kappa = 1 and the stationary variance kT/kappa = 1, everything depends on the damping ratio
# zeta = gamma / (2 sqrt(m kappa)): omega = 2 zeta and gamma/m = 4 zeta^2.
def damped_C(t, zeta):
    """Exact stationary position correlation C(t)/C(0) of the damped harmonic oscillator."""
    g2, w = 2 * zeta**2, 2 * zeta  # gamma/2m, omega
    if abs(zeta - 1) < 1e-12:
        return (1 + w * t) * np.exp(-w * t)
    if zeta > 1:  # e^{-g2 t}[cosh + (g2/O) sinh], written with decaying exponentials only
        O = np.sqrt(g2**2 - w**2)
        slow, fast = np.exp(-(g2 - O) * t), np.exp(-(g2 + O) * t)
        return 0.5 * (1 + g2 / O) * slow + 0.5 * (1 - g2 / O) * fast
    w1 = np.sqrt(w**2 - g2**2)
    return np.exp(-g2 * t) * (np.cos(w1 * t) + g2 / w1 * np.sin(w1 * t))


def damped_rate(zeta):
    """Long-time decay rate of C(t), in units of the Smoluchowski rate kappa/gamma."""
    zeta = np.asarray(zeta, float)
    over = 2 * zeta**2 * (1 - np.sqrt(np.clip(1 - 1 / zeta**2, 0, None)))
    return np.where(zeta >= 1, over, 2 * zeta**2)


# --- Part IV: other solvable potentials (A = -U', B = D) --------------------------
def V_from(U1, U2):
    """Symmetrised potential V = U'^2/2D - U''/2 from U' and U''."""
    return lambda x: U1(x) ** 2 / (2 * D) - U2(x) / 2


# radial motion: U = k x^2/2 - c ln x on x > 0, c = (d - 1) D/2; k = 0 is the Bessel process
def radial_p(r, t, r0, d, k=0.0):
    """Radius of a d-dimensional free (k = 0) or OU motion: (r/s2)(r/m)^mu e^{-(r^2+m^2)/2 s2} I_mu(r m/s2)."""
    mu = d / 2 - 1
    m, s2 = (r0, D * t) if k == 0 else (r0 * np.exp(-k * t), 0.5 * D / k * -np.expm1(-2 * k * t))
    return (r / s2) * (r / m) ** mu * np.exp(-((r - m) ** 2) / (2 * s2)) * ive(mu, r * m / s2)


def radial_ps(r, d, k=K):
    """Chi distribution: stationary density of the radius of a d-dimensional OU motion."""
    ell = np.sqrt(D / k)
    return 2 * r ** (d - 1) * np.exp(-((r / ell) ** 2)) / (ell**d * gamma(d / 2))


def radial_laguerre(r, t, r0, d, k=K, N=80):
    """Gardiner's (5.2.135) for general d: p_s sum_n L_n^(mu)(z) L_n^(mu)(z0)/h_n e^{-2nkt}, z = k r^2/D."""
    mu, z, z0 = d / 2 - 1, k * r**2 / D, k * r0**2 / D
    tot, hn = 0.0, 1.0
    for n in range(N):
        tot = tot + eval_genlaguerre(n, mu, z) * eval_genlaguerre(n, mu, z0) / hn * np.exp(-2 * n * k * t)
        hn *= (n + 1 + mu) / (n + 1)
    return radial_ps(r, d, k) * tot


def radial_reach(a, r0, d):
    """Probability that the free d-dimensional motion ever comes within a of the origin, from r0 > a."""
    return 1.0 if d <= 2 else (a / r0) ** (d - 2)


# soft V: U = a ln cosh(x/w), s = a/D, far-field speed v = a/w, ell = D/v, tau = D/v^2
def sech(u):
    """1/cosh u without overflow for large |u|."""
    e = np.exp(-np.abs(u))
    return 2 * e / (1 + e * e)


def lncosh_ps(x, s, w):
    return sech(x / w) ** (2 * s) / (w * beta_fn(0.5, s))


def lncosh_levels(s, w):
    return np.array([D / (2 * w**2) * n * (2 * s - n) for n in range(int(np.ceil(s)))])


def lncosh_P(x, n, s, w):
    """Forward bound state n < s: sech^{2s-n}(x/w) P_n^{(s-n,s-n)}(tanh(x/w)) (unnormalised)."""
    return sech(x / w) ** (2 * s - n) * eval_jacobi(n, s - n, s - n, np.tanh(x / w))


# a well with its own walls: U = -a ln cos(x/w) on |x| < pi w/2, s = a/D
def lncos_ps(x, s, w):
    return np.cos(x / w) ** (2 * s) / (w * beta_fn(0.5, s + 0.5))


def lncos_lam(n, s, w):
    return D / (2 * w**2) * n * (n + 2 * s)


def lncos_p(x, t, x0, s, w, N=120):
    """Gegenbauer expansion p_s sum_n C_n^(s)(sin x/w) C_n^(s)(sin x0/w)/h_n e^{-lambda_n t}, h_n by quadrature."""
    y = np.linspace(-np.pi * w / 2, np.pi * w / 2, 20001)[1:-1]
    ps_y = lncos_ps(y, s, w)
    tot = 0.0
    for n in range(N):
        Cy = eval_gegenbauer(n, s, np.sin(y / w))
        hn = trapz(ps_y * Cy**2, y)
        tot = tot + eval_gegenbauer(n, s, np.sin(x / w)) * eval_gegenbauer(n, s, np.sin(x0 / w)) / hn * np.exp(-lncos_lam(n, s, w) * t)
    return lncos_ps(x, s, w) * tot


# --- Part V: arbitrary potentials ----------------------------------------------------
def jump_spectrum(U, a, b, n, m=6, vectors=False):
    """Symmetrised generator of a jump process on [a, b] (reflecting ends) with rates (D/2h^2) e^{-(U_j - U_i)/D}
    between neighbours. Detailed balance with pi_i ∝ e^{-2U_i/D} is exact, so the zero mode is exact and
    exponentially small rates come out with relative accuracy. Returns (x, eigenvalues[, symmetrised vectors])."""
    h = (b - a) / n
    x = a + h * (np.arange(n) + 0.5)
    u = U(x) / D
    diag = np.zeros(n)
    diag[:-1] += np.exp(-(u[1:] - u[:-1]))
    diag[1:] += np.exp(-(u[:-1] - u[1:]))
    c = D / (2 * h**2)
    out = eigh_tridiagonal(c * diag, -c * np.ones(n - 1), eigvals_only=not vectors, select="i", select_range=(0, m - 1))
    return (x, *out) if vectors else (x, out)


def dw_U(beta, a=1.0):
    """Quartic double well with barrier Delta U = beta D/2 between minima at -a, a."""
    dU = beta * D / 2
    return lambda x: dU * ((x / a) ** 2 - 1) ** 2


def dw_curvatures(beta, a=1.0):
    """U''(a) at the minima and |U''(0)| at the barrier top."""
    dU = beta * D / 2
    return 8 * dU / a**2, 4 * dU / a**2


def dw_kramers(beta, a=1.0):
    """lambda_1 = 2 r, r = sqrt(U''_a |U''_b|)/2 pi e^{-2 Delta U/D} (Gardiner 5.2.174 with D -> D/2)."""
    ka, kb = dw_curvatures(beta, a)
    return 2 * np.sqrt(ka * kb) / (2 * np.pi) * np.exp(-beta)


def dw_gardiner_9218(beta, a=1.0):
    """Gardiner (9.2.18) with D -> D/2: lambda_1 = (D/2)/(n_a n_c int_a^c dx/p_s)."""
    U = dw_U(beta, a)
    x = np.linspace(-4 * a, 4 * a, 400001)
    w = np.exp(-2 * (U(x) - U(0.0)) / D)  # p_s up to a constant, scaled to avoid overflow
    Z = trapz(w, x)
    m = np.abs(x) <= a
    return (D / 2) / (0.25 * Z * trapz(1 / w[m], x[m]))


def dw_eigen(beta, a=1.0, n=6000, m=6):
    return jump_spectrum(dw_U(beta, a), -3 * a, 3 * a, n, m, vectors=True)


# --- figures -----------------------------------------------------------------
def fig_reflecting():
    x = np.linspace(0, L, 1001)
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})

    ax = axs[0]
    for t, c in zip(SNAP_T, RAMP):
        ax.plot(x, images(x, t, X0, +1), color=c, label=f"{t:g}")
    ax.axhline(1 / L, color=INK, ls="--", lw=1.0, label=r"$p_s$")
    ax.set(xlabel=r"$x/L$", ylabel=r"$L\,p(x,t\,|\,x_0,0)$", xlim=(0, L), ylim=(0, 4.2))
    snapshot_legend(ax)
    panel_label(ax, "(a)")

    ax = axs[1]
    t = np.logspace(-3, 0.5, 300)
    m1 = refl_mean(t, X0)
    sd = np.sqrt(refl_second(t, X0) - m1**2)
    ax.plot(t, m1, color=BLUE, label=r"$\langle x\rangle/L$")
    ax.plot(t, sd, color=ORANGE, label=r"$\sigma_x/L$")
    ax.plot(t, np.sqrt(D * t), color=MUTED, ls=":", lw=1.0)
    ax.axhline(L / 2, color=MUTED, ls="--", lw=0.9)
    ax.axhline(L / np.sqrt(12), color=MUTED, ls="--", lw=0.9)
    ax.text(2.8, L / 2 + 0.012, r"$L/2$", color=INK2, fontsize=7.5, ha="right", va="bottom")
    ax.text(2.8, L / np.sqrt(12) - 0.012, r"$L/\sqrt{12}$", color=INK2, fontsize=7.5, ha="right", va="top")
    ax.text(0.2, 0.56, r"$\sqrt{Dt}$", color=INK2, fontsize=7.5, ha="right")
    ax.set(xscale="log", xlabel=r"$Dt/L^2$", ylim=(0, 0.6), xlim=(t[0], t[-1]))
    ax.legend(loc="lower right", handlelength=1.5, bbox_to_anchor=(1.02, 0.0))
    panel_label(ax, "(b)")

    ax = axs[2]
    tau = np.linspace(0, 1.0, 300)
    C = refl_autocov(tau) / refl_autocov(np.array([0.0]))[0]
    ax.plot(tau, C, color=BLUE, label="exact (all odd modes)")
    ax.plot(tau, (96 / np.pi**4) * np.exp(-lam(1) * tau), color=INK, ls="--", lw=1.0,
            label=r"$(96/\pi^4)\,e^{-\lambda_1\tau}$")
    ax.set(yscale="log", xlabel=r"$D\tau/L^2$", ylabel=r"$C(\tau)/C(0)$", xlim=(0, 1), ylim=(5e-3, 1.1))
    ax.legend(loc="lower left", handlelength=1.5)
    panel_label(ax, "(c)")

    fig.savefig(FIG / "reflecting.pdf")
    plt.close(fig)


def fig_absorbing_density():
    x = np.linspace(0, L, 1001)
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"wspace": 0.28})
    for ax, cond in zip(axs, [False, True]):
        for t, c in zip(SNAP_T, RAMP):
            p = images(x, t, X0, -1)
            y = p / survival(np.array([t]), X0)[0] if cond else p
            ax.plot(x, y, color=c, label=f"{t:g}")
        ax.set(xlabel=r"$x/L$", xlim=(0, L), ylim=(0, 4.2))
    axs[1].plot(x, qsd(x), color=INK, ls="--", lw=1.0, label="QSD")
    axs[0].set_ylabel(r"$L\,p(x,t\,|\,x_0,0)$")
    axs[1].set_ylabel(r"$L\,q(x,t)=L\,p/S$")
    snapshot_legend(axs[0])
    snapshot_legend(axs[1])
    panel_label(axs[0], "(a)")
    panel_label(axs[1], "(b)")
    fig.savefig(FIG / "absorbing_density.pdf")
    plt.close(fig)


def fig_absorbing_exit(mc):
    start, t_exit, wall = mc["hist"]
    n = t_exit.size
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})

    ax = axs[0]
    t = np.linspace(1e-4, 1.5, 400)
    ax.plot(t, survival(t, X0), color=INK, label="exact")
    ax.plot(t, (4 / np.pi) * np.sin(np.pi * X0 / L) * np.exp(-lam(1) * t), color=MUTED, ls="--", lw=1.0,
            label=r"$\frac{4}{\pi}\sin\frac{\pi x_0}{L}\,e^{-\lambda_1 t}$")
    tg = np.linspace(0.05, 1.4, 12)
    mc_dots(ax, tg, [(t_exit > s).mean() for s in tg], INK, label="simulation")
    ax.set(yscale="log", xlabel=r"$Dt/L^2$", ylabel=r"$S(t\,|\,x_0)$", xlim=(0, 1.5), ylim=(1e-3, 1.5))
    ax.legend(loc="lower left", handlelength=1.5)
    panel_label(ax, "(a)")

    ax = axs[1]
    t = np.linspace(1e-4, 0.8, 500)
    j0, jL = exit_flux0(t, X0), exit_fluxL(t, X0)
    ax.plot(t, j0 + jL, color=INK, label=r"$f=j_0+j_L$")
    ax.plot(t, j0, color=C_EXIT0, label=r"$j_0$ (exit at 0)")
    ax.plot(t, jL, color=C_EXITL, label=r"$j_L$ (exit at $L$)")
    bins = np.linspace(0, 0.8, 33)
    mid = 0.5 * (bins[1:] + bins[:-1])
    for w, c in [(0, C_EXIT0), (1, C_EXITL)]:
        h, _ = np.histogram(t_exit[wall == w], bins)
        mc_dots(ax, mid, h / (n * np.diff(bins)), c)
    ax.set(xlabel=r"$Dt/L^2$", ylabel=r"$(L^2/D)\times$ density", xlim=(0, 0.8), ylim=(0, None))
    ax.legend(loc="upper right", handlelength=1.5)
    panel_label(ax, "(b)")

    ax = axs[2]
    t = np.linspace(2e-3, 0.6, 400)
    for x0, c, lbl in [(X0, INK, rf"$x_0={X0:g}L$"), (0.5 * L, AQUA, r"$x_0=L/2$")]:
        ax.plot(t, (exit_flux0(t, x0) + exit_fluxL(t, x0)) / survival(t, x0), color=c, label=lbl)
    ax.axhline(lam(1), color=MUTED, ls="--", lw=1.0)
    ax.text(0.59, lam(1) - 0.4, r"$\lambda_1=\pi^2D/2L^2$", color=INK2, fontsize=7.5, ha="right", va="top")
    ax.set(xlabel=r"$Dt/L^2$", ylabel=r"$\kappa(t)\,L^2/D$", xlim=(0, 0.6), ylim=(0, 8))
    ax.legend(loc="upper right", handlelength=1.5)
    panel_label(ax, "(c)")

    fig.savefig(FIG / "absorbing_exit.pdf")
    plt.close(fig)


def fig_absorbing_exit_times(mc):
    x0s, start, t_exit, wall = mc["grid"]
    x = np.linspace(0, L, 400)
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"wspace": 0.28})

    ax = axs[0]
    ax.plot(x, 1 - x / L, color=C_EXIT0, label=r"$\pi_0=1-x_0/L$")
    ax.plot(x, x / L, color=C_EXITL, label=r"$\pi_L=x_0/L$")
    piL = np.array([wall[start == i].mean() for i in range(len(x0s))])
    mc_dots(ax, x0s, 1 - piL, C_EXIT0)
    mc_dots(ax, x0s, piL, C_EXITL, label="simulation")
    ax.set(xlabel=r"$x_0/L$", ylabel="splitting probability", xlim=(0, L), ylim=(0, 1.02))
    ax.legend(loc="upper center", handlelength=1.5)
    panel_label(ax, "(a)")

    ax = axs[1]
    ax.plot(x, T_mean(x), color=INK, label=r"$T=\langle T\rangle$")
    ax.plot(x, np.sqrt(T_var(x)), color=INK, ls="--", lw=1.0, label=r"$\sigma_T$")
    ax.plot(x, T_given0(x), color=C_EXIT0, label=r"$T_0$ (exit at 0)")
    ax.plot(x, T_givenL(x), color=C_EXITL, label=r"$T_L$ (exit at $L$)")
    T = [t_exit[start == i].mean() for i in range(len(x0s))]
    T0 = [t_exit[(start == i) & (wall == 0)].mean() for i in range(len(x0s))]
    TL = [t_exit[(start == i) & (wall == 1)].mean() for i in range(len(x0s))]
    mc_dots(ax, x0s, T, INK)
    mc_dots(ax, x0s, T0, C_EXIT0)
    mc_dots(ax, x0s, TL, C_EXITL)
    ax.set(xlabel=r"$x_0/L$", ylabel=r"time $\times\,D/L^2$", xlim=(0, L), ylim=(0, 0.48))
    ax.legend(loc="upper center", handlelength=1.5, ncol=2, columnspacing=1.2)
    panel_label(ax, "(b)")

    fig.savefig(FIG / "absorbing_exit_times.pdf")
    plt.close(fig)


def fig_mixed(mc):
    x0s, start, t_exit, _ = mc["grid"]
    _, t_hist, _ = mc["hist"]
    x = np.linspace(0, L, 1001)
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})

    ax = axs[0]
    for t, c in zip(SNAP_T, RAMP):
        ax.plot(x, mix_images(x, t, X0) / mix_survival(np.array([t]), X0)[0], color=c, label=f"{t:g}")
    ax.plot(x, mix_qsd(x), color=INK, ls="--", lw=1.0, label="QSD")
    ax.set(xlabel=r"$x/L$", ylabel=r"$L\,q(x,t)=L\,p/S$", xlim=(0, L), ylim=(0, 4.2))
    snapshot_legend(ax)
    panel_label(ax, "(a)")

    ax = axs[1]
    t = np.linspace(1e-4, 2.5, 500)
    ax.plot(t, mix_flux(t, X0), color=INK, label="exact")
    ax.plot(t, lam_mix(0) * (4 / np.pi) * np.cos(np.pi * X0 / (2 * L)) * np.exp(-lam_mix(0) * t),
            color=MUTED, ls="--", lw=1.0, label="slowest mode")
    bins = np.linspace(0, 2.5, 26)
    h, _ = np.histogram(t_hist, bins)
    mc_dots(ax, 0.5 * (bins[1:] + bins[:-1]), h / (t_hist.size * np.diff(bins)), INK, label="simulation")
    ax.set(xlabel=r"$Dt/L^2$", ylabel=r"$f(t\,|\,x_0)\,L^2/D$", xlim=(0, 2.5), ylim=(0, 1.5))
    ax.legend(loc="upper right", handlelength=1.5)
    panel_label(ax, "(b)")

    ax = axs[2]
    xx = np.linspace(0, L, 400)
    ax.plot(xx, mix_T(xx), color=INK, label=r"$T$")
    ax.plot(xx, np.sqrt(mix_T_var(xx)), color=INK, ls="--", lw=1.0, label=r"$\sigma_T$")
    ax.plot(xx, T_mean(xx), color=MUTED, ls=":", lw=1.2, label=r"$T$, both absorbing")
    mc_dots(ax, x0s, [t_exit[start == i].mean() for i in range(len(x0s))], INK)
    mc_dots(ax, x0s, [t_exit[start == i].std() for i in range(len(x0s))], INK)
    ax.set(xlabel=r"$x_0/L$", ylabel=r"time $\times\,D/L^2$", xlim=(0, L), ylim=(0, 1.15))
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), handlelength=1.5)
    panel_label(ax, "(c)")

    fig.savefig(FIG / "mixed.pdf")
    plt.close(fig)


def fig_halfline_reflecting():
    x = np.linspace(0, 12, 1201)
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"wspace": 0.28})

    ax = axs[0]
    for t, c in zip(SNAP_TH, RAMP):
        ax.plot(x, half_line(x, t, X0H, +1), color=c, label=f"{t:g}")
    ax.set(xlabel=r"$x/x_0$", ylabel=r"$x_0\,p(x,t\,|\,x_0,0)$", xlim=(0, 12), ylim=(0, 1.9))
    snapshot_legend(ax, title=r"$Dt/x_0^2$")
    panel_label(ax, "(a)")

    ax = axs[1]
    t = np.logspace(-3, 3, 300)
    m1 = hl_mean(t, X0H)
    ax.plot(t, m1, color=BLUE, label=r"$\langle x\rangle/x_0$")
    ax.plot(t, np.sqrt(X0H**2 + D * t - m1**2), color=ORANGE, label=r"$\sigma_x/x_0$")
    tl = np.logspace(0, 3, 50)
    ts = np.logspace(-3, -0.5, 50)
    ax.plot(tl, np.sqrt(2 * D * tl / np.pi), color=MUTED, ls=":", lw=1.0)
    ax.plot(tl, np.sqrt((1 - 2 / np.pi) * D * tl), color=MUTED, ls=":", lw=1.0)
    ax.plot(ts, np.sqrt(D * ts), color=MUTED, ls=":", lw=1.0)
    ax.text(30, 2.3 * np.sqrt(2 * 30 / np.pi), r"$\sqrt{2Dt/\pi}$", color=INK2, fontsize=7.5, ha="right")
    ax.text(60, 0.42 * np.sqrt((1 - 2 / np.pi) * 60), r"$\sqrt{(1-2/\pi)Dt}$", color=INK2, fontsize=7.5)
    ax.text(3e-3, 0.11, r"$\sqrt{Dt}$", color=INK2, fontsize=7.5)
    ax.set(xscale="log", yscale="log", xlabel=r"$Dt/x_0^2$", xlim=(t[0], t[-1]), ylim=(0.02, 50))
    ax.legend(loc="upper left", handlelength=1.5)
    panel_label(ax, "(b)")

    fig.savefig(FIG / "halfline_reflecting.pdf")
    plt.close(fig)


def fig_halfline_absorbing():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})

    ax = axs[0]
    x = np.linspace(0, 12, 1201)
    for t, c in zip(SNAP_TH, RAMP):
        ax.plot(x, half_line(x, t, X0H, -1), color=c, label=f"{t:g}")
    ax.set(xlabel=r"$x/x_0$", ylabel=r"$x_0\,p(x,t\,|\,x_0,0)$", xlim=(0, 8), ylim=(0, 1.9))
    snapshot_legend(ax, title=r"$Dt/x_0^2$")
    panel_label(ax, "(a)")

    ax = axs[1]
    u = np.linspace(0, 4.5, 500)
    for t, c in zip(SNAP_TH, RAMP):
        s = np.sqrt(D * t)
        q = half_line(u * s, t, X0H, -1) / hl_survival(t, X0H)
        ax.plot(u, s * q, color=c, label=f"{t:g}")
    ax.plot(u, u * np.exp(-(u**2) / 2), color=INK, ls="--", lw=1.0, label="Rayleigh")
    ax.set(xlabel=r"$x/\sqrt{Dt}$", ylabel=r"$\sqrt{Dt}\,q(x,t)$", xlim=(0, 4.5), ylim=(0, 1.9))
    snapshot_legend(ax, title=r"$Dt/x_0^2$")
    panel_label(ax, "(b)")

    ax = axs[2]
    t = np.logspace(-2, 4, 300)
    ax.plot(t, hl_survival(t, X0H), color=INK, label="exact")
    tl = np.logspace(0, 4, 50)
    ax.plot(tl, X0H * np.sqrt(2 / (np.pi * D * tl)), color=MUTED, ls="--", lw=1.0,
            label=r"$x_0\sqrt{2/\pi Dt}$")
    ax.set(xscale="log", yscale="log", xlabel=r"$Dt/x_0^2$", ylabel=r"$S(t\,|\,x_0)$",
           xlim=(t[0], t[-1]), ylim=(5e-3, 1.5))
    ax.legend(loc="lower left", handlelength=1.5)
    panel_label(ax, "(c)")

    fig.savefig(FIG / "halfline_absorbing.pdf")
    plt.close(fig)


def fig_free_summary():
    """All A = 0 geometries side by side, from x0 = 0.3L."""
    t_snap = 0.1
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    xf = np.linspace(-0.8, 1.6, 1201)
    xh = np.linspace(0, 1.6, 801)
    xi = np.linspace(0, L, 501)
    for ax, sign in zip(axs[:2], [+1, -1]):
        ax.plot(xf, heat(xf - X0, t_snap), color=INK, ls="--", lw=1.0, label="none")
        ax.plot(xh, half_line(xh, t_snap, X0, sign), color=BLUE, label=r"$0$")
        ax.plot(xi, images(xi, t_snap, X0, sign), color=ORANGE, label=r"$0$ and $L$")
        for w in (0, L):
            ax.axvline(w, color=AXIS, lw=0.8)
        ax.set(xlabel=r"$x/L$", xlim=(-0.8, 1.6), ylim=(0, 2.45))
    axs[1].plot(xi, mix_images(xi, t_snap, X0), color=AQUA, label=r"refl. $0$, abs. $L$")
    axs[0].set_ylabel(r"$L\,p(x,t\,|\,x_0,0)$")
    axs[0].set_title("reflecting walls", color=INK2, fontsize=8)
    axs[1].set_title("absorbing walls", color=INK2, fontsize=8)
    for ax in axs[:2]:
        ax.legend(loc="upper right", bbox_to_anchor=(1.04, 1.02), handlelength=1.5)
    panel_label(axs[0], "(a)")
    panel_label(axs[1], "(b)")

    ax = axs[2]
    t = np.linspace(1e-3, 2, 400)
    ax.plot(t, hl_survival(t, X0), color=BLUE, label="wall at 0")
    ax.plot(t, survival(t, X0), color=ORANGE, label=r"walls at 0, $L$")
    ax.plot(t, mix_survival(t, X0), color=AQUA, label=r"refl. 0, abs. $L$")
    ax.set(yscale="log", xlabel=r"$Dt/L^2$", ylabel=r"$S(t\,|\,x_0)$", xlim=(0, 2), ylim=(1e-4, 1.5))
    ax.legend(loc="lower left", handlelength=1.5)
    panel_label(ax, "(c)")

    fig.savefig(FIG / "free_summary.pdf")
    plt.close(fig)


def pe_legend(ax, title=r"$vL/D$", **kw):
    kw = {"loc": "upper left", "handlelength": 1.5, **kw}
    leg = ax.legend(title=title, title_fontsize=7.5, **kw)
    leg.get_title().set_color(INK2)


def fig_drift_reflecting():
    x = np.linspace(0, L, 1001)
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})

    ax = axs[0]
    for v, c in zip(PE, DIVERGING):
        ax.plot(x, dr_stationary(x, v), color=c, label=f"{v:g}")
    ax.set(xlabel=r"$x/L$", ylabel=r"$L\,p_s(x)$", xlim=(0, L), ylim=(0, 8.5))
    pe_legend(ax, loc="upper center")
    panel_label(ax, "(a)")

    ax = axs[1]
    v = 4.0
    for t, c in zip(SNAP_T, RAMP):
        ax.plot(x, dr_modes(x, t, X0, v), color=c, label=f"{t:g}")
    ax.plot(x, dr_stationary(x, v), color=INK, ls="--", lw=1.0, label=r"$p_s$")
    ax.set(xlabel=r"$x/L$", ylabel=r"$L\,p(x,t\,|\,x_0,0)$", xlim=(0, L), ylim=(0, 8.5))
    snapshot_legend(ax, loc="upper left", bbox_to_anchor=(-0.02, 1.04))
    panel_label(ax, "(b)")

    ax = axs[2]
    t = np.linspace(0, 0.6, 200)
    for v, c in zip(PE, DIVERGING):
        ax.plot(t, np.abs(dr_mean(t, X0, v) - dr_mean_s(v)), color=c, label=f"{v:g}")
    ax.set(yscale="log", xlabel=r"$Dt/L^2$", ylabel=r"$|\langle x(t)\rangle-\langle x\rangle_s|/L$",
           xlim=(0, 0.6), ylim=(1e-5, 1))
    panel_label(ax, "(c)")

    fig.savefig(FIG / "drift_reflecting.pdf")
    plt.close(fig)


def fig_drift_absorbing():
    x = np.linspace(1e-6, L - 1e-6, 1001)
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.6), gridspec_kw={"wspace": 0.28, "hspace": 0.5})
    (a1, a2), (a3, a4) = axs

    for v, c in zip(PE, DIVERGING):
        a1.plot(x, da_qsd(x, v), color=c, label=f"{v:g}")
        a2.plot(x, da_piL(x, v), color=c, label=f"{v:g}")
        a3.plot(x, da_T(x, v), color=c, label=f"{v:g}")
    a1.set(xlabel=r"$x/L$", ylabel=r"$L\,q_\infty(x)$", xlim=(0, L), ylim=(0, 4.5))
    pe_legend(a1, loc="upper center", ncol=2, columnspacing=1.0)
    a2.set(xlabel=r"$x_0/L$", ylabel=r"$\pi_L(x_0)$", xlim=(0, L), ylim=(0, 1.02))
    a3.set(xlabel=r"$x_0/L$", ylabel=r"$T(x_0)\,D/L^2$", xlim=(0, L), ylim=(0, 0.3))

    for v, c in zip(PE[2:], DIVERGING[2:]):
        a4.plot(x, da_TL(x, v), color=c, label=rf"$T_L$, ${v:g}$")
    for v, c in zip(PE[:2], DIVERGING[:2]):
        a4.plot(x, da_TL(x, v), color=c, ls=(0, (2, 2)), lw=1.5, label=rf"$T_L$, ${v:g}$")
    a4.set(xlabel=r"$x_0/L$", ylabel=r"$T_L(x_0)\,D/L^2$", xlim=(0, L), ylim=(0, 0.36))
    pe_legend(a4, loc="upper right")
    for ax, lbl in zip([a1, a2, a3, a4], "abcd"):
        panel_label(ax, f"({lbl})")

    fig.savefig(FIG / "drift_absorbing.pdf")
    plt.close(fig)


def fig_drift_mixed():
    x = np.linspace(0, L, 1001)
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})

    ax = axs[0]
    for v, c in zip(PE, DIVERGING):
        ax.plot(x, dm_qsd(x, v), color=c, label=f"{v:g}")
    ax.set(xlabel=r"$x/L$", ylabel=r"$L\,q_\infty(x)$", xlim=(0, L), ylim=(0, 4.5))
    pe_legend(ax, loc="upper right", bbox_to_anchor=(1.04, 1.04))
    panel_label(ax, "(a)")

    ax = axs[1]
    pe = np.linspace(-6, 6, 240)
    ax.plot(pe, [dm_modes(v, 1)[0][0] for v in pe], color=INK, label="refl. 0, abs. $L$")
    ax.plot(pe, [1 / dm_T(0.0, v) for v in pe], color=INK, ls=":", lw=1.2, label=r"$1/T(0)$")
    ax.plot(pe, da_lam1(pe), color=ORANGE, label="same-type walls")
    ax.axvline(-1, color=AXIS, lw=0.8)
    ax.set(yscale="log", xlabel=r"$vL/D$", ylabel=r"slowest rate $\times L^2/D$", xlim=(-6, 6), ylim=(1e-4, 50))
    ax.legend(loc="lower right", handlelength=1.5)
    panel_label(ax, "(b)")

    ax = axs[2]
    for v, c in zip(PE, DIVERGING):
        ax.plot(x, dm_T(x, v), color=c, label=f"{v:g}")
    ax.set(yscale="log", xlabel=r"$x_0/L$", ylabel=r"$T(x_0)\,D/L^2$", xlim=(0, L), ylim=(1e-2, 2e2))
    panel_label(ax, "(c)")

    fig.savefig(FIG / "drift_mixed.pdf")
    plt.close(fig)


def fig_drift_halfline_reflecting():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"wspace": 0.28})

    # Fixed v < 0: drift units ell = D/|v|, tau = D/v^2, start at x0 = ell.
    ax = axs[0]
    v = -1.0
    ell, tau = D / abs(v), D / v**2
    xi = np.linspace(0, 5, 1001)
    for s, c in zip([0.05, 0.3, 1.0, 3.0, 10.0], RAMP):
        ax.plot(xi, ell * dhr_p(xi * ell, s * tau, ell, v), color=c, label=f"{s:g}")
    ax.plot(xi, 2 * np.exp(-2 * xi), color=INK, ls="--", lw=1.0, label=r"$p_s$")
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,p(x,t\,|\,x_0{=}\ell,0)$", xlim=(0, 5), ylim=(0, 2.1))
    snapshot_legend(ax, title=r"$t/\tau$")
    panel_label(ax, "(a)")

    ax = axs[1]
    t = np.logspace(-2, 2, 120)
    xs = np.linspace(0, 260, 52001)
    for v, c in zip(V_HALF, DIVERGING):
        m = [trapz(xs * dhr_p(xs, ti, X0H, v), xs) for ti in t]
        ax.plot(t, m, color=c, label=f"{v:g}")
    ax.set(xscale="log", yscale="log", xlabel=r"$Dt/x_0^2$", ylabel=r"$\langle x(t)\rangle/x_0$",
           xlim=(t[0], t[-1]), ylim=(0.3, 200))
    pe_legend(ax, title=r"$vx_0/D$", loc="upper left")
    panel_label(ax, "(b)")

    fig.savefig(FIG / "drift_halfline_reflecting.pdf")
    plt.close(fig)


def fig_drift_halfline_absorbing():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    # Panels (a) and (c) are at fixed v < 0, in drift units ell = D/|v|, tau = D/v^2.
    v = -1.0
    ell, tau = D / abs(v), D / v**2

    ax = axs[0]
    xi = np.linspace(0, 8, 801)
    for s, c in zip([0.3, 1.0, 3.0, 10.0, 30.0], RAMP):
        q = dha_p(xi * ell, s * tau, ell, v) / dha_survival(s * tau, ell, v)
        ax.plot(xi, ell * q, color=c, label=f"{s:g}")
    ax.plot(xi, xi * np.exp(-xi), color=INK, ls="--", lw=1.0, label="edge QSD")
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,q(x,t)$", xlim=(0, 8), ylim=(0, 1.5))
    snapshot_legend(ax, title=r"$t/\tau$")
    panel_label(ax, "(a)")

    ax = axs[1]
    t = np.logspace(-3, 2, 400)
    for vv, c in zip(V_HALF, DIVERGING):
        ax.plot(t, inverse_gaussian(t, X0H, vv), color=c, label=f"{vv:g}")
    ax.set(xscale="log", yscale="log", xlabel=r"$Dt/x_0^2$", ylabel=r"$f(t\,|\,x_0)\,x_0^2/D$",
           xlim=(t[0], t[-1]), ylim=(1e-4, 1))
    pe_legend(ax, title=r"$vx_0/D$", loc="lower left", handlelength=1.0, borderaxespad=0.2)
    panel_label(ax, "(b)")

    ax = axs[2]
    edge = 1 / (2 * tau)
    for frac, c in zip([0.2, 0.5, 0.8], RAMP[0::2]):
        ax.plot(xi, ell * dha_qsd(xi * ell, frac * edge, v), color=c, label=f"{frac:g}")
    ax.plot(xi, ell * dha_qsd(xi * ell, edge, v), color=INK, ls="--", lw=1.0, label="1 (edge)")
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,\phi_\lambda(x)$", xlim=(0, 8), ylim=(0, 0.5))
    snapshot_legend(ax, title=r"$2\lambda\tau$")
    panel_label(ax, "(c)")

    fig.savefig(FIG / "drift_halfline_absorbing.pdf")
    plt.close(fig)


def fig_spectra():
    """Eigenvalue ladders of six examples, each in its own natural rate unit."""
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    top = 10.0
    well = [m[0] for m in ou_modes(2.5, "mixed")[:6]]
    cols = [
        ("refl. interval\n" r"unit $\pi^2D/2L^2$", [0, 1, 4, 9], None, (0, 1)),
        ("abs. interval\n" r"unit $\pi^2D/2L^2$", [1, 4, 9], None, (1, 4)),
        ("free half-line\n" "(no gap)", [], 0.0, None),
        ("drift into wall\n" r"unit $1/2\tau$", [0], 1.0, (0, 1)),
        ("OU line\n" r"unit $k$", list(range(0, 11)), None, (0, 1)),
        ("harmonic well, abs.\n" r"at $2.5\ell$; unit $k$", well, None, (well[0], well[1])),
    ]
    for i, (label, levels, cont, gap) in enumerate(cols):
        x0_, x1_ = i + 0.18, i + 0.82
        if cont is not None:
            ax.add_patch(plt.Rectangle((x0_, cont), x1_ - x0_, top - cont, color=BLUE, alpha=0.15, lw=0))
            ax.plot([x0_, x1_], [cont, cont], color=BLUE, lw=1.2)
        for lv in levels:
            if lv <= top:
                ax.plot([x0_, x1_], [lv, lv], color=INK, lw=1.5)
        if gap is not None:
            ax.annotate("", xy=(i + 0.5, gap[1]), xytext=(i + 0.5, gap[0]),
                        arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.2, shrinkA=0, shrinkB=0))
        ax.text(i + 0.5, -0.9, label, ha="center", va="top", fontsize=7, color=INK2)
    ax.annotate(rf"$\lambda_0\approx{well[0]:.3f}\,k$", xy=(5.3, well[0]), xytext=(5.3, 3.3), ha="center",
                fontsize=7, color=INK2, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set(xlim=(0, len(cols)), ylim=(-0.3, top), ylabel=r"$\lambda$ / unit")
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    fig.savefig(FIG / "spectra.pdf")
    plt.close(fig)


def level_segments(ax, x, Vx, E, color, region=None):
    """Draw the level E where V < E (the classically allowed region), optionally only where region is True."""
    ok = Vx < E if region is None else (Vx < E) & region
    edges = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
    for i0, i1 in zip(edges[::2], edges[1::2]):
        ax.plot([x[i0], x[i1 - 1]], [E, E], color=color, lw=1.6, solid_capstyle="butt")


def fig_operators():
    """Schrodinger potentials V = A^2/2D + A'/2 and their levels (units of k); blue even, orange odd."""
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3), gridspec_kw={"wspace": 0.35})
    ax = axs[0]
    xi = np.linspace(-3.6, 3.6, 2001)
    Vx = V_ou(xi * ELL) / K
    ax.plot(xi, Vx, color=INK, lw=1.2)
    for n in range(6):
        level_segments(ax, xi, Vx, n, BLUE if n % 2 == 0 else ORANGE)
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$V/k$, $\lambda/k$", xlim=(-3.6, 3.6), ylim=(-0.8, 5.6))
    ax.plot([], [], color=BLUE, label="even")
    ax.plot([], [], color=ORANGE, label="odd")
    ax.legend(loc="lower right", handlelength=1.2, bbox_to_anchor=(1.04, -0.02))
    panel_label(ax, "(a)")

    ax = axs[1]
    kappa = 16.0
    Lc = 2 * np.pi * cos_R(kappa)
    u = np.linspace(-0.5, 0.5, 4001)
    Vx = V_cos(kappa)(u * Lc) / K
    ax.plot(u, Vx, color=INK, lw=1.2)
    ev = cos_spectrum(kappa)[0][:4]
    bottom = np.abs(u) < 0.25
    for m_, E in enumerate(ev):
        level_segments(ax, u, Vx, E, BLUE if m_ % 2 == 0 else ORANGE, bottom)
        if m_ > 0:
            level_segments(ax, u, Vx, E, ORANGE if m_ % 2 == 0 else BLUE, ~bottom)
    ax.set(xlabel=r"$x/L$", xlim=(-0.5, 0.5), ylim=(-0.8, 5.6))
    panel_label(ax, "(b)")

    ax = axs[2]
    Lc = 4.0 * ELL
    xi = np.linspace(-Lc / 2, Lc / 2, 2001) / ELL
    Vx = V_ou(xi * ELL) / K
    ax.plot(xi * ELL / Lc, Vx, color=INK, lw=1.2)
    for s in (-0.5, 0.5):
        ax.annotate("", xy=(s, 5.5), xytext=(s, V_ou(Lc / 2) / K),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2, shrinkA=0, shrinkB=0))
    for kind, c in [("reflecting", BLUE), ("absorbing", ORANGE)]:
        for nu, _, _ in ou_modes(Lc / (2 * ELL), kind)[:4]:
            if nu < 5.6:
                level_segments(ax, xi * ELL / Lc, Vx, nu, c)
    ax.set(xlabel=r"$x/L$", xlim=(-0.5, 0.5), ylim=(-0.8, 5.6))
    panel_label(ax, "(c)")
    fig.savefig(FIG / "operators.pdf")
    plt.close(fig)


def walls_path(x0=1.0, T=4.0, dt=1e-3):
    """One Brownian path (D = 1) that hits 0 well inside [0, T] and dips well below it, found by a deterministic
    scan over seeds, so the three wall treatments in fig_walls show something."""
    n = int(T / dt)
    t = np.linspace(0, T, n + 1)
    for seed in range(1000):
        x = x0 + np.concatenate([[0], np.cumsum(np.random.default_rng(seed).standard_normal(n) * np.sqrt(D * dt))])
        hit = np.argmax(x <= 0) if np.any(x <= 0) else None
        if hit and 1.0 < t[hit] < 2.0 and x.min() < -0.8 and x.max() > 2.2:
            return t, x, hit
    raise RuntimeError("no suitable path")


def fig_walls():
    """The same path with a hard wall, a cliff, and on a ring."""
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 3.9), gridspec_kw={"height_ratios": [1, 1.5], "wspace": 0.3,
                                                                     "hspace": 0.35})
    t, x, hit = walls_path()
    # top row: the potentials
    ax = axs[0, 0]
    ax.fill_between([-0.6, 0], -0.2, 1.0, hatch="////", facecolor="white", edgecolor=AXIS, lw=0)
    ax.plot([0, 0, 3], [1.0, 0, 0], color=INK, lw=1.4)
    ax.set_title(r"hard wall: $\mathcal{U}\to\infty$ for $x<0$", color=INK2, fontsize=8)
    ax = axs[0, 1]
    ax.plot([3, 0, 0], [0, 0, -0.9], color=INK, lw=1.4)
    ax.annotate("", xy=(0.0, -1.0), xytext=(0.0, -0.55), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(-0.15, -0.55, "falls off", ha="right", va="center", fontsize=7, color=INK2)
    ax.set_title(r"cliff: removed at $x=0$", color=INK2, fontsize=8)
    for ax in axs[0, :2]:
        ax.set(xlim=(-0.6, 3), ylim=(-1.05, 1.05))
        ax.axis("off")
    ax = axs[0, 2]
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=INK, lw=1.4)
    ax.plot([np.cos(0.6)], [np.sin(0.6)], "o", ms=6, color=BLUE, mec="white", mew=0.8)
    ax.set(xlim=(-1.6, 1.6), ylim=(-1.15, 1.15), aspect="equal")
    ax.axis("off")
    ax.set_title(r"ring of circumference $L$", color=INK2, fontsize=8)
    # bottom row: one path, three treatments
    ax = axs[1, 0]
    ax.axhspan(-1.5, 0, color=GRID, alpha=0.6, lw=0)
    ax.plot(t, np.abs(x), color=BLUE, lw=1.0)
    ax.set(xlabel=r"$Dt/x_0^2$", ylabel=r"$x/x_0$", ylim=(-1.5, 3.2), xlim=(0, 4))
    ax = axs[1, 1]
    ax.axhspan(-1.5, 0, color=GRID, alpha=0.6, lw=0)
    ax.plot(t[hit:], x[hit:], color=MUTED, lw=0.7, ls=":")
    ax.plot(t[:hit + 1], x[:hit + 1], color=BLUE, lw=1.0)
    ax.plot([t[hit]], [0], "x", color=ORANGE, ms=7, mew=1.6)
    ax.set(xlabel=r"$Dt/x_0^2$", ylim=(-1.5, 3.2), xlim=(0, 4))
    ax = axs[1, 2]
    Lr = 2.0
    xw = np.mod(x, Lr)
    jumps = np.flatnonzero(np.abs(np.diff(xw)) > Lr / 2)
    ax.plot(t, x, color=MUTED, lw=0.7, ls=":", label="lift (winding)", zorder=1)
    for i0, i1 in zip(np.r_[0, jumps + 1], np.r_[jumps + 1, len(x)]):
        ax.plot(t[i0:i1], xw[i0:i1], color=BLUE, lw=1.0, zorder=3, label="on the ring" if i0 == 0 else None)
    for y in (0, Lr):
        ax.axhline(y, color=AXIS, lw=0.8)
    ax.set(xlabel=r"$Dt/x_0^2$", ylim=(-1.5, 3.2), xlim=(0, 4))
    ax.legend(loc="upper left", handlelength=1.5)
    for a_, lbl in zip(axs[1], "abc"):
        panel_label(a_, f"({lbl})")
    fig.savefig(FIG / "walls.pdf")
    plt.close(fig)


def fig_overdamped():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"wspace": 0.3})
    ax = axs[0]
    t = np.linspace(0, 5, 1001)
    for z, c in zip([0.25, 1.0, 3.0, 10.0], RAMP[1:]):
        ax.plot(t, damped_C(t, z), color=c, label=f"{z:g}")
    ax.plot(t, np.exp(-t), color=INK, ls="--", lw=1.0, label="Smoluchowski")
    ax.set(xlabel=r"$t\,\mathcal{K}/\gamma$", ylabel=r"$C(t)/C(0)$", xlim=(0, 5), ylim=(-0.75, 1.05))
    leg = ax.legend(title=r"damping ratio $\zeta$", title_fontsize=7.5, loc="upper right", handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(a)")

    ax = axs[1]
    z = np.logspace(np.log10(0.2), np.log10(30), 400)
    ax.plot(z, damped_rate(z), color=INK, label="exact")
    zz = z[z > 1.3]
    ax.plot(zz, 1 + 1 / (4 * zz**2), color=ORANGE, ls="--", lw=1.0, label=r"corrected, $1+1/4\zeta^2$")
    ax.axhline(1, color=MUTED, ls=":", lw=1.0, label="Smoluchowski")
    ax.axvline(1, color=MUTED, lw=0.8)
    ax.text(1.08, 0.12, "critical", fontsize=7, color=INK2)
    ax.set(xscale="log", xlabel=r"damping ratio $\zeta=\gamma/2\sqrt{m\mathcal{K}}$", ylabel=r"slowest rate $/(\mathcal{K}/\gamma)$",
           xlim=(0.2, 30), ylim=(0, 2.2))
    ax.legend(loc="upper right", handlelength=1.5)
    panel_label(ax, "(b)")
    fig.savefig(FIG / "overdamped.pdf")
    plt.close(fig)


def fig_reductions():
    """Map of the reductions: arrows point from a problem to the simpler one it reduces to."""
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.set(xlim=(0, 12), ylim=(0, 7.4))
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.35", fc="white", ec=AXIS, lw=0.9)
    hub = dict(boxstyle="round,pad=0.35", fc="#eaf2fc", ec=BLUE, lw=1.0)
    nodes = {
        "kramers": (6.0, 7.0, "underdamped particle in a potential\n(Kramers' equation, $x$ and $v$)", box),
        "any": (6.0, 5.85, "overdamped: any $A(x)$, $B(x)$", box),
        "sym": (6.0, 4.55, r"$B=D$, $A=-U'$:  $H=-\frac{1}{2}D\partial_x^2+V$" "\n" r"$V=U'^2/2D-U''/2$", box),
        "si": (1.55, 3.05, "shape-invariant $V$\n(Part IV: log, ln cosh,\n$-$ln cos, Morse, ...)", box),
        "ou": (4.55, 3.05, "harmonic $V$:\nOU (Part III)", box),
        "drift": (7.45, 3.05, "constant $V$:\ndrift $v$ (Part II)", box),
        "gen": (10.45, 3.05, "any other $V$ (Part V):\nquadratures, bounds,\nKramers, numerics", box),
        "free": (6.0, 1.45, "$V=0$: free diffusion, any walls,\ncircle (Part I)", box),
        "gauss": (6.0, 0.3, "the Gaussian on the line", hub),
    }
    for key, (x, y, text, style) in nodes.items():
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=INK, bbox=style, zorder=3)

    def arrow(a, b):
        (x0, y0), (x1, y1) = nodes[a][:2], nodes[b][:2]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=16, shrinkB=16), zorder=2)

    def label(x, y, text, ha="left"):
        ax.text(x, y, text, fontsize=6.6, color=BLUE, ha=ha, va="center", zorder=4)

    for a, b in [("kramers", "any"), ("any", "sym"), ("sym", "si"), ("sym", "ou"), ("sym", "drift"), ("sym", "gen"),
                 ("si", "ou"), ("ou", "free"), ("drift", "free"), ("free", "gauss")]:
        arrow(a, b)
    label(6.15, 6.42, "eliminate $v$ ($\\gamma^2\\gg m\\,\\mathcal{U}''$)")
    label(6.15, 5.27, "Lamperti $y=\\int dx/\\sqrt{B}$, then symmetrise $p=\\sqrt{p_s}\\,\\psi$")
    label(3.2, 4.05, "read off $V$", ha="right")
    label(3.3, 2.6, "radius of\n$d$-dim OU", ha="center")
    label(4.95, 1.98, "$y=xe^{kt}$, clock $\\theta(t)$\n(walls at the bottom)", ha="right")
    label(7.05, 2.2, "gauge $e^{\\alpha(x-x_0)-t/2\\tau}$\n(boost on a circle)", ha="left")
    label(6.15, 0.95, "images, cover, fold, cut")
    ax.text(10.45, 1.75, "symmetry still helps:\nfold, cover, cut;\nDoob, partner $U\\leftrightarrow-U$",
            ha="center", va="center", fontsize=6.6, color=INK2)
    ax.text(1.55, 1.75, "partner $U\\leftrightarrow-U$\n(ladders, doublets);\nDoob: conditioning",
            ha="center", va="center", fontsize=6.6, color=INK2)
    fig.savefig(FIG / "reductions.pdf")
    plt.close(fig)


def fig_solvable():
    """Gallery of shape-invariant potentials: symmetrised V (black), levels where V < lambda, continuum shaded."""
    fig, axs = plt.subplots(2, 3, figsize=(6.5, 4.3), gridspec_kw={"wspace": 0.32, "hspace": 0.62})
    s_m, s_c, s_t, sig, d = 3.5, 3.5, 2.0, 2.0, 5.0
    c_r = (d - 1) * D / 2
    panels = [
        (r"harmonic: $\frac{1}{2}kx^2$", np.linspace(-3.4, 3.4, 2001), V_ou, [n * K for n in range(6)], None, (-0.8, 5.6)),
        (r"harmonic $+$ log, $d=5$", np.linspace(0.05, 3.6, 2001),
         V_from(lambda x: K * x - c_r / x, lambda x: K + c_r / x**2), [2 * n * K for n in range(3)], None, (-3, 5.6)),
        (r"linear $+$ log: $vx-c\ln x$", np.linspace(0.1, 30, 3001),
         V_from(lambda x: 1.0 - sig * D / x, lambda x: sig * D / x**2),
         [0.5 * (1 - (sig / (n + sig)) ** 2) for n in range(12)], 0.5, (-0.6, 0.85)),
        (r"Morse: $vx+U_0e^{-x/w}$", np.linspace(-3, 5, 2001),
         V_from(lambda x: s_m - np.exp(-x), lambda x: np.exp(-x)), [0.5 * n * (2 * s_m - n) for n in range(4)],
         0.5 * s_m**2, (-2.2, 8.0)),
        (r"$a\ln\cosh(x/w)$", np.linspace(-5, 5, 2001),
         V_from(lambda x: s_c * np.tanh(x), lambda x: s_c / np.cosh(x) ** 2), lncosh_levels(s_c, 1.0),
         0.5 * s_c**2, (-2.2, 8.0)),
        (r"$-a\ln\cos(x/w)$", np.linspace(-1.5, 1.5, 2001),
         V_from(lambda x: s_t * np.tan(x), lambda x: s_t / np.cos(x) ** 2), [lncos_lam(n, s_t, 1.0) for n in range(4)],
         None, (-1.5, 13.0)),
    ]
    for ax, (title, x, V, levels, edge, ylim) in zip(axs.flat, panels):
        Vx = V(x)
        if edge is not None:
            ax.axhspan(edge, ylim[1], color=BLUE, alpha=0.12, lw=0)
            ax.axhline(edge, color=BLUE, lw=0.8)
        ax.plot(x, Vx, color=INK, lw=1.2)
        for E in levels:
            level_segments(ax, x, Vx, E, BLUE)
        ax.set(xlim=(x[0], x[-1]), ylim=ylim)
        ax.set_title(title, color=INK2, fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$V$, $\lambda$")
    for ax, lbl in zip(axs.flat, "abcdef"):
        panel_label(ax, f"({lbl})")
    fig.savefig(FIG / "solvable.pdf")
    plt.close(fig)


def fig_radial():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    dims = [1.0, 2.0, 3.0, 5.0]
    ax = axs[0]
    r = np.linspace(1e-4, 4, 800)
    for d, c in zip(dims, RAMP[1:]):
        ax.plot(r, radial_p(r, 1.0, 1.0, d), color=c, label=f"{d:g}")
    ax.set(xlabel=r"$r/r_0$", ylabel=r"$r_0\,p(r,t\,|\,r_0)$", xlim=(0, 4), ylim=(0, 0.75))
    leg = ax.legend(title=r"$d$", title_fontsize=7.5, loc="upper right", handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(a)")

    ax = axs[1]
    r = np.linspace(0, 3.5, 700) * ELL
    for d, c in zip(dims, RAMP[1:]):
        ax.plot(r / ELL, ELL * radial_ps(r, d), color=c, label=f"{d:g}")
    ax.set(xlabel=r"$r/\ell$", ylabel=r"$\ell\,p_s(r)$", xlim=(0, 3.5), ylim=(0, 1.25))
    panel_label(ax, "(b)")

    ax = axs[2]
    dd = np.linspace(1, 8, 300)
    for ratio, c in [(0.5, BLUE), (0.2, ORANGE)]:
        ax.plot(dd, [radial_reach(ratio, 1.0, d) for d in dd], color=c, label=rf"$a/r_0={ratio:g}$")
        di = np.array([3.0, 4.0, 5.0, 6.0])
        mc_dots(ax, di, [radial_reach_quad(ratio, d) for d in di], c)
    ax.set(xlabel=r"dimension $d$", ylabel=r"$\Pr(\text{ever reach } r=a)$", xlim=(1, 8), ylim=(0, 1.05))
    ax.legend(loc="upper right", handlelength=1.5)
    panel_label(ax, "(c)")
    fig.savefig(FIG / "radial.pdf")
    plt.close(fig)


def radial_reach_quad(a, d, r0=1.0, R=1e6):
    """Splitting probability for reaching a before R (corrected Gardiner 5.2.190), with psi = r^{2c/D}."""
    f = lambda u: np.exp((2 - d) * u)  # dy/psi = y^{1-d} dy, integrated over u = ln y
    return quad(f, np.log(r0), np.log(R), limit=400)[0] / quad(f, np.log(a), np.log(R), limit=400)[0]


def fig_lncosh():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    ell = 1.0  # D/v
    ax = axs[0]
    x = np.linspace(-6, 6, 1201) * ell
    for s, c in zip([0.25, 1.0, 4.0], RAMP[1::1]):
        ax.plot(x / ell, ell * lncosh_ps(x, s, s * ell), color=c, label=f"{s:g}")
    ax.plot(x / ell, np.exp(-2 * np.abs(x) / ell), color=INK, ls="--", lw=1.0, label=r"$v|x|$")
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,p_s(x)$", xlim=(-6, 6), ylim=(0, 1.1))
    leg = ax.legend(title=r"$s=a/D$", title_fontsize=7.5, loc="upper right", handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(a)")

    ax = axs[1]
    s = np.linspace(1e-3, 6, 600)
    ax.axhspan(1, 1.2, color=BLUE, alpha=0.12, lw=0)
    ax.axhline(1, color=BLUE, lw=0.8)
    for n in range(0, 6):
        m = s > n
        ax.plot(s[m], 1 - (1 - n / s[m]) ** 2, color=INK, lw=1.4)
    for n in (1, 2):
        m = s > 2 * n
        ax.plot(s[m], 2 * n / s[m], color=MUTED, ls=":", lw=1.0)
    ax.text(4.2, 1.06, "continuum", fontsize=7, color=INK2)
    ax.set(xlabel=r"$s=w/\ell$", ylabel=r"$\lambda\cdot2\tau$", xlim=(0, 6), ylim=(-0.05, 1.2))
    panel_label(ax, "(b)")

    ax = axs[2]
    s = 3.5
    x = np.linspace(-12, 12, 1201) * ell
    for n, c in zip(range(4), RAMP[1:]):
        P = lncosh_P(x, n, s, s * ell)
        ax.plot(x / ell, P / np.max(np.abs(P)), color=c, label=f"{n}")
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$P_n(x)$ (scaled)", xlim=(-12, 12), ylim=(-1.1, 1.1))
    leg = ax.legend(title=r"$n$ ($s=3.5$)", title_fontsize=7.5, loc="upper right", handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(c)")
    fig.savefig(FIG / "lncosh.pdf")
    plt.close(fig)


def fig_lncos():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    w = L / np.pi
    ax = axs[0]
    x = np.linspace(-0.5, 0.5, 1001)[1:-1] * L
    for s, c in zip([0.1, 0.5, 2.0, 8.0], RAMP[1:]):
        ax.plot(x / L, L * lncos_ps(x, s, w), color=c, label=f"{s:g}")
    ax.set(xlabel=r"$x/L$", ylabel=r"$L\,p_s(x)$", xlim=(-0.5, 0.5), ylim=(0, 5.6))
    leg = ax.legend(title=r"$s=a/D$", title_fontsize=7.5, loc="upper right", handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(a)")

    ax = axs[1]
    s = np.logspace(-2, 2, 300)
    for n, c in zip([1, 2, 3], [BLUE, ORANGE, AQUA]):
        ax.plot(s, n**2 + 2 * s * n, color=c, label=f"$n={n}$")
        ax.plot(s, 2 * s * n, color=c, ls=":", lw=1.0)
        ax.axhline(n**2, color=c, ls="--", lw=0.7)
    ax.set(xscale="log", yscale="log", xlabel=r"$s=a/D$", ylabel=r"$\lambda_n\,/\,(\pi^2D/2L^2)$", xlim=(1e-2, 1e2),
           ylim=(0.5, 1e3))
    ax.legend(loc="upper left", handlelength=1.5)
    panel_label(ax, "(b)")

    ax = axs[2]
    s, x0 = 1.0, 0.35 * L
    x = np.linspace(-0.5, 0.5, 801)[1:-1] * L
    for t, c in zip([0.002, 0.006, 0.02, 0.05], RAMP[1:]):
        ax.plot(x / L, L * lncos_p(x, t, x0, s, w), color=c, label=f"{t:g}")
    ax.plot(x / L, L * lncos_ps(x, s, w), color=INK, ls="--", lw=1.0, label=r"$p_s$")
    ax.set(xlabel=r"$x/L$", ylabel=r"$L\,p(x,t\,|\,x_0)$", xlim=(-0.5, 0.5), ylim=(0, 9.5))
    snapshot_legend(ax, loc="upper left", bbox_to_anchor=(-0.02, 1.04))
    panel_label(ax, "(c)")
    fig.savefig(FIG / "lncos.pdf")
    plt.close(fig)


def dw_population(beta, t, a=1.0, n=1500):
    """Probability on the right of the barrier at times t, from a point start at the left minimum."""
    U = dw_U(beta, a)
    x, lam, v = jump_spectrum(U, -3 * a, 3 * a, n, m=n, vectors=True)
    sq = np.exp(-(U(x) - U(a)) / D)  # sqrt(pi) up to a constant
    j = np.argmin(np.abs(x + a))
    right = (x > 0) @ (sq[:, None] * v)  # sum_i>0 sqrt(pi_i) v_in
    coef = right * v[j] / sq[j]
    return np.array([np.sum(coef * np.exp(-lam * tt)) for tt in np.atleast_1d(t)]), lam


def dw_committor(beta, x, a=1.0):
    """pi_c(x): probability of reaching +a before -a (corrected Gardiner 5.2.190 / 9.1.10)."""
    U = dw_U(beta, a)
    y = np.linspace(-a, a, 200001)
    cum = cumulative_trapezoid(np.exp(2 * (U(y) - U(0.0)) / D), y, initial=0)
    return np.interp(x, y, cum / cum[-1], left=0.0, right=1.0)


def fig_double_well():
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.6), gridspec_kw={"wspace": 0.3, "hspace": 0.5})
    betas = [2.0, 4.0, 8.0]
    ax = axs[0, 0]
    x = np.linspace(-2, 2, 801)
    for b, c in zip(betas, RAMP[1:]):
        w = np.exp(-2 * dw_U(b)(x) / D)
        ax.plot(x, w / trapz(w, x), color=c, label=f"{b:g}")
    ax.set(xlabel=r"$x/a$", ylabel=r"$a\,p_s(x)$", xlim=(-2, 2), ylim=(0, 1.75))
    leg = ax.legend(title=r"$2\Delta U/D$", title_fontsize=7.5, loc="upper center", handlelength=1.5, ncol=3,
                    columnspacing=0.8)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(a)")

    ax = axs[0, 1]
    bs = np.linspace(1, 16, 31)
    lams = np.array([dw_eigen(b, n=3000, m=4)[1] for b in bs])
    ka = np.array([dw_curvatures(b)[0] for b in bs])
    ax.plot(bs, lams[:, 1] / ka, color=BLUE, label=r"$\lambda_1$")
    ax.plot(bs, lams[:, 2] / ka, color=ORANGE, label=r"$\lambda_2$")
    ax.plot(bs, lams[:, 3] / ka, color=AQUA, label=r"$\lambda_3$")
    ax.plot(bs, [dw_kramers(b) for b in bs] / ka, color=MUTED, ls="--", lw=1.0, label="Kramers")
    bi = np.array([2.0, 6.0, 10.0, 14.0])
    mc_dots(ax, bi, [dw_gardiner_9218(b) / dw_curvatures(b)[0] for b in bi], BLUE, label="(9.2.18)")
    ax.axhline(0.5, color=MUTED, ls=":", lw=0.8)
    ax.axhline(1.0, color=MUTED, ls=":", lw=0.8)
    ax.set(yscale="log", xlabel=r"$2\Delta U/D$", ylabel=r"$\lambda\,/\,U''(a)$", xlim=(1, 16), ylim=(1e-8, 3))
    ax.legend(loc="lower left", handlelength=1.5, ncol=2, columnspacing=0.8)
    panel_label(ax, "(b)")

    ax = axs[1, 0]
    b = 8.0
    t = np.logspace(-3, 4, 300)
    Nc, lam = dw_population(b, t)
    ax.plot(t, Nc, color=INK, label="exact")
    ax.plot(t, 0.5 * -np.expm1(-lam[1] * t), color=ORANGE, ls="--", lw=1.0, label=r"$\frac{1}{2}(1-e^{-\lambda_1t})$")
    ax.set(xscale="log", xlabel=r"$Dt/a^2$", ylabel=r"$\Pr(x(t)>0)$", xlim=(1e-3, 1e4), ylim=(0, 0.55))
    ax.legend(loc="upper left", handlelength=1.5)
    panel_label(ax, "(c)")

    ax = axs[1, 1]
    for b, c in [(2.0, RAMP[1]), (8.0, RAMP[3])]:
        x, lam, v = dw_eigen(b, n=3000, m=2)
        Q1 = v[:, 1] / np.exp(-(dw_U(b)(x) - dw_U(b)(1.0)) / D)
        Q1 /= Q1[np.argmin(np.abs(x - 1.0))]
        m = np.abs(x) < 2
        ax.plot(x[m], Q1[m], color=c, label=rf"$Q_1$, ${b:g}$")
        ax.plot(x[m], 2 * dw_committor(b, x[m]) - 1, color=c, ls="--", lw=1.0)
    ax.set(xlabel=r"$x/a$", ylabel=r"$Q_1$, $2\pi_c-1$", xlim=(-2, 2), ylim=(-1.6, 1.6))
    ax.legend(loc="upper left", handlelength=1.5)
    panel_label(ax, "(d)")
    fig.savefig(FIG / "double_well.pdf")
    plt.close(fig)


# --- Part VI: interfaces and singular boundaries ---------------------------------------
# A step in B at x = 0, A = 0 on either side. J is continuous and p(0+) = K p(0-). Smoothing the
# step over a width eps and adding the drift chi B'/2 gives K = (D-/D+)^(1 - chi): chi = 0 is Ito,
# 1/2 Stratonovich, 1 a thermal particle. The propagator is a partial image.
IF_DM, IF_DP, IF_A = 0.25, 1.0, 1.0  # D_-, D_+, and the half-width a of the box
CONVENTIONS = [("thermal", 1.0, BLUE), ("Stratonovich", 0.5, AQUA), ("Ito", 0.0, ORANGE)]  # name, chi, colour
CONV_LABEL = {"thermal": "thermal", "Stratonovich": "Stratonovich", "Ito": r"It$\hat{\rm o}$"}


def iface_K(chi, Dm=IF_DM, Dp=IF_DP):
    return (Dm / Dp) ** (1 - chi)


def iface_r(K, Dm=IF_DM, Dp=IF_DP):
    """Weight of the image; the transmitted Gaussian has 1 - r. eta = K sqrt(D+/D-)."""
    eta = K * np.sqrt(Dp / Dm)
    return (eta - 1) / (eta + 1)


def iface_p(x, t, x0, K, Dm=IF_DM, Dp=IF_DP):
    """Propagator from x0 > 0: incident and image Gaussians (variance D+ t) on the right, the transmitted one,
    centred at x0 sqrt(D-/D+) with variance D- t, on the left."""
    r = iface_r(K, Dm, Dp)
    g = lambda y, Dd: np.exp(-(y**2) / (2 * Dd * t)) / np.sqrt(2 * np.pi * Dd * t)
    return np.where(x > 0, g(x - x0, Dp) + r * g(x + x0, Dp), (1 - r) * g(x - x0 * np.sqrt(Dm / Dp), Dm))


def iface_split(x0, K, a=IF_A, Dm=IF_DM, Dp=IF_DP):
    """pi_+(x0) on (-a, a), both walls absorbing: piecewise linear, continuous, K D+ pi'(0+) = D- pi'(0-)."""
    sp = 1 / (a + K * Dp * a / Dm)
    return np.where(x0 > 0, 1 - sp * (a - x0), K * Dp * sp / Dm * (x0 + a))


def iface_T(x0, K, a=IF_A, Dm=IF_DM, Dp=IF_DP):
    """Mean exit time from (-a, a): -x^2/D_pm + slope x + T(0), continuous, with K D+ T'(0+) = D- T'(0-)."""
    sp = a * a * (1 / Dp - 1 / Dm) / (a + K * Dp * a / Dm)
    T0 = a * a / Dp - sp * a
    return np.where(x0 > 0, -(x0**2) / Dp + sp * x0 + T0, -(x0**2) / Dm + K * Dp / Dm * sp * x0 + T0)


def iface_B(x, eps, Dm=IF_DM, Dp=IF_DP):
    """The step smoothed over a width eps, and its derivative."""
    return Dm + (Dp - Dm) * 0.5 * (1 + np.tanh(x / eps)), (Dp - Dm) * 0.5 / eps / np.cosh(x / eps) ** 2


def iface_fd(chi, eps, x0, t, t0=0.01, Lb=5.0, n=8001, dt=2e-4):
    """Smoothed step with drift chi B'/2: conservative finite differences, Crank-Nicolson from a free Gaussian at t0
    (the interface is still 5 widths away then)."""
    x = np.linspace(-Lb, Lb, n)
    h = x[1] - x[0]
    B = iface_B(x, eps)[0]
    Ah = 0.5 * chi * iface_B(0.5 * (x[1:] + x[:-1]), eps)[1]
    ci, cj = 0.5 * Ah + B[:-1] / (2 * h), 0.5 * Ah - B[1:] / (2 * h)  # J_{i+1/2} = ci p_i + cj p_{i+1}
    main = np.zeros(n)
    main[:-1] -= ci / h
    main[1:] += cj / h
    M = sparse.diags([ci / h, main, -cj / h], [-1, 0, 1], format="csc")
    I = sparse.identity(n, format="csc")
    lu = splu(I - 0.5 * dt * M)
    Mp = I + 0.5 * dt * M
    p = np.exp(-((x - x0) ** 2) / (2 * IF_DP * t0)) / np.sqrt(2 * np.pi * IF_DP * t0)
    for _ in range(int(round((t - t0) / dt))):
        p = lu.solve(Mp @ p)
    return x, p


def iface_box_levels(K, m=4, a=IF_A, Dm=IF_DM, Dp=IF_DP):
    """Nonzero rates of the box (-a, a) with reflecting walls: K sqrt(D+) tan th+ + sqrt(D-) tan th- = 0 with
    th_pm = a sqrt(2 lambda/D_pm), multiplied through by the cosines."""
    F = lambda lam: (K * np.sqrt(Dp) * np.sin(a * np.sqrt(2 * lam / Dp)) * np.cos(a * np.sqrt(2 * lam / Dm))
                     + np.sqrt(Dm) * np.sin(a * np.sqrt(2 * lam / Dm)) * np.cos(a * np.sqrt(2 * lam / Dp)))
    grid = np.linspace(1e-6, 60, 60001)
    Fg = F(grid)
    roots = [brentq(F, grid[i], grid[i + 1]) for i in np.flatnonzero(np.sign(Fg[1:]) != np.sign(Fg[:-1]))]
    return np.array(roots[:m])


def iface_jump_levels(K, n=4000, m=4, a=IF_A, Dm=IF_DM, Dp=IF_DP):
    """The same from a jump process on n cells with detailed balance pi = 1 (left), K (right). Bulk bonds have
    conductance B/2h^2; the interface bond sqrt(K) D- D+ / h^2 (D- + K D+), which makes the flux continuous."""
    h = 2 * a / n
    xc = -a + (np.arange(n) + 0.5) * h
    pi = np.where(xc > 0, K, 1.0)
    c = np.where(xc[:-1] + h / 2 > 0, Dp, Dm) / (2 * h * h)
    c[n // 2 - 1] = np.sqrt(K) * Dm * Dp / (h * h * (Dm + K * Dp))
    out = np.zeros(n)
    out[:-1] += c * np.sqrt(pi[1:] / pi[:-1])
    out[1:] += c * np.sqrt(pi[:-1] / pi[1:])
    return eigh_tridiagonal(out, -c, eigvals_only=True, select="i", select_range=(1, m))


@cache
def mc_interface(n_paths=1000, eps=0.03, dt=4e-5):
    """Euler paths of the smoothed step, dx = chi B'/2 dt + sqrt(B) dW, all three conventions at once, with a
    Brownian-bridge test at the walls +-a (where B is constant). Returns x0s and, per convention, the exit wall."""
    rng = np.random.default_rng(20261027)
    x0s = np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]) * IF_A
    chis = np.array([c for _, c, _ in CONVENTIONS])
    x = np.tile(np.repeat(x0s, n_paths), chis.size)
    chi = np.repeat(chis, x0s.size * n_paths)
    right = np.zeros(x.size, bool)
    alive = np.arange(x.size)
    while alive.size:
        xa = x[alive]
        B, B1 = iface_B(xa, eps)
        xb = xa + 0.5 * chi[alive] * B1 * dt + np.sqrt(B * dt) * rng.standard_normal(alive.size)
        inside = np.abs(xb) < IF_A
        pR = np.where(inside, np.exp(-2 * (IF_A - xa) * np.clip(IF_A - xb, 0, None) / (IF_DP * dt)), 0.0)
        pL = np.where(inside, np.exp(-2 * (IF_A + xa) * np.clip(IF_A + xb, 0, None) / (IF_DM * dt)), 0.0)
        u = rng.random(alive.size)
        outR = (xb >= IF_A) | (inside & (u < pR))
        outL = (xb <= -IF_A) | (inside & ~outR & (u < pR + pL))
        right[alive[outR]] = True
        x[alive] = xb
        alive = alive[~(outR | outL)]
    return x0s, right.reshape(chis.size, x0s.size, n_paths)


def fig_interface():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    a = IF_A
    ax = axs[0]
    x = np.linspace(-1.5, 2.5, 2001) * a
    ax.axvspan(-1.5, 0, color=MUTED, alpha=0.12, lw=0)
    for t, c in zip([0.04, 0.1, 0.25, 0.6], RAMP[1:]):
        ax.plot(x / a, a * iface_p(x, t, 0.5 * a, 1.0), color=c, label=f"{t:g}")
    ax.set(xlabel=r"$x/a$", ylabel=r"$a\,p(x,t\,|\,x_0)$", xlim=(-1.5, 2.5), ylim=(0, 2.1))
    snapshot_legend(ax, title=r"$D_+t/a^2$")
    panel_label(ax, "(a)")

    ax = axs[1]
    ax.axvspan(-1, 0, color=MUTED, alpha=0.12, lw=0)
    xs = np.array([-1, 0, 0, 1]) * a
    for name, chi, c in CONVENTIONS:
        K = iface_K(chi)
        ax.plot(xs / a, a * np.array([1, 1, K, K]) / (a * (1 + K)), color=c, label=CONV_LABEL[name])
    ax.set(xlabel=r"$x/a$", ylabel=r"$a\,p_s(x)$", xlim=(-1, 1), ylim=(0, 1.2))
    ax.legend(loc="upper right", handlelength=1.5)
    panel_label(ax, "(b)")

    ax = axs[2]
    ax.axvspan(-1, 0, color=MUTED, alpha=0.12, lw=0)
    x0s, right = mc_interface()
    xx = np.linspace(-1, 1, 401) * a
    for i, (name, chi, c) in enumerate(CONVENTIONS):
        ax.plot(xx / a, iface_split(xx, iface_K(chi)), color=c, label=CONV_LABEL[name])
        mc_dots(ax, x0s / a, right[i].mean(1), c)
    ax.set(xlabel=r"$x_0/a$", ylabel=r"$\pi_+(x_0)$", xlim=(-1, 1), ylim=(0, 1))
    ax.legend(loc="upper left", handlelength=1.5)
    panel_label(ax, "(c)")
    fig.savefig(FIG / "interface.pdf")
    plt.close(fig)


# The square-root process y = r^2: A = dD - 2ky, B = 4Dy on y > 0, any real d (Feller 1951; CIR).
# y = 0 is an exit for d <= 0, regular for 0 < d < 2, an entrance for d >= 2. Lengths in ell^2 = D/k.
def sqrt_scales(t, y0):
    """ybar_t = y0 e^{-2kt} and sigma_t^2 = (ell^2/2)(1 - e^{-2kt}), the per-component OU variance."""
    return y0 * np.exp(-2 * K * t), 0.5 * ELL**2 * -np.expm1(-2 * K * t)


def sqrt_p(y, t, y0, d, sign=+1):
    """p_+ (I_mu: reflected, or the only solution for d >= 2) or p_- (I_{-mu}: absorbed at 0), mu = d/2 - 1."""
    mu = d / 2 - 1
    m, s2 = sqrt_scales(t, y0)
    return ((y / m) ** (mu / 2) * np.exp(-((np.sqrt(y) - np.sqrt(m)) ** 2) / (2 * s2))
            * ive(sign * mu, np.sqrt(y * m) / s2) / (2 * s2))


def sqrt_z(t, y0):
    return y0 / ELL**2 / np.expm1(2 * K * t)


def sqrt_S(t, y0, d):
    """Survival of the absorbed process, P(1 - d/2, z_t); 1 for d >= 2, where the origin is never reached."""
    t = np.asarray(t, float)
    return np.ones_like(t) if d >= 2 else gammainc(1 - d / 2, sqrt_z(t, y0))


def sqrt_ps(y, d):
    return y ** (d / 2 - 1) * np.exp(-y / ELL**2) / (gamma(d / 2) * ELL**d)


def sqrt_qsd(y, lam, d):
    """QSD with decay rate lam in (0, (2-d)k]: M(1 + lam/2k, 2 - d/2, -y/ell^2), normalised with
    int_0^inf M(a, b, -z) dz = (b - 1)/(a - 1)."""
    a, b = 1 + lam / (2 * K), 2 - d / 2
    return hyp1f1(a, b, -y / ELL**2) * (a - 1) / ((b - 1) * ELL**2)


def sqrt_T(y0, d):
    """Mean time to reach 0, int S dt (finite for d < 2)."""
    return quad(lambda t: sqrt_S(t, y0, d), 0, np.inf, limit=400)[0]


def sqrt_branching_step(y, dt, rng):
    """Exact transition of the d = 0 process (Feller's branching diffusion): Poisson(ybar/2 sigma^2) gamma shapes."""
    m, s2 = sqrt_scales(dt, y)
    N = rng.poisson(m / (2 * s2))
    return np.where(N > 0, rng.gamma(np.maximum(N, 1), 2 * s2), 0.0)


def sqrt_paths(T=2.5, dt=0.002, y0=1.0, seed=7):
    """Paths for the figure: d = 3 and d = 1 as squared radii of exact OU steps, the d = 1 path also stopped at its
    first zero, and d = 0 by exact branching steps."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T + dt / 2, dt)
    a_, s = np.exp(-K * dt), np.sqrt(0.5 * ELL**2 * -np.expm1(-2 * K * dt))
    x = np.zeros((t.size, 4))
    x[0] = [np.sqrt(y0), 0, 0, np.sqrt(y0)]  # three components for d = 3, the last for d = 1
    yb = np.zeros(t.size)
    yb[0] = y0
    for i in range(1, t.size):
        x[i] = a_ * x[i - 1] + s * rng.standard_normal(4)
        yb[i] = sqrt_branching_step(yb[i - 1], dt, rng)
    y3, y1 = (x[:, :3] ** 2).sum(1), x[:, 3] ** 2
    hit = np.flatnonzero(np.sign(x[1:, 3]) != np.sign(x[:-1, 3]))
    t_hit = t[hit[0] + 1] if hit.size else np.inf
    return t, y3, y1, t_hit, yb


def fig_sqrt():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    ax = axs[0]
    t, y3, y1, t_hit, yb = sqrt_paths(seed=2422)  # a draw in which each path shows its boundary within kt < 2.5
    i = int(np.argmax(yb == 0))
    ax.plot(t, y3 / ELL**2, color=BLUE, lw=0.9, label=r"$d=3$, entrance")
    ax.plot(t, y1 / ELL**2, color=AQUA, lw=0.9, label=r"$d=1$, regular")
    ax.plot(t[: i + 1], yb[: i + 1] / ELL**2, color=ORANGE, lw=0.9, label=r"$d=0$, exit")
    ax.plot([t[i], t[-1]], [0, 0], color=ORANGE, lw=1.8, solid_capstyle="butt")
    ax.plot([t_hit], [0], "x", color=INK, ms=5, mew=1.3, zorder=4)
    ax.set(xlabel=r"$kt$", ylabel=r"$y/\ell^2$", xlim=(0, t[-1]), ylim=(-0.08, 3.6))
    ax.legend(loc="upper right", handlelength=1.2, fontsize=6.5)
    panel_label(ax, "(a)")

    ax = axs[1]
    for (lo, hi), lbl in [((-2, 0), "exit"), ((0, 2), "regular"), ((2, 4), "entrance")]:
        ax.axvspan(lo, hi, color=[ORANGE, MUTED, BLUE][[-2, 0, 2].index(lo)], alpha=0.10, lw=0)
        ax.text(0.5 * (lo + hi), 8.45, lbl, ha="center", fontsize=7, color=INK2, clip_on=False)
    d = np.linspace(-2, 4, 601)
    for n in range(5):
        ax.plot(d[d > 0], 2 * n * K + 0 * d[d > 0], color=BLUE, lw=1.2, label="reflected" if n == 0 else None)
    for n in range(4):
        ax.plot(d[d < 2], (2 * n + 2 - d[d < 2]) * K, color=ORANGE, lw=1.2, label="absorbed" if n == 0 else None)
    ax.text(2.12, 0.75, "reflected", fontsize=7, color=BLUE)
    ax.text(-1.92, 0.75, "absorbed", fontsize=7, color=ORANGE)
    ax.set(xlabel=r"$d$", ylabel=r"$\lambda_n/k$", xlim=(-2, 4), ylim=(-0.3, 8.2))
    panel_label(ax, "(b)")

    ax = axs[2]
    y = np.logspace(-5, np.log10(4), 400) * ELL**2
    tt, y0 = 0.25, ELL**2
    for dd, sgn, c, ls, lbl in [(3.0, +1, BLUE, "-", r"$d=3$"), (1.0, +1, AQUA, "-", r"$d=1$, reflected"),
                                 (1.0, -1, AQUA, "--", r"$d=1$, absorbed"), (-1.0, -1, ORANGE, "--", r"$d=-1$")]:
        ax.plot(y / ELL**2, ELL**2 * sqrt_p(y, tt, y0, dd, sgn), color=c, ls=ls, lw=1.2, label=lbl)
    ax.set(xscale="log", yscale="log", xlabel=r"$y/\ell^2$", ylabel=r"$\ell^2\,p(y,t\,|\,y_0)$", xlim=(1e-5, 4),
           ylim=(1e-3, 30))
    ax.legend(loc="lower left", handlelength=1.8, fontsize=6.5)
    panel_label(ax, "(c)")
    fig.savefig(FIG / "sqrt.pdf")
    plt.close(fig)


def feller_type(f, g, a, x0, cuts=(1e-12, 1e-16)):
    """Gardiner's (5.2.25)-(5.2.29) at a left boundary a: are f, g, h1 = f int_{x0}^x g and h2 = g int_{x0}^x f
    integrable on (a, x0)? An integral counts as finite if it no longer changes between the two cutoffs."""
    u0 = np.log(x0 - a)
    on_log = lambda h, c: quad(lambda u: h(a + np.exp(u)) * np.exp(u), np.log(c), u0, limit=400)[0]
    G = lambda x: on_log(g, x - a)  # int_x^{x0} g = -int_{x0}^x g; the sign does not matter for integrability
    F = lambda x: on_log(f, x - a)
    finite = lambda h: abs(on_log(h, cuts[1]) - on_log(h, cuts[0])) <= 1e-3 * max(1.0, abs(on_log(h, cuts[1])))
    fL, gL = finite(f), finite(g)
    if fL and gL:
        return "regular"
    if not gL and finite(lambda x: f(x) * G(x)):
        return "exit"
    if gL and finite(lambda x: g(x) * F(x)):
        return "entrance"
    return "natural"


# A current through infinity: the quadratic integrate-and-fire neuron, A = x^2 + I, B = D, reinjected at -infinity.
# theta = 2 arctan x maps the line onto a circle; both infinities are theta = pi, where the noise vanishes.
QIF_D = [1.0, 0.25]
QIF_I = [-1.0, 0.0, 1.0]


def qif_T(I, Dq):
    """Mean period 1/J_s = sqrt(2 pi/D) int_0^inf u^{-1/2} exp[-(2/D)(u^3/12 + I u)] du, with u = w^2."""
    f = lambda w: 2 * np.exp(-(2 / Dq) * (w**6 / 12 + I * w**2))
    pre = np.sqrt(2 * np.pi / Dq)
    if I >= 0:
        return pre * quad(f, 0, np.inf, limit=400)[0]
    w = np.sqrt(2 * np.sqrt(-I))  # below threshold, I = -c^2, the integrand peaks at u = 2c
    return pre * (quad(f, 0, w, limit=400)[0] + quad(f, w, np.inf, limit=400)[0])


def qif_ps(x, I, Dq):
    """(2 J_s/D) int_0^inf exp[phi(x) - phi(x + u)] du, phi = (2/D)(x^3/3 + I x)."""
    J = 1 / qif_T(I, Dq)

    def one(xx):
        # far out the integrand decays within u ~ D/2(x^2 + I); below threshold it can rise again, up to
        # u = c - x where phi has its minimum (the barrier top +c). Knots at both.
        e = lambda u: np.exp(-(2 / Dq) * (u**3 / 3 + xx * u**2 + (xx**2 + I) * u))
        s = min(1.0, Dq / (2 * abs(xx**2 + I) + 1e-300))
        knots = sorted({0.0, 40 * s} | ({np.sqrt(-I) - xx} if I < 0 and np.sqrt(-I) > xx else set()))
        return (sum(quad(e, lo, hi, limit=200)[0] for lo, hi in zip(knots[:-1], knots[1:]))
                + quad(e, knots[-1], np.inf, limit=200)[0])

    return 2 * J / Dq * np.array([one(xx) for xx in np.atleast_1d(x)])


@cache
def mc_qif(I, Dq, n=1000, T=60.0, burn=5.0, dt=2e-3):
    """Firing rate from Euler steps of the theta equation (bounded coefficients); a spike is a winding of theta."""
    rng = np.random.default_rng(20261030 + int(100 * I) + int(100 * Dq))
    th = rng.uniform(-np.pi, np.pi, n)
    sq = np.sqrt(Dq * dt)

    def run(duration):
        for _ in range(int(round(duration / dt))):
            c, s = np.cos(th), np.sin(th)
            th[:] += (1 - c + I * (1 + c) - 0.5 * Dq * s * (1 + c)) * dt + sq * (1 + c) * rng.standard_normal(n)

    run(burn)
    start = th.copy()
    run(T)
    rate = (th - start) / (2 * np.pi * T)
    return rate.mean(), rate.std() / np.sqrt(n)


def fig_qif():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    ax = axs[0]
    x = np.linspace(-5, 5, 401)
    for I, c in zip(QIF_I, [BLUE, AQUA, ORANGE]):
        ax.plot(x, qif_ps(x, I, 1.0), color=c, label=f"{I:g}")
        J = 1 / qif_T(I, 1.0)
        xt = x[np.abs(x) > 2]
        ax.plot(xt[xt > 0], J / xt[xt > 0] ** 2, color=c, ls=":", lw=1.0)
        ax.plot(xt[xt < 0], J / xt[xt < 0] ** 2, color=c, ls=":", lw=1.0)
    ax.set(xlabel=r"$x$", ylabel=r"$p_s(x)$", xlim=(-5, 5), ylim=(0, 0.75))
    leg = ax.legend(title=r"$I$ ($D=1$)", title_fontsize=7.5, loc="upper right", handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(a)")

    ax = axs[1]
    Is = np.linspace(-2, 3, 201)
    for Dq, c in zip(QIF_D, [BLUE, ORANGE]):
        ax.plot(Is, [1 / qif_T(I, Dq) for I in Is], color=c, label=rf"$D={Dq:g}$")
        cc = np.sqrt(-Is[Is < -0.3])
        ax.plot(Is[Is < -0.3], cc / np.pi * np.exp(-8 * cc**3 / (3 * Dq)), color=c, ls=":", lw=1.0)
        Imc = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0] if Dq == 1.0 else [-0.25, 0.0, 0.5, 1.0, 2.0]
        mc_dots(ax, Imc, [mc_qif(I, Dq)[0] for I in Imc], c)
    Ip = Is[Is > 0]
    ax.plot(Ip, np.sqrt(Ip) / np.pi, color=INK, ls="--", lw=1.0, label=r"$\sqrt{I}/\pi$")
    ax.set(yscale="log", xlabel=r"$I$", ylabel=r"firing rate $J_s$", xlim=(-2, 3), ylim=(1e-4, 1))
    ax.legend(loc="lower right", handlelength=1.5)
    panel_label(ax, "(b)")

    ax = axs[2]
    th = np.linspace(-np.pi, np.pi, 401)[1:-1]
    for I, c in zip(QIF_I, [BLUE, AQUA, ORANGE]):
        xx = np.tan(th / 2)
        ax.plot(th / np.pi, qif_ps(xx, I, 1.0) * (1 + xx**2) / 2, color=c, label=f"{I:g}")
        ax.plot([1], [0.5 / qif_T(I, 1.0)], "o", ms=3.5, color=c)
    ax.set(xlabel=r"$\theta/\pi$", ylabel=r"$p_s(\theta)$", xlim=(-1, 1), ylim=(0, 0.9))
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    panel_label(ax, "(c)")
    fig.savefig(FIG / "qif.pdf")
    plt.close(fig)


TAILS = [0.25, 0.5, 0.75]  # initial tail parameters: beta ell (drift) and nu (OU)


@cache
def attraction_rates():
    """Measured long-time decay rates of S against the tail parameter, in units of the bottom of the L^2
    spectrum (1/2 tau for drift, k for OU)."""
    v, s_ = -1.0, [0.25, 0.5, 0.75, 1.5]
    x = np.linspace(0, 250, 50001)
    drift = []
    for s in s_:
        t1, t2 = (40.0, 60.0) if s < 1 else (300.0, 400.0)
        l1, l2 = (dha_conditioned_exp(x, t, s, v)[1] for t in (t1, t2))
        drift.append((l1 - l2) / (t2 - t1) / (v**2 / (2 * D)))
    ou = [np.log(ou_hl_evolve(lomax(s), [ELL], 15 * TAU)[1] / ou_hl_evolve(lomax(s), [ELL], 20 * TAU)[1]) / (5 * TAU * K)
          for s in s_]
    return s_, drift, ou


def fig_attraction():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.3), gridspec_kw={"wspace": 0.42})
    colors = RAMP[1:4]
    ax = axs[0]
    v, t = -1.0, 60.0
    x = np.linspace(0, 250, 50001)
    for s, c in zip(TAILS, colors):
        ax.plot(x, dha_conditioned_exp(x, t, s, v)[0], color=c, label=f"{s:g}")
    p_point = dha_p(x, t, 1.0, v)
    ax.plot(x, p_point / trapz(p_point, x), color=INK, label="point")
    for s in TAILS + [1.0]:
        ax.plot(x, dha_qsd(x, dha_lam_of_tail(s, v), v), color=MUTED, ls="--", lw=0.9)
    ax.set(yscale="log", xlabel=r"$x/\ell$", ylabel=r"$\ell\,q(x,t)$", xlim=(0, 14), ylim=(1e-5, 1.5))
    leg = ax.legend(title=r"tail $\beta\ell$", title_fontsize=7.5, loc="lower left", bbox_to_anchor=(0.05, 0),
                    handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(a)")

    ax = axs[1]
    t = 12 * TAU
    x = np.logspace(-2, 2, 300) * ELL
    for s, c in zip(TAILS, colors):
        p, S = ou_hl_evolve(lomax(s), x, t, ny=100001)
        ax.plot(x / ELL, ELL * p / S, color=c, label=f"{s:g}")
    ax.plot(x / ELL, ELL * ou_half(x, t, X0_OU * ELL, -1) / ou_hl_survival(t, X0_OU * ELL), color=INK, label="point")
    for s in TAILS + [1.0]:
        ax.plot(x / ELL, ELL * ou_qsd_nu(x, s), color=MUTED, ls="--", lw=0.9)
    ax.set(xscale="log", yscale="log", xlabel=r"$x/\ell$", ylabel=r"$\ell\,q(x,t)$", xlim=(1e-2, 1e2), ylim=(1e-7, 3))
    leg = ax.legend(title=r"tail $\nu$", title_fontsize=7.5, loc="lower left", handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(b)")

    ax = axs[2]
    s = np.linspace(0, 2, 201)
    ax.plot(s, np.where(s < 1, s * (2 - s), 1), color=BLUE, label="drift")
    ax.plot(s, np.minimum(s, 1), color=ORANGE, label="OU")
    s_, drift, ou = attraction_rates()
    mc_dots(ax, s_, drift, BLUE)
    mc_dots(ax, s_, ou, ORANGE)
    ax.set(xlabel=r"tail: $\beta\ell$ (drift), $\nu$ (OU)", ylabel=r"$\lambda/\lambda_{\mathrm{b}}$", xlim=(0, 2),
           ylim=(0, 1.15))
    ax.legend(loc="lower right", handlelength=1.5)
    panel_label(ax, "(c)")
    fig.savefig(FIG / "qsd_attraction.pdf")
    plt.close(fig)


def fig_ou_line():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"wspace": 0.28})
    ax = axs[0]
    xi = np.linspace(-2.5, 3.5, 1201)
    for s_, c in zip([0.05, 0.2, 0.5, 1.0, 3.0], RAMP):
        ax.plot(xi, ELL * ou_line(xi * ELL, s_ * TAU, X0_OU * ELL), color=c, label=f"{s_:g}")
    ax.plot(xi, np.exp(-(xi**2)) / np.sqrt(np.pi), color=INK, ls="--", lw=1.0, label=r"$p_s$")
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,p(x,t\,|\,x_0,0)$", xlim=(-2.5, 3.5), ylim=(0, 1.9))
    snapshot_legend(ax, title=r"$t/\tau$")
    panel_label(ax, "(a)")

    ax = axs[1]
    s_ = np.linspace(0, 4, 200)
    m1 = X0_OU * np.exp(-s_)
    m2 = (X0_OU**2 - 0.5) * np.exp(-2 * s_)
    ax.plot(s_, m1, color=BLUE, label=r"$\langle x\rangle/\ell$, rate $k$")
    ax.plot(s_, m2, color=ORANGE, label=r"$(\langle x^2\rangle-\ell^2/2)/\ell^2$, rate $2k$")
    ax.set(yscale="log", xlabel=r"$t/\tau$", xlim=(0, 4), ylim=(1e-3, 5))
    ax.legend(loc="upper right", handlelength=1.5)
    panel_label(ax, "(b)")
    fig.savefig(FIG / "ou_line.pdf")
    plt.close(fig)


def ou_folded_mean(t, x0):
    m, var = ou_mean_var(t, x0)
    sd = np.sqrt(var)
    return m * erf(m / (np.sqrt(2) * sd)) + sd * np.sqrt(2 / np.pi) * np.exp(-(m**2) / (2 * var))


def fig_ou_halfline_reflecting():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"wspace": 0.28})
    ax = axs[0]
    xi = np.linspace(0, 3.5, 1001)
    for s_, c in zip([0.05, 0.2, 0.5, 1.0, 3.0], RAMP):
        ax.plot(xi, ELL * ou_half(xi * ELL, s_ * TAU, X0_OU * ELL, +1), color=c, label=f"{s_:g}")
    ax.plot(xi, 2 * np.exp(-(xi**2)) / np.sqrt(np.pi), color=INK, ls="--", lw=1.0, label=r"$p_s$")
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,p(x,t\,|\,x_0,0)$", xlim=(0, 3.5), ylim=(0, 1.9))
    snapshot_legend(ax, title=r"$t/\tau$")
    panel_label(ax, "(a)")

    ax = axs[1]
    s_ = np.linspace(1e-3, 4, 300)
    ax.plot(s_, np.abs(ou_folded_mean(s_ * TAU, X0_OU * ELL) / ELL - 1 / np.sqrt(np.pi)), color=BLUE,
            label=r"$|\langle x\rangle-\langle x\rangle_s|/\ell$")
    ax.plot(s_, (X0_OU**2 - 0.5) * np.exp(-2 * s_), color=ORANGE, label=r"$(\langle x^2\rangle-\ell^2/2)/\ell^2$")
    ax.plot(s_, X0_OU * np.exp(-s_), color=MUTED, ls=":", lw=1.2, label=r"line: $\langle x\rangle/\ell$ (rate $k$)")
    ax.set(yscale="log", xlabel=r"$t/\tau$", xlim=(0, 4), ylim=(1e-4, 5))
    ax.legend(loc="lower left", handlelength=1.5)
    panel_label(ax, "(b)")
    fig.savefig(FIG / "ou_halfline_reflecting.pdf")
    plt.close(fig)


def fig_ou_halfline_absorbing():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    ax = axs[0]
    xi = np.linspace(0, 3.5, 1001)
    for s_, c in zip([0.1, 0.3, 0.6, 1.0, 3.0], RAMP):
        q = ou_half(xi * ELL, s_ * TAU, X0_OU * ELL, -1) / ou_hl_survival(s_ * TAU, X0_OU * ELL)
        ax.plot(xi, ELL * q, color=c, label=f"{s_:g}")
    ax.plot(xi, ELL * ou_hl_qsd(xi * ELL), color=INK, ls="--", lw=1.0, label="QSD")
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,q(x,t)$", xlim=(0, 3.5), ylim=(0, 2.4))
    snapshot_legend(ax, title=r"$t/\tau$")
    panel_label(ax, "(a)")

    ax = axs[1]
    s_ = np.linspace(1e-3, 6, 300)
    ax.plot(s_, ou_hl_survival(s_ * TAU, X0_OU * ELL), color=INK, label="exact")
    ax.plot(s_, 2 / np.sqrt(np.pi) * X0_OU * np.exp(-s_), color=MUTED, ls="--", lw=1.0,
            label=r"$\frac{2}{\sqrt{\pi}}\frac{x_0}{\ell}e^{-t/\tau}$")
    ax.set(yscale="log", xlabel=r"$t/\tau$", ylabel=r"$S(t\,|\,x_0)$", xlim=(0, 6), ylim=(1e-2, 1.5))
    ax.legend(loc="lower left", handlelength=1.5)
    panel_label(ax, "(b)")

    ax = axs[2]
    xi0 = np.logspace(-2, 2, 60)
    ax.plot(xi0, [ou_hl_T(v * ELL) / TAU for v in xi0], color=INK, label="exact")
    ax.plot(xi0[xi0 > 1], np.log(2 * xi0[xi0 > 1]) + np.euler_gamma / 2, color=MUTED, ls="--", lw=1.0,
            label=r"$\ln(2x_0/\ell)+\gamma/2$")
    ax.set(xscale="log", xlabel=r"$x_0/\ell$", ylabel=r"$T(x_0)/\tau$", xlim=(1e-2, 1e2), ylim=(0, 5.5))
    ax.legend(loc="upper left", handlelength=1.5)
    panel_label(ax, "(c)")
    fig.savefig(FIG / "ou_halfline_absorbing.pdf")
    plt.close(fig)


def length_legend(ax, **kw):
    kw = {"loc": "upper right", "handlelength": 1.5, **kw}
    leg = ax.legend(title=r"$L/\ell$", title_fontsize=7.5, **kw)
    leg.get_title().set_color(INK2)


def fig_ou_reflecting():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"wspace": 0.28})
    ax = axs[0]
    for Lxi, c in zip(L_OU, RAMP[1:]):
        xi = np.linspace(0, Lxi, 400)
        w = np.exp(-(xi**2))
        ax.plot(xi, w / trapz(w, xi), color=c, label=f"{Lxi:g}")
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,p_s(x)$", xlim=(0, 4), ylim=(0, 2.4))
    length_legend(ax)
    panel_label(ax, "(a)")

    ax = axs[1]
    Lxi, x0 = 2.0, 1.5
    xi = np.linspace(0, Lxi, 801)
    for s_, c in zip([0.05, 0.1, 0.3, 0.6, 2.0], RAMP):
        ax.plot(xi, ELL * ou_p(xi * ELL, s_ * TAU, x0 * ELL, Lxi, "reflecting"), color=c, label=f"{s_:g}")
    w = np.exp(-(xi**2))
    ax.plot(xi, w / trapz(w, xi), color=INK, ls="--", lw=1.0, label=r"$p_s$")
    ax.set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,p(x,t\,|\,x_0,0)$", xlim=(0, Lxi), ylim=(0, 2.4))
    snapshot_legend(ax, title=r"$t/\tau$", loc="upper left", bbox_to_anchor=(-0.02, 1.04))
    panel_label(ax, "(b)")
    fig.savefig(FIG / "ou_reflecting.pdf")
    plt.close(fig)


def fig_ou_absorbing():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    u = np.linspace(1e-4, 1, 400)
    for Lxi, c in zip(L_OU, RAMP[1:]):
        axs[0].plot(u * Lxi, ELL * ou_qsd(u * Lxi * ELL, Lxi, "absorbing"), color=c, label=f"{Lxi:g}")
        axs[1].plot(u, ou_piL(u * Lxi * ELL, Lxi), color=c, label=f"{Lxi:g}")
        axs[2].plot(u, ou_T(u * Lxi * ELL, Lxi, "absorbing") / TAU, color=c, label=f"{Lxi:g}")
    xi = np.linspace(0, 4, 400)
    axs[0].plot(xi, ELL * ou_hl_qsd(xi * ELL), color=INK, ls="--", lw=1.0, label=r"$\infty$")
    axs[0].set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,q_\infty(x)$", xlim=(0, 4), ylim=(0, 4.2))
    length_legend(axs[0])
    axs[1].set(xlabel=r"$x_0/L$", ylabel=r"$\pi_L(x_0)$", xlim=(0, 1), ylim=(0, 1.02))
    axs[2].set(yscale="log", xlabel=r"$x_0/L$", ylabel=r"$T(x_0)/\tau$", xlim=(0, 1), ylim=(1e-3, 3))
    for ax, lbl in zip(axs, "abc"):
        panel_label(ax, f"({lbl})")
    fig.savefig(FIG / "ou_absorbing.pdf")
    plt.close(fig)


def fig_ou_mixed():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    u = np.linspace(0, 1 - 1e-4, 400)
    for Lxi, c in zip(L_OU, RAMP[1:]):
        axs[0].plot(u * Lxi, ELL * ou_qsd(u * Lxi * ELL, Lxi, "mixed"), color=c, label=f"{Lxi:g}")
        axs[1].plot(u, ou_T(u * Lxi * ELL, Lxi, "mixed") / TAU, color=c, label=f"{Lxi:g}")
    xi = np.linspace(0, 4, 400)
    axs[0].plot(xi, 2 * np.exp(-(xi**2)) / np.sqrt(np.pi), color=INK, ls="--", lw=1.0, label=r"$p_s$, $\infty$")
    axs[0].set(xlabel=r"$x/\ell$", ylabel=r"$\ell\,q_\infty(x)$", xlim=(0, 4), ylim=(0, 3.4))
    length_legend(axs[0])
    axs[1].set(yscale="log", xlabel=r"$x_0/L$", ylabel=r"$T(x_0)/\tau$", xlim=(0, 1), ylim=(1e-3, 1e7))

    ax = axs[2]
    Ls = np.linspace(0.3, 4.2, 60)
    ax.plot(Ls**2, [ou_T(0.0, Lxi, "mixed")[0] / TAU for Lxi in Ls], color=INK, label=r"$T(0)$")
    ax.plot(Ls**2, 1 / ou_kramers(Ls), color=MUTED, ls="--", lw=1.0, label="Kramers")
    Lp = np.array([1.0, 2.0, 3.0, 4.0])
    mc_dots(ax, Lp**2, [1 / ou_modes(Lxi, "mixed")[0][0] for Lxi in Lp], ORANGE, label=r"$1/\lambda_0$")
    ax.set(yscale="log", xlabel=r"$kL^2/D=\Delta U/(D/2)$", ylabel=r"time$/\tau$", xlim=(0, 18), ylim=(0.1, 1e7))
    ax.legend(loc="upper left", handlelength=1.5)
    for a_, lbl in zip(axs, "abc"):
        panel_label(a_, f"({lbl})")
    fig.savefig(FIG / "ou_mixed.pdf")
    plt.close(fig)


def fig_ou_summary():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    Ls = np.logspace(np.log10(0.2), np.log10(4.5), 50)
    rate = {kind: [ou_modes(Lxi, kind, 2)[1 if kind == "reflecting" else 0][0] for Lxi in Ls]
            for kind in ["reflecting", "absorbing", "mixed"]}
    ax.plot(Ls, rate["reflecting"], color=BLUE, label="both reflecting (gap)")
    ax.plot(Ls, rate["absorbing"], color=ORANGE, label="both absorbing")
    ax.plot(Ls, rate["mixed"], color=AQUA, label=r"refl. $0$, abs. $L$")
    small = Ls < 1.2
    ax.plot(Ls[small], np.pi**2 / (2 * Ls[small] ** 2), color=MUTED, ls=":", lw=1.0)
    ax.plot(Ls[small], np.pi**2 / (8 * Ls[small] ** 2), color=MUTED, ls=":", lw=1.0)
    ax.plot(Ls[Ls > 1.5], ou_kramers(Ls[Ls > 1.5]), color=MUTED, ls="--", lw=1.0)
    for y in (1, 2):
        ax.axhline(y, color=MUTED, ls="--", lw=0.8)
    ax.text(0.42, 45, r"$\frac{\pi^2D}{2L^2}$", fontsize=7.5, color=INK2)
    ax.text(0.23, 3.0, r"$\frac{\pi^2D}{8L^2}$", fontsize=7.5, color=INK2)
    ax.text(3.0, 2.25, r"$2k$", fontsize=7.5, color=INK2)
    ax.text(3.0, 1.1, r"$k$", fontsize=7.5, color=INK2)
    ax.text(2.0, 2e-3, "Kramers", fontsize=7.5, color=INK2)
    ax.set(xscale="log", yscale="log", xlabel=r"$L/\ell$", ylabel=r"slowest rate $\times\,\tau$",
           xlim=(0.2, 4.5), ylim=(1e-6, 200))
    ax.legend(loc="lower left", handlelength=1.5)
    fig.savefig(FIG / "ou_summary.pdf")
    plt.close(fig)


def fig_circle():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    ax = axs[0]
    y = np.linspace(-0.5, 0.5, 801)
    for t, c in zip(SNAP_C, RAMP):
        ax.plot(y, L * circ_images(y * L, t, 0.0), color=c, label=f"{t:g}")
    ax.axhline(1, color=INK, ls="--", lw=1.0, label=r"$p_s$")
    ax.set(xlabel=r"$(x-x_0)/L$", ylabel=r"$L\,p(x,t\,|\,x_0,0)$", xlim=(-0.5, 0.5), ylim=(0, 9.5))
    snapshot_legend(ax)
    panel_label(ax, "(a)")

    ax = axs[1]
    t = 0.15
    for a_ in (-1.5, 0.5):
        ax.axvspan(a_, a_ + 1, color=GRID, alpha=0.6, lw=0)
    yy = np.linspace(-1.75, 1.75, 1401)
    ax.plot(yy, L * heat(yy * L, t), color=MUTED, lw=1.2, label=r"line: $g$")
    yc = np.linspace(-0.5, 0.5, 401)
    for n in (-1, 1):
        ax.plot(yc, L * heat((yc + n) * L, t), color=ORANGE, ls="--", lw=1.0,
                label=r"$n=\pm1$ terms" if n == 1 else None)
    ax.plot(yc, L * circ_images(yc * L, t, 0.0), color=BLUE, label="circle: sum of all")
    ax.set(xlabel=r"lift $(\tilde x-x_0)/L$", ylabel=r"$L\,p$", xlim=(-1.75, 1.75), ylim=(0, 1.75))
    ax.legend(loc="upper right", bbox_to_anchor=(1.06, 1.06), handlelength=1.5)
    panel_label(ax, "(b)")

    ax = axs[2]
    for n in range(4):
        lv = n**2
        if n == 0:
            ax.plot([0.15, 0.85], [lv, lv], color=BLUE, lw=1.8)
        else:
            ax.plot([0.15, 0.49], [lv, lv], color=BLUE, lw=1.8)
            ax.plot([0.51, 0.85], [lv, lv], color=ORANGE, lw=1.8)
            ax.plot([2.15, 2.85], [lv, lv], color=ORANGE, lw=1.8)
        ax.plot([1.15, 1.85], [lv, lv], color=BLUE, lw=1.8)
    for i, lbl in enumerate(["circle,\n" r"length $L$", "reflecting\n" r"$(0,L/2)$", "absorbing\n" r"$(0,L/2)$"]):
        ax.text(i + 0.5, -0.6, lbl, ha="center", va="top", fontsize=7, color=INK2)
    ax.plot([], [], color=BLUE, label=r"even about $x_0$")
    ax.plot([], [], color=ORANGE, label=r"odd about $x_0$")
    ax.legend(loc="upper left", handlelength=1.5)
    ax.set(xlim=(0, 3), ylim=(-0.4, 12.5), ylabel=r"$\lambda\,/\,(2\pi^2D/L^2)$")
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    panel_label(ax, "(c)")
    fig.savefig(FIG / "circle.pdf")
    plt.close(fig)


def fig_drift_circle():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.6})
    ax = axs[0]
    pe = 40.0
    xs = np.linspace(0, 1, 241)
    ts = np.linspace(0.0015, 0.12, 320)
    P = np.array([L * circ_images(xs * L, t, 0.5 * L, v=pe * D / L) for t in ts])
    cmap = LinearSegmentedColormap.from_list("ramp", ["#ffffff"] + RAMP)
    im = ax.pcolormesh(ts, xs, P.T, cmap=cmap, vmin=0, vmax=3, shading="auto", rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.08, ticks=[0, 1, 2, 3])
    cb.ax.set_title(r"$L\,p$", color=INK2, fontsize=8)
    cb.outline.set_visible(False)
    ax.set(xlabel=r"$Dt/L^2$", ylabel=r"$x/L$", xlim=(0, 0.12), ylim=(0, 1))
    ax.grid(False)
    panel_label(ax, "(a)")

    ax = axs[1]
    q = 2 * np.pi * np.arange(-3, 4)
    for v, c in zip(PE[2:], DIVERGING[2:]):
        ax.plot(0.5 * q**2, v * q, "o", ms=4.5, mfc=c, mec="white", mew=0.8, color=c, label=f"{v:g}", zorder=3)
    ax.axvline(2 * np.pi**2, color=INK, ls=":", lw=1.0)
    ax.text(2 * np.pi**2 + 5, -100, "gap", fontsize=7, color=INK2)
    ax.set(xlabel=r"$\mathrm{Re}\,\lambda\;L^2/D$", ylabel=r"$\mathrm{Im}\,\lambda\;L^2/D$",
           xlim=(-10, 190), ylim=(-110, 120))
    pe_legend(ax, loc="upper center", bbox_to_anchor=(0.6, 1.06), ncol=3, columnspacing=0.8)
    panel_label(ax, "(b)")

    ax = axs[2]
    u = np.linspace(1e-3, 1 - 1e-3, 300)
    for v, c in zip(PE, DIVERGING):
        ax.plot(u, da_T(u * L, v), color=c, label=f"{v:g}")
    ax.set(xlabel=r"$d/L$", ylabel=r"$T(d)\;D/L^2$", xlim=(0, 1), ylim=(0, 0.42))
    pe_legend(ax, loc="upper center", ncol=3, columnspacing=0.8)
    panel_label(ax, "(c)")
    fig.savefig(FIG / "drift_circle.pdf")
    plt.close(fig)


def fig_ou_circle():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.42})
    ax = axs[0]
    u = np.linspace(-0.5, 0.5, 801)
    for Lc, c in zip(SAW_L, RAMP[1:]):
        ax.plot(u, Lc * saw_ps(u * Lc, Lc), color=c, label=f"{Lc:g}")
    ax.set(xlabel=r"$x/L$", ylabel=r"$L\,p_s(x)$", xlim=(-0.5, 0.5), ylim=(0, 2.7))
    length_legend(ax)
    panel_label(ax, "(a)")

    ax = axs[1]
    Ls = np.linspace(1.2, 9, 80)
    for kind, c, lbl in [("reflecting", BLUE, "even (refl.)"), ("absorbing", ORANGE, "odd (abs.)")]:
        s0 = 1 if kind == "reflecting" else 0
        lv = np.array([[m_[0] for m_ in ou_modes(Lc / (2 * ELL), kind, 8)[s0:s0 + 4]] for Lc in Ls])
        for j in range(4):
            ax.plot(Ls, lv[:, j], color=c, label=lbl if j == 0 else None)
    for n in range(1, 4):
        Lf = Ls[Ls < 5 * n]
        ax.plot(Lf, 2 * np.pi**2 * n**2 / Lf**2, color=MUTED, ls=":", lw=1.0)
    for n in range(1, 8):
        ax.axhline(n, color=MUTED, ls="--", lw=0.6)
    ax.set(xlabel=r"$L/\ell$", ylabel=r"$\lambda/k$", xlim=(1.2, 9), ylim=(0, 7.5))
    ax.legend(loc="lower right", handlelength=1.2, ncol=2, columnspacing=0.6, bbox_to_anchor=(1.04, -0.02))
    panel_label(ax, "(b)")

    ax = axs[2]
    Ls = np.linspace(0.3, 9, 120)
    ax.plot(Ls, saw_Deff(Ls) / D, color=INK, label="exact")
    Lk = Ls[Ls > 3.5]
    ax.plot(Lk, 0.5 * Lk**2 * ou_kramers(Lk / 2) * K / D, color=MUTED, ls="--", lw=1.0,
            label=r"$\frac{1}{2}L^2\lambda_0^{\mathrm{mix}}$")
    Lp = [1.0, 3.0, 5.0, 7.0]
    mc_dots(ax, Lp, [saw_ring(Lc)[2] / D for Lc in Lp], ORANGE, label="discretised")
    ax.set(yscale="log", xlabel=r"$L/\ell$", ylabel=r"$D_{\mathrm{eff}}/D$", xlim=(0, 9), ylim=(1e-6, 3))
    ax.legend(loc="lower left", handlelength=1.5)
    panel_label(ax, "(c)")
    fig.savefig(FIG / "ou_circle.pdf")
    plt.close(fig)


def fig_cos_circle():
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.6), gridspec_kw={"wspace": 0.3, "hspace": 0.55})
    kappa_colors = [RAMP[1], RAMP[2], RAMP[4]]

    ax = axs[0, 0]
    kappa = 3.0
    Lc = 2 * np.pi * cos_R(kappa)
    for f, c in zip([0.0, 0.5, 1.0, 1.5, 3.0], RAMP):
        x, ps, _ = circle_ps(cos_phi(kappa, f), Lc, n=800)
        u = np.append(x / Lc, 1.0)
        u = np.where(u > 0.5, u - 1, u)
        order = np.argsort(u, kind="stable")
        ax.plot(u[order], Lc * np.append(ps, ps[0])[order], color=c, label=f"{f:g}")
    ax.set(xlabel=r"$x/L$", ylabel=r"$L\,p_s(x)$", xlim=(-0.5, 0.5), ylim=(0, 4.4))
    leg = ax.legend(title=r"$v/kR$", title_fontsize=7.5, loc="upper left", handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(a)")

    ax = axs[0, 1]
    ks = np.logspace(np.log10(0.05), 2, 60)
    lv = np.array([cos_spectrum(kp)[0][1:5] for kp in ks])
    for j in range(4):
        ax.plot(ks, lv[:, j], color=BLUE, label="exact (doublets)" if j == 0 else None)
    for m_ in range(1, 4):
        ax.plot(ks, m_**2 / ks, color=MUTED, ls=":", lw=1.0, label=r"free circle" if m_ == 1 else None)
    for n in range(1, 5):
        ax.axhline(n, color=MUTED, ls="--", lw=0.6, label=r"$nk$" if n == 1 else None)
    ax.set(xscale="log", yscale="log", xlabel=r"$\kappa=2R^2/\ell^2$", ylabel=r"$\lambda/k$",
           xlim=(0.05, 100), ylim=(0.5, 40))
    ax.legend(loc="upper right", handlelength=1.5)
    panel_label(ax, "(b)")

    fs = np.linspace(0, 3, 121)
    for kp, c in zip(KAPPAS, kappa_colors):
        tr = np.array([cos_transport(kp, f)[:2] for f in fs])
        axs[1, 0].plot(fs, tr[:, 0] / (K * cos_R(kp)), color=c, label=f"{kp:g}")
        axs[1, 1].plot(fs, tr[:, 1] / D, color=c, label=f"{kp:g}")
    fd = np.linspace(1, 3, 200)
    ax = axs[1, 0]
    ax.plot(fd, np.sqrt(fd**2 - 1), color=INK, ls="--", lw=1.0, label="no noise")
    ax.plot(fs, fs, color=MUTED, ls=":", lw=1.0, label="no well")
    ax.set(xlabel=r"$v/kR$", ylabel=r"$v_{\rm w}/kR$", xlim=(0, 3), ylim=(0, 3))
    leg = ax.legend(title=r"$\kappa$", title_fontsize=7.5, loc="upper left", handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(c)")

    ax = axs[1, 1]
    ax.axhline(1, color=MUTED, ls=":", lw=1.0)
    Vm, Dm, _ = mc_tilt()
    mc_dots(ax, [1.0], [Dm / D], kappa_colors[1], label=r"simulation, $\kappa=3$")
    ax.set(yscale="log", xlabel=r"$v/kR$", ylabel=r"$D_{\mathrm{eff}}/D$", xlim=(0, 3), ylim=(1e-7, 300))
    leg = ax.legend(title=r"$\kappa$", title_fontsize=7.5, loc="lower right", handlelength=1.5)
    leg.get_title().set_color(INK2)
    panel_label(ax, "(d)")
    fig.savefig(FIG / "cos_circle.pdf")
    plt.close(fig)


# --- Monte Carlo (each with its own seed, so sections can run alone) ----------
@cache
def mc_absorbing():
    rng = np.random.default_rng(20261026)
    x0s = np.round(np.arange(0.1, 0.95, 0.1), 1)
    return {
        "grid": (x0s, *simulate_exits(x0s, 4000, 1e-4, rng)),
        "hist": simulate_exits([X0], 50000, 1e-4, rng),
    }


@cache
def mc_tilt(kappa=3.0, f=1.0, n_paths=4000, dt=2e-3, T=200.0):
    """Lift of the tilted cosine circle: Euler paths from the bottom, 10 tau burn-in, then T.
    Returns V, D_eff and the relative standard error of the variance."""
    rng = np.random.default_rng(20261029)
    R = cos_R(kappa)
    v = f * K * R
    x = np.zeros(n_paths)

    def run(duration):
        steps = int(round(duration / dt))
        for s in range(0, steps, 2000):
            for w in rng.standard_normal((min(2000, steps - s), n_paths)) * np.sqrt(D * dt):
                x[:] += (v - K * R * np.sin(x / R)) * dt + w

    run(10 * TAU)
    start = x.copy()
    run(T)
    d = x - start
    return d.mean() / T, d.var() / T, np.sqrt(2 / n_paths)


@cache
def mc_mixed():
    rng = np.random.default_rng(20261027)
    x0s = np.round(np.arange(0.0, 0.95, 0.1), 1)
    return {
        "grid": (x0s, *simulate_exits(x0s, 4000, 2e-4, rng, lower="reflecting")),
        "hist": simulate_exits([X0], 20000, 2e-4, rng, lower="reflecting"),
    }


def report_mc(mc, T_exact, piL_exact=None):
    x0s, start, t_exit, wall = mc["grid"]
    print("  simulation (Euler + Brownian bridge), +/- one standard error:")
    for i, x0 in enumerate(x0s):
        te, w = t_exit[start == i], wall[start == i]
        line = f"    x0={x0:.1f}  <T>={te.mean():.4f} +/- {te.std() / np.sqrt(te.size):.4f} (exact {T_exact(x0):.4f})"
        if piL_exact is not None:
            se = np.sqrt(w.mean() * (1 - w.mean()) / w.size)
            line += f"  pi_L={w.mean():.4f} +/- {se:.4f} (exact {piL_exact(x0):.4f})"
        print(line)


# --- checks ------------------------------------------------------------------
def check_reflecting():
    ok = True
    x = np.linspace(0, L, 4001)
    for t in [0.02, 0.2]:
        diff = np.max(np.abs(images(x, t, X0, +1) - modes(x, t, X0, "reflecting", N=2000)))
        ok &= check(f"images vs modes, max |diff| at t={t}", diff, 0, 1e-8)
    p = images(x, 0.05, X0, +1)
    ok &= check("normalisation at t=0.05", trapz(p, x), 1, 1e-6)
    ok &= check("<x> series vs quadrature at t=0.05", refl_mean(np.array([0.05]), X0)[0], trapz(x * p, x), 1e-6)
    ok &= check("<x^2> series vs quadrature at t=0.05", refl_second(np.array([0.05]), X0)[0], trapz(x * x * p, x), 1e-6)
    ok &= check("C(0) = L^2/12", refl_autocov(np.array([0.0]))[0], L**2 / 12, 1e-6)
    ok &= check("<x(0)> = x0", refl_mean(np.array([0.0]), X0)[0], X0, 1e-4)
    return ok


def check_absorbing():
    ok = True
    x = np.linspace(0, L, 4001)
    for t in [0.02, 0.2]:
        diff = np.max(np.abs(images(x, t, X0, -1) - modes(x, t, X0, "absorbing", N=2000)))
        ok &= check(f"images vs modes, max |diff| at t={t}", diff, 0, 1e-8)
    p = images(x, 0.1, X0, -1)
    ok &= check("S(t) series vs quadrature at t=0.1", survival(np.array([0.1]), X0)[0], trapz(p, x), 1e-6)
    t = np.linspace(1e-5, 6, 200001)
    j0, jL = exit_flux0(t, X0, M=12), exit_fluxL(t, X0, M=12)
    f = j0 + jL
    ok &= check("int f dt = 1", trapz(f, t), 1, 1e-5)
    ok &= check("pi_L = int j_L dt", trapz(jL, t), X0 / L, 1e-5)
    ok &= check("<T> = x0(L-x0)/D", trapz(t * f, t), T_mean(X0), 1e-5)
    ok &= check("Var T", trapz(t * t * f, t) - trapz(t * f, t) ** 2, T_var(X0), 1e-5)
    ok &= check("T_0", trapz(t * j0, t) / trapz(j0, t), T_given0(X0), 1e-5)
    ok &= check("T_L", trapz(t * jL, t) / trapz(jL, t), T_givenL(X0), 1e-5)
    tt = np.array([0.05, 0.3])
    n = np.arange(1, 4002, 2)[:, None]
    f_series = (2 * np.pi * D / L**2) * (n * np.sin(n * np.pi * X0 / L) * np.exp(-lam(n) * tt)).sum(0)
    ok &= check("f images vs -dS/dt series", np.max(np.abs(exit_flux0(tt, X0) + exit_fluxL(tt, X0) - f_series)), 0, 1e-8)
    ok &= check("QSD average of T = 1/lambda_1", trapz(qsd(x) * T_mean(x), x), 1 / lam(1), 1e-6)
    report_mc(mc_absorbing(), T_mean, lambda x0: x0 / L)
    return ok


def check_mixed():
    ok = True
    x = np.linspace(0, L, 4001)
    for t in [0.02, 0.2]:
        diff = np.max(np.abs(mix_images(x, t, X0) - mix_modes(x, t, X0, N=2000)))
        ok &= check(f"images vs modes, max |diff| at t={t}", diff, 0, 1e-8)
    p = mix_images(x, 0.1, X0)
    ok &= check("S(t) series vs quadrature at t=0.1", mix_survival(np.array([0.1]), X0)[0], trapz(p, x), 1e-6)
    tt = np.array([0.05, 0.3, 1.0])
    n = np.arange(4000)[:, None]
    f_series = (np.pi * D / (2 * L**2)) * ((-1.0) ** n * (2 * n + 1) * np.cos((2 * n + 1) * np.pi * X0 / (2 * L))
                                           * np.exp(-lam_mix(n) * tt)).sum(0)
    ok &= check("f images vs -dS/dt series", np.max(np.abs(mix_flux(tt, X0) - f_series)), 0, 1e-8)
    t = np.linspace(1e-5, 40, 400001)
    f = mix_flux(t, X0, M=30)
    ok &= check("int f dt = 1", trapz(f, t), 1, 1e-5)
    ok &= check("<T> = (L^2-x0^2)/D", trapz(t * f, t), mix_T(X0), 1e-5)
    ok &= check("Var T", trapz(t * t * f, t) - trapz(t * f, t) ** 2, mix_T_var(X0), 1e-5)
    ok &= check("QSD average of T = 1/lambda_0", trapz(mix_qsd(x) * mix_T(x), x), 1 / lam_mix(0), 1e-6)
    report_mc(mc_mixed(), mix_T)
    return ok


def check_halfline_reflecting():
    ok = True
    x = np.linspace(0, 40, 400001)
    for t in [0.5, 5.0]:
        p = half_line(x, t, X0H, +1)
        ok &= check(f"normalisation at t={t}", trapz(p, x), 1, 1e-6)
        ok &= check(f"<x> folded-normal formula vs quadrature at t={t}", hl_mean(t, X0H), trapz(x * p, x), 1e-6)
        ok &= check(f"<x^2> = x0^2 + Dt at t={t}", trapz(x * x * p, x), X0H**2 + D * t, 1e-6)
    return ok


def check_halfline_absorbing():
    ok = True
    x = np.linspace(0, 40, 400001)
    for t in [0.5, 5.0]:
        ok &= check(f"S = erf(x0/sqrt(2Dt)) vs quadrature at t={t}", hl_survival(t, X0H),
                    trapz(half_line(x, t, X0H, -1), x), 1e-6)
    t = np.linspace(1e-6, 10, 1000001)
    ok &= check("int_0^10 f dt = 1 - S(10)", trapz(levy_smirnov(t, X0H), t), 1 - hl_survival(10.0, X0H), 1e-6)
    ok &= check("mode of f = x0^2/3D", t[np.argmax(levy_smirnov(t, X0H))], X0H**2 / (3 * D), 1e-5)
    t_med = X0H**2 / (2 * D * erfinv(0.5) ** 2)
    ok &= check(f"S(t_med) = 1/2 at t_med = {t_med:.4f} x0^2/D", hl_survival(t_med, X0H), 0.5, 1e-12)
    T = 1e4
    u = np.linspace(0, 6, 601)
    q = np.sqrt(D * T) * half_line(u * np.sqrt(D * T), T, X0H, -1) / hl_survival(T, X0H)
    ok &= check(f"Rayleigh limit, max |diff| at Dt/x0^2={T:g}", np.max(np.abs(q - u * np.exp(-(u**2) / 2))), 0, 1e-3)
    return ok


def check_line():
    ok = True
    s, y = 1.0, 0.5
    t = np.linspace(1e-6, 60, 2000001)
    k = np.sqrt(2 * s / D)
    ok &= check("resolvent int e^{-st} g dt = e^{-k|y|}/(Dk)", trapz(np.exp(-s * t) * heat(y, t), t),
                np.exp(-k * y) / (D * k), 1e-6)
    return ok


def check_drift_reflecting():
    ok = True
    x = np.linspace(0, L, 4001)
    for v in [-4.0, 2.0]:
        for t in [0.02, 0.2]:
            ok &= check(f"v={v:g}: normalisation at t={t}", trapz(dr_modes(x, t, X0, v), x), 1, 1e-6)
        pf = lambda y, t, v=v: dr_modes(y, t, X0, v)
        ok &= check(f"v={v:g}: PDE residual at t=0.05", fpe_residual(pf, np.linspace(0.01, 0.99, 99), 0.05, v), 0, 1e-4)
        J = max(abs(current(pf, np.array([0.0]), 0.05, v)[0]), abs(current(pf, np.array([L]), 0.05, v)[0]))
        ok &= check(f"v={v:g}: wall currents at t=0.05", J, 0, 1e-4)
        ok &= check(f"v={v:g}: <x>_s formula vs quadrature", dr_mean_s(v), trapz(x * dr_stationary(x, v), x), 1e-6)
        ok &= check(f"v={v:g}: <x(t)> modes vs quadrature at t=0.05", dr_mean(np.array([0.05]), X0, v)[0],
                    trapz(x * dr_modes(x, 0.05, X0, v), x), 1e-6)
        ok &= check(f"v={v:g}: <x(t)> ~ x0 + vt at t=1e-3", trapz(x * dr_modes(x, 1e-3, X0, v, N=2000), x),
                    X0 + v * 1e-3, 1e-4)
    return ok


def check_drift_absorbing():
    ok = True
    x = np.linspace(0, L, 4001)
    for v in [-4.0, 2.0]:
        pf = lambda y, t, v=v: da_p(y, t, X0, v)
        ok &= check(f"v={v:g}: PDE residual at t=0.05", fpe_residual(pf, np.linspace(0.01, 0.99, 99), 0.05, v), 0, 1e-4)
        ok &= check(f"v={v:g}: S series vs quadrature at t=0.1", da_survival(np.array([0.1]), X0, v)[0],
                    trapz(da_p(x, 0.1, X0, v), x), 1e-6)
        t = np.linspace(1e-5, 6, 200001)
        j0, jL = da_flux0(t, X0, v, M=12), da_fluxL(t, X0, v, M=12)
        ok &= check(f"v={v:g}: int (j_0 + j_L) dt = 1", trapz(j0 + jL, t), 1, 1e-5)
        ok &= check(f"v={v:g}: pi_L", trapz(jL, t), da_piL(X0, v), 1e-5)
        ok &= check(f"v={v:g}: T = (L pi_L - x0)/v", trapz(t * (j0 + jL), t), da_T(X0, v), 1e-5)
        ok &= check(f"v={v:g}: T_0", trapz(t * j0, t) / trapz(j0, t), da_T0(X0, v), 1e-5)
        ok &= check(f"v={v:g}: T_L", trapz(t * jL, t) / trapz(jL, t), da_TL(X0, v), 1e-5)
        ok &= check(f"v={v:g}: QSD normalisation", trapz(da_qsd(x, v), x), 1, 1e-6)
        ok &= check(f"v={v:g}: QSD average of T = 1/lambda_1", trapz(da_qsd(x, v) * da_T(x, v), x), 1 / da_lam1(v), 1e-6)
    ok &= check("T_L even in v", abs(da_TL(X0, 3.0) - da_TL(X0, -3.0)), 0, 1e-12)
    return ok


def check_drift_mixed():
    ok = True
    x = np.linspace(0, L, 4001)
    for v in [-4.0, -0.5, 3.0]:
        pf = lambda y, t, v=v: dm_p(y, t, X0, v)
        ok &= check(f"v={v:g}: PDE residual at t=0.05", fpe_residual(pf, np.linspace(0.01, 0.99, 99), 0.05, v), 0, 1e-4)
        ok &= check(f"v={v:g}: J(0) = 0", abs(current(pf, np.array([0.0]), 0.05, v)[0]), 0, 1e-4)
        ok &= check(f"v={v:g}: p(L) = 0", abs(pf(np.array([L]), 0.05)[0]), 0, 1e-8)
        ok &= check(f"v={v:g}: <x(t)> ~ x0 + vt at t=1e-3", trapz(x * dm_p(x, 1e-3, X0, v, N=200), x),
                    X0 + v * 1e-3, 1e-4)
        lams, c = dm_survival_coeffs(X0, v)
        ok &= check(f"v={v:g}: S(0.05) modes vs quadrature of p", (c * np.exp(-lams * 0.05)).sum(),
                    trapz(dm_p(x, 0.05, X0, v), x), 1e-6)
        ok &= check(f"v={v:g}: T = int S dt = sum c_n/lambda_n", (c / lams).sum(), dm_T(X0, v), 1e-4)
        ok &= check(f"v={v:g}: QSD average of T = 1/lambda_0", trapz(dm_qsd(x, v) * dm_T(x, v), x),
                    1 / dm_modes(v)[0][0], 1e-5)
    v = -4.0
    print(f"  v={v:g}: lambda_0 = {dm_modes(v)[0][0]:.5g}, 1/T(0) = {1 / dm_T(0.0, v):.5g},"
          f" (2v^2/D) e^(-2|v|L/D) = {2 * v**2 / D * np.exp(-2 * abs(v) * L / D):.5g}")
    return ok


def check_drift_halfline_reflecting():
    ok = True
    x = np.linspace(0, 120, 240001)
    for v in [-1.0, 1.0]:
        pf = lambda y, t, v=v: dhr_p(y, t, X0H, v)
        for t in [0.5, 5.0]:
            ok &= check(f"v={v:g}: normalisation at t={t}", trapz(pf(x, t), x), 1, 1e-6)
        ok &= check(f"v={v:g}: PDE residual at t=0.5", fpe_residual(pf, np.linspace(0.05, 6, 120), 0.5, v), 0, 1e-4)
        ok &= check(f"v={v:g}: J(0) = 0 at t=0.5", abs(current(pf, np.array([0.0]), 0.5, v)[0]), 0, 1e-5)
    xs = np.linspace(0, 10, 1001)
    ok &= check("v=-1: p -> 2|v|/D e^{-2|v|x/D} at t=60",
                np.max(np.abs(dhr_p(xs, 60.0, X0H, -1.0) - 2 * np.exp(-2 * xs))), 0, 1e-3)
    return ok


def check_drift_halfline_absorbing():
    ok = True
    x = np.linspace(0, 120, 240001)
    for v in [-1.0, 0.5]:
        pf = lambda y, t, v=v: dha_p(y, t, X0H, v)
        ok &= check(f"v={v:g}: PDE residual at t=0.5", fpe_residual(pf, np.linspace(0.05, 6, 120), 0.5, v), 0, 1e-4)
        ok &= check(f"v={v:g}: S formula vs quadrature at t=2", dha_survival(2.0, X0H, v), trapz(pf(x, 2.0), x), 1e-6)
    t = np.linspace(1e-6, 400, 4000001)
    f = inverse_gaussian(t, X0H, -1.0)
    ok &= check("v=-1: int f dt = 1", trapz(f, t), 1, 1e-6)
    ok &= check("v=-1: <T> = x0/|v|", trapz(t * f, t), X0H / 1.0, 1e-6)
    ok &= check("v=-1: Var T = x0 D/|v|^3", trapz(t * t * f, t) - trapz(t * f, t) ** 2, X0H * D, 1e-5)
    ok &= check("v=+0.5: Pr(absorbed) = e^{-2 v x0/D}", trapz(inverse_gaussian(t, X0H, 0.5), t), np.exp(-2 * 0.5 * X0H / D), 1e-6)
    v, T = -1.0, 200.0
    asym = 2 * X0H / v**2 * np.sqrt(D / (2 * np.pi)) * T**-1.5 * np.exp(abs(v) / D * X0H - v**2 * T / (2 * D))
    ok &= check(f"v=-1: S / asymptote at t={T:g}", dha_survival(T, X0H, v) / asym, 1, 0.03)
    xs = np.linspace(0, 60, 600001)
    for frac in [0.5, 1.0]:
        lam_ = frac * v**2 / (2 * D)
        phi = dha_qsd(xs, lam_, v)
        ok &= check(f"QSD lambda={frac:g} edge: normalisation", trapz(phi, xs), 1, 1e-5)
        h = xs[1] - xs[0]
        res = -v * np.gradient(phi, h) + 0.5 * D * np.gradient(np.gradient(phi, h), h) + lam_ * phi
        ok &= check(f"QSD lambda={frac:g} edge: eigen-equation residual", np.max(np.abs(res[5:-5])), 0, 1e-4)
    return ok


def check_drift_line():
    ok = True
    a, v = 2.0, 0.7  # level a above the start x0 = 0, drift toward it
    t = np.linspace(1e-6, 400, 4000001)
    f = inverse_gaussian(t, a, -v)
    ok &= check("first passage to a: <T> = a/v", trapz(t * f, t), a / v, 1e-5)
    ok &= check("drift away: Pr(reach a) = e^{-2va/D}", trapz(inverse_gaussian(t, a, v), t), np.exp(-2 * v * a / D), 1e-6)
    return ok


def check_ou_line():
    ok = True
    x = np.linspace(-8, 8, 4001)
    for t in [0.2, 0.4]:
        ok &= check(f"Mehler sum vs Gaussian at t={t}",
                    np.max(np.abs(ou_mehler(x, t, X0_OU, N=300) - ou_line(x, t, X0_OU))), 0, 1e-8)
        th = ou_theta(t)
        mapped = np.exp(K * t) * heat(np.exp(K * t) * x - X0_OU, th)
        ok &= check(f"map y = x e^kt: free Gaussian at theta(t), t={t}", np.max(np.abs(mapped - ou_line(x, t, X0_OU))), 0, 1e-12)
    pf = lambda y, t: ou_line(y, t, X0_OU)
    ok &= check("PDE residual at t=0.3", ou_residual(pf, np.linspace(-3, 4, 141), 0.3), 0, 1e-4)
    return ok


def check_ou_halfline_reflecting():
    ok = True
    x = np.linspace(0, 10, 100001)
    pf = lambda y, t: ou_half(y, t, X0_OU, +1)
    for t in [0.1, 1.0]:
        p_ = pf(x, t)
        ok &= check(f"normalisation at t={t}", trapz(p_, x), 1, 1e-8)
        ok &= check(f"<x^2> = x0^2 e^-2kt + ell^2(1-e^-2kt)/2 at t={t}", trapz(x * x * p_, x),
                    X0_OU**2 * np.exp(-2 * t) + 0.5 * (1 - np.exp(-2 * t)), 1e-8)
        ok &= check(f"folded-normal <x> vs quadrature at t={t}", ou_folded_mean(t, X0_OU), trapz(x * p_, x), 1e-8)
        th = ou_theta(t)
        ok &= check(f"map from Part I half-line at t={t}",
                    np.max(np.abs(np.exp(K * t) * half_line(np.exp(K * t) * x, th, X0_OU, +1) - p_)), 0, 1e-12)
    ok &= check("PDE residual at t=0.3", ou_residual(pf, np.linspace(0.05, 4, 80), 0.3), 0, 1e-4)
    ok &= check("J(0) = 0", abs(ou_current(pf, np.array([0.0]), 0.3)[0]), 0, 1e-6)
    return ok


def check_ou_halfline_absorbing():
    ok = True
    x = np.linspace(0, 10, 100001)
    pf = lambda y, t: ou_half(y, t, X0_OU, -1)
    for t in [0.2, 1.5]:
        ok &= check(f"S = erf(xi0/sqrt(e^2kt-1)) vs quadrature at t={t}", ou_hl_survival(t, X0_OU), trapz(pf(x, t), x), 1e-8)
    ok &= check("PDE residual at t=0.3", ou_residual(pf, np.linspace(0.05, 4, 80), 0.3), 0, 1e-4)
    t = np.linspace(0, 40, 400001)
    S = ou_hl_survival(np.maximum(t, 1e-12), X0_OU)
    ok &= check("T = int S dt vs erfcx formula", trapz(S, t), ou_hl_T(X0_OU), 1e-6)
    ok &= check("QSD normalisation", trapz(ou_hl_qsd(x), x), 1, 1e-8)
    q = pf(x, 8.0) / ou_hl_survival(8.0, X0_OU)
    ok &= check("q -> QSD at t=8 tau", np.max(np.abs(q - ou_hl_qsd(x))), 0, 1e-5)
    c = ou_hl_T(1000 * ELL) / TAU - np.log(1000)
    ok &= check("T ~ tau [ln(x0/ell) + ln 2 + gamma/2]", c, np.log(2) + np.euler_gamma / 2, 1e-5)
    # the continuum of QSDs, 0 < nu <= 1
    xs = np.linspace(0.05, 6, 120)
    for nu in [0.3, 0.7]:
        q = lambda y, nu=nu: ou_qsd_nu(y, nu)
        ok &= check(f"QSD nu={nu}: normalisation", quad(q, 0, np.inf, limit=800)[0], 1, 1e-8)
        h = 1e-4
        res = (0.5 * D * (q(xs + h) - 2 * q(xs) + q(xs - h)) / h**2
               + K * ((xs + h) * q(xs + h) - (xs - h) * q(xs - h)) / (2 * h) + nu * K * q(xs))
        ok &= check(f"QSD nu={nu}: eigen-equation residual", np.max(np.abs(res)), 0, 1e-6)
        big = np.logspace(1, 4, 50) * ELL
        ok &= check(f"QSD nu={nu}: positive, tail exponent", np.polyfit(np.log(big), np.log(q(big)), 1)[0], -1 - nu, 1e-3)
        xg = np.linspace(1e-3, 20, 400)
        p, S = ou_hl_evolve(q, xg, TAU)
        ok &= check(f"QSD nu={nu}: evolves as e^(-nu k t) q", np.max(np.abs(p - np.exp(-nu * K * TAU) * q(xg))), 0, 1e-10)
    ok &= check("nu = 1 is the Rayleigh QSD", np.max(np.abs(ou_qsd_nu(xs, 1.0) - ou_hl_qsd(xs))), 0, 1e-12)
    ok &= check("nu = 1.5 solution changes sign", float(np.min(ou_qsd_nu(np.linspace(0.01, 20, 400), 1.5)) < 0), 1, 0)
    return ok


def check_operator():
    """The symmetrised operator H = -(D/2) d^2 + V, solved by finite differences, against the spectra found
    elsewhere from the forward equation."""
    ok = True
    Lr, v = 1.0, 3.0
    ev = h_interval(lambda x: v**2 / (2 * D) + 0 * x, 0, Lr, 4000, robin=(v / D, v / D), m=5)
    want = np.array([0] + [v**2 / (2 * D) + 0.5 * D * (n * np.pi / Lr) ** 2 for n in range(1, 5)])
    ok &= check("drift, reflecting interval: constant V = v^2/2D with Robin walls psi' = (v/D) psi",
                np.max(np.abs(ev - want) / np.maximum(1, want)), 0, 1e-5)
    Lx = 2.0 * ELL
    ev = h_interval(V_ou, 0, Lx, 4000, robin=(0.0, -K * Lx / D), m=5)
    want = np.array([m[0] for m in ou_modes(Lx / ELL, "reflecting")[:5]]) * K
    ok &= check("OU, reflecting interval: Robin psi' = (A/D) psi vs Kummer roots", np.max(np.abs(ev - want) / np.maximum(1, want)),
                0, 1e-5)
    ev = h_interval(V_ou, 0, Lx, 4000, m=5)
    want = np.array([m[0] for m in ou_modes(Lx / ELL, "absorbing")[:5]]) * K
    ok &= check("OU, absorbing interval: Dirichlet vs Kummer roots", np.max(np.abs(ev - want) / np.maximum(1, want)), 0, 1e-5)
    for kappa in [3.0, 16.0]:
        Lc = 2 * np.pi * cos_R(kappa)
        ev = h_circle(V_cos(kappa), Lc, 2000, m=7)
        want = np.sort(np.concatenate(cos_spectrum(kappa)))[:7] * K
        ok &= check(f"cosine circle, kappa={kappa:g}: V with wells at the bottom and the top of U, vs Fourier",
                    np.max(np.abs(ev - want) / np.maximum(1, want)), 0, 1e-4)
    Lc = 4.0 * ELL
    ev = h_circle(V_ou, Lc, 2000, delta=K * Lc / 2, m=6)
    want = np.sort([m[0] for kind in ("reflecting", "absorbing") for m in ou_modes(Lc / (2 * ELL), kind)[:4]])[:6] * K
    ok &= check("sawtooth circle: harmonic V plus (kL/2) delta at the cusp", np.max(np.abs(ev - want) / np.maximum(1, want)),
                0, 2e-3)
    return ok


def check_overdamped():
    """The full (x, v) model against its overdamped limit."""
    from scipy.integrate import solve_ivp
    ok = True
    t = np.linspace(0, 6, 61)
    for z in [0.25, 1.0, 3.0]:
        g, w2 = 4 * z**2, 4 * z**2  # gamma/m and omega^2 in units kappa/gamma = 1
        sol = solve_ivp(lambda s, y: [y[1], -g * y[1] - w2 * y[0]], (0, 6), [1.0, 0.0], t_eval=t, rtol=1e-11, atol=1e-13)
        ok &= check(f"zeta={z:g}: closed-form C(t) solves the regression equation C'' + (gamma/m)C' + omega^2 C = 0",
                    np.max(np.abs(sol.y[0] - damped_C(t, z))), 0, 1e-8)
    ok &= check("critical damping: slowest rate = 2 x Smoluchowski", float(damped_rate(1.0)), 2, 1e-12)
    ok &= check("zeta=10: slowest rate = (kappa/gamma)(1 + 1/4zeta^2), Gardiner (6.4.108)", float(damped_rate(10.0)),
                1 + 1 / 400, 2e-5)
    # Maxwell-Boltzmann solves Kramers' equation in a double well: 0 = -v p_x + (1/m) d_v[(U' + gamma v) p] + (gamma kT/m^2) p_vv
    m, gam, kT = 0.7, 1.3, 0.9
    Phi = lambda x: (x**2 - 1) ** 2
    dPhi = lambda x: 4 * x * (x**2 - 1)
    p = lambda x, v: np.exp(-(0.5 * m * v**2 + Phi(x)) / kT)
    X, Vv = np.meshgrid(np.linspace(-1.5, 1.5, 61), np.linspace(-2, 2, 61))
    h = 1e-4
    px = (p(X + h, Vv) - p(X - h, Vv)) / (2 * h)
    flux = lambda x, v: (dPhi(x) + gam * v) * p(x, v)
    dv_flux = (flux(X, Vv + h) - flux(X, Vv - h)) / (2 * h)
    pvv = (p(X, Vv + h) - 2 * p(X, Vv) + p(X, Vv - h)) / h**2
    res = -Vv * px + dv_flux / m + gam * kT / m**2 * pvv
    ok &= check("Maxwell-Boltzmann is stationary for Kramers' equation (relative residual)", np.max(np.abs(res)) / np.max(np.abs(px)),
                0, 1e-5)
    return ok


def rel_err(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want) / np.maximum(1, np.abs(want))))


def check_solvable():
    """Spectra of the shape-invariant potentials from finite differences of H, and the partner identities."""
    ok = True
    for d in [3.0, 5.0]:
        c = (d - 1) * D / 2
        ev = h_interval(V_from(lambda x: K * x - c / x, lambda x: K + c / x**2), 1e-6, 12 * ELL, 20000, m=4)
        ok &= check(f"harmonic + log, d={d:g}: lambda_n = 2nk", rel_err(ev, 2 * K * np.arange(4)), 0, 1e-5)
    for sig, v in [(1.5, 1.0), (3.0, 2.0)]:
        ev = h_interval(V_from(lambda x: v - sig * D / x, lambda x: sig * D / x**2), 1e-6, 300.0, 60000, m=4)
        want = v**2 / (2 * D) * (1 - (sig / (np.arange(4) + sig)) ** 2)
        ok &= check(f"linear + log, sigma={sig:g}: v^2/2D [1 - sigma^2/(n+sigma)^2]", rel_err(ev, want), 0, 1e-4)
    for s, w, U0 in [(3.5, 1.0, 1.0), (2.2, 0.5, 3.0)]:
        v = s * D / w
        nb = int(np.floor(s - 0.5)) + 1  # levels at least 0.5 below the edge; those nearer are too extended for the box
        # start the box where the wall is steep but V is still moderate: a huge V costs absolute eigenvalue accuracy
        ev = h_interval(V_from(lambda x: v - U0 / w * np.exp(-x / w), lambda x: U0 / w**2 * np.exp(-x / w)),
                        -3 * w, 60.0, 60000, m=nb)
        ok &= check(f"Morse, s={s:g}: (D/2w^2) n (2s - n)", rel_err(ev, [D / (2 * w**2) * n * (2 * s - n) for n in range(nb)]),
                    0, 1e-3)
    # The partner of drift A is drift -A (potential -U): V~ = U'^2/2D + U''/2. Shape invariance: V~(s) = V(s') + R(s).
    partner = lambda U1, U2: (lambda y: U1(y) ** 2 / (2 * D) + U2(y) / 2)
    x = np.linspace(0.3, 1.4, 50)
    s, c = 2.7, 2.0
    cases = [
        ("ln cosh, s -> s-1, R = (D/2w^2)(2s-1)",
         partner(lambda y: s * D * np.tanh(y), lambda y: s * D / np.cosh(y) ** 2),
         V_from(lambda y: (s - 1) * D * np.tanh(y), lambda y: (s - 1) * D / np.cosh(y) ** 2), D / 2 * (2 * s - 1)),
        ("ln cos, s -> s+1, R = (D/2w^2)(2s+1)",
         partner(lambda y: s * D * np.tan(y), lambda y: s * D / np.cos(y) ** 2),
         V_from(lambda y: (s + 1) * D * np.tan(y), lambda y: (s + 1) * D / np.cos(y) ** 2), D / 2 * (2 * s + 1)),
        ("harmonic + log, c -> c+D, R = 2k",
         partner(lambda y: K * y - c / y, lambda y: K + c / y**2),
         V_from(lambda y: K * y - (c + D) / y, lambda y: K + (c + D) / y**2), 2 * K),
        ("harmonic, R = k (the Part III ladder)",
         partner(lambda y: K * y, lambda y: K + 0 * y), V_from(lambda y: K * y, lambda y: K + 0 * y), K),
    ]
    for name, Vt, Vs, R in cases:
        ok &= check(f"partner identity: {name}", np.max(np.abs(Vt(x) - Vs(x) - R)), 0, 1e-12)
    return ok


def check_radial():
    ok = True
    t, r0 = 0.7, 1.3
    r = np.linspace(0.01, 6, 300)
    th = np.linspace(0, 2 * np.pi, 4001)
    num = np.array([trapz(np.exp(-(rr**2 + r0**2 - 2 * rr * r0 * np.cos(th)) / (2 * D * t)), th) * rr / (2 * np.pi * D * t)
                    for rr in r])
    ok &= check("d=2, k=0: closed form = angular integral of the 2D Gaussian", np.max(np.abs(radial_p(r, t, r0, 2.0) - num)), 0, 1e-12)
    tt, x0 = 0.4, 1.1
    rr = np.linspace(0.01, 4, 200) * ELL
    ok &= check("d=1, k>0: OU reflected at the bottom (images)", np.max(np.abs(radial_p(rr, tt, x0, 1.0, K) - ou_half(rr, tt, x0, +1))),
                0, 1e-12)
    ok &= check("d=3, k>0: e^{kt}(r/r0) x OU absorbed at the bottom (Doob, h = x)",
                np.max(np.abs(radial_p(rr, tt, x0, 3.0, K) - np.exp(K * tt) * rr / x0 * ou_half(rr, tt, x0, -1))), 0, 1e-12)
    ok &= check("d=2.6: closed form = Laguerre series (Gardiner 5.2.135 generalised)",
                np.max(np.abs(radial_p(rr, tt, x0, 2.6, K) - radial_laguerre(rr, tt, x0, 2.6))), 0, 1e-10)
    ok &= check("d=2.6: chi density normalised", quad(lambda y: radial_ps(y, 2.6), 0, np.inf)[0], 1, 1e-10)
    ok &= check("d=2: chi density = Gardiner's Rayleigh p_s (5.2.129)",
                np.max(np.abs(radial_ps(rr, 2.0) - 2 * K * rr / D * np.exp(-K * rr**2 / D))), 0, 1e-14)
    d = 2.6
    c = (d - 1) * D / 2
    pf = lambda y, t: radial_p(y, t, x0, d, K)
    h, dt = 2e-4, 1e-5
    y = np.linspace(0.2, 3.0, 80)
    A = lambda y: -K * y + c / y
    res = ((pf(y, tt + dt) - pf(y, tt - dt)) / (2 * dt) + (A(y + h) * pf(y + h, tt) - A(y - h) * pf(y - h, tt)) / (2 * h)
           - 0.5 * D * (pf(y + h, tt) - 2 * pf(y, tt) + pf(y - h, tt)) / h**2)
    ok &= check("d=2.6: PDE residual with drift -kr + (d-1)D/2r", np.max(np.abs(res)) / np.max(pf(y, tt)), 0, 1e-4)
    for d in [3.0, 4.5]:
        ok &= check(f"d={d:g}: Pr(reach a) = (a/r0)^(d-2) vs splitting quadrature", radial_reach_quad(0.5, d), radial_reach(0.5, 1.0, d), 1e-5)
    return ok


def check_lncosh():
    ok = True
    for s, w in [(2.5, 1.0), (4.3, 0.7)]:
        nb = int(np.ceil(s))
        ev = h_interval(V_from(lambda x: s * D / w * np.tanh(x / w), lambda x: s * D / w**2 / np.cosh(x / w) ** 2),
                        -40 * w, 40 * w, 20000, m=nb)
        ok &= check(f"s={s:g}: bound states (D/2w^2) n (2s - n), n < s", rel_err(ev, lncosh_levels(s, w)), 0, 1e-4)
        ok &= check(f"s={s:g}: p_s normalised", quad(lambda y: lncosh_ps(y, s, w), -np.inf, np.inf)[0], 1, 1e-10)
        x = np.linspace(-3, 3, 601)
        A = lambda y: -s * D / w * np.tanh(y / w)
        h = 1e-4
        for n in range(1, nb):
            P = lambda y, n=n: lncosh_P(y, n, s, w)
            LP = -(A(x + h) * P(x + h) - A(x - h) * P(x - h)) / (2 * h) + 0.5 * D * (P(x + h) - 2 * P(x) + P(x - h)) / h**2
            ok &= check(f"s={s:g}, n={n}: Jacobi eigenfunction, forward residual",
                        np.max(np.abs(LP + lncosh_levels(s, w)[n] * P(x))) / np.max(np.abs(P(x))), 0, 1e-5)
    return ok


def check_lncos():
    ok = True
    for s, w in [(1.5, 1.0), (3.0, 0.5)]:
        ev = h_interval(V_from(lambda x: s * D / w * np.tan(x / w), lambda x: s * D / w**2 / np.cos(x / w) ** 2),
                        -np.pi * w / 2, np.pi * w / 2, 20000, m=5)
        ok &= check(f"s={s:g}: (D/2w^2) n (n + 2s)", rel_err(ev, [lncos_lam(n, s, w) for n in range(5)]), 0, 1e-4)
        ok &= check(f"s={s:g}: p_s normalised", quad(lambda y: lncos_ps(y, s, w), -np.pi * w / 2, np.pi * w / 2)[0], 1, 1e-10)
        x = np.linspace(-1.2 * w, 1.2 * w, 241)
        A = lambda y: -s * D / w * np.tan(y / w)
        h = 1e-4
        for n in [1, 2]:
            Q = lambda y, n=n: eval_gegenbauer(n, s, np.sin(y / w))
            LQ = A(x) * (Q(x + h) - Q(x - h)) / (2 * h) + 0.5 * D * (Q(x + h) - 2 * Q(x) + Q(x - h)) / h**2
            ok &= check(f"s={s:g}, n={n}: Gegenbauer eigenfunction, backward residual",
                        np.max(np.abs(LQ + lncos_lam(n, s, w) * Q(x))) / np.max(np.abs(Q(x))), 0, 1e-5)
        k, half = s * D / w**2, np.pi * w / 2
        var = quad(lambda y: y * y * lncos_ps(y, s, w), -half, half)[0]
        lam1 = lncos_lam(1, s, w)
        ok &= check(f"s={s:g}: lambda_1 = k + pi^2 D/2L^2", lam1, k + np.pi**2 * D / (2 * (2 * half) ** 2), 1e-12)
        ok &= check(f"s={s:g}: Bakry-Emery k <= lambda_1 <= D/2 Var (both hold)", float(k <= lam1 <= D / (2 * var)), 1, 0)
    s, w = 1.0, L / np.pi
    x0, x = 0.35 * L, np.linspace(-0.5, 0.5, 4001)[1:-1] * L
    pf = lambda y, t: lncos_p(y, t, x0, s, w)
    ok &= check("Gegenbauer propagator normalised at Dt/L^2=0.01", trapz(pf(x, 0.01), x), 1, 1e-6)
    for t in [0.005, 0.05]:
        ok &= check(f"<sin(x/w)> = sin(x0/w) e^(-lambda_1 t) exactly (Q_1 = 2s sin), Dt/L^2={t:g}",
                    trapz(np.sin(x / w) * pf(x, t), x), np.sin(x0 / w) * np.exp(-lncos_lam(1, s, w) * t), 1e-6)
    A = lambda y: -s * D / w * np.tan(y / w)
    y = np.linspace(-0.4, 0.4, 81) * L
    h, dt, t = 2e-4, 1e-6, 0.01
    res = ((pf(y, t + dt) - pf(y, t - dt)) / (2 * dt) + (A(y + h) * pf(y + h, t) - A(y - h) * pf(y - h, t)) / (2 * h)
           - 0.5 * D * (pf(y + h, t) - 2 * pf(y, t) + pf(y - h, t)) / h**2)
    ok &= check("Gegenbauer propagator: PDE residual", np.max(np.abs(res)) / np.max(pf(y, t)), 0, 1e-4)
    # y = (1 + sin(x/w))/2 turns this into Wright-Fisher with symmetric mutation (Ito: dy = y' dx + y''/2 D dt)
    xs = np.linspace(-1.4, 1.4, 9) * w
    th = xs / w
    eps = D / w**2
    drift_y = np.cos(th) / (2 * w) * A(xs) - np.sin(th) / (4 * w**2) * D
    yv = (1 + np.sin(th)) / 2
    ok &= check("Lamperti: drift of y = (1 + sin x/w)/2 is u(1 - 2y), u = (D/w^2)(2s+1)/4",
                np.max(np.abs(drift_y - eps * (2 * s + 1) / 4 * (1 - 2 * yv))), 0, 1e-12)
    ok &= check("Lamperti: diffusion of y is (D/w^2) y(1-y)",
                np.max(np.abs((np.cos(th) / 2) ** 2 * eps - eps * yv * (1 - yv))), 0, 1e-12)
    # s = 1/2, w = 1 is the polar angle theta = x + pi/2 of rotational diffusion on the sphere (D_r = D/2)
    theta = np.linspace(0.1, np.pi - 0.1, 50)
    xx = theta - np.pi / 2
    ok &= check("s=1/2: drift -(D/2) tan(x) = D_r cot(theta), the polar-angle drift on the sphere",
                np.max(np.abs(-0.5 * D * np.tan(xx) - 0.5 * D / np.tan(theta))), 0, 1e-12)
    from scipy.special import eval_legendre
    ok &= check("s=1/2: Gegenbauer C_n^(1/2)(sin x) = Legendre P_n(-cos theta), n <= 5",
                max(np.max(np.abs(eval_gegenbauer(n, 0.5, np.sin(xx)) - eval_legendre(n, -np.cos(theta)))) for n in range(6)),
                0, 1e-12)
    ok &= check("s=1/2: rates (D/2) n(n+1) = D_r n(n+1); lambda_1 = 2 D_r (Debye)", lncos_lam(1, 0.5, 1.0), 2 * (D / 2), 1e-14)
    return ok


def check_double_well():
    ok = True
    l8a = dw_eigen(8.0, n=3000, m=2)[1][1]
    l8b = dw_eigen(8.0, n=6000, m=2)[1][1]
    ok &= check("beta=8: lambda_1 converged (3000 vs 6000 sites)", l8a / l8b, 1, 1e-4)
    l12 = dw_eigen(12.0, m=2)[1][1]
    ok &= check("beta=12: lambda_1 vs Gardiner (9.2.18)", l12 / dw_gardiner_9218(12.0), 1, 1e-3)
    l16 = dw_eigen(16.0, m=2)[1][1]
    ok &= check("beta=16: lambda_1 / Kramers -> 1 (corrections ~1/beta)", l16 / dw_kramers(16.0), 1, 0.03)
    U = dw_U(12.0)
    x = np.linspace(-4, 4, 400001)
    psi = np.exp(-2 * (U(x) - U(0.0)) / D)
    inner = cumulative_trapezoid(psi, x, initial=0)
    m = np.abs(x) <= 1.0
    T = 2 / D * trapz(inner[m] / psi[m], x[m])
    ok &= check("beta=12: lambda_1 T(-a -> a)/2 = 1 (one hop either way at rate lambda_1/2)", l12 * T / 2, 1, 1e-3)
    lam = dw_eigen(80.0, m=4)[1]
    ka, kb = dw_curvatures(80.0)
    ok &= check("beta=80: lambda_2 -> |U''(b)| (the barrier-top well of V)", lam[2] / kb, 1, 0.03)
    ok &= check("beta=80: lambda_3 -> U''(a) (first excited state in each well)", lam[3] / ka, 1, 0.05)
    xg, lam, v = dw_eigen(8.0, n=3000, m=2)
    Q1 = v[:, 1] / np.exp(-dw_U(8.0)(xg) / D)
    Q1 /= Q1[np.argmin(np.abs(xg - 1.0))]
    mm = np.abs(xg) < 1.5
    ok &= check("beta=8: slowest backward eigenfunction Q_1 = 2 pi_c - 1 (the committor)",
                np.max(np.abs(Q1[mm] - (2 * dw_committor(8.0, xg[mm]) - 1))), 0, 2e-2)
    t = np.array([1.0, 10.0, 100.0, 1000.0])
    Nc, lam = dw_population(8.0, t)
    ok &= check("beta=8: Pr(x>0) from the left minimum = (1 - e^{-lambda_1 t})/2 after the intrawell transient",
                np.max(np.abs(Nc - 0.5 * -np.expm1(-lam[1] * t))), 0, 2e-3)
    return ok


def check_attraction():
    ok = True
    v = -1.0  # drift toward the wall, ell = tau = 1
    x = np.linspace(0, 250, 50001)
    beta0 = 0.5
    q, _ = dha_conditioned_exp(x, 60.0, beta0, v)
    lam_ = dha_lam_of_tail(beta0, v)
    ok &= check("drift, start e^(-x/2ell): q -> QSD with the same tail at t=60 tau",
                np.max(np.abs(q - dha_qsd(x, lam_, v))), 0, 1e-5)
    l1, l2 = (dha_conditioned_exp(x, t, beta0, v)[1] for t in (40.0, 60.0))
    ok &= check("drift, start e^(-x/2ell): S decays at beta ell (2 - beta ell)/2 tau", (l1 - l2) / 20, lam_, 1e-4)
    err = [np.max(np.abs(dha_conditioned_exp(x, t, 2.0, v)[0] - dha_qsd(x, v**2 / (2 * D), v))) for t in (100.0, 400.0)]
    ok &= check("drift, start e^(-2x/ell): q -> edge QSD, error ~ 1/t (ratio 400 vs 100)", err[1] / err[0], 0.25, 0.05)
    xg = np.linspace(1e-3, 30, 300) * ELL
    for nu0, nu in [(0.5, 0.5), (1.5, 1.0)]:
        p, S = ou_hl_evolve(lomax(nu0), xg, 20 * TAU)
        ok &= check(f"OU, start ~x^(-1-{nu0:g}): q -> QSD nu={nu:g} at t=20 tau", np.max(np.abs(p / S - ou_qsd_nu(xg, nu))),
                    0, 1e-4)
        S1 = ou_hl_evolve(lomax(nu0), xg[:1], 15 * TAU)[1]
        ok &= check(f"OU, start ~x^(-1-{nu0:g}): S decays at {nu:g} k", np.log(S1 / S) / (5 * TAU), nu * K, 1e-3)
    return ok


def check_ou_interval(kind):
    ok = True
    Lxi, x0 = 2.0, 1.2
    x = np.linspace(0, Lxi * ELL, 8001)
    pf = lambda y, t: ou_p(y, t, x0, Lxi, kind)
    ok &= check(f"{kind}: PDE residual at t=0.05", ou_residual(pf, np.linspace(0.05, 1.95, 60), 0.05), 0, 1e-4)
    if kind in ("reflecting", "mixed"):
        ok &= check(f"{kind}: J(0) = 0", abs(ou_current(pf, np.array([0.0]), 0.05)[0]), 0, 1e-5)
    if kind == "reflecting":
        ok &= check(f"{kind}: J(L) = 0", abs(ou_current(pf, np.array([Lxi * ELL]), 0.05)[0]), 0, 1e-5)
        ok &= check(f"{kind}: normalisation at t=0.05", trapz(pf(x, 0.05), x), 1, 1e-6)
    else:
        ok &= check(f"{kind}: p(L) = 0", abs(pf(np.array([Lxi * ELL]), 0.05)[0]), 0, 1e-8)
        nus, c = ou_survival_coeffs(x0, Lxi, kind)
        ok &= check(f"{kind}: S(0.1) modes vs quadrature", (c * np.exp(-nus * 0.1)).sum(), trapz(pf(x, 0.1), x), 1e-6)
        ok &= check(f"{kind}: T = sum c_n/lambda_n vs F formula", (c / nus).sum() * TAU, ou_T(x0, Lxi, kind)[0], 1e-4)
        ok &= check(f"{kind}: QSD average of T = 1/lambda_min", trapz(ou_qsd(x, Lxi, kind) * ou_T(x, Lxi, kind), x),
                    TAU / nus[0], 1e-4)
    ok &= check(f"{kind}: <x(t)> ~ x0 e^-kt at t=0.02", trapz(x * pf(x, 0.02), x), x0 * np.exp(-K * 0.02), 1e-5)
    if kind == "absorbing":
        # pi_L = int j_L dt with j_L = -(D/2) dp/dx at L from the modes; the flux through L
        # is negligible before t0, where the mode sum would converge slowly.
        tot, t0 = 0.0, 0.002
        for nu, Qf, nrm in ou_modes(Lxi, kind):
            dQ = ou_Q(nu, Lxi, "odd")[1] / ELL
            tot += -0.5 * D * np.exp(-(Lxi**2)) * dQ * Qf(x0 / ELL) / nrm * np.exp(-nu * K * t0) / (nu * K)
        ok &= check(f"{kind}: pi_L = erfi ratio vs integrated flux at L", tot, ou_piL(x0, Lxi), 1e-5)
    # limits of the slowest rate
    lo, hi = (1, 2.0) if kind == "reflecting" else (0, {"absorbing": 1.0, "mixed": None}[kind])
    small = {"reflecting": np.pi**2 / 2, "absorbing": np.pi**2 / 2, "mixed": np.pi**2 / 8}[kind]
    ok &= check(f"{kind}: slowest rate x L^2/D at L=0.1 ell -> {small:.4f}", ou_modes(0.1, kind)[lo][0] * 0.1**2, small, 2e-2)
    if hi is not None:
        ok &= check(f"{kind}: slowest rate at L=5 ell -> {hi:g} k", ou_modes(5.0, kind)[lo][0], hi, 1e-5)
    else:
        ok &= check("mixed: lambda_0 T(0) -> 1 at L=3 ell", ou_modes(3.0, kind)[0][0] * ou_T(0.0, 3.0, kind)[0], 1, 2e-3)
    if kind == "absorbing":
        r = [m[0] for m in ou_modes(Lxi, "reflecting")[1:6]]
        a = [m[0] for m in ou_modes(Lxi, "absorbing")[:5]]
        ok &= check("reflecting spectrum = absorbing spectrum + k", np.max(np.abs(np.array(r) - np.array(a) - 1)), 0, 1e-9)
    return ok


def check_circle():
    ok = True
    x = np.linspace(0, L, 2001)
    for t in [0.005, 0.05]:
        ok &= check(f"image sum = mode sum at Dt/L^2={t}", np.max(np.abs(circ_images(x, t, X0) - circ_modes(x, t, X0))),
                    0, 1e-10)
        ok &= check(f"normalisation at Dt/L^2={t}", trapz(circ_images(x, t, X0), x), 1, 1e-10)
    pf = lambda y, t: circ_images(y, t, X0)
    ok &= check("PDE residual at Dt/L^2=0.02", fpe_residual(pf, np.linspace(0, L, 101), 0.02, 0.0), 0, 1e-4)
    a, t = L / 2, 0.01
    y = np.linspace(0, a, 501)
    n = np.arange(1, 400)[:, None]
    refl = 1 / a + (2 / a) * (np.cos(n * np.pi * y / a) * np.exp(-0.5 * D * (n * np.pi / a) ** 2 * t)).sum(0)
    ok &= check("fold: circle = (1/2) reflecting interval (0, L/2) started at a wall",
                np.max(np.abs(circ_modes(X0 + y, t, X0) - refl / 2)), 0, 1e-10)
    q1, t = 2 * np.pi / L, 0.03
    ok &= check("<cos q1(x - x0)> = e^{-lambda_1 t}", trapz(np.cos(q1 * (x - X0)) * pf(x, t), x),
                np.exp(-0.5 * D * q1**2 * t), 1e-10)
    ok &= check("first winding: L^2/T_w = D", L**2 / winding_T(lambda y: 0 * y, L, 0.0), D, 1e-8)
    return ok


def check_drift_circle():
    ok = True
    x = np.linspace(0, L, 2001)
    for v in [-4.0, 7.0]:
        a = v / D
        pf = lambda y, t, v=v: circ_images(y, t, X0, v)
        ok &= check(f"v={v:g}: boosted images = complex-eigenvalue modes",
                    np.max(np.abs(pf(x, 0.02) - circ_modes(x, 0.02, X0, v))), 0, 1e-10)
        ok &= check(f"v={v:g}: PDE residual", fpe_residual(pf, np.linspace(0, L, 101), 0.02, v), 0, 1e-4)
        q1, t = 2 * np.pi / L, 0.03
        c1 = trapz(np.cos(q1 * x) * pf(x, t), x) + 1j * trapz(np.sin(q1 * x) * pf(x, t), x)
        want = np.exp(1j * q1 * X0 - (0.5 * D * q1**2 - 1j * v * q1) * t)
        ok &= check(f"v={v:g}: <e^(iq1 x)> = e^(iq1 x0 - lambda t), lambda = Dq^2/2 - ivq (mode m = -1)",
                    abs(c1 - want), 0, 1e-10)
        ok &= check(f"v={v:g}: J = v/L at Dt/L^2=1", current(pf, np.array([0.37]), 1.0, v)[0], v / L, 1e-8)
        u = lambda y, t: np.exp(-a * (y - X0) + v**2 * t / (2 * D)) * pf(y, t)
        y0 = np.array([0.21])
        ok &= check(f"v={v:g}: gauge lands on a twisted circle, u(x+L)/u(x) = e^(-vL/D)",
                    (u(y0 + L, 0.05) / u(y0, 0.05))[0], np.exp(-a * L), 1e-10)
        printed, corrected = gardiner_J(lambda y: 2 * a * y, L, 1 / L)
        ok &= check(f"v={v:g}: (5.2.46) from (5.2.45) gives J = v/L", corrected, v / L, 1e-8)
        ok &= check(f"v={v:g}: (5.2.46) as printed = -2 x that", printed / corrected, -2, 1e-10)
        Tw = winding_T(lambda y: -2 * a * y, L, 0.0)
        ok &= check(f"v={v:g}: first winding T_w = L tanh(vL/D)/v", Tw, L * np.tanh(a * L) / v, 1e-7)
        ok &= check(f"v={v:g}: renewal V = L tanh(vL/D)/T_w = v", L * np.tanh(a * L) / Tw, v, 1e-7)
    return ok


def check_ou_circle():
    ok = True
    for Lc in [2.0, 3.0]:
        x = np.linspace(-Lc / 2, Lc / 2, 20001)
        ok &= check(f"L={Lc:g} ell: p_s normalisation", trapz(saw_ps(x, Lc), x), 1, 1e-8)
        # the cusp puts a kink in the integrand, so Simpson is only second order here
        xg, ps, J = circle_ps(lambda y: -(saw_wrap(y, Lc) / ELL) ** 2, Lc, ns=20001)
        ok &= check(f"L={Lc:g} ell: cut-invariant p_s = truncated Gaussian, J = 0",
                    np.max(np.abs(ps - saw_ps(xg, Lc))) + abs(J), 0, 1e-5)
    Lc, x0 = 3.0, 0.4
    pf = lambda y, t: saw_p(y, t, x0, Lc)
    x = np.linspace(-Lc / 2, Lc / 2, 30001)
    for t in [0.05, 0.3]:
        ok &= check(f"folded propagator: normalisation at t={t}", trapz(pf(x, t), x), 1, 1e-7)
    ok &= check("folded propagator: PDE residual away from the cusp",
                ou_residual(pf, np.linspace(-1.4, 1.4, 113), 0.05), 0, 1e-4)
    t, h = 0.3, 1e-4
    pm, pp = pf(Lc / 2 - h * np.arange(3), t), pf(-Lc / 2 + h * np.arange(3), t)
    Jm = -K * (Lc / 2) * pm[0] - 0.5 * D * (3 * pm[0] - 4 * pm[1] + pm[2]) / (2 * h)
    Jp = K * (Lc / 2) * pp[0] - 0.5 * D * (-3 * pp[0] + 4 * pp[1] - pp[2]) / (2 * h)
    ok &= check("cusp: J continuous across it (5.2.15)", Jm - Jp, 0, 1e-6)
    for a in [0.75, 2.5]:
        r = [m[0] for m in ou_modes(a, "reflecting")[1:6]]
        ab = [m[0] for m in ou_modes(a, "absorbing")[:5]]
        ok &= check(f"L={2 * a:g} ell: doublets split by exactly k", np.max(np.abs(np.array(r) - np.array(ab) - 1)), 0, 1e-9)
    for Lc in [3.0, 6.0]:
        U2D = lambda y, Lc=Lc: (saw_wrap(y, Lc) / ELL) ** 2
        ok &= check(f"L={Lc:g} ell: Lifson-Jackson = L^2/T_w (Gardiner 5.2.158 on the cover)",
                    saw_Deff(Lc), Lc**2 / winding_T(U2D, Lc, 0.0), 1e-7)
        ok &= check(f"L={Lc:g} ell: T_w the same from x0 = 0.3 L", winding_T(U2D, Lc, 0.3 * Lc) / winding_T(U2D, Lc, 0.0),
                    1, 1e-7)
        ok &= check(f"L={Lc:g} ell: Lifson-Jackson = ring discretisation", saw_ring(Lc)[2] / saw_Deff(Lc), 1, 1e-3)
    lam0 = ou_modes(4.0, "mixed")[0][0] * K
    ok &= check("L=8 ell: D_eff = L^2 lambda_0^mix(L/2)/2 (cusp crossed with probability 1/2)",
                saw_Deff(8.0) / (0.5 * 64 * lam0), 1, 1e-4)
    return ok


def check_cos_circle():
    ok = True
    for kappa, f in [(3.0, 0.0), (3.0, 0.7), (1.0, 2.0)]:
        R = cos_R(kappa)
        Lc, phi, tag = 2 * np.pi * R, cos_phi(kappa, f), f"kappa={kappa:g}, v/kR={f:g}"
        x, ps, J = circle_ps(phi, Lc)
        V, Deff, c = cos_transport(kappa, f)
        ok &= check(f"{tag}: cut-invariant p_s = Fourier null vector", np.max(np.abs(ps - cos_density(c, x, kappa))), 0, 1e-9)
        ok &= check(f"{tag}: = Gardiner (5.2.47) with the seam at 0", np.max(np.abs(ps - gardiner_ps(phi, Lc, x))), 0, 1e-7)
        ok &= check(f"{tag}: V = L J_s = Lambda'(0)", Lc * J, V, 1e-9)
        xb = (np.arange(4000) + 0.5) * Lc / 4000
        _, Vr, Dr = ring_transport(f * K * R - K * R * np.sin(xb / R), Lc)
        ok &= check(f"{tag}: V, ring discretisation", Vr, V, 1e-6)
        ok &= check(f"{tag}: D_eff = Lambda''(0) vs ring", Dr, Deff, 1e-5)
        if f == 0:
            ok &= check(f"{tag}: von Mises", np.max(np.abs(ps - np.exp(kappa * np.cos(x / R)) / (Lc * i0(kappa)))), 0, 1e-10)
            ok &= check(f"{tag}: D_eff = D/I0(kappa)^2", Deff, D / i0(kappa) ** 2, 1e-9)
            U2D = lambda y: -kappa * np.cos(y / R)
            ok &= check(f"{tag}: L^2/T_w (Gardiner 5.2.158 on the cover)", Lc**2 / winding_T(U2D, Lc, 0.0),
                        D / i0(kappa) ** 2, 1e-8)
            ok &= check(f"{tag}: T_w the same from x0 = 0.37 L", winding_T(U2D, Lc, 0.37 * Lc) / winding_T(U2D, Lc, 0.0),
                        1, 1e-7)
        else:
            printed, corrected = gardiner_J(phi, Lc, ps[0])
            ok &= check(f"{tag}: (5.2.46) from (5.2.45) = J_s", corrected, J, 1e-7)
    kappa, f = 3.0, 1e-4
    ok &= check("small tilt: V/v = D_eff/D (Einstein relation)", cos_transport(kappa, f)[0] / (f * K * cos_R(kappa)),
                1 / i0(kappa) ** 2, 1e-6)
    kappa = 20.0
    Lc = 2 * np.pi * cos_R(kappa)
    ok &= check("kappa=20: D_eff / [(k L^2/pi) e^{-2 kappa}] (Kramers, 5.2.174) ~ 1 - 1/4kappa",
                (D / i0(kappa) ** 2) / (K * Lc**2 / np.pi * np.exp(-2 * kappa)), 1 - 1 / (4 * kappa), 1e-3)
    ev_e, ev_o = cos_spectrum(3.0)
    ok &= check("kappa=3: even and odd sectors isospectral (exact doublets)", np.max(np.abs(ev_e[1:6] - ev_o[:5])), 0, 1e-9)
    R = cos_R(3.0)
    Lc, n = 2 * np.pi * R, 1200
    ring_ev = np.sort(-np.linalg.eigvals(ring_ops(-K * R * np.sin((np.arange(n) + 0.5) * Lc / n / R), Lc)[1].toarray()).real)
    ok &= check("kappa=3: lowest levels vs ring discretisation", np.max(np.abs(ring_ev[1:7:2] - ev_e[1:4])), 0, 1e-3)
    ok &= check("kappa -> 0: lambda_1 kappa -> k", cos_spectrum(0.01)[0][1] * 0.01, 1, 1e-4)
    ok &= check("kappa -> infinity: lambda_1 -> k", cos_spectrum(200.0, N=200)[0][1], 1, 5e-3)
    Vm, Dm, rel = mc_tilt()
    V, Deff, _ = cos_transport(3.0, 1.0)
    ok &= check(f"simulation, kappa=3, v=kR: V (rel. s.e. of D_eff {rel:.3f})", Vm, V, 0.01)
    ok &= check("simulation, kappa=3, v=kR: D_eff", Dm, Deff, 0.07)
    return ok


def two_wall_quadrature(psi_over_B, inv_psi, y):
    """Splitting probability pi_b and mean exit time T on a grid y from Gardiner's (5.2.190, corrected) and
    (5.2.158). A repeated node (zero width) lets the integrands jump there without error."""
    I1 = cumulative_trapezoid(inv_psi, y, initial=0)
    I2 = cumulative_trapezoid(inv_psi * cumulative_trapezoid(psi_over_B, y, initial=0), y, initial=0)
    pi_b = I1 / I1[-1]
    return pi_b, 2 * (pi_b * I2[-1] - I2)


def check_interface():
    ok = True
    t, x0, h = 0.3, 0.5 * IF_A, 1e-7
    for name, K in [(n, iface_K(c)) for n, c, _ in CONVENTIONS] + [("K=2.7", 2.7)]:
        p = lambda y, K=K: iface_p(y, t, x0, K)
        ok &= check(f"{name}: propagator normalised", quad(p, -20, 0)[0] + quad(p, 0, 20)[0], 1, 1e-10)
        ok &= check(f"{name}: p(0+)/p(0-) = K", p(h) / p(-h), K, 1e-5)
        flux = IF_DP * (p(2 * h) - p(h)) / (IF_DM * (p(-h) - p(-2 * h)))
        ok &= check(f"{name}: D+ p'(0+) = D- p'(0-) (J continuous)", flux, 1, 1e-4)
        y, hh, dt = np.concatenate([np.linspace(-1.5, -0.05, 30), np.linspace(0.05, 2.5, 50)]), 1e-4, 1e-6
        Bv = np.where(y > 0, IF_DP, IF_DM)
        pt = lambda tt, K=K: iface_p(y, tt, x0, K)
        res = (pt(t + dt) - pt(t - dt)) / (2 * dt) - 0.5 * Bv * (iface_p(y + hh, t, x0, K) - 2 * pt(t)
                                                                + iface_p(y - hh, t, x0, K)) / hh**2
        ok &= check(f"{name}: free equation on each side (residual)", np.max(np.abs(res)), 0, 1e-5)
    ok &= check("potential step at fixed D: r = -tanh(dU/D) with K = e^{-2 dU/D}",
                iface_r(np.exp(-2 * 0.7 / D), D, D), -np.tanh(0.7 / D), 1e-14)
    for name, chi, _ in CONVENTIONS:
        errs = []
        for eps in [0.02, 0.01]:
            x, p = iface_fd(chi, eps, x0, t)
            ex = iface_p(x, t, x0, iface_K(chi))
            errs.append(np.max(np.abs(p - ex)[np.abs(x) > 0.1]) / np.max(ex))
        ok &= check(f"{name}: smoothed step with drift {chi:g} B'/2 (finite differences, eps=0.01a) -> closed form, "
                    "max rel. error", errs[1], 0, 0.015)
        ok &= check(f"{name}: that error halves with eps (O(eps))", errs[0] / errs[1], 2, 0.25)
    # sharp step: piecewise-constant psi = B p_s on a grid with 0 repeated
    y = np.concatenate([np.linspace(-IF_A, 0, 2001), np.linspace(0, IF_A, 2001)])
    left = np.arange(y.size) < 2001
    for name, chi, _ in CONVENTIONS:
        K = iface_K(chi)
        ps = np.where(left, 1.0, K)
        pi_b, T = two_wall_quadrature(ps, 1 / (np.where(left, IF_DM, IF_DP) * ps), y)
        ok &= check(f"{name}: pi_+(x0) vs Gardiner (5.2.190) quadrature", np.max(np.abs(pi_b - iface_split(y, K))), 0, 1e-12)
        ok &= check(f"{name}: T(x0) vs Gardiner (5.2.158) quadrature", np.max(np.abs(T - iface_T(y, K))), 0, 1e-10)
    for name, chi, mean in [("thermal", 1.0, (IF_DP + IF_DM) / 2), ("Stratonovich", 0.5, np.sqrt(IF_DP * IF_DM)),
                            ("Ito", 0.0, 2 / (1 / IF_DP + 1 / IF_DM))]:
        ok &= check(f"{name}: T(0) = a^2/<B>_s = a^2/(its mean of D+, D-)", float(iface_T(0.0, iface_K(chi))),
                    IF_A**2 / mean, 1e-14)
        ok &= check(f"{name}: box rates, transcendental roots vs jump process (4000 cells)",
                    rel_err(iface_box_levels(iface_K(chi)), iface_jump_levels(iface_K(chi))), 0, 1e-5)
    x0s, right = mc_interface()
    eps = 0.03
    ys = np.linspace(-IF_A, IF_A, 40001)
    for i, (name, chi, _) in enumerate(CONVENTIONS):
        Bs = iface_B(ys, eps)[0]
        pi_s = np.interp(x0s, ys, two_wall_quadrature(Bs ** (chi - 1), Bs ** (-chi), ys)[0])
        frac = right[i].mean(1)
        se = np.sqrt(np.maximum(frac * (1 - frac), 1e-4) / right.shape[2])
        print(f"  simulation, {name}: pi_+ = {np.round(frac, 3)} (smoothed exact {np.round(pi_s, 3)}; "
              f"sharp {np.round(iface_split(x0s, iface_K(chi)), 3)})")
        ok &= check(f"{name}: simulated pi_+ vs the smoothed quadrature, max |error|/s.e.",
                    np.max(np.abs(frac - pi_s) / se), 0, 3.5)
    return ok


def check_sqrt():
    ok = True
    t, y0 = 0.4, 1.0
    for d in [3.0, 1.3, 0.5]:
        ok &= check(f"d={d:g}: p_+ normalised", quad(lambda y: sqrt_p(y, t, y0, d, +1), 0, np.inf, limit=200)[0], 1, 1e-9)
    for d in [1.3, 1.0, 0.0, -1.0, -2.5]:
        ok &= check(f"d={d:g}: int p_- = P(1 - d/2, z_t)", quad(lambda y: sqrt_p(y, t, y0, d, -1), 0, np.inf, limit=200)[0],
                    float(sqrt_S(t, y0, d)), 1e-9)
    ok &= check("d=0: Pr(y_t = 0) = e^{-z_t} (branching diffusion)", float(1 - sqrt_S(t, y0, 0.0)), np.exp(-sqrt_z(t, y0)), 1e-12)
    for d, s in [(3.0, 1), (1.3, 1), (1.3, -1), (0.0, -1), (-1.0, -1)]:
        pf = lambda yy, tt, d=d, s=s: sqrt_p(yy, tt, y0, d, s)
        J = lambda yy, d=d, pf=pf, h=1e-4: ((d * D - 2 * K * yy) * pf(yy, t)
                                            - (4 * D * (yy + h) * pf(yy + h, t) - 4 * D * (yy - h) * pf(yy - h, t)) / (4 * h))
        y = np.linspace(0.05, 4, 100)
        res = (pf(y, t + 1e-6) - pf(y, t - 1e-6)) / 2e-6 + (J(y + 1e-4) - J(y - 1e-4)) / 2e-4
        ok &= check(f"d={d:g}, p_{'+' if s > 0 else '-'}: PDE residual", np.max(np.abs(res)) / np.max(pf(y, t)), 0, 1e-4)
    yy = np.linspace(0.01, 5, 200)
    ok &= check("d=2.6: p_+(y) = radial p(sqrt y)/(2 sqrt y) of sec:radial",
                np.max(np.abs(sqrt_p(yy, t, y0, 2.6) - radial_p(np.sqrt(yy), t, np.sqrt(y0), 2.6, K) / (2 * np.sqrt(yy)))), 0, 1e-12)
    for d in [1.3, 0.0, -1.0]:
        ok &= check(f"d={d:g}: p_- = e^(-(2-d)kt) (y0/y)^(1-d/2) p_+ in dimension 4-d (Doob)",
                    np.max(np.abs(sqrt_p(yy, t, y0, d, -1)
                                  - np.exp(-(2 - d) * K * t) * (y0 / yy) ** (1 - d / 2) * sqrt_p(yy, t, y0, 4 - d))), 0, 1e-12)
    d = 1.3
    Jp = lambda yy, s, h: ((d * D - 2 * K * yy) * sqrt_p(yy, t, y0, d, s)
                           - (4 * D * (yy + h) * sqrt_p(yy + h, t, y0, d, s) - 4 * D * (yy - h) * sqrt_p(yy - h, t, y0, d, s)) / (4 * h))
    dS = (sqrt_S(t + 1e-6, y0, d) - sqrt_S(t - 1e-6, y0, d)) / 2e-6
    ok &= check("d=1.3, reflected: J(0+) = 0", Jp(1e-9, +1, 1e-12), 0, 1e-4)
    ok &= check("d=1.3, absorbed: J(0+) = dS/dt (the outflow)", Jp(1e-9, -1, 1e-12), float(dS), 1e-4)
    for s, want, what in [(+1, d / 2 - 1, "reflected p ~ y^(d/2-1)"), (-1, 0.0, "absorbed p -> a nonzero constant")]:
        slope = np.log(sqrt_p(1e-8, t, y0, d, s) / sqrt_p(1e-6, t, y0, d, s)) / np.log(1e-2)
        ok &= check(f"d=1.3, {what} (log-slope at 0)", slope, want, 1e-3)
    slope = np.log((sqrt_p(1e-8, t, y0, d, -1) / sqrt_ps(1e-8, d)) / (sqrt_p(1e-6, t, y0, d, -1) / sqrt_ps(1e-6, d))) / np.log(1e-2)
    ok &= check("d=1.3, absorbed: p/p_s ~ y^(1-d/2) -> 0 (the absorbing condition where B vanishes)", slope, 1 - d / 2, 1e-3)
    for d in [-1.0, 0.0, 1.0, 1.5]:
        psi = lambda y, d=d: y ** (d / 2) * np.exp(-y / ELL**2)
        inner = lambda z, psi=psi: quad(lambda w: 2 * psi(w) / (4 * D * w), z, np.inf, limit=200)[0]
        TG = quad(lambda z: inner(z) / psi(z), 0, y0, limit=200)[0]
        ok &= check(f"d={d:g}: T(y0) = int S dt vs Gardiner (5.2.161) quadrature", sqrt_T(y0, d), TG, 1e-7)
    for d in [1.3, 3.0]:
        for n in [1, 2]:
            Q = lambda y, n=n, d=d: eval_genlaguerre(n, d / 2 - 1, y / ELL**2)
            y, h = np.linspace(0.1, 4, 50), 1e-4
            LQ = (d * D - 2 * K * y) * (Q(y + h) - Q(y - h)) / (2 * h) + 2 * D * y * (Q(y + h) - 2 * Q(y) + Q(y - h)) / h**2
            ok &= check(f"reflected, d={d:g}, n={n}: Laguerre, backward residual at 2nk", np.max(np.abs(LQ + 2 * n * K * Q(y))), 0, 1e-4)
    for d in [1.3, 0.0, -1.0]:
        for n in [0, 1, 2]:
            Q = lambda y, n=n, d=d: y ** (1 - d / 2) * eval_genlaguerre(n, 1 - d / 2, y / ELL**2)
            y, h = np.linspace(0.1, 4, 50), 1e-4
            LQ = (d * D - 2 * K * y) * (Q(y + h) - Q(y - h)) / (2 * h) + 2 * D * y * (Q(y + h) - 2 * Q(y) + Q(y - h)) / h**2
            ok &= check(f"absorbed, d={d:g}, n={n}: y^(1-d/2) L_n^(1-d/2), backward residual at (2n+2-d)k",
                        np.max(np.abs(LQ + (2 * n + 2 - d) * K * Q(y))) / np.max(np.abs(Q(y))), 0, 1e-4)
        tl = 6.0 / K
        rate = -(np.log(sqrt_S(tl + 1e-3, y0, d)) - np.log(sqrt_S(tl - 1e-3, y0, d))) / 2e-3
        ok &= check(f"d={d:g}: S decays at (2-d)k", float(rate), (2 - d) * K, 1e-4)
    for d in [1.0, 0.0, -1.0]:
        lam0 = (2 - d) * K
        for lam in [0.3 * lam0, lam0]:
            q = lambda y, lam=lam, d=d: sqrt_qsd(y, lam, d)
            y, h = np.linspace(0.05, 5, 100), 1e-4
            J = lambda yy, q=q, d=d: (d * D - 2 * K * yy) * q(yy) - (4 * D * (yy + h) * q(yy + h) - 4 * D * (yy - h) * q(yy - h)) / (4 * h)
            res = -(J(y + h) - J(y - h)) / (2 * h) + lam * q(y)
            tag = f"d={d:g}, lambda={lam / K:.2f}k"
            ok &= check(f"QSD {tag}: forward eigen-equation residual", np.max(np.abs(res)) / np.max(q(y)), 0, 1e-5)
            ok &= check(f"QSD {tag}: normalised", quad(q, 0, np.inf, limit=400)[0], 1, 1e-8)
            ok &= check(f"QSD {tag}: positive", float(np.min(q(np.linspace(0, 60, 2000))) > 0), 1, 0)
            if lam < lam0:
                tail = np.log(q(400.0) / q(40.0)) / np.log(10)
                ok &= check(f"QSD {tag}: tail exponent -1 - lambda/2k", tail, -1 - lam / (2 * K), 2e-2)
    y = np.linspace(0.01, 6, 200)
    d, tl = 0.5, 8.0 / K
    ok &= check("d=0.5: p_-/S from a point start -> e^{-y/ell^2}/ell^2 (Yaglom)",
                np.max(np.abs(sqrt_p(y, tl, y0, d, -1) / sqrt_S(tl, y0, d) - np.exp(-y / ELL**2) / ELL**2)), 0, 1e-4)
    # the samplers behind the figure: means against the exact ones
    rng = np.random.default_rng(1)
    ys = np.full(200000, 1.0)
    for _ in range(50):
        ys = sqrt_branching_step(ys, 0.01, rng)
    ok &= check("d=0 exact sampler: <y_t> = y0 e^{-2kt} at kt=0.5", ys.mean(), np.exp(-2 * K * 0.5), 0.01)
    ok &= check("d=0 exact sampler: Pr(y_t = 0) = e^{-z_t} at kt=0.5", np.mean(ys == 0), np.exp(-sqrt_z(0.5, 1.0)), 0.01)
    return ok


def check_qif():
    ok = True
    for I, Dq in [(1.0, 0.5), (0.0, 1.0), (-0.5, 0.5)]:
        # Gardiner (5.2.160) on (-X, X), reflecting at -X and absorbing at X, plus the deterministic flights 1/X
        X, n = 50.0, 400001
        x = np.linspace(-X, X, n)
        phi = (2 / Dq) * (x**3 / 3 + I * x)
        dphi, hx = np.diff(phi), x[1] - x[0]
        seg = hx * -np.expm1(-dphi) / dphi
        inner = np.zeros(n)
        for i in range(n - 2, -1, -1):  # inner(x_i) = int_{x_i}^X e^{phi(x_i) - phi(y)} dy, stably
            inner[i] = seg[i] + np.exp(-dphi[i]) * inner[i + 1]
        Tg = 2 / Dq * trapz(inner, x) + 2 / X
        ok &= check(f"I={I:g}, D={Dq:g}: 1/J_s single integral vs (5.2.160) on a grid", qif_T(I, Dq), Tg, 1e-4)
    ok &= check("D -> 0, I = 1: period -> pi/sqrt(I)", qif_T(1.0, 1e-3), np.pi, 1e-6)
    ok &= check("I = 0: period = sqrt(2 pi)/3 6^(1/6) Gamma(1/6) D^(-1/3)", qif_T(0.0, 0.37),
                np.sqrt(2 * np.pi) / 3 * 6 ** (1 / 6) * gamma(1 / 6) * 0.37 ** (-1 / 3), 1e-9)
    ok &= check("I = -1, D = 0.1: J_s / Kramers (c/pi) e^{-8c^3/3D} -> 1", (1 / qif_T(-1.0, 0.1)) / (np.exp(-8 / 0.3) / np.pi), 1, 0.03)
    for I in [1.0, -1.0]:
        J = 1 / qif_T(I, 1.0)
        X = 30.0
        x = np.linspace(-X, X, 6001)
        ok &= check(f"I={I:g}, D=1: p_s normalised (grid plus the tails 2 J_s/X)", trapz(qif_ps(x, I, 1.0), x) + 2 * J / X, 1, 2e-4)
        ok &= check(f"I={I:g}, D=1: x^2 p_s -> J_s at x = +-200", float(np.max(np.abs(qif_ps(np.array([-200.0, 200.0]), I, 1.0)
                                                                                  * 200**2 / J - 1))), 0, 1e-3)
        ok &= check(f"I={I:g}, D=1: theta density at theta = pi is J_s/2", float(qif_ps(np.array([1e4]), I, 1.0)[0] * (1 + 1e8) / 2), J / 2,
                    1e-4)
    xs, Dq, I = np.linspace(-3, 3, 13), 0.7, 0.4
    th = 2 * np.arctan(xs)
    ito = 2 * (xs**2 + I) / (1 + xs**2) - 2 * Dq * xs / (1 + xs**2) ** 2
    ok &= check("theta = 2 arctan x: Ito drift = 1 - cos + I(1 + cos) - (D/2) sin (1 + cos)",
                np.max(np.abs(ito - (1 - np.cos(th) + I * (1 + np.cos(th)) - 0.5 * Dq * np.sin(th) * (1 + np.cos(th))))), 0, 1e-12)
    ok &= check("theta = 2 arctan x: noise 2 sqrt(D)/(1 + x^2) = sqrt(D)(1 + cos)",
                np.max(np.abs(2 / (1 + xs**2) - (1 + np.cos(th)))), 0, 1e-12)
    for Dq in QIF_D:
        for I in ([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0] if Dq == 1.0 else [-0.25, 0.0, 0.5, 1.0, 2.0]):
            r, se = mc_qif(I, Dq)
            ok &= check(f"simulation of theta, I={I:g}, D={Dq:g}: firing rate (s.e. {se:.4f})", r, 1 / qif_T(I, Dq), 3.5 * se)
    # regular at infinity: the free (drifting) circle seen through x = tan(theta/2)
    v, Dq = 0.7, 1.0
    x, h = np.linspace(-6, 6, 25), 1e-5
    A = lambda y: 0.5 * v * (1 + y**2) + 0.25 * Dq * y * (1 + y**2)
    Bp = lambda y: 0.25 * Dq * (1 + y**2) ** 2 / (np.pi * (1 + y**2))  # B times the Cauchy density
    J = A(x) / (np.pi * (1 + x**2)) - 0.5 * (Bp(x + h) - Bp(x - h)) / (2 * h)
    ok &= check("tan chart of the drifting circle: Cauchy p_s carries the constant current v/2 pi", np.max(np.abs(J - v / (2 * np.pi))), 0, 1e-8)
    f = lambda y: np.exp(-(4 * v / Dq) * np.arctan(y)) / (1 + y**2)
    g = lambda y: 8 * np.exp((4 * v / Dq) * np.arctan(y)) / (Dq * (1 + y**2))
    ok &= check("tan chart: int f and int g over the line are finite (both ends regular)",
                float(np.isfinite(quad(f, -np.inf, np.inf)[0]) and np.isfinite(quad(g, -np.inf, np.inf)[0])), 1, 0)
    return ok


def check_feller():
    ok = True
    for d, want in [(-1.0, "exit"), (0.0, "exit"), (1.0, "regular"), (3.0, "entrance")]:
        f = lambda y, d=d: y ** (-d / 2) * np.exp(y / ELL**2)
        g = lambda y, d=d: y ** (d / 2 - 1) * np.exp(-y / ELL**2) / (2 * D)
        got = feller_type(f, g, 0.0, 1.0)
        ok &= check(f"square-root process, d={d:g}: y=0 is {got} (Gardiner 5.2.25-29)", float(got == want), 1, 0)
        # the same process in r = sqrt(y): B = D, A = (d-1)D/2r - kr; the type does not depend on the chart
        fr = lambda r, d=d: r ** (1 - d) * np.exp(K * r * r / D)
        gr = lambda r, d=d: 2 * r ** (d - 1) * np.exp(-K * r * r / D) / D
        got = feller_type(fr, gr, 0.0, 1.0)
        ok &= check(f"radius, d={d:g}: r=0 is {got} too", float(got == want), 1, 0)
    fc = lambda x: (1 + x) ** -4 * np.exp(2 * x)  # Gardiner's X + A <-> 2X, (5.2.58) with a = 1
    gc = lambda x: 2 * (1 + x) ** 3 * np.exp(-2 * x) / x
    got = feller_type(fc, gc, 0.0, 1.0)
    ok &= check(f"Gardiner's X + A <-> 2X: x=0 is {got} (5.2.60)", float(got == "exit"), 1, 0)
    mu_g, sg = 0.3, 1.0  # geometric Brownian motion, dx = mu x dt + sigma x dW
    got = feller_type(lambda x: x ** (-2 * mu_g / sg**2), lambda x: 2 * x ** (2 * mu_g / sg**2 - 2) / sg**2, 0.0, 1.0)
    ok &= check(f"geometric Brownian motion: x=0 is {got} (5.2.24)", float(got == "natural"), 1, 0)
    a0, b, sg = 0.8, 0.5, 0.9
    xs = np.linspace(0.2, 3, 15)
    ok &= check("y = ln x maps dx = (a0 - bx)dt + sigma x dW to the Morse drift a0 e^{-y} - b - sigma^2/2",
                np.max(np.abs((a0 - b * xs) / xs - 0.5 * sg**2 - (a0 * np.exp(-np.log(xs)) - b - 0.5 * sg**2))), 0, 1e-14)
    w = 1.0
    ev = h_interval(V_from(lambda x: -0.5 * D / w * np.tan(x / w), lambda x: -0.5 * D / w**2 / np.cos(x / w) ** 2),
                    -np.pi * w / 2, np.pi * w / 2, 20000, m=4)
    ok &= check("neutral Wright-Fisher (s = -1/2, absorbed): rates (D/2w^2) n(n+1), n >= 1 (the sphere's, less 0)",
                rel_err(ev, [lncos_lam(n, 0.5, w) for n in range(1, 5)]), 0, 1e-4)
    return ok


# --- Gardiner's P_lambda and Q_lambda in every example ------------------------------------
# P solves the forward eigen-equation (5.2.75), Q the backward one (5.2.76), and int P_n Q_m = delta_nm (5.2.79).
# Where the stationary current vanishes P = p_s Q, with p_s taken from the reflecting problem when a wall absorbs
# (5.2.91); on an unbounded domain p_s is the zero-current solution, normalisable or not.
def pq_gram(Ps, Qs, x):
    """max |int P_n Q_m dx - delta_nm| by the trapezoid rule on x (complex values allowed)."""
    G = np.array([[np.sum(0.5 * (P * Q)[1:] * np.diff(x) + 0.5 * (P * Q)[:-1] * np.diff(x)) for Q in Qs] for P in Ps])
    return np.max(np.abs(G - np.eye(len(Ps))))


def pq_residual(P, Q, lam_, A, B, x, h=1e-4):
    """Residuals of -(AP)' + (BP)''/2 = -lam P and A Q' + B Q''/2 = -lam Q (callables), the larger one, relative
    to (1 + |lam|) max|P| and (1 + |lam|) max|Q|, the size of each term."""
    LP = (-(A(x + h) * P(x + h) - A(x - h) * P(x - h)) / (2 * h)
          + 0.5 * (B(x + h) * P(x + h) - 2 * B(x) * P(x) + B(x - h) * P(x - h)) / h**2)
    LQ = A(x) * (Q(x + h) - Q(x - h)) / (2 * h) + 0.5 * B(x) * (Q(x + h) - 2 * Q(x) + Q(x - h)) / h**2
    return max(np.max(np.abs(LP + lam_ * P(x))) / np.max(np.abs(P(x))),
               np.max(np.abs(LQ + lam_ * Q(x))) / np.max(np.abs(Q(x)))) / (1 + abs(lam_))


def hermite_h(N, xi):
    """Gardiner's (5.2.118), Q_n = H_n(xi)/sqrt(2^n n!) for n < N, by the stable recurrence."""
    h = [np.ones_like(xi), np.sqrt(2) * xi]
    for n in range(1, N - 1):
        h.append(np.sqrt(2 / (n + 1)) * xi * h[n] - np.sqrt(n / (n + 1)) * h[n - 1])
    return h[:N]


def check_pq():
    ok = True
    BD = lambda y: D + 0 * y
    const = lambda c: (lambda y: c + 0 * y)

    # Part I: Q_n = sqrt2 cos, sqrt2 sin, P_n = Q_n/L; the half-lines and the line as k-integrals
    x = np.linspace(0, L, 20001)
    fam = {"reflecting": [const(1.0)] + [lambda y, n=n: np.sqrt(2) * np.cos(n * np.pi * y / L) for n in range(1, 6)],
           "absorbing": [lambda y, n=n: np.sqrt(2) * np.sin(n * np.pi * y / L) for n in range(1, 7)],
           "mixed": [lambda y, n=n: np.sqrt(2) * np.cos((2 * n + 1) * np.pi * y / (2 * L)) for n in range(6)]}
    for name, Qs in fam.items():
        ok &= check(f"free, {name} interval: int P_n Q_m = delta_nm with P_n = Q_n/L",
                    pq_gram([Q(x) / L for Q in Qs], [Q(x) for Q in Qs], x), 0, 1e-8)
    k = np.linspace(0, 40, 40001)
    xs, t = np.linspace(0.05, 4, 80), 0.5
    for sign, f, name in [(+1, np.cos, "reflecting"), (-1, np.sin, "absorbing")]:
        p = trapz_rows(2 / np.pi * f(np.outer(xs, k)) * f(k * X0H) * np.exp(-0.5 * D * k**2 * t), k)
        ok &= check(f"free half-line, {name}: int dk P_k(x) Q_k(x0) e^(-lambda t), P_k = (2/pi) Q_k, = images",
                    np.max(np.abs(p - half_line(xs, t, X0H, sign))), 0, 1e-10)
    kk = np.linspace(-40, 40, 80001)
    xl = np.linspace(-3, 3, 121)
    p = trapz_rows(np.cos(np.outer(xl - X0H, kk)) * np.exp(-0.5 * D * kk**2 * t), kk) / (2 * np.pi)
    ok &= check("free line: P_k = e^{ikx}/2pi, Q_k = e^{-ikx} give the Gaussian", np.max(np.abs(p - heat(xl - X0H, t))), 0, 1e-10)

    # Part II on the line and half-lines: P = sqrt(p_s/N) u, Q = u/sqrt(N p_s), p_s ~ e^{2 alpha x}
    v = -1.3
    a = v / D
    gauge = np.exp(a * (xl - X0H)) * trapz_rows(np.cos(np.outer(xl - X0H, kk)) * np.exp(-(v**2 / (2 * D) + 0.5 * D * kk**2) * t),
                                                kk) / (2 * np.pi)
    fourier = trapz_rows(np.cos(np.outer(xl - X0H - v * t, kk)) * np.exp(-0.5 * D * kk**2 * t), kk) / (2 * np.pi)
    ok &= check("drift line: gauge modes (real lambda) give g(x - x0 - vt)", np.max(np.abs(gauge - heat(xl - X0H - v * t, t))),
                0, 1e-10)
    ok &= check("drift line: Fourier modes (lambda = Dk^2/2 + ivk) give the same", np.max(np.abs(fourier - gauge)), 0, 1e-10)
    for v in [-1.0, 1.0]:
        a = v / D
        # u_k = cos kx + (a/k) sin kx, written with sinc so that k = 0 is on the grid; 1/N_k = (2/pi) k^2/(k^2 + a^2)
        u = lambda y: np.cos(np.outer(y, k)) + a * y[:, None] * np.sinc(np.outer(y, k) / np.pi)
        invN = 2 / np.pi * k**2 / (k**2 + a**2)
        decay = np.exp(-(v**2 / (2 * D) + 0.5 * D * k**2) * t)
        p = np.exp(a * (xs - X0H)) * trapz_rows(u(xs) * u(np.array([X0H])) * invN * decay, k)
        if v < 0:
            p = p + 2 * abs(a) * np.exp(2 * a * xs)  # P_0 Q_0 = p_s
        ok &= check(f"drift half-line, reflecting, v={v:g}: Robin continuum{' + P_0 = p_s' if v < 0 else ''} = Robin image",
                    np.max(np.abs(p - dhr_p(xs, t, X0H, v))), 0, 1e-10)
        p = np.exp(a * (xs - X0H)) * trapz_rows(2 / np.pi * np.sin(np.outer(xs, k)) * np.sin(k * X0H) * decay, k)
        ok &= check(f"drift half-line, absorbing, v={v:g}: P_k = sqrt(2p_s/pi) sin kx, Q_k = sqrt(2/pi p_s) sin kx",
                    np.max(np.abs(p - dha_p(xs, t, X0H, v))), 0, 1e-10)

    # Part II on the interval (p_s of the reflecting interval throughout) and the circle
    xi = np.linspace(0.01, L - 0.01, 99)
    for v in [-2.0, 3.0]:
        a = v / D
        ps = lambda y, v=v: dr_stationary(y, v)
        kn = np.arange(1, 6) * np.pi / L
        Nn = 0.5 * L * (1 + (a / kn) ** 2)
        lams = [0.0] + list(v**2 / (2 * D) + 0.5 * D * kn**2)
        sets = {"reflecting": ([ps] + [lambda y, k=k, N=N: np.sqrt(ps(y) / N) * dr_phi(y, k, a) for k, N in zip(kn, Nn)],
                               [const(1.0)] + [lambda y, k=k, N=N: dr_phi(y, k, a) / np.sqrt(N * ps(y)) for k, N in zip(kn, Nn)],
                               lams),
                "absorbing": ([lambda y, k=k: np.sqrt(2 * ps(y) / L) * np.sin(k * y) for k in kn],
                              [lambda y, k=k: np.sqrt(2 / (L * ps(y))) * np.sin(k * y) for k in kn], lams[1:]),
                "mixed": ([lambda y, u=u, N=N: np.sqrt(ps(y) / N) * u(y) for _, u, N in dm_modes(v)[:6]],
                          [lambda y, u=u, N=N: u(y) / np.sqrt(N * ps(y)) for _, u, N in dm_modes(v)[:6]],
                          [m[0] for m in dm_modes(v)[:6]])}
        for name, (Ps, Qs, ls) in sets.items():
            ok &= check(f"drift {name} interval, v={v:g}: int P_n Q_m = delta_nm", pq_gram([P(x) for P in Ps], [Q(x) for Q in Qs], x),
                        0, 1e-7)
            ok &= check(f"drift {name} interval, v={v:g}: forward and backward eigen-equations",
                        max(pq_residual(P, Q, l, const(v), BD, xi) for P, Q, l in zip(Ps, Qs, ls)), 0, 1e-5)
        Ps, Qs, _ = sets["mixed"]
        h = 1e-6
        walls = max(max(abs(v * P(0.0) - 0.5 * D * (P(h) - P(-h)) / (2 * h)), abs(Q(h) - Q(-h)) / (2 * h), abs(P(L)), abs(Q(L)))
                    for P, Q in zip(Ps, Qs))
        ok &= check(f"drift mixed interval, v={v:g}: J(P) = Q' = 0 at 0, P = Q = 0 at L", walls, 0, 1e-7)
    xc = np.linspace(0, L, 4001)
    for v in [0.0, 3.0]:
        q = lambda m: 2 * np.pi * m / L
        ms = range(-3, 4)
        Ps = [lambda y, m=m: np.exp(1j * q(m) * y) / L for m in ms]
        Qs = [lambda y, m=m: np.exp(-1j * q(m) * y) for m in ms]
        ok &= check(f"circle, v={v:g}: P_m = e^(iq_m x)/L, Q_m = e^(-iq_m x) biorthonormal",
                    pq_gram([P(xc) for P in Ps], [Q(xc) for Q in Qs], xc), 0, 1e-12)
        ok &= check(f"circle, v={v:g}: eigen-equations with lambda_m = Dq_m^2/2 + ivq_m",
                    max(pq_residual(P, Q, 0.5 * D * q(m) ** 2 + 1j * v * q(m), const(v), BD, xc[::40])
                        for P, Q, m in zip(Ps, Qs, ms)), 0, 1e-5)

    # Part III: Hermite (5.2.118) on the line and half-lines; Kummer on the intervals; the sawtooth; the cosine well
    xo = np.linspace(-9, 9, 36001) * ELL
    H = hermite_h(8, xo / ELL)
    pso = np.exp(-((xo / ELL) ** 2)) / (ELL * np.sqrt(np.pi))
    ok &= check("OU line: Q_n = H_n/sqrt(2^n n!), P_n = p_s Q_n, biorthonormal", pq_gram([pso * Q for Q in H], H, xo), 0, 1e-10)
    xh = xo[xo >= 0]
    H = hermite_h(12, xh / ELL)
    for par, name in [(0, "reflecting"), (1, "absorbing")]:
        Qs = H[par::2]
        ok &= check(f"OU half-line, {name}: {'even' if par == 0 else 'odd'} Q_n, P_n = 2 x (line p_s) Q_n, biorthonormal",
                    pq_gram([2 * pso[xo >= 0] * Q for Q in Qs], Qs, xh), 0, 1e-10)
    xg, x0, t = np.linspace(0.02, 4, 60) * ELL, 1.5 * ELL, 0.3
    Hx, H0 = hermite_h(120, xg / ELL), hermite_h(120, np.array([x0 / ELL]))
    psx = 2 * np.exp(-((xg / ELL) ** 2)) / (ELL * np.sqrt(np.pi))
    for sign, par, name in [(+1, 0, "reflecting"), (-1, 1, "absorbing")]:
        p = sum(psx * Hx[n] * H0[n] * np.exp(-n * K * t) for n in range(par, 120, 2))
        ok &= check(f"OU half-line, {name}: sum P_n(x) Q_n(x0) e^(-nkt) = images", np.max(np.abs(p - ou_half(xg, t, x0, sign))), 0, 1e-10)
    Lxi = 2.0
    c = 2 / (ELL * np.sqrt(np.pi) * erf(Lxi))  # p_s = c e^{-xi^2}, the reflecting interval's
    xg = np.linspace(0, Lxi * ELL, 20001)
    xr = np.linspace(0.01, Lxi - 0.01, 99) * ELL
    for kind in ["reflecting", "absorbing", "mixed"]:
        md = ou_modes(Lxi, kind)[:6]
        Qs = [lambda y, Qf=Qf, N=N: Qf(y / ELL) / np.sqrt(c * N) for _, Qf, N in md]
        Ps = [lambda y, Q=Q: c * np.exp(-((y / ELL) ** 2)) * Q(y) for Q in Qs]
        ok &= check(f"OU {kind} interval, L=2 ell: Q_n normalised by int p_s Q_n^2 = 1, P_n = p_s Q_n, biorthonormal",
                    pq_gram([P(xg) for P in Ps], [Q(xg) for Q in Qs], xg), 0, 1e-7)
        ok &= check(f"OU {kind} interval: eigen-equations", max(pq_residual(P, Q, nu * K, lambda y: -K * y, BD, xr)
                                                               for P, Q, (nu, _, _) in zip(Ps, Qs, md)), 0, 1e-5)
    ok &= check("OU reflecting interval: Q_0 = 1", np.max(np.abs(ou_modes(Lxi, "reflecting")[0][1](xg / ELL) / np.sqrt(c * ou_modes(Lxi, "reflecting")[0][2]) - 1)),
                0, 1e-7)
    Lc = 3.0 * ELL
    a = Lc / (2 * ELL)
    c = 2 / (ELL * np.sqrt(np.pi) * erf(a))
    sec = {"R": ou_modes(a, "reflecting"), "A": ou_modes(a, "absorbing")}
    Qe = [lambda y, Qf=Qf, N=N: Qf(np.abs(y) / ELL) / np.sqrt(c * N) for _, Qf, N in sec["R"]]
    Qo = [lambda y, Qf=Qf, N=N: np.sign(y) * Qf(np.abs(y) / ELL) / np.sqrt(c * N) for _, Qf, N in sec["A"]]
    xs = np.linspace(-Lc / 2, Lc / 2, 20001)
    Qs = Qe[:4] + Qo[:4]
    ok &= check("sawtooth circle: Q = Q^R(|x|), sgn(x) Q^A(|x|), P = p_s Q, biorthonormal on the circle",
                pq_gram([saw_ps(xs, Lc) * Q(xs) for Q in Qs], [Q(xs) for Q in Qs], xs), 0, 1e-7)
    x0, t = 0.4 * ELL, 0.2
    p = (sum(saw_ps(xs, Lc) * Q(xs) * Q(x0) * np.exp(-nu * K * t) for Q, (nu, _, _) in zip(Qe, sec["R"]))
         + sum(saw_ps(xs, Lc) * Q(xs) * Q(x0) * np.exp(-nu * K * t) for Q, (nu, _, _) in zip(Qo, sec["A"])))
    ok &= check("sawtooth circle: sum P Q(x0) e^(-lambda t) = the glued propagator", np.max(np.abs(p - saw_p(xs, t, x0, Lc))), 0, 1e-10)
    kappa = 3.0
    R = cos_R(kappa)
    m = np.arange(-40, 41)
    xg = np.linspace(0, 2 * np.pi * R, 201)
    for f in [0.0, 0.7]:
        lam_, Vr = np.linalg.eig(cos_fourier(kappa, f, N=40)[1])
        W = np.linalg.inv(Vr)
        order = np.argsort(-lam_.real)[:5]
        P = lambda y, j: (Vr[:, j] * np.exp(1j * np.outer(y / R, m))).sum(1)
        Q = lambda y, j: (W[j, :] * np.exp(-1j * np.outer(y / R, m))).sum(1)
        A = lambda y: f * K * R - K * R * np.sin(y / R)
        ok &= check(f"cosine circle, v/kR={f:g}: right/left eigenvectors of the recurrence are P_lambda, Q_lambda",
                    max(pq_residual(lambda y: P(y, j), lambda y: Q(y, j), -lam_[j], A, BD, xg) for j in order), 0, 1e-5)
        res = max(pq_residual(lambda y: P(y, j), lambda y: P(y, j) / P(y, order[0]), -lam_[j], A, BD, xg) for j in order[1:])
        if f == 0:
            ok &= check(f"cosine circle, v=0: P_lambda/p_s is the backward eigenfunction (P = p_s Q)", res, 0, 1e-5)
        else:
            ok &= check(f"cosine circle, v/kR={f:g}: P_lambda/p_s is not (residual of order 1)", float(res > 0.1), 1, 0)

    # Part IV: Laguerre, the Bessel continuum, lncosh bound states, Gegenbauer
    d = 2.6
    mu = d / 2 - 1
    rg = np.linspace(0, 8, 40001) * ELL
    Qs = [np.sqrt(gamma(n + 1) * gamma(d / 2) / gamma(n + d / 2)) * eval_genlaguerre(n, mu, (rg / ELL) ** 2) for n in range(6)]
    ok &= check(f"radial, d={d:g}: Q_n = sqrt(n! G(d/2)/G(n+d/2)) L_n^(mu)(r^2/ell^2), P_n = p_s Q_n, biorthonormal",
                pq_gram([radial_ps(rg, d) * Q for Q in Qs], Qs, rg), 0, 1e-8)
    t, r0 = 0.7, 1.3
    kap = np.linspace(0, 30, 30001)[1:]
    rr = np.linspace(0.05, 5, 100)
    p = rr ** (mu + 1) * r0 ** (-mu) * trapz_rows(kap * jv(mu, np.outer(rr, kap)) * jv(mu, kap * r0) * np.exp(-0.5 * D * kap**2 * t), kap)
    ok &= check(f"radial, d={d:g}, k=0: int dlambda P_lambda Q_lambda e^(-lambda t), Bessel J_mu, = closed form",
                np.max(np.abs(p - radial_p(rr, t, r0, d))), 0, 1e-6)
    s, w = 2.5, 1.0
    xg = np.linspace(-40, 40, 160001) * w
    ps = lncosh_ps(xg, s, w)
    Qs = [np.cosh(xg / w) ** n * eval_jacobi(n, s - n, s - n, np.tanh(xg / w)) for n in range(3)]
    Qs = [Q / np.sqrt(trapz(ps * Q * Q, xg)) for Q in Qs]
    ok &= check(f"lncosh, s={s:g}: bound states Q_n = cosh^n P_n^(s-n,s-n)(tanh), P_n = p_s Q_n, biorthogonal",
                pq_gram([ps * Q for Q in Qs], Qs, xg), 0, 1e-8)
    xr = np.linspace(-3, 3, 121) * w
    ok &= check(f"lncosh, s={s:g}: backward residual of Q_n", max(
        pq_residual(lambda y, n=n: lncosh_P(y, n, s, w), lambda y, n=n: np.cosh(y / w) ** n * eval_jacobi(n, s - n, s - n, np.tanh(y / w)),
                    lncosh_levels(s, w)[n], lambda y: -s * D / w * np.tanh(y / w), BD, xr) for n in range(3)), 0, 1e-5)
    for s in [1.5, 0.3]:
        w = 1.0
        xg = np.linspace(-np.pi * w / 2, np.pi * w / 2, 200001)
        Qs = [np.sqrt(gamma(n + 1) * (n + s) * gamma(2 * s) / (s * gamma(n + 2 * s))) * eval_gegenbauer(n, s, np.sin(xg / w))
              for n in range(6)]
        ok &= check(f"lncos, s={s:g}: Q_n = sqrt(n!(n+s)G(2s)/sG(n+2s)) C_n^(s)(sin x/w), P_n = p_s Q_n, biorthonormal",
                    pq_gram([lncos_ps(xg, s, w) * Q for Q in Qs], Qs, xg), 0, 1e-6)
        ok &= check(f"lncos, s={s:g}: Q_1 = sqrt(2s+2) sin(x/w)", np.max(np.abs(Qs[1] - np.sqrt(2 * s + 2) * np.sin(xg / w))), 0, 1e-12)

    # Part V: the double well
    xg, lam_, vec = dw_eigen(8.0, n=3000, m=2)
    h = xg[1] - xg[0]
    ps = np.exp(-2 * dw_U(8.0)(xg) / D)
    ps /= ps.sum() * h
    Q1 = vec[:, 1] / np.sqrt(h * ps)  # Q = psi/sqrt(p_s), so that int p_s Q_1^2 = sum psi^2 = 1
    Q1 *= np.sign(Q1[np.argmin(np.abs(xg - 1.0))])
    q1 = 2 * dw_committor(8.0, xg) - 1
    ok &= check("double well, beta=8: int p_s (2 pi_c - 1)^2 = 1, so the committor is already normalised", h * np.sum(ps * q1**2), 1, 1e-3)
    mm = np.abs(xg) < 1.5
    ok &= check("double well, beta=8: normalised Q_1 = 2 pi_c - 1", np.max(np.abs(Q1[mm] - q1[mm])), 0, 2e-2)
    xg, lam_, vec = dw_eigen(80.0, m=4)
    ok &= check("double well, beta=80: the lambda_2 mode psi_2 sits on the hilltop (weight in |x| < a/4)",
                np.sum(vec[np.abs(xg) < 0.25, 2] ** 2), 1, 1e-3)

    # Part VI: the interface box, and the square-root process reflected and absorbed
    a, Dm, Dp = IF_A, IF_DM, IF_DP
    xl, xr = np.linspace(-a, 0, 20001), np.linspace(0, a, 20001)
    for chi in [1.0, 0.0]:
        Kc = iface_K(chi)
        lams = [0.0] + list(iface_box_levels(Kc, m=4))
        kp_, km_ = (np.sqrt(2 * np.array(lams) / Dd) for Dd in (Dp, Dm))
        Ql = [np.cos(q * (a + xl)) / np.cos(q * a) for q in km_]
        Qr = [np.cos(q * (a - xr)) / np.cos(q * a) for q in kp_]
        psl, psr = 1 / (a * (1 + Kc)), Kc / (a * (1 + Kc))
        nrm = [np.sqrt(psl * trapz(l * l, xl) + psr * trapz(r * r, xr)) for l, r in zip(Ql, Qr)]
        G = np.array([[(psl * trapz(Ql[i] * Ql[j], xl) + psr * trapz(Qr[i] * Qr[j], xr)) / (nrm[i] * nrm[j]) for j in range(5)]
                      for i in range(5)])
        ok &= check(f"interface box, chi={chi:g}: Q = cos(k(a - |x|))/cos(ka), P = p_s Q, biorthonormal",
                    np.max(np.abs(G - np.eye(5))), 0, 1e-8)
    y0, t = ELL**2, 0.3
    yg = np.linspace(0.05, 5, 60) * ELL**2
    zz = lambda y: y / ELL**2
    d = 1.2
    mu = d / 2 - 1
    cr = lambda n: np.sqrt(gamma(n + 1) * gamma(d / 2) / gamma(n + d / 2))
    G = np.array([[quad(lambda y: sqrt_ps(y, d) * cr(n) * cr(j) * eval_genlaguerre(n, mu, zz(y)) * eval_genlaguerre(j, mu, zz(y)),
                        0, np.inf, limit=200)[0] for j in range(5)] for n in range(5)])
    ok &= check(f"square root, d={d:g}, reflected: Q_n = sqrt(n! G(d/2)/G(n+d/2)) L_n^(mu)(z), biorthonormal",
                np.max(np.abs(G - np.eye(5))), 0, 1e-8)
    p = sum(sqrt_ps(yg, d) * cr(n) ** 2 * eval_genlaguerre(n, mu, zz(yg)) * eval_genlaguerre(n, mu, zz(y0)) * np.exp(-2 * n * K * t)
            for n in range(80))
    ok &= check(f"square root, d={d:g}, reflected: sum P_n Q_n(y0) e^(-2nkt) = p_+", np.max(np.abs(p - sqrt_p(yg, t, y0, d, +1))), 0, 1e-10)
    for d in [1.2, -0.5]:
        b = 1 - d / 2
        ca = lambda n, b=b: np.sqrt(gamma(n + 1) / gamma(n + 1 + b))
        Pn = lambda y, n, b=b: ca(n) * np.exp(-zz(y)) * eval_genlaguerre(n, b, zz(y)) / ELL**2
        Qn = lambda y, n, b=b: ca(n) * zz(y) ** b * eval_genlaguerre(n, b, zz(y))
        G = np.array([[quad(lambda y: Pn(y, n) * Qn(y, j), 0, np.inf, limit=200)[0] for j in range(5)] for n in range(5)])
        ok &= check(f"square root, d={d:g}, absorbed: P_n = e^(-z) L_n^(b)(z)/ell^2, Q_n = z^b L_n^(b)(z) (b = 1-d/2), biorthonormal",
                    np.max(np.abs(G - np.eye(5))), 0, 1e-8)
        p = sum(Pn(yg, n) * Qn(y0, n) * np.exp(-(2 * n + 2 - d) * K * t) for n in range(80))
        ok &= check(f"square root, d={d:g}, absorbed: sum P_n Q_n(y0) e^(-(2n+2-d)kt) = p_-",
                    np.max(np.abs(p - sqrt_p(yg, t, y0, d, -1))), 0, 1e-10)
    return ok


# name -> (title, checks, figure makers)
SECTIONS = {
    "reflecting": ("Interval, reflecting walls", check_reflecting, [fig_reflecting]),
    "absorbing": ("Interval, absorbing walls", check_absorbing,
                  [fig_absorbing_density, lambda: fig_absorbing_exit(mc_absorbing()),
                   lambda: fig_absorbing_exit_times(mc_absorbing())]),
    "mixed": ("Interval, reflecting at 0 and absorbing at L", check_mixed, [lambda: fig_mixed(mc_mixed())]),
    "halfline_reflecting": ("Half-line, reflecting at 0", check_halfline_reflecting, [fig_halfline_reflecting]),
    "halfline_absorbing": ("Half-line, absorbing at 0", check_halfline_absorbing, [fig_halfline_absorbing]),
    "line": ("Whole line, and summary of A = 0", check_line, [fig_free_summary]),
    "circle": ("Circle", check_circle, [fig_circle]),
    "drift_reflecting": ("Drift, interval, reflecting walls", check_drift_reflecting, [fig_drift_reflecting]),
    "drift_absorbing": ("Drift, interval, absorbing walls", check_drift_absorbing, [fig_drift_absorbing]),
    "drift_mixed": ("Drift, interval, reflecting at 0 and absorbing at L", check_drift_mixed, [fig_drift_mixed]),
    "drift_halfline_reflecting": ("Drift, half-line, reflecting at 0", check_drift_halfline_reflecting,
                                  [fig_drift_halfline_reflecting]),
    "drift_halfline_absorbing": ("Drift, half-line, absorbing at 0", check_drift_halfline_absorbing,
                                 [fig_drift_halfline_absorbing]),
    "drift_line": ("Drift, whole line", check_drift_line, []),
    "drift_circle": ("Drift, circle", check_drift_circle, [fig_drift_circle]),
    "overdamped": ("Underdamped particle vs the overdamped (Smoluchowski) equation", check_overdamped, [fig_overdamped]),
    "reductions": ("Map of the reductions", lambda: True, [fig_reductions]),
    "walls": ("Physical walls: one path with a hard wall, a cliff, and on a ring", lambda: True, [fig_walls]),
    "operator": ("Symmetrised (Schrodinger) form of the operator", check_operator, [fig_operators]),
    "spectra": ("Spectral-gap figure", lambda: True, [fig_spectra]),
    "attraction": ("Which QSD a start reaches (drift and OU half-lines)", check_attraction, [fig_attraction]),
    "ou_line": ("OU, whole line", check_ou_line, [fig_ou_line]),
    "ou_halfline_reflecting": ("OU, half-line reflecting at the bottom", check_ou_halfline_reflecting,
                               [fig_ou_halfline_reflecting]),
    "ou_halfline_absorbing": ("OU, half-line absorbing at the bottom", check_ou_halfline_absorbing,
                              [fig_ou_halfline_absorbing]),
    "ou_reflecting": ("OU, interval reflecting", lambda: check_ou_interval("reflecting"), [fig_ou_reflecting]),
    "ou_absorbing": ("OU, interval absorbing", lambda: check_ou_interval("absorbing"), [fig_ou_absorbing]),
    "ou_mixed": ("OU, interval reflecting at 0 and absorbing at L", lambda: check_ou_interval("mixed"),
                 [fig_ou_mixed, fig_ou_summary]),
    "ou_circle": ("OU on the circle (sawtooth)", check_ou_circle, [fig_ou_circle]),
    "cos_circle": ("Cosine well on the circle, with tilt", check_cos_circle, [fig_cos_circle]),
    "solvable": ("Part IV: shape-invariant potentials", check_solvable, [fig_solvable]),
    "radial": ("Part IV: logarithmic potentials, radial motion", check_radial, [fig_radial]),
    "lncosh": ("Part IV: soft V, a ln cosh(x/w)", check_lncosh, [fig_lncosh]),
    "lncos": ("Part IV: well with its own walls, -a ln cos(x/w)", check_lncos, [fig_lncos]),
    "double_well": ("Part V: quartic double well", check_double_well, [fig_double_well]),
    "interface": ("Part VI: a step in the diffusion coefficient", check_interface, [fig_interface]),
    "sqrt": ("Part VI: where B vanishes, the square-root process", check_sqrt, [fig_sqrt]),
    "qif": ("Part VI: a current through infinity (QIF neuron)", check_qif, [fig_qif]),
    "feller": ("Part VI: Feller's classification", check_feller, []),
    "pq": ("Gardiner's P_lambda and Q_lambda in every example", check_pq, []),
}


def main(names):
    FIG.mkdir(exist_ok=True)
    unknown = set(names) - set(SECTIONS)
    if unknown:
        sys.exit(f"unknown section(s) {sorted(unknown)}; choose from {list(SECTIONS)}")
    ok = True
    for name in names or SECTIONS:
        title, checks, figures = SECTIONS[name]
        print(title)
        ok &= checks()
        for make in figures:
            make()
    print("All formula checks passed." if ok else "SOME CHECKS FAILED.")
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main(sys.argv[1:])
