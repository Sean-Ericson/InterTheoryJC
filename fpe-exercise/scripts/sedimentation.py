"""Figure for sedimentation-key.tex, plus numerical checks of every result in the key.

The model is a colloid settling in a cell of height L: dx = -v dt + sqrt(D) dW on 0 < x < L,
with h = D/2v the gravitational length. The closed cell has reflecting walls, the sticky cell
absorbing ones. Units: D = L = 1, so v = 1/2h, and time is in L^2/D unless stated.
The spectra are also checked against finite-difference matrices of the backward operator.

    python scripts/sedimentation.py

writes figures/sed_modes.pdf and prints each check next to its expected value. Uses numpy,
scipy and matplotlib (the conda env `research` has all three).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

FIG = Path(__file__).resolve().parent.parent / "figures"
D, L = 1.0, 1.0

# --- style (as in fpe-examples/scripts/make_figures.py) ---------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

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
    }
)


# --- formulas from the key -------------------------------------------------------------
def k(n):
    return n * np.pi / L


def lam(n, h):
    """Shared by both cells for n >= 1; only the closed cell has lambda_0 = 0."""
    return 0.5 * D * (1 / (4 * h**2) + k(n) ** 2)


def p_s(x, h):
    return np.exp(-x / h) / (h * (1 - np.exp(-L / h)))


def phi(n, x, h):
    return np.cos(k(n) * x) - np.sin(k(n) * x) / (2 * h * k(n))


def Nn(n, h):
    return 0.5 * L * (1 + 1 / (4 * h**2 * k(n) ** 2))


def Q_closed(n, x, h):
    return np.exp(x / (2 * h)) * phi(n, x, h)


def P_closed(n, x, h):
    return np.exp(-x / (2 * h)) * phi(n, x, h) / Nn(n, h)


def Q_sticky(n, x, h):
    return np.exp(x / (2 * h)) * np.sin(k(n) * x)


def P_sticky(n, x, h):
    return 2 / L * np.exp(-x / (2 * h)) * np.sin(k(n) * x)


def qsd(x, h):
    a, k1 = 1 / (2 * h), k(1)
    return (a**2 + k1**2) / (k1 * (1 + np.exp(-a * L))) * np.exp(-a * x) * np.sin(k1 * x)


def pi_L(x, h):
    return np.expm1(x / h) / np.expm1(L / h)


def T_stick(x, h):
    v = D / (2 * h)
    return (x - L * pi_L(x, h)) / v


def mass_sticky(n, h):
    a = 1 / (2 * h)
    return 2 / L * k(n) * (1 - (-1) ** n * np.exp(-a * L)) / (a**2 + k(n) ** 2)


def p_sticky(x, t, x0, h, nmax=400):
    return sum(P_sticky(n, x, h) * Q_sticky(n, x0, h) * np.exp(-lam(n, h) * t) for n in range(1, nmax + 1))


def S_sticky(t, x0, h, nmax=400):
    return sum(mass_sticky(n, h) * Q_sticky(n, x0, h) * np.exp(-lam(n, h) * t) for n in range(1, nmax + 1))


def J(f, x, h, eps=1e-6):
    """Current A p - (1/2) D p' with A = -v."""
    return -D / (2 * h) * f(x) - 0.5 * D * (f(x + eps) - f(x - eps)) / (2 * eps)


# --- finite-difference backward operators ----------------------------------------------
def fd_backward(h, M, walls):
    """-v Q' + (1/2) D Q'' on a grid; 'closed' uses ghost points for Q' = 0, 'sticky' Q = 0."""
    v = D / (2 * h) if np.isfinite(h) else 0.0
    dx = L / M
    lo, di, up = 0.5 * D / dx**2 + v / (2 * dx), -D / dx**2, 0.5 * D / dx**2 - v / (2 * dx)
    if walls == "sticky":
        n = M - 1
        return np.diag(np.full(n, di)) + np.diag(np.full(n - 1, lo), -1) + np.diag(np.full(n - 1, up), 1)
    n = M + 1
    K = np.diag(np.full(n, di)) + np.diag(np.full(n - 1, lo), -1) + np.diag(np.full(n - 1, up), 1)
    K[0, 1] = D / dx**2
    K[-1, -2] = D / dx**2
    return K


def fd_rates(h, M, walls, count):
    w = np.sort(-np.real(np.linalg.eigvals(fd_backward(h, M, walls))))
    return w[:count]


def check(label, got, want, tol=1e-6):
    ok = np.all(np.abs(np.asarray(got) - np.asarray(want)) <= tol * (1 + np.abs(np.asarray(want))))
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {got} (expected {want})")


# --- checks ----------------------------------------------------------------------------
def checks():
    # (a) Perrin's length for a silica sphere, radius 0.5 um, in water at 293 K
    kT, r, drho, g = 1.380649e-23 * 293, 0.5e-6, 1.0e3, 9.81
    print("(a) closed cell: stationary state")
    mg = 4 / 3 * np.pi * r**3 * drho * g
    vs = mg / (6 * np.pi * 1.0e-3 * r)
    check("h for the silica sphere (um)", kT / mg * 1e6, 0.79, 1e-2)
    check("settling speed (um/s), water at 1.0 mPa s", vs * 1e6, 0.55, 1e-2)
    check("h/v (s)", kT / mg / vs, 1.4, 2e-2)
    check("L = 20h: 1/lambda_1 in h/v", 1 / (0.25 + np.pi**2 / 400), 3.6, 2e-2)
    check("L = 8h: ceiling share 1/(1+e^4)", 1 / (1 + np.exp(4)), 0.018, 2e-2)
    for h in (0.1, 0.5, 3.0):
        print(f" L/h = {L / h:g}")
        check("int p_s", quad(p_s, 0, L, args=(h,))[0], 1.0)
        check("<x>_s = h - L/(e^(L/h)-1)", quad(lambda x: x * p_s(x, h), 0, L)[0], h - L / np.expm1(L / h))
        xs = np.linspace(0.02, 0.98, 9)
        check("J[p_s] = 0", np.max(np.abs(J(lambda x: p_s(x, h), xs, h))), 0.0, 1e-6)

    print("(b) closed cell: modes")
    for h in (0.1, 0.5, 3.0):
        print(f" L/h = {L / h:g}")
        xs = np.linspace(0.0, L, 11)
        e = 1e-4
        for n in (1, 2, 3):
            Q = lambda x: Q_closed(n, x, h)
            res = -D / (2 * h) * (Q(xs + e) - Q(xs - e)) / (2 * e) + 0.5 * D * (Q(xs + e) - 2 * Q(xs) + Q(xs - e)) / e**2
            res += lam(n, h) * Q(xs)
            check(f"backward residual n={n}", np.max(np.abs(res)) / (1 + np.max(np.abs(Q(xs)))), 0.0, 1e-4)
            e1 = 1e-5
            dQ = lambda x: (Q(x + e1) - Q(x - e1)) / (2 * e1)
            scale = np.max(np.abs(dQ(np.linspace(0, L, 101))))
            check(f"Q_{n}'(0), Q_{n}'(L) relative to max|Q_{n}'|", [dQ(0.0) / scale, dQ(L) / scale], [0, 0], 1e-6)
            check(f"P_{n} = p_s Q_{n} / int p_s Q_{n}^2", np.max(np.abs(P_closed(n, xs, h) - p_s(xs, h) * Q(xs)
                  / quad(lambda x: p_s(x, h) * Q(x) ** 2, 0, L)[0])), 0.0, 1e-9)
        G = np.array([[quad(lambda x: P_closed(n, x, h) * Q_closed(m, x, h), 0, L)[0] for m in range(1, 5)] for n in range(1, 5)])
        check("biorthogonality n,m = 1..4", np.max(np.abs(G - np.eye(4))), 0.0, 1e-9)
        check("int P_n = 0", [quad(lambda x: P_closed(n, x, h), 0, L)[0] for n in (1, 2, 3)], [0, 0, 0], 1e-9)
        want = np.r_[0.0, lam(np.arange(1, 6), h)]
        check("finite-difference spectrum, closed", fd_rates(h, 1500, "closed", 6), want, 2e-4)
    # amplitude of the modes for a start at the ceiling grows like e^(L/2h)
    h = 0.05
    check("Q_1(L) e^(-L/2h) = -1 (cos pi)", Q_closed(1, L, h) * np.exp(-L / (2 * h)), -1.0, 1e-9)

    print("(c) sticky cell: modes and QSD")
    for h in (0.1, 0.5, 3.0):
        print(f" L/h = {L / h:g}")
        xs = np.linspace(0.0, L, 11)
        e = 1e-4
        for n in (1, 2, 3):
            Q = lambda x: Q_sticky(n, x, h)
            res = -D / (2 * h) * (Q(xs + e) - Q(xs - e)) / (2 * e) + 0.5 * D * (Q(xs + e) - 2 * Q(xs) + Q(xs - e)) / e**2
            res += lam(n, h) * Q(xs)
            check(f"backward residual n={n}", np.max(np.abs(res)) / (1 + np.max(np.abs(Q(xs)))), 0.0, 1e-4)
        G = np.array([[quad(lambda x: P_sticky(n, x, h) * Q_sticky(m, x, h), 0, L)[0] for m in range(1, 5)] for n in range(1, 5)])
        check("biorthogonality n,m = 1..4", np.max(np.abs(G - np.eye(4))), 0.0, 1e-9)
        check("P_n = p_s Q_n (L/2 int p_s ...)", np.max(np.abs(P_sticky(2, xs, h) - p_s(xs, h) * Q_sticky(2, xs, h)
              / quad(lambda x: p_s(x, h) * Q_sticky(2, x, h) ** 2, 0, L)[0])), 0.0, 1e-9)
        check("finite-difference spectrum, sticky", fd_rates(h, 1500, "sticky", 5), lam(np.arange(1, 6), h), 2e-4)
        check("int qsd", quad(qsd, 0, L, args=(h,))[0], 1.0)
        check("qsd = P_1 / int P_1", np.max(np.abs(qsd(xs, h) - P_sticky(1, xs, h) / mass_sticky(1, h))), 0.0, 1e-9)
        xg = np.linspace(0, L, 200001)
        xstar = xg[np.argmax(qsd(xg, h))]
        check("QSD peak: tan(pi x*/L) = 2 pi h/L", np.tan(np.pi * xstar / L), 2 * np.pi * h / L, 1e-3)
        Jq = lambda x: J(lambda y: qsd(y, h), x, h)
        check("total outflux of the QSD = lambda_1", Jq(L) - Jq(0.0), lam(1, h), 1e-6)
        check("ceiling/floor = e^(-L/2h)", Jq(L) / -Jq(0.0), np.exp(-L / (2 * h)), 1e-6)
        check("int qsd pi_L = 1/(1+e^(L/2h))", quad(lambda x: qsd(x, h) * pi_L(x, h), 0, L)[0], 1 / (1 + np.exp(L / (2 * h))))
        # survival: modes against finite differences
        M = 800
        K = fd_backward(h, M, "sticky")
        xgr = np.arange(1, M) * L / M
        for x0 in (0.3, 0.9):
            ts = np.array([0.02, 0.1, 0.3])
            fd = np.array([np.interp(x0, xgr, expm(K * t) @ np.ones(M - 1)) for t in ts])
            check(f"S(t | x0={x0}), modes vs finite differences", S_sticky(ts, x0, h), fd, 2e-4)
        tl = 3 / (lam(2, h) - lam(1, h)) + 0.5
        check("S ~ Q_1(x0) int P_1 e^(-lambda_1 t) at late times", S_sticky(tl, 0.5, h),
              Q_sticky(1, 0.5, h) * mass_sticky(1, h) * np.exp(-lam(1, h) * tl), 1e-3)

    print("(d) where and when it sticks")
    for h in (0.1, 0.5, 3.0):
        print(f" L/h = {L / h:g}")
        v = D / (2 * h)
        xs = np.linspace(0.05, 0.95, 7)
        e = 1e-4
        Tpp = (T_stick(xs + e, h) - 2 * T_stick(xs, h) + T_stick(xs - e, h)) / e**2
        Tp = (T_stick(xs + e, h) - T_stick(xs - e, h)) / (2 * e)
        check("-v T' + (D/2) T'' = -1", np.max(np.abs(-v * Tp + 0.5 * D * Tpp + 1)), 0.0, 1e-4)
        check("T(0) = T(L) = 0", [T_stick(0.0, h), T_stick(L, h)], [0, 0], 1e-12)
        pp = (pi_L(xs + e, h) - 2 * pi_L(xs, h) + pi_L(xs - e, h)) / e**2
        p1 = (pi_L(xs + e, h) - pi_L(xs - e, h)) / (2 * e)
        check("-v pi' + (D/2) pi'' = 0", np.max(np.abs(-v * p1 + 0.5 * D * pp)) / (1 + np.max(np.abs(pp))), 0.0, 1e-6)
        check("int qsd T = 1/lambda_1", quad(lambda x: qsd(x, h) * T_stick(x, h), 0, L)[0], 1 / lam(1, h))

    print("(e) currents map one cell onto the other")
    for h in (0.1, 0.5, 3.0):
        print(f" L/h = {L / h:g}")
        xs = np.linspace(0.05, 0.95, 7)
        for n in (1, 2, 3):
            Jn = lambda x: J(lambda y: P_closed(n, y, h), x, h)
            check(f"J[P_{n} closed] = (D k_n/2) P_{n} sticky", np.max(np.abs(Jn(xs) - 0.5 * D * k(n) * P_sticky(n, xs, h))), 0.0, 1e-6)
            check(f"J[P_{n} closed] = 0 at both walls", [Jn(0.0), Jn(L)], [0, 0], 1e-6)
            e = 1e-5
            check(f"P_{n} closed = J'/lambda", np.max(np.abs((Jn(xs + e) - Jn(xs - e)) / (2 * e) / lam(n, h) - P_closed(n, xs, h))), 0.0, 1e-4)
        # conditioned never to stick: drift A + D (ln Q_1)' = (pi D / L) cot(pi x / L), whatever v
        e = 1e-6
        drift = -D / (2 * h) + D * (np.log(Q_sticky(1, xs + e, h)) - np.log(Q_sticky(1, xs - e, h))) / (2 * e)
        check("Q-process drift = (pi D/L) cot(pi x/L)", np.max(np.abs(drift - np.pi * D / L / np.tan(np.pi * xs / L))), 0.0, 1e-6)


# --- figure ----------------------------------------------------------------------------
def figure():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
    h = L / 8
    v = D / (2 * h)

    # (a) barometric p_s (closed) against the QSD (sticky), L = 8h
    x = np.linspace(0, L, 600)
    ax = axs[0]
    ax.plot(x / h, p_s(x, h) * h, color=INK, label=r"$p_s$, closed")
    ax.plot(x / h, qsd(x, h) * h, color=BLUE, label="QSD, sticky")
    ax.set_xlabel(r"$x/h$")
    ax.set_ylabel(r"density $\times\,h$")
    ax.set_title(r"(a) $L=8h$", loc="left")
    ax.legend(loc="upper right")

    # (b) sticky cell, released near the ceiling: p/S settles on the QSD
    ax = axs[1]
    x0 = 0.9 * L
    xi = np.linspace(1e-4, L - 1e-4, 400)
    for tv, c in ((1, AQUA), (2.5, BLUE), (4, ORANGE), (8, INK)):
        t = tv * h / v
        p = p_sticky(xi, t, x0, h)
        ax.plot(xi / h, p / S_sticky(t, x0, h) * h, color=c, lw=1.2, label=rf"$vt/h={tv:g}$")
    ax.plot(xi / h, qsd(xi, h) * h, color=MUTED, ls="--", lw=1.0, label="QSD")
    ax.set_ylim(0, 0.75)
    ax.set_xlabel(r"$x/h$")
    ax.set_ylabel(r"$p/S\,\times\,h$")
    ax.set_title(r"(b) sticky, from $x_0=0.9L$", loc="left")
    ax.legend(loc="upper right", handlelength=1.4)

    # (c) the two spectra against L/h: curves are lambda_n, markers finite differences
    ax = axs[2]
    r = np.linspace(0, 12, 200)
    for n in range(1, 4):
        ax.plot(r, 0.5 * ((r / 2) ** 2 + (n * np.pi) ** 2), color=MUTED, lw=1.0)
    ax.plot(r, 0 * r, color=MUTED, lw=1.0)
    rm = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
    for i, rr in enumerate(rm):
        hh = L / rr if rr > 0 else np.inf
        cl = fd_rates(hh, 600, "closed", 4)
        st = fd_rates(hh, 600, "sticky", 3)
        ax.plot([rr] * 4, cl, "o", ms=4.5, mfc="none", mec=INK, mew=0.9, label="closed" if i == 0 else None)
        ax.plot([rr] * 3, st, "x", ms=4, color=ORANGE, mew=1.1, label="sticky" if i == 0 else None)
    ax.set_ylim(-3, 78)
    ax.set_xlabel(r"$L/h$")
    ax.set_ylabel(r"$\lambda_n L^2/D$")
    ax.set_title("(c) the two spectra", loc="left")
    ax.legend(loc="upper left")

    fig.tight_layout(w_pad=1.0)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "sed_modes.pdf")
    print(f"wrote {FIG / 'sed_modes.pdf'}")


if __name__ == "__main__":
    checks()
    figure()
