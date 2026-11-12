"""Figure for wright-fisher-key.tex, plus numerical checks of every result in the key.

The model is the Wright-Fisher diffusion with symmetric mutation, A = u(1-2x) and
B = x(1-x)/N on 0 < x < 1, with time in generations and theta = 2Nu. The diffusion results are
checked three ways: by algebra on polynomials, against finite-difference solutions, and against
the discrete Wright-Fisher chain (mutation, then binomial resampling of N copies), whose
transition matrix is built exactly.

    python scripts/wright_fisher.py

writes figures/wf_modes.pdf and prints each check next to its expected value. Uses numpy,
scipy and matplotlib (the conda env `research` has all three).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import betaln, eval_gegenbauer
from scipy.stats import binom

FIG = Path(__file__).resolve().parent.parent / "figures"

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
def lam(n, theta, N):
    return n * (n - 1 + 2 * theta) / (2 * N)


def p_s(x, theta):
    """Beta(theta, theta) density: the stationary state with mutation."""
    return np.exp((theta - 1) * np.log(x * (1 - x)) - betaln(theta, theta))


def Es(F, theta):
    """Stationary average of F, with the end singularities of p_s handled by quad's weight."""
    return quad(F, 0, 1, weight="alg", wvar=(theta - 1, theta - 1))[0] * np.exp(-betaln(theta, theta))


def backward(c, u, N):
    """L^dagger on the polynomial with coefficients c (constant term first)."""
    c = np.r_[np.asarray(c, float), 0.0, 0.0]
    A = np.array([u, -2 * u])
    B = np.array([0.0, 1.0, -1.0]) / N
    return poly.polyadd(poly.polymul(A, poly.polyder(c)), 0.5 * poly.polymul(B, poly.polyder(c, 2)))


def Pn0(n, x):
    """Interior forward mode without mutation, n >= 2: a Gegenbauer polynomial in 1-2x."""
    return eval_gegenbauer(n - 2, 1.5, 1 - 2 * np.asarray(x, float))


def hn0(n):
    return quad(lambda x: x * (1 - x) * Pn0(n, x) ** 2, 0, 1)[0]


def Qn0(n, x):
    """Backward mode without mutation, normalised so that int P_n Q_n = 1."""
    x = np.asarray(x, float)
    return x * (1 - x) * Pn0(n, x) / hn0(n)


def survival(x0, t, N, nmax=60):
    """S(t | x0) without mutation, from the mode sum (only even n carry probability)."""
    t = np.atleast_1d(t)
    out = np.zeros_like(t, dtype=float)
    for n in range(2, nmax + 1, 2):
        mass = quad(lambda x: Pn0(n, x), 0, 1)[0]
        out += Qn0(n, x0) * mass * np.exp(-lam(n, 0, N) * t)
    return out


def T_abs(x, N):
    """Mean time to fixation or loss, from (5.2.154)."""
    x = np.asarray(x, float)
    return -2 * N * (x * np.log(x) + (1 - x) * np.log(1 - x))


# --- the discrete Wright-Fisher chain --------------------------------------------------
def chain(N, u):
    """Transition matrix: mutation with probability u each way, then binomial resampling."""
    i = np.arange(N + 1)
    xm = i / N + u * (1 - 2 * i / N)
    return binom.pmf(i[None, :], N, xm[:, None])


def chain_stationary(N, u):
    w, V = np.linalg.eig(chain(N, u).T)
    v = np.real(V[:, np.argmin(np.abs(w - 1))])
    return v / v.sum()


def check(label, got, want, tol=1e-6):
    ok = np.all(np.abs(np.asarray(got) - np.asarray(want)) <= tol * (1 + np.abs(np.asarray(want))))
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {got} (expected {want})")


# --- checks ----------------------------------------------------------------------------
def checks():
    N = 200
    print("(a) one generation of the chain")
    for theta in (0.5, 2.0):
        u = theta / (2 * N)
        T = chain(N, u)
        x = np.arange(N + 1) / N
        dx = T @ x - x
        var = T @ x**2 - (T @ x) ** 2
        check(f"theta={theta}: mean change = u(1-2x)", np.max(np.abs(dx - u * (1 - 2 * x))), 0.0, 1e-12)
        check(f"theta={theta}: variance = x(1-x)/N + O(u/N)", np.max(np.abs(var - x * (1 - x) / N)) * N**2, 0.0, 3 * theta)

    print("(b) stationary state with mutation")
    for theta in (0.3, 1.0, 3.0):
        check(f"theta={theta}: int p_s", Es(lambda x: 1.0 + 0 * x, theta), 1.0)
        Exx = Es(lambda x: x * (1 - x), theta)
        check(f"theta={theta}: E_s[x(1-x)] = theta/2(2theta+1)", Exx, theta / (2 * (2 * theta + 1)))
        pi = chain_stationary(N, theta / (2 * N))
        x = np.arange(N + 1) / N
        check(f"theta={theta}: chain heterozygosity ~ theta/(2theta+1)", 2 * pi @ (x * (1 - x)), theta / (2 * theta + 1), 2e-2)
    # Feller's test at x = 0, with f ~ x^-theta, g ~ x^(theta-1), from the full A and B
    s = np.linspace(np.log(1e-40), np.log(0.5), 400001)
    xg = np.exp(s)
    for theta, want in ((0.5, "regular"), (2.0, "entrance"), (0.0, "exit")):
        f = (xg * (1 - xg)) ** (-theta) / 0.25 ** (-theta)
        g = 2 * N / (xg * (1 - xg) * f)
        cum = lambda F: np.concatenate([[0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(xg))])
        tail = lambda F, cut: np.sum((0.5 * (F[1:] + F[:-1]) * np.diff(xg))[xg[1:] < cut])
        h1 = f * np.abs(cum(g) - cum(g)[-1])
        h2 = g * np.abs(cum(f) - cum(f)[-1])
        fin = lambda F: tail(np.abs(F), 1e-20) < 1e-3 * (1 + tail(np.abs(F), 0.5))
        if fin(f) and fin(g):
            got = "regular"
        elif not fin(g) and fin(h1):
            got = "exit"
        elif fin(g) and fin(h2):
            got = "entrance"
        else:
            got = "natural"
        print(f"  [{'ok' if got == want else 'FAIL'}] Feller type at 0, theta={theta}: {got} (expected {want})")

    print("(c) eigenfunctions with mutation")
    for theta in (0.3, 1.0, 3.0):
        u = theta / (2 * N)
        c = theta / (2 * (2 * theta + 1))
        Q = {1: [1, -2], 2: [-c, 1, -1]}
        for n, q in Q.items():
            res = poly.polyadd(backward(q, u, N), lam(n, theta, N) * np.r_[q, 0, 0])
            check(f"theta={theta}: L^dagger Q_{n} + lambda_{n} Q_{n}", np.max(np.abs(res)), 0.0, 1e-14)
        # leading coefficient of L^dagger x^n gives lambda_n
        for n in range(1, 7):
            check(f"theta={theta}: x^{n} coefficient", -backward(np.eye(n + 1)[n], u, N)[n], lam(n, theta, N), 1e-12)
        G = [[Es(lambda x: poly.polyval(x, Q[m]) * poly.polyval(x, Q[n]), theta) for n in (1, 2)] for m in (1, 2)]
        check(f"theta={theta}: int p_s Q_1 Q_2", G[0][1], 0.0, 1e-9)
        check(f"theta={theta}: int p_s Q_1^2 = 1/(2theta+1)", G[0][0], 1 / (2 * theta + 1), 1e-9)
        check(f"theta={theta}: int p_s Q_2 = 0", Es(lambda x: poly.polyval(x, Q[2]), theta), 0.0, 1e-9)
        # the chain's eigenvalues: exactly (1-2u)^k prod(1-i/N); their rates approach lambda_k
        mu = np.sort(np.real(np.linalg.eigvals(chain(N, u))))[::-1][:7]
        k = np.arange(7)
        exact = (1 - 2 * u) ** k * np.array([np.prod(1 - np.arange(j) / N) for j in k])
        check(f"theta={theta}: chain eigenvalues = (1-2u)^k prod(1-i/N)", mu, exact, 1e-8)
        check(f"theta={theta}: chain rates -ln(mu_k) ~ lambda_k", -np.log(mu[1:]), lam(k[1:], theta, N), 1.5e-2)

    print("(d) no mutation")
    check("lambda_0 = lambda_1 = 0, lambda_2 = 1/N", [lam(0, 0, N), lam(1, 0, N), lam(2, 0, N)], [0, 0, 1 / N], 1e-12)
    xs = np.linspace(0.03, 0.97, 9)
    for n in range(2, 7):
        q = np.polynomial.Polynomial.fit(xs, Qn0(n, xs), n).convert().coef
        res = poly.polyadd(backward(q, 0.0, N), lam(n, 0, N) * np.r_[q, 0, 0])
        check(f"backward residual n={n}", np.max(np.abs(res)), 0.0, 1e-8)
        # forward: (1/2N) (x(1-x) P)'' = -lambda P, with x(1-x) P_n proportional to Q_n
        Bp = poly.polymul([0, 1, -1], np.polynomial.Polynomial.fit(xs, Pn0(n, xs), n - 2).convert().coef) / N
        fres = 0.5 * poly.polyval(xs, poly.polyder(Bp, 2)) + lam(n, 0, N) * Pn0(n, xs)
        check(f"forward residual n={n}", np.max(np.abs(fres)) * N, 0.0, 1e-8)
    G = np.array([[quad(lambda x: Pn0(n, x) * Qn0(m, x), 0, 1)[0] for m in range(2, 7)] for n in range(2, 7)])
    check("interior biorthogonality, n,m = 2..6", np.max(np.abs(G - np.eye(5))), 0.0, 1e-9)
    check("P_2 = 1, Q_2 = 6x(1-x)", np.max(np.abs(Qn0(2, xs) - 6 * xs * (1 - xs))) + np.max(np.abs(Pn0(2, xs) - 1)), 0.0, 1e-12)
    check("int (1-2x) 30x(1-x)(1-2x) dx", quad(lambda x: 30 * x * (1 - x) * (1 - 2 * x) ** 2, 0, 1)[0], 1.0)
    check("lambda_3 - lambda_2, lambda_4 - lambda_2", [lam(3, 0, 1) - lam(2, 0, 1), lam(4, 0, 1) - lam(2, 0, 1)], [2, 5], 1e-12)
    # full P_2 = 1 - delta_0/2 - delta_1/2 is orthogonal to both zero modes
    check("int P_2 Q for Q = 1 and Q = x, atoms included", [1 - 0.5 - 0.5, 0.5 - 0.5 * 1], [0, 0], 1e-12)
    # current out of the uniform QSD
    h = 1e-6
    J = lambda x: -0.5 * ((x + h) * (1 - x - h) - (x - h) * (1 - x + h)) / (2 * h) / N
    check("J(0), J(1) of the uniform QSD = -1/2N, 1/2N", [J(0.0) * N, J(1.0) * N], [-0.5, 0.5], 1e-8)
    check("total outflux = lambda_2", (J(1.0) - J(0.0)) * N, 1.0, 1e-8)

    # survival: mode sum against finite differences and against the chain. The degenerate
    # diffusion makes the grid error first order, so extrapolate from two grids.
    def fd_survival(x0, ts, M):
        xg = np.arange(1, M) / M
        D2 = (np.diag(np.ones(M - 2), -1) - 2 * np.eye(M - 1) + np.diag(np.ones(M - 2), 1)) * M**2
        K = np.diag(xg * (1 - xg) / (2 * N)) @ D2
        return np.array([np.interp(x0, xg, expm(K * t) @ np.ones(M - 1)) for t in ts])

    for x0 in (0.1, 0.5):
        ts = np.array([0.25, 0.5, 1.0, 2.0, 4.0]) * N
        fd = 2 * fd_survival(x0, ts, 800) - fd_survival(x0, ts, 400)
        check(f"x0={x0}: survival, modes vs finite differences", survival(x0, ts, N), fd, 2e-4)
        lead = 6 * x0 * (1 - x0) * np.exp(-ts[2:] / N)
        check(f"x0={x0}: survival ~ 6x0(1-x0)e^(-t/N) for t >= N", survival(x0, ts[2:], N), lead, 2e-2 if x0 == 0.1 else 2e-3)
    Nc = 400
    T0 = chain(Nc, 0.0)
    for x0 in (0.1, 0.5):
        v = np.zeros(Nc + 1)
        v[int(x0 * Nc)] = 1.0
        got, want = [], []
        for t in range(1, 2 * Nc + 1):
            v = v @ T0
            if t in (Nc // 2, Nc, 2 * Nc):
                got.append(v[1:-1].sum())
                want.append(survival(x0, t, Nc)[0])
        check(f"x0={x0}: survival, modes vs the chain (N={Nc})", np.array(got), np.array(want), 1e-2)
    # fixation probability and mean absorption time in the chain
    Tii = T0[1:-1, 1:-1]
    fix = np.linalg.solve(np.eye(Nc - 1) - Tii, T0[1:-1, -1])
    check("chain fixation probability = x0", np.max(np.abs(fix - np.arange(1, Nc) / Nc)), 0.0, 1e-9)
    Tm = np.linalg.solve(np.eye(Nc - 1) - Tii, np.ones(Nc - 1))
    check("chain mean absorption time from 1/2 ~ 2N ln 2", Tm[Nc // 2 - 1], 2 * Nc * np.log(2), 1e-2)
    w, V = np.linalg.eig(Tii.T)
    qsd = np.abs(np.real(V[:, np.argmax(np.real(w))]))
    qsd = qsd / qsd.sum() * Nc
    bulk = (np.arange(1, Nc) > 0.05 * Nc) & (np.arange(1, Nc) < 0.95 * Nc)
    check("chain QSD ~ uniform for 0.05 < x < 0.95 (max deviation)", np.max(np.abs(qsd[bulk] - 1)), 0.0, 2e-2)
    print(f"         (within a few copies of fixation the chain's QSD dips: {np.round(qsd[:3], 3)})")
    check("chain QSD decay rate ~ 1/N", -np.log(np.max(np.real(w))) * Nc, 1.0, 1e-2)
    # mean absorption time
    xs2 = np.linspace(0.05, 0.95, 7)
    hh = 1e-4
    Tpp = (T_abs(xs2 + hh, N) - 2 * T_abs(xs2, N) + T_abs(xs2 - hh, N)) / hh**2
    check("(x(1-x)/2N) T'' = -1", np.max(np.abs(xs2 * (1 - xs2) / (2 * N) * Tpp + 1)), 0.0, 1e-5)
    check("T(1/2) = 2N ln 2", T_abs(0.5, N), 2 * N * np.log(2))
    check("int T over the uniform QSD = N = 1/lambda_2", quad(lambda x: T_abs(x, N), 0, 1)[0], N)

    print("(e) the limit theta -> 0")
    for theta in (1e-3, 1e-4):
        c = theta / (2 * (2 * theta + 1))
        PQ = lambda x: p_s(x, theta) * (x * (1 - x) - c) / (theta / 2)
        xi = np.linspace(0.1, 0.9, 9)
        check(f"theta={theta}: p_s Q_2 / (theta/2) = 1 inside", np.max(np.abs(PQ(xi) - 1)), 0.0, 10 * theta)
        # mass near x = 0: with y = x^theta, p_s dx = (1-x)^(theta-1) dy / (theta B(theta,theta))
        eps = 0.01
        reg = lambda x: (1 - x) ** (theta - 1) * np.exp(-betaln(theta, theta)) / theta
        near = lambda F: quad(lambda y: reg(y ** (1 / theta)) * F(y ** (1 / theta)), 0, eps**theta)[0]
        check(f"theta={theta}: mass of p_s Q_2/(theta/2) on (0,0.01) -> -1/2 + 0.01",
              near(lambda x: (x * (1 - x) - c) / (theta / 2)), -0.5 + eps, 20 * theta)
        check(f"theta={theta}: p_s mass on (0, 0.01) -> 1/2", near(lambda x: 1.0), 0.5, 20 * theta)


# --- figure ----------------------------------------------------------------------------
def figure():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))

    # (a) stationary densities with mutation; dots are the exact chain, N = 100
    N = 100
    x = np.linspace(1e-3, 1 - 1e-3, 800)
    ax = axs[0]
    for theta, c in ((0.25, BLUE), (1.0, INK), (4.0, ORANGE)):
        ax.plot(x, p_s(x, theta), color=c, label=rf"$\theta={theta:g}$")
        pi = chain_stationary(N, theta / (2 * N))
        i = np.arange(1, N)[::4]
        ax.plot(i / N, pi[i] * N, "o", ms=2.2, color=c)
    ax.set_ylim(0, 3.2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$p_s(x)$")
    ax.set_title(r"(a) with mutation, $\theta=2Nu$", loc="left")
    ax.legend(loc="upper center")

    # (b) no mutation: the chain conditioned on survival flattens to the uniform QSD
    N = 200
    T0 = chain(N, 0.0)
    v = np.zeros(N + 1)
    v[N // 10] = 1.0
    ax = axs[1]
    marks = {20: 0.1, 60: 0.3, 200: 1.0, 600: 3.0}
    shades = [AQUA, BLUE, ORANGE, INK]
    for t in range(1, max(marks) + 1):
        v = v @ T0
        if t in marks:
            q = v[1:-1] / v[1:-1].sum() * N
            ax.plot(np.arange(1, N) / N, q, color=shades[list(marks).index(t)], label=rf"$t={marks[t]:g}N$")
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, 5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$p/S$")
    ax.set_title(r"(b) $u=0$, $x_0=0.1$: toward the QSD", loc="left")
    ax.legend(loc="upper right", handlelength=1.4)

    # (c) survival: chain against the mode sum and the leading mode
    ax = axs[2]
    tt = np.linspace(0, 4, 400)
    for x0, c in ((0.1, BLUE), (0.5, ORANGE)):
        v = np.zeros(N + 1)
        v[int(x0 * N)] = 1.0
        ts, S = [0.0], [1.0]
        for t in range(1, 4 * N + 1):
            v = v @ T0
            if t % 20 == 0:
                ts.append(t / N)
                S.append(v[1:-1].sum())
        ax.plot(tt[1:], survival(x0, tt[1:] * N, N), color=c, label=rf"$x_0={x0:g}$")
        ax.plot(tt, 6 * x0 * (1 - x0) * np.exp(-tt), color=c, ls="--", lw=1.0)
        ax.plot(ts[::2], S[::2], "o", ms=2.4, color=c)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 2)
    ax.set_xlabel(r"$t/N$")
    ax.set_ylabel(r"$S(t)$")
    ax.set_title("(c) survival: modes, leading mode, chain", loc="left")
    ax.legend(loc="lower left")

    fig.tight_layout(w_pad=1.0)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "wf_modes.pdf")
    print(f"wrote {FIG / 'wf_modes.pdf'}")


if __name__ == "__main__":
    checks()
    figure()
