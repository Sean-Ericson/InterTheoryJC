# SDE Explorer

Simulate sample paths of a 1D stochastic differential equation and watch its probability density evolve under the matching Fokker–Planck equation, side by side. Notation follows Gardiner, *Handbook of Stochastic Methods*:

- SDE: dx = a(x,t) dt + b(x,t) dW(t)
- Fokker–Planck: ∂ₜp = −∂ₓ[A p] + ½ ∂ₓ²[B p], with A = a and B = b² (eq. 4.3.20)

For a Stratonovich SDE, A becomes a + ½ b ∂ₓb (eq. 4.3.40).

## Running it

```sh
npm install
npm run dev      # local dev server
npm test         # numerical checks against exact solutions
npm run build    # static site in dist/ (works from any sub-path, e.g. GitHub Pages)
```

## What's in it

- **Presets** with section references: Wiener, Ornstein–Uhlenbeck, multiplicative noise (GBM), double well, decay from an unstable state, absorbing walls, drift toward an absorbing wall, escape from a metastable well, the X + A ⇌ 2X reaction (5.2.58), the Feller/CIR process, a tilted periodic potential, and a periodically forced double well.
- **Custom formulas** for a(x,t) and b(x,t). Any symbol other than `x` and `t` becomes a parameter with its own slider.
- **Boundaries** set separately for each side: open, reflecting, absorbing, or periodic. They apply to both the paths and the solver.
- **Overlays:**
  - exact solutions where one is known (Wiener, OU, GBM, absorbing walls, drift toward a wall on the half-line);
  - the stationary density (§5.2.2), including the current-carrying solution for periodic boundaries;
  - with absorbing walls, the quasi-stationary density instead (see below);
  - mean ± s.d., variance and survival probability over time;
  - two-time and stationary autocorrelation functions (see below).
- **Start in the stationary state** instead of N(x₀, σ₀²). Both the density and the sample paths then start from pₛ, or from φ₀ when there are absorbing walls.

## Quasi-stationary distributions

With an absorbing wall there is no stationary density: probability leaks out, and the surviving mass S(t) decays. The density of the paths that survive, q = p/S, obeys ∂ₜq = 𝓛q + κ(t)q with exit rate κ = −Ṡ/S. Its stationary state is the quasi-stationary density (QSD) φ₀: the positive solution of 𝓛φ₀ = −λ₀φ₀ that vanishes on the absorbing walls (§5.2.5).

- **Condition on survival** switches the density panel, histogram and heatmap from p to q = p/S. The right-hand chart then shows the exit rate κ(t), which tends to λ₀.
- **Quasi-stationary density** overlays φ₀. When the view is not conditioned, it shows S(t)·φ₀ instead, because p → S(t)φ₀ at long times. The header gives λ₀, the decay rate of S, and the gap λ₁ − λ₀, the rate at which q approaches φ₀.
- **Quasi-stationary φ₀** in the initial-condition section starts the run in φ₀: q then stays put, κ = λ₀ from the start, and S(t) = e^{−λ₀t} exactly.

φ₀ is the ground state of the solver's own discretised operator, so the Fokker–Planck curve converges to it exactly. It is found by Sturm-sequence bisection and inverse iteration (`src/qsd.ts`). On an unbounded side the solver needs a wall. If φ₀ reaches an open edge, the app reports that there is no QSD: on the half-line there is none when drift is absent or points away from the absorbing wall.

The new presets:

- **Drift toward an absorbing wall:** dx = −μ dt + σ dW on the half-line, with an exact solution by images. With D = σ², the notes' convention (B = D):
  - Conditioned on survival, q creeps toward the edge QSD (μ/D)² x e^{−μx/D}, and κ ≈ μ²/2D + 3/2t.
  - Both approaches are algebraic, because λ = μ²/2D sits at the edge of a continuum of QSDs.
  - The finite domain's far edge picks one QSD out of that continuum, with λ₀ slightly above μ²/2D.
- **Escape from a metastable well:** q settles into the well within about 1/(λ₁ − λ₀), while S decays at the much slower Kramers rate λ₀.

## Autocorrelation

The bottom row (**Show → Autocorrelation**) has two charts.

- **Left, two-time covariance ⟨X(s + τ), X(s)⟩.** Here s is the playback time, so scrubbing moves it. The chart compares three estimates:
  - **Fokker–Planck.** Following §3.7.4 (eq. 3.7.46), the Markov property reduces the two-time average to a one-time problem: F = (x − mₛ) p(x, s) evolves under the same equation, and ⟨X(s + τ), X(s)⟩ = ∫ x F dx (`src/correlation.ts`).
  - **Sample paths:** the covariance of the recorded positions at s and s + τ.
  - **Exact,** from the regression theorem ⟨X(t), X(s)⟩ = e^{B(t−s)} var X(s) (eq. 3.7.61). This holds wherever the Itô drift is linear, A + Bx: Wiener, OU and GBM.

  The stationary G(τ) is overlaid dotted. Once s is past the transient, the two-time curve lies on it.
- **Right, stationary correlation coefficient ρ(τ) = G(τ)/G(0) on a log scale.** G(τ) comes from the same calculation started in pₛ. The dashed line is exp(−λ₁τ), where λ₁ is the slowest relaxation rate of the discretised operator (§5.2.5), so a single-exponential process lies on it. With absorbing walls, G is the quasi-stationary analogue: started in φ₀, among survivors, decaying at λ₁ − λ₀.

With absorbing walls, all covariances are among the paths that survive to s + τ. On a periodic domain x wraps round, so the panel shows none. Starting in the stationary state makes the sample paths a stationary ensemble, and the left chart then estimates G(τ) directly.

## Deep links

You can open a preset paused at a given time, which is handy for discussion:

`?preset=double-well&t=20`, `?preset=gbm&interp=strat&t=1.5`, `?preset=walls&cond=1&start=stationary`, `&theme=light|dark`

Preset ids: `wiener`, `ou`, `gbm`, `double-well`, `unstable`, `walls`, `halfline`, `escape`, `chemical`, `cir`, `periodic`, `resonance`.

A link can also carry a whole process:

`?a=-k*x&b=sqrt(D)&p=k:1,D:1&lo=0&hi=2&bc=reflecting,absorbing&x0=0&T=50&cond=1`

- `a`, `b`: the drift and noise formulas.
- `p`: parameter values, as `name:value` pairs. A parameter left out is set to 1.
- `lo`, `hi`: the domain.
- `bc`: the left and right walls.
- `x0`, `s0`: the initial Gaussian's centre and spread.
- `T`: the run length.

Plot Lab's **Open in SDE Explorer** buttons use these links (`applyProcessLink` in `src/settings.ts`).

## Plot Lab

[Plot Lab](../plot-lab) has the interactive figures of the Fokker–Planck examples notes: the exact results, one page per section. Each of its pages links here with the same process, and the header here links back.

- During development, Plot Lab's dev server runs on port 5174 and this one on 5173.
- Built, the two are deployed side by side.
- `VITE_PLOT_LAB_URL` overrides the address of the link.

## Numerics

- **Paths:** Milstein scheme. At absorbing walls, a Brownian-bridge correction catches crossings that happen between time steps.
- **Density:**
  - finite volumes with Scharfetter–Gummel fluxes, so it stays positive, conserves probability and handles b → 0;
  - Crank–Nicolson time stepping with a Rannacher start-up;
  - periodic boundaries solved with a cyclic tridiagonal (Sherman–Morrison) method.
- **Shared start:** the paths and the solver both begin from the same Gaussian N(x₀, σ₀²). Its width is at least two grid cells.
