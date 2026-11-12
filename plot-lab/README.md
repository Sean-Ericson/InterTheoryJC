# Plot Lab

Interactive versions of the figures in the Fokker–Planck examples notes (`../fpe-examples`). There is one page per section of the notes that has a figure, in the notes' order. Each page shows the notes' curves at the notes' parameter values, and the parameters the notes fixed become sliders.

Plot Lab is the companion to **SDE Explorer** (`../sde-explorer`), and the two link to each other:

| | SDE Explorer | Plot Lab |
|---|---|---|
| Shows | one process running forward in time | the exact results of the notes |
| Method | sample paths and a numerical Fokker–Planck solver | closed forms, mode sums and spectra |
| Process | any drift and noise | the notes' examples |

Every figure page has an **Open in SDE Explorer** button. It opens the same process (formulas, parameters, domain and walls) as a simulation.

## Running it

```sh
npm install
npm run dev      # http://localhost:5174 (SDE Explorer's dev server is 5173; the links between them use these ports)
npm test         # every formula against the notes' own script, plus the exit-time simulation
npm run build    # static site in dist/ (works from any sub-path, e.g. GitHub Pages)
```

For the links between the apps to work once built, deploy the two side by side, as `…/plot-lab/` and `…/sde-explorer/`. You can also set `VITE_SDE_EXPLORER_URL` at build time; SDE Explorer reads `VITE_PLOT_LAB_URL`.

## Pages

| Part | Pages (figure files) |
|---|---|
| Toolkit | where the equation comes from (`walls`, `overdamped`); symmetrised operator (`operators`); spectral gap (`spectra`); which QSD a start reaches (`qsd_attraction`) |
| I · Free diffusion | half-line, reflecting / absorbing; interval, reflecting / absorbing (`absorbing_density`, `absorbing_exit`, `absorbing_exit_times`) / mixed; circle; summary |
| II · Constant drift | the same six geometries (`drift_*`) |
| III · Ornstein–Uhlenbeck | line; half-lines; intervals; sawtooth circle; cosine well (`cos_circle`); summary (`ou_*`) |
| IV · Solvable potentials | shape-invariant gallery (`solvable`); radial motion (`radial`); a ln cosh (`lncosh`); −a ln cos (`lncos`) |
| V · Arbitrary potentials | the double well (`double_well`) |
| VI · Interfaces and singular boundaries | a step in D (`interface`); the square-root process, where B vanishes (`sqrt`); the QIF neuron's current through infinity (`qif`) |
| VII · Reductions | the map of reductions (`reductions`), with each box linking to its page |

## How the pages work

- **Notes' curves.** The families of curves (snapshot times, drifts, lengths) are drawn at the notes' values, in the notes' colours: a light-to-dark blue ramp for time, and blue–grey–red for a drift swept through zero.
- **Slider curve.** The heavy ink curve follows the sliders. A ▶ button sweeps a time (or other) parameter from one end of its range to the other.
- **Charts.**
  - Hover any chart for a readout of every curve.
  - Where a chart says "Drag to move …", pressing or dragging on it sets that parameter.
- **Address bar.** It always holds the current parameters, so a link reproduces the view, e.g. `?page=ou-mixed&L=3`.
- **Reset to the notes' values** restores the figure as printed.
- **Captions.** Each card carries the notes' caption for that subfigure. *Key results* repeats the section's main formulas.

## Numerics

The closed forms are ported from `fpe-examples/scripts/make_figures.py`. The figures that need more than a closed form use:

- **Image and mode sums.** These are truncated where their terms fall below double precision, rather than at fixed counts.
- **Spectral solver** (`src/math/spectrum.ts`):
  - Used for two-wall OU intervals, both sectors of the sawtooth circle, and the cosine well.
  - p = √pₛ ψ turns the forward operator into H = −(D/2)∂ₓ² + V. H is discretised by finite differences, so the matrix is symmetric tridiagonal.
  - Its levels come from Sturm bisection, its modes from inverse iteration.
- **Exponentially small rates.**
  - The Kramers escape rate of the harmonic well is found by shooting with Kummer's series. Finite differences carry an absolute O(h²) error in the energy, which swamps a rate of 10⁻⁸.
  - The double well uses the notes' jump-process discretisation, whose detailed balance is exact.
- **Tilted cosine well.** Its transport (V, D_eff) comes from perturbation theory in Fourier space. The matrices are tridiagonal, so each solve is O(N).

## Tests

`tests/notes.test.ts` checks every ported function against values from the notes' script, at sample points. The values are stored in `tests/fixtures/notes-reference.json`. Tolerances are mostly 10⁻¹⁰–10⁻¹²; looser ones carry a comment explaining why.

When the notes change, regenerate the fixture from `plot-lab/`, in the conda env `research`:

```sh
python -B scripts/notes_reference.py ../fpe-examples/scripts tests/fixtures/notes-reference.json
```

Then run `npm test`.

## Adding a figure

When the notes gain a figure:

1. Port its formulas to `src/math/`. Add sample values to `scripts/notes_reference.py` and a test.
2. Declare the page with `figurePage({...})` (`src/figures/framework.ts`). The declaration gives:
   - the parameters' defaults (the notes' values);
   - the sidebar controls;
   - a `model(values)` for the shared computation;
   - one panel per subfigure, each with a caption and a `spec(model, values)` that returns a `ChartSpec`;
   - optionally, a link to SDE Explorer.

   See `src/figures/part1.ts` for small examples.
3. Add the page to `FIGURES` in `src/figures/index.ts`, in the notes' order.

The framework builds the sidebar, cards, URL state, play buttons, linked hover and the SDE Explorer link. Colours are assigned by role, never by series index:

- ink is the process, and the slider's curve;
- blue and orange are the first and second entity (the walls at 0 and L);
- aqua is the stationary or quasi-stationary state;
- `seq1`–`seq5` and `div1`–`div5` are the notes' ramps.

## Free diffusion, absorbing walls (§6)

This page is built by hand (`src/pages/interval/`). Besides the three figures of the notes, it simulates exit times.

- **Charts:**
  - survival S(t) on a linear or log axis;
  - the exit-time density f, split into the fluxes j₀ and j_L through each wall;
  - the exit rate κ = f/S, together with κ from x₀ = L/2;
  - the density at t\*;
  - the mean exit times and σ_T against x₀;
  - the notes' density snapshots, unconditioned and conditioned;
  - the splitting probabilities.
- **Tiles** give λ₁, ⟨T⟩, σ_T, the median, P(exit at L) and the two conditional means, each with the simulated estimate under it.
- **Controls:**
  - L, D and x₀;
  - the starting density: a point, uniform, or φ₀;
  - the time axis, with **Fit**;
  - overlays: split by wall, slowest mode, walls separately (the half-line results, exact at early times), simulation, and mean & median.
- **Interaction.**
  - Hovering a time chart moves a crosshair on all three time charts, and the density follows the hovered time.
  - Dragging on a time chart pins t\*.
  - Dragging on an x₀ chart moves the start.
- **Numerics.**
  - The sine series is used for Dt/L² ≥ 0.05 and images below that; they agree to 10⁻¹⁰.
  - The simulation uses exact Gaussian steps, with a Brownian-bridge test for crossings between steps.
- **Deep-link keys:**
  - `L`, `D`, `x0`, `start` (`point`, `uniform`, `qsd`);
  - `tScale`, `sScale`, `fScale`, `tMin`, `tMax`, `t`;
  - `split`, `slowest`, `oneWall`, `mc`, `marks`;
  - `paths`, `seed`.

  For example, `?page=interval&x0=0.05&tScale=log&tMin=0.0001&tMax=2&sScale=log&oneWall=1` shows the early-time Lévy–Smirnov regime against the exponential tail.
