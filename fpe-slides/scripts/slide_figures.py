"""Slide versions of figures from ../fpe-examples, relabelled for the talk.

The notes' own figure functions draw everything; this script only swaps some
label strings before saving, so the slides stay in sync with the notes.

    python fpe-slides/scripts/slide_figures.py     (conda env `research`)

writes PDFs to fpe-slides/figures/.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "fpe-examples" / "scripts"))

import make_figures as mf  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.text import Text  # noqa: E402

OUT = HERE.parent / "figures"

# On the slides densities are plotted against x/L without the factor L
DENSITY = r"$L\,p(x,t\,|\,x_0,0)$"
QSD = r"$L\,q_\infty(x)$"


def first_three_in_a_row(fig):
    """Turn a 2x2 figure into a 1x3 row like the other example figures, dropping the fourth panel."""
    axs = fig.get_axes()
    fig.delaxes(axs[3])
    fig.set_size_inches(6.5, 2.2)
    for ax, spec in zip(axs[:3], fig.add_gridspec(1, 3, wspace=0.42)):
        ax.set_subplotspec(spec)


# name of the slide PDF -> (figure function from the notes, {old label: new label}[, layout change])
RELABEL = {
    "absorbing_density_slides.pdf": (
        mf.fig_absorbing_density,
        {
            DENSITY: r"$p(x,t\,|\,x_0,0)$",
            r"$L\,q(x,t)=L\,p/S$": r"$\tilde p(x,t)=p/S$",
        },
    ),
    "walls_slides.pdf": (
        mf.fig_walls,
        {r"ring of circumference $L$": r"ring of circumference $L=2x_0$"},
    ),
    "reflecting_slides.pdf": (mf.fig_reflecting, {DENSITY: r"$p(x,t\,|\,x_0,0)$"}),
    "drift_reflecting_slides.pdf": (
        mf.fig_drift_reflecting,
        {r"$L\,p_s(x)$": r"$p_s(x)$", DENSITY: r"$p(x,t\,|\,x_0,0)$"},
    ),
    "drift_absorbing_slides.pdf": (
        mf.fig_drift_absorbing,
        {QSD: r"$\tilde p_\infty(x)$"},
        first_three_in_a_row,
    ),
    "drift_mixed_slides.pdf": (mf.fig_drift_mixed, {QSD: r"$\tilde p_\infty(x)$"}),
}


def relabelled(make, name, swaps, layout=None):
    """Run one of the notes' figure functions, relabel its text, and save it under OUT/name."""
    save = Figure.savefig
    unused = set(swaps)

    def patched(fig, _fname, *args, **kwargs):
        for t in fig.findobj(Text):
            s = t.get_text()
            for old, new in swaps.items():
                if old in s:
                    s = s.replace(old, new)
                    unused.discard(old)
            t.set_text(s)
        if layout:
            layout(fig)
        save(fig, OUT / name, *args, **kwargs)

    Figure.savefig = patched
    try:
        make()
    finally:
        Figure.savefig = save
    if unused:
        sys.exit(f"{name}: labels not found (did the notes change?): {sorted(unused)}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, (make, swaps, *layout) in RELABEL.items():
        relabelled(make, name, swaps, *layout)
        print(f"wrote {OUT / name}")
