# Plot Policy (guidance for clear, decent figures)

Guidance for agents producing matplotlib figures in this workspace (pepsy
examples, notebooks, benchmark figures).

This is **guidance, not a rigid template.** The goal is simply: produce clear,
legible, good-looking plots. Use judgment and adapt to the data — do not
mechanically copy every setting below. The concrete style in the last section
is a recommended starting point (matching the owner's taste), not a mandate.

## What "decent and clear" means (the actual requirements)

Aim for these principles on every figure:

- **Readable text.** Axis labels and ticks large enough to read in a paper/slide
  (labels ~14–17 pt, ticks ~12–13 pt). Don't ship default-tiny fonts.
- **Labeled axes.** Every axis has a label with units/meaning; use LaTeX math
  (`r'$\delta Z^2$'`) where it helps. Add a legend whenever there is more than
  one series.
- **Sensible layout.** Reasonable figure size (e.g. ~10×6 single panel), a light
  grid to guide the eye, and `bbox_inches='tight'` / `tight_layout()` so nothing
  is clipped.
- **Good color.** For a parameter sweep, sample a perceptual colormap
  (`plasma`, `viridis`, `cividis`) so the series read in order. Avoid clashing
  hand-picked colors and don't rely on the raw default cycle for many curves.
- **Distinguish reference from data.** Plot an exact/analytic/reference curve
  distinctly (e.g. solid black, no marker) so it stands out from approximate data.
- **Right scale.** Use a log axis for errors / quantities spanning orders of
  magnitude; set limits when they improve readability.
- **Show uncertainty.** If you have error bars, draw them (`errorbar(..., capsize=...)`).
- **Save reusably.** For figures worth keeping, save a vector format (PDF/SVG)
  and, if a raster is needed, PNG at `dpi=300`. Keep image files out of `src/`.

If a simple plot communicates the point clearly, a plain `fig, ax` with labels,
a legend, and a grid is entirely fine — clarity beats decoration.

## Recommended house style (a good default, not required)

This reproduces the owner's preferred look; use it as a convenient starting
point and tweak freely.

```python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'lines.linewidth': 2.1,
    'lines.marker': 'o',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.figsize': (10, 6),
})
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# One curve per swept parameter, colored from a perceptual colormap.
bonds = [4, 8, 16, 32, 64]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(bonds)))

# Optional exact/reference curve: solid black, no marker.
# plt.plot(x, exact, label='exact', color='black', linestyle='-', marker='', alpha=0.8)

for i, bnd in enumerate(bonds):
    plt.plot(
        x, y[i],
        label=rf'$D={bnd}$',
        color=colors[i],
        marker='>', markersize=10,
        linewidth=2, alpha=0.5, linestyle='-',
    )
    # with uncertainties: plt.errorbar(x, y[i], yerr=err[i], capsize=5, ...)

plt.xlabel(r'depth', fontsize=17, labelpad=10)
plt.ylabel(r'$Z^2$', fontsize=17, labelpad=10)
# plt.yscale('log')   # for errors / wide dynamic range
plt.legend(loc='best', frameon=True, shadow=True, fontsize=13)
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
plt.savefig('figure.pdf', bbox_inches='tight')
plt.show()
```

In notebooks, crisp inline output helps: `%config InlineBackend.figure_formats = ['svg']`.

For stacked diagnostics, `plt.subplots(n, 1, sharex=True)` with the same
label/grid conventions per axis works well; mark regime boundaries with
`ax.axvline(..., color='0.35', linestyle='--', linewidth=1)`.

## Minimal checklist

- [ ] Labeled axes (units/meaning), legend if >1 series, readable font sizes.
- [ ] Light grid; nothing clipped (`tight` layout / bbox).
- [ ] Perceptual colormap for parameter sweeps; reference curve distinct.
- [ ] Log scale for errors / wide ranges; error bars drawn when available.
- [ ] Keep-worthy figures saved as vector (PDF/SVG) + PNG `dpi=300`, not under `src/`.
