# 2026-10-01 — Metric cap and final gate compression compatibility

Classification: **adopt** public dimensional compression APIs. No compatibility
shim, installed-library edits, dependency changes, or new approximate method.

## Installed capabilities

Activated the existing `envs/py312` environment and inspected:

| Package | Version |
| --- | --- |
| Quimb | 1.15.1.dev66+ge927f06e1 |
| Autoray | 0.11.1.dev3+g1b476b305 |
| Cotengra | 0.8.3.dev7+g1d7fd333f |
| Cotengrust | 0.2.1 |
| Symmray | 0.4.1.dev7+g83fb22865 |
| Torch | 2.6.0+cu124 |
| JAX | 0.10.2 |

Installed MPS/MPO `compress` explicitly accepts `form`; PEPS/PEPO `compress`
accepts lattice row/column sweeps and split options, so forwarding `form`
reaches an incompatible split driver. `TensorNetwork1D` is publicly exposed.
The final gate helper now supplies `form="left"` only for that class; lattice
compression retains its own public defaults. The existing `compress_all_`
fallback is unchanged. Array conversion and backend dispatch remain upstream.

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The requested
[Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
returned an internal error; used the official repository and installed native
array implementation/probe. The Quimb changelog describes newer split cutoff
defaults; this task preserves the installed policy and changes only dimensional
option dispatch. Cotengra optimizer/CMA-ES and Cotengrust fallback policy are
unchanged.

## Numerical evidence

Dense NumPy/Torch regression: final chi-one compression of a two-site
entangled PEPS retains the known dominant Schmidt term, with unchanged input
and Torch complex128 CPU arrays. Exact optimizer target regression reconstructs
both Schmidt terms independently despite inherited final compression options.
Native U1 fermionic PEPO regression checks native arrays, operator norm and
cross overlap at an uncapped rank. This does not establish truncated native
gradient correctness or GPU execution.

Cap precedence regressions check actual metric arguments, public direct calls,
run records, initial/later sweep contractions and backend-specific overrides.
Named constructor/per-call caps outrank stored mappings; per-call mappings
retain precedence. The resolved cap supplies delegated defaults, while explicit
backend-specific mappings remain overrides.

Validation results and resource limitations are recorded in the
[handoff](../../../history/2026-10-01-metric-gate-fixes.md).
