# 2026-09-30 — MPS DMRG1 aliases generic DMRG

## Scope and implementation

User requested that `MpsOptimizer(mode="dmrg1")` and `mode="dmrg"` mean
one-site FIT from the initialized guess by default. DMRG1 now normalizes to
generic DMRG without named-schedule metadata at construction and mode changes.
Explicit block sizes, guess policies, convergence controls, adjacent budgets,
copy/shot replay and conditional events follow the same generic policy.
Removed the old DMRG1 growth reservation, rank latch and product-fermion-only
two-site exception. The public diagnostic `dmrg1_one_site_locked` remains
present on ordinary gate FIT records as `False` for compatibility. Named
DMRG2/3 and the FIT numerical kernels are unchanged. Other optimizers' own
DMRG1 policies are outside this request.

The initialized guess can open bond support; subsequent default one-site
sweeps keep that support fixed. The default is still `guess-src`, eight
maximum sweeps, automatic tolerance/cutoff, and patience two.

## Upstream audit

Checked official sources on 2026-09-30:

- [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html):
  recent decomposition cutoff and randomized-SVD restrictions do not require
  a new alias-specific compression path. Existing Pepsy adapters remain.
- [Autoray repository](https://github.com/jcmgray/autoray): retain existing
  backend dispatch; this change adds no array conversions.
- [Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
  [changelog](https://cotengra.readthedocs.io/en/latest/changelog.html): retain
  the existing contraction optimizer and fallback policy.
- [Symmray arrays documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
  returned a tool internal error. Used the accessible official
  [repository](https://github.com/jcmgray/symmray) and installed capabilities.

Installed environment: Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Cotengrust
`0.2.1`, Symmray `0.4.1.dev7+g83fb22865`, NumPy `2.5.2`, Torch
`2.6.0+cu124`. No environment or dependency changes.

Inspected installed signatures of `MatrixProductState.gate_with_auto_swap`,
`Tensor.split`, `tensor_network_1d_compress`, the SRC compressor and
`FIT.run_gate`. SRC exposes `seed`, `max_bond` and compression options;
`Tensor.split` exposes cutoff mode and rank caps. Symmray exposes
`linalg.svd_truncated`; Autoray resolves NumPy SVD through its registry.
The alias reuses these existing adapters and native guess/FIT paths.

Classification: **adopt** the existing generic DMRG dispatch for DMRG1;
**defer** unrelated upstream changes. No new compatibility shim or prototype.

## Validation

Numerical alias regressions compare states, canonical centers, FIT diagnostics
and sweep block sizes on a truncated five-site circuit, via constructor,
`set_mode`, `run(mode=...)` and copy entry points. Default and explicit block
sizes two/three are covered. A product-state entangling reference separately
checks that guess initialization opens bonds before one-site refinement.
Existing domain tests cover native fermions, controls, batching, convergence,
non-unitary scale, backend handling and the unchanged DMRG2/3 schedules.

Final command results and limitations are recorded in the
[session handoff](../../../history/2026-09-30-mps-dmrg1-alias.md).
The discussed examples notebook retains its prior outputs; it was not rerun
and those results do not measure the new DMRG1 behavior.
