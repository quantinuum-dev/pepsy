# 2026-09-30 — Tree DMRG defaults to one-node refinement

## Implemented

`TreeOptimizer` now defaults to `fit_block_size=1`. Generic `mode="dmrg"`
and its `fit` alias refine one tree tensor at a time from the initialized
guess, without multi-node FIT warm-up. The exact layered target remains
separate; guess construction can still split tensors and open bond support.
The change is one constructor default, not a modification of TreeFIT kernels.

Explicit larger `fit_block_size` values remain honored. Named `dmrg1`,
`dmrg2`, `dmrg3`, and `mix` keep their existing policies; unlike the earlier
MPS cleanup, this request did not remove Tree's legacy `dmrg1` mode.
TreePepsOptimizer passes its own explicit block size (default two) and is
unchanged. Standalone TreeFIT also retains its own defaults.
Constructor `mode="auto"` remains the general replay default.

The four-iteration budget remains: each iteration is inward plus outward,
so it permits eight directional passes. Default dense initialization is SRC;
`auto` uses direct compression for native fermionic data. Seed zero, random
strength zero, auto tolerances, minimum two iterations, patience one and
opt-in diagnostics are unchanged. See the complete
[default-settings table](../../api/optimizers/tree_fit.md).

## Compatibility audit

Inspected installed `TreeFIT.run_gate` (block size two at the standalone
level; optimizer forwards its own selection), `Tensor.split` and effective
TreeOptimizer defaults. No dependency installation or global dispatch change.
Installed dependency versions match the earlier
[GPU audit](2026-09-30-mps-autoray-gpu-audit.md#scope-and-environment):
Autoray 0.11.1.dev3+g1b476b305, Quimb 1.15.1.dev66+ge927f06e1,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865 and Torch 2.6.0+cu124.

Rechecked official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
Symmray's abelian-array page was unavailable again; installed native behavior
was checked with U1/U1U1 reference tests. Explicit Pepsy cutoff semantics
remain in place despite newer upstream split defaults.
Classification: **adopt** the existing one-node FIT kernel as the optimizer
default; **defer** unrelated upstream changes. No new shim or approximation.

## Regression coverage

- Eight dense cases exercise construction, copying, run-time mode selection,
  and the `fit` alias on NumPy and Torch CPU. Each local FIT call is intercepted
  and must contain exactly one node. An entangling nonlocal circuit must match
  the exact state, with valid canonical metadata and bond cap.
- Four native cases exercise U1/U1U1 and complex64/complex128 with the automatic
  direct guess. They preserve native arrays and fermionic metadata, match the
  direct reference at sufficient chi, and report one-node-only traces.
- The old generic warm-up regression now requests block size two explicitly,
  retaining its existing `(2,2,1,1)` assertion and numerical intent.

Final validation counts and limitations are recorded in the
[handoff](../../../history/2026-09-30-tree-dmrg-one-site-default.md).
