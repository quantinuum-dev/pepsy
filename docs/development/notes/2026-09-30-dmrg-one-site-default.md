# 2026-09-30 — Generic MPS DMRG defaults to one-site FIT

The requested policy is implemented in `MpsOptimizer.run`: omitted/None
`fit_block_size` selects one-site FIT for generic `dmrg` and its `fit` alias.
Named DMRG schedules still resolve their own block sizes. Explicit two-/three-
site updates and standalone `FIT.run_gate` defaults are unchanged.
The private `_run_dmrg` default also matches the public generic policy.
Native one-site window preparation bypasses dense bond padding and preserves
the current sectors until the existing native guess preparation runs. This
avoids the old padding guard rejecting otherwise supported native FIT calls.

One-site refinement keeps the initialized bond spaces and avoids block SVD
updates. It need not remove redundant bond dimensions from an SRC guess.
Target preparation and guess construction can still use SVD. No claim of
SVD-free replay or measured speedup is made. The single-pair exact shortcut
requires a two-site FIT block and cannot override a one-site selection.

## Upstream audit

Installed versions inspected in the existing Python 3.12 environment:
Pepsy 0.5.0, Quimb 1.15.1.dev66+ge927f06e1,
Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f,
Cotengrust 0.2.1, Symmray 0.4.1.dev7+g83fb22865.
Inspected `Tensor.split` and `FIT.run_gate` signatures and resolved Autoray's
NumPy QR, SVD, and norm dispatch. `FIT.run_gate` still defaults to block size
two. No dependency, backend registration, or numerical kernel was changed.

- **Adopt:** existing one-site FIT route and existing public decomposition
  dispatch, with native preparation kept out of the dense-padding path.
- **Defer:** upstream algorithm/default migrations outside this request.
  No compatibility shim or prototype was needed.

Consulted the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The [Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
returned an internal error; the official repository and installed native
paths were used instead. Quimb's newer cutoff/decomposition policies do not
require changing this local option resolver.

Validation and remaining limits are recorded in the
[session handoff](../../../history/2026-09-30-dmrg-one-site-default.md).
