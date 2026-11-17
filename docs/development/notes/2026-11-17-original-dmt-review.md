# Original DMT implementation and API review

The default density path implements the connected-residual construction in
[White et al., Sec. III](https://arxiv.org/pdf/1707.01506). It canonicalizes
the active MPO bond, protects the boundary trace maps, subtracts the
disconnected baseline, truncates the free block, and refactors the resulting
thin factors. Exact constraints and the residual cutoff act on distinct
subspaces. Independent dense block and local-marginal tests cover both Pauli
and ordinary MPO storage.

The two storage APIs intentionally have different historical cutoff defaults:
ordinary MPO compression/replay use `auto/auto`; Pauli APIs use `0.0/rel`.
Explicit `auto/auto` or `0.0/rel` selects comparable settings in either basis.
One Pauli gate caps only the acted-on bond; an initially oversized input needs
a full sweep when the global cap matters. The first `MpoOptimizer` DMT gate
does perform a full sweep. Default full-region checks can be disabled; bond
guards remain, while density defects are then unmeasured. The optimized
one-bond refiner is not a full-MPO constrained variational optimizer.
The relaxed fallback uses zero-cutoff weighted/ordinary SVD seeds at the
requested cap; the paper residual cutoff does not apply to those seeds.

## Precision fix

A finite Python `constraint_weight` can be too large for the tensor dtype.
The relaxed metric has $g=(1+w)^{-1/2}$ and uses both $g$ and $1/g$.
With a complex64 MPO and `w=1e100`, the old path overflowed and raised a
backend `LinAlgError`; at `w=1e308`, complex128 has the same representability
problem. Relaxed mode now rejects values above $1/\operatorname{tiny}-1$
for the real component dtype before modifying the input. This guard preserves
the dtype and does not affect strict/paper compression or ordinary weights.

## Upstream audit

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray). The linked Symmray
array documentation was unavailable. Installed versions: Quimb 1.15.0,
Autoray 0.8.4, Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1,
Symmray 0.4.1.dev15+g0374aaa3c, NumPy 2.5.2, SciPy 1.18.0,
Torch 2.9.1, and JAX 0.8.2. The installed `array_split` supports explicit
method, absorb, cutoff, cutoff mode, bond cap, and info; Autoray dispatches
native SVD for NumPy, Torch, and JAX. Classification: **adopt** existing
native decomposition and dtype handling; **defer** CuPy validation and
native Symmray support. No installed dependency was changed.

Validation results and the existing unrelated full-suite blocker are in the
[session handoff](../../../history/2026-11-17-original-dmt-review.md).
