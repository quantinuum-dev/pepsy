# 2026-10-30 — batched dense PEPO simple update

Baseline: develop `e658bed6`, with pre-existing unrelated modifications
preserved. Changes are local, uncommitted and unpublished. This continues
the [saved-factor QR work](2026-10-30-pepo-qr-reuse.md).

Implemented `operators.PreparedSimpleUpdate`: copied target/weights, static
ordered layouts, shape-compatible disjoint-site batches, and independent
network batches. Array kernels use the registered QR/SVD derivatives, exact
phase conventions, conditioned weights and detached RMS scale extraction.
The dense RMS helper is shared with ordinary `renorm_gauge`; native block
handling remains in its original implementation.

The shape-retaining path checks every member's requested rank; a change
replays sequential adaptive `gate_simple`. It does not approximate cutoff
decisions with a common rank. Public boundaries reject unsupported arrays
and gates. The [owning API guide](../docs/api/operators/gates.md#prepared-dense-simple-update)
documents usage and limitations.

AOT-eager captures forward and registered backward together. The attempted
Inductor route was removed after CUDA complex gradients disagreed with the
physical matrix reference (largest absolute error 0.424 in the small test).
CUDA graph capture separately failed with cuSOLVER/checked-solve operations.
No relaxed tolerances or derivative substitutions were used to accept it.

The unchanged-environment upstream audit from the QR task applies: Torch
2.14.1+cu126, Quimb 1.15.1.dev101, Autoray 0.11.1.dev14, Cotengra 0.8.3,
Symmray 0.4.1.dev25. Classification: adopt batched array execution; prototype
AOT-eager; defer Inductor/CUDA graphs and JAX migration.

The [downstream record](../../gaugy_examples/research/diagnostics/2026-10-30-pepo-hv-circuit-gpu/README.md)
owns full finite-PBC benchmarks and source snapshots. The matched 30-step
downstream fit takes 110.21 seconds versus 1092.56 seconds, with final local
cost differing by `3.32e-12` and parameters by at most `1.26e-10`.

Final CPU selection: 106 passed, four CUDA cases deselected. It covers
prepared replay, ordinary SU scale/zero-weight behavior and public package
layout. Reconstruction and derivative comparisons include renormalization
on/off, per-member cutoff modes, rank changes, an overcomplete input bond,
automatic cutoff resolution, and exact zero external support. That last
case emits the expected warning about regularized singular-QR derivatives.
Ruff over `src tests` passes. Downstream Gaugy's broader selection passes
132 tests; the final five prepared-engine tests pass after guards were added.
Four final CUDA reconstruction/gradient cases pass, covering eager and
AOT-eager with both gesvd and gesvdj. Evidence is retained in the linked
diagnostic. No full-suite result is inferred, and the three previously reproduced downstream native-Z2
sector-map failures remain unresolved.
