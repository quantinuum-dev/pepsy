# 2026-09-28 — Cluster construction correctness review

## Scope and findings

Review of the uncommitted cluster MPO/Pauli PEPO work on `develop`, baseline
`4e398e4`. Existing unrelated optimizer/sampler changes are preserved.
No staging, commits or publication.

The coefficient-binding probe reproduced three issues in cluster MPOs:

- Positional tensor parameters with a Python time step left empty singleton
  targets on NumPy while interacting targets used Torch, causing subtraction
  to fail with mixed-backend matrix multiplication.
- Callable coefficients returning tensors had the same problem. Inspecting
  the parameter container cannot reveal a tensor captured by a callback.
- Cluster coefficient resolution rejected `parameters=None` even when an
  `MPOParameter` supplied a valid default.

## Fix

Fixed-mode interval and graph paths evaluate their local ordered targets, align
backend/dtype using the existing shared converter, then perform connected
subtraction. The initial inference from raw parameter containers is removed.
No user callback is evaluated merely to infer a backend, and no numerical data
is retained in a structural cache. Native tensors determine precision, so host
identity matrices do not force float64. Complex host targets promote the common
representation without discarding imaginary components. Mixed tensor backends
are rejected. Declared parameter defaults are honored; absent required
parameters keep their previous error behavior.

Only `mpo_product.py` changes numerically in this review. The mathematical
partition recurrence, symmetry matcher, PEPO construction and SVD-free split
kernels are unchanged. See the [MPO API guide](../../api/operators/mpo_cluster.md).

**Adopt:** existing `infer_backend_converter_from_sample`, Autoray dtype
inspection and casting. The same-environment upstream audit from the
[implementation](2026-09-28-fixed-cluster-autodiff.md#backendcompiler-audit)
is reused. Installed `get_common_dtype` implementation and converter signature
were inspected; no installed packages or backend registrations were changed.

## Independent checks

`tests/test_cluster_correctness_review.py` enumerates square-lattice set
partitions directly, embeds matrices through computational-basis indices, and
uses SciPy matrix exponentials. It does not call production cluster,
partition, embedding or symmetry helpers for its reference. At p=2,3,4 the
result agrees with both MPO and PEPO, with reuse enabled/disabled and uniform
or spatially varying couplings. This covers disjoint residual products and the
four-site loop as well as tree clusters.

Further regressions compare interval/graph MPO values and Torch gradients for
positional/default/callable bindings at zero and nonzero coupling, repeated
calls and complex host/tensor steps. Callback counts agree between host/tensor
steps. Float32 preservation, host-complex promotion, and a JAX end-to-end value
and gradient through target alignment are checked. The JAX check is autodiff,
not whole-builder JIT.

The expanded new regression file passed **15 tests** (7.34 s), CPU-only.
The broader focused gate passed **374 tests**, with two existing deprecation
warnings (92.83 s), alongside the separate final 15-case review run that
included the last two complex/JAX checks. Full Ruff, whitespace and relative
documentation link checks passed. Commands and exact scope are recorded in the
[handoff](../../../history/2026-09-28-cluster-correctness-review.md).
The preceding full CPU suite (5241 passed, 105 skipped) predates this fix;
no new full-suite or GPU result is claimed. Existing limits on exact bond size,
whole-builder machine-code capture and unsupported native fixed MPO paths
remain as documented in the implementation note.
