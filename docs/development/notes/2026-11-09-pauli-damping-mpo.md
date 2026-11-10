# 2026-11-09 — Reusable one-dimensional Pauli damping MPO

`PauliDampingMPO` makes the existing exact weight filter independently
constructible, inspectable and reusable. Geometry rules and 2D work were
explicitly deferred by the user before implementation.

## Representation and ownership

The [public API](../../api/operators/pauli_mpo.md#reusable-damping-mpo) is
exported by `pepsy.operators` and implemented in `operators.pauli_damping`.
`PauliMPO.apply_daoe` constructs the same object for one application;
`apply_damping` accepts an existing object. The old private counter/application
helper was replaced rather than duplicated.

The physical axes of the exported Quimb MPO label input/output Pauli basis
elements, each of dimension four. They are distinct from the dimension-two
bra/ket axes of an ordinary physical qubit MPO. Internal storage keeps only
the physical diagonal `(left, right, 4)`. This is the exact counting
construction in [DAOE, p. 2](https://arxiv.org/pdf/2004.05177), with bond
dimension at most `max_weight + 1`.

Structural parameters are frozen and validated. Native application uses
backend-dispatched diagonal contractions; sparse application scales only
stored terms. Neither path enumerates native Pauli words, normalizes,
compresses nor evolves the Hamiltonian. Backend constants are rebuilt on
application, avoiding cached autodiff graphs. Exported cores and Quimb arrays
are independent copies, so modifying them does not change later reuse.
Integer references are promoted to floating point as in the preceding filter.

No-op filters preserve existing compression reports and return independent
operator copies unless `inplace=True`; actual filtering clears stale reports.
Separate SVD diagnostics measure numerical loss relative to the filtered
target, not the physical error of finite-damping dynamics.

## Upstream audit and decisions

The installed environment was rechecked and is unchanged from the
[preceding audit](2026-11-09-pauli-mpo-daoe.md#installed-upstream-audit):
Quimb 1.15.0, Autoray 0.8.4, Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1,
Symmray 0.4.1.dev15+g0374aaa3c, Torch 2.9.1 and JAX 0.8.2. Installed Pepsy
metadata remains 0.5.0 while source declares 0.6.0. No dependencies changed.

Reconsulted the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The Symmray
`abelian_arrays.html` page remains inaccessible. Native Symmray is outside
the dense damping contract and is rejected by capability checks.

Installed `MatrixProductOperator` accepts `arrays`, `shape='lrud'`,
`upper_ind_id` and `lower_ind_id`; the implementation uses these public
arguments and removes singleton open-boundary axes, including the one-site
case. Public `ar.astype(x, dtype_name, **kwargs)` and the existing shared
sample-based converter preserve the preceding dtype/device policy.

| Decision | Classification |
| --- | --- |
| Reuse the exact weight counter and direct diagonal contraction | Adopt |
| Export through public Quimb MPO construction; one public namespace owner | Adopt |
| Reusable frozen filter and coefficient-space MPO boundary | Prototype implemented |
| New global dispatch shims or installed-package edits | Defer; none needed |
| Geometry filters, 2D, finite-density and fermionic variants | Defer |
| Scalable local Heisenberg integrator and transport benchmarks | Defer |

## Validation

- `test_pauli_damping_mpo.py`, `test_pauli_mpo.py`,
  `test_pauli_mpo_review.py`, `test_pauli_mpo_daoe.py`: **123 passed**.
  The preceding real-core JAX adjoint test still emits one complex-to-real
  warning; values and gradients match its independent reference.
- Explicit superoperator matrices and diagonal cores match all Pauli words
  for one, two and four sites, including zero damping and identity filters.
- Genuine Quimb MPO-on-MPS application matches native application and direct
  coefficients. Exact compression reconstructs the target; lossy compression
  reports bound its independently measured Hilbert-Schmidt error.
- Ownership tests mutate exported NumPy cores/Quimb arrays and reuse the
  original filter. NumPy/Torch single/double real/complex export preserves
  dtype; Torch device metadata and JAX JIT export are checked. Existing
  Torch/JAX coefficient-gradient and 64-site no-expansion regressions pass.
- `test_public_api.py` + `test_package_layout.py`: **54 passed, 1 failed**.
  The unchanged failure is `test_package_version_matches_installed_distribution`
  from installed 0.5.0 versus source 0.6.0 metadata.
- Both new/updated API examples execute, including the ten-period trajectory
  with final autocorrelation approximately `0.7462753773`.
- Ruff, local Markdown file-target checks and `git diff --check` pass.
  Full suite, accelerator/CuPy execution and physical transport accuracy
  were not tested. Earlier broad MPO failures are recorded in the preceding
  audit; they were not rerun for this additive operator-only step.

## Additional pre-commit validation

The later user request to test more adds eight-period three-site trajectories
at `gamma=0`, `0.12` and `0.7`. Every period matches an independent dense
unitary/Pauli-projection reference; spin correlations conserve total spin,
and the Hilbert-Schmidt norm contracts only when damping is nonzero.
Real and complex Torch inputs also pass finite-difference `gradcheck`.

The expanded four-module Pauli selection gives **128 passed, 1 skipped**
on the default single CPU device. The skipped nondefault-device test passes
separately with two logical JAX CPU devices. These are placement checks,
not GPU validation. The existing real-core JAX adjoint warning remains.

Publication validation and the broader baseline comparison are recorded in
the [commit handoff](../../../history/2026-11-09-pauli-damping-validation-commit.md).
