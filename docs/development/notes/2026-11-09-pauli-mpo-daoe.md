# 2026-11-09 — PauliMPO readiness and first DAOE filter

The native representation supports an exact DAOE weight filter after focused
algebra/backend fixes; it does not yet provide a scalable evolution driver.
See the [public contract](../../api/operators/pauli_mpo.md#daoe-weight-filter).
The real-core compatibility fixes also address the failure observed in the
preceding read-only lattice-DMT capability probe; geometry work stays separate.

## Review and decisions

| Area | Finding and action | Classification |
| --- | --- | --- |
| Native algebra | Subtraction applied the minus sign on every core, reversing its meaning on even chains; scale one boundary core only. Commutator regressions cover one through four sites. | Adopt |
| Real/complex conversion | Keyword `dtype` failed with installed positional `astype` adapters. Adopt public `ar.astype` and preserve float32 → complex64 precision. | Adopt |
| Constants | Backend-only `like` did not preserve dtype/device. Reuse the shared backend converter for masks, zeros and basis tensors. | Adopt |
| Dense readout | Installed Autoray string-namespace caching aliases NumPy and Torch/JAX namespaces. Contract local native cores directly for dense backends, preserving gradients and grouped row/column order. No upstream globals or installed packages changed. | Compatibility shim |
| Gates/channels | Keep in-place channels core-backed, clear stale gate/channel compression reports, and reject floating/bool/string gate sites. Complex local gates are converted at matching complex precision. | Adopt |
| Partial SVD | An empty returned tail incorrectly reported zero loss after a rank-limited solve. Include the local reconstruction residual and uncomputed rank. Cutoff selection still sees only the computed spectrum. | Adopt |
| DAOE | Implement the exact diagonal Pauli-weight counting automaton and sparse equivalent, with compression as a separate operation. | Prototype stage implemented |
| Scalable dynamics / DMT | Local gates still form a global gate MPO and multiply cores. Local operator TEBD/TDVP, lattice-DMT protected truncation and transport reproduction are absent. | Defer |

The filter is [arXiv:2004.05177, Eq. (1)](https://arxiv.org/abs/2004.05177).
Its counter saturates at `max_weight`; each further nonidentity label adds
`exp(-gamma)`. Summing all final counter states leaves low-weight strings
unchanged. The counter bond dimension is at most `max_weight + 1`.
This is a diagonal superoperator on coefficient space, not multiplication
by an ordinary physical operator or a Kraus-channel representation.

No filter normalization or implicit SVD is introduced. Coefficient gradients
are supported; damping and threshold are static parameters. Exact filtering
does not make finite-damping dynamics equal to physical unitary evolution.

## Installed upstream audit

| Dependency | Installed metadata |
| --- | --- |
| Pepsy | 0.5.0; checkout/import source is 0.6.0 |
| Quimb | 1.15.0 |
| Autoray | 0.8.4; below the checkout's declared `>=0.9` |
| Cotengra | 0.8.3.dev8+g8954240f2 |
| Cotengrust | 0.2.1 |
| Symmray | 0.4.1.dev15+g0374aaa3c |
| Torch / JAX | 2.9.1 / 0.8.2 |

Consulted the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The requested Symmray `abelian_arrays.html` page was unavailable; installed
source and the repository were used instead. No new native Symmray behavior
is claimed.

Installed signatures/source checked: `ar.astype(x, dtype_name, **kwargs)`,
`ar.get_namespace(like=None, device=None, dtype=None, submodule=None)`,
the shared sample-based backend converter, Quimb `to_dense(**contract_opts)`
and Cotengra `Contractor.__call__`. Autoray 0.8.4 keys namespaces requested by
backend strings with `str` rather than the backend name: a NumPy contraction
can therefore contaminate a subsequent Torch contraction in this environment.
Changing the shared environment or global namespace cache is outside this task.
The Pauli dense readout uses backend-dispatched local einsums instead.

## Validation on this checkout

- `test_pauli_mpo.py`, `test_pauli_mpo_review.py`, `test_pauli_mpo_daoe.py`:
  **78 passed**. One JAX complex-to-real adjoint warning occurs in the
  real-core dense-gradient reference; values and gradients agree.
- New checks compare every coefficient through four sites, identity/trace
  preservation, weight versus diameter, no-op ownership, filter composition,
  integer promotion, Torch real/complex single/double precision gradients,
  JAX JIT/gradients and a dense Heisenberg/filter/compress period.
- A 64-site `(I+X)` product is filtered with word materialization patched to
  fail; normalized identity and all-X coefficients agree with the formula.
- The documented ten-period reference example executes; final autocorrelation
  is approximately `0.7462753773`. This is not a transport benchmark.
- `test_public_api.py` + `test_package_layout.py`: **54 passed, 1 failed**.
  `test_package_version_matches_installed_distribution` sees installed metadata
  0.5.0 versus the unchanged 0.6.0 project declaration at baseline `ad523b6`.
- Broader `test_mpo.py` + `test_mpo_automaton.py`: **133 passed, 17 failed,
  1 skipped**. Restoring baseline `HEAD`'s entire `pauli_mpo` module in memory
  and rerunning produces exactly the same 17 failures and counts. See below.
- Ruff over `src tests`, `git diff --check` and local Markdown file-target
  checks pass; full suite and device-specific runs not run.

Baseline-reproduced failures in `test_mpo.py`:

- `test_general_local_operator_keeps_parameter_coefficient_differentiable`
- `test_graded_parameter_phase_remains_in_autodiff_graph`
- `test_mpo_basis_square_lattice_aligns_paulis_and_preserves_autodiff`
- `test_exp_mpo_uses_to_backend_before_contraction_and_compression`
- `test_compress_mpo_product_reports_native_sector_compression`
- `test_compress_mpo_product_native_dmrg_restores_mpo_boundary`
- `test_higher_order_block_sparse_symmetry_matches_dense_mpo` (four sectors)
- `test_native_symmray_mpo_to_dense_restores_physical_basis_order`
- `test_u1_symmetry_supports_degenerate_physical_sectors`
- `test_mpo_basis_preserves_symmetry_for_parameterized_exponential`
- `test_parameterized_pauli_hamiltonian_preserves_torch_autograd`
- `test_parameterized_observable_expectation_preserves_torch_autograd`

Baseline-reproduced failures in `test_mpo_automaton.py`:

- `test_backend_delinearization_preserves_torch_gradients`
- `test_backend_delinearization_supports_jax_arrays`

## Limits and next boundary

Uniform dense NumPy and CPU Torch/JAX paths were checked. CuPy, accelerators,
native Symmray DAOE, mixed-backend/mixed-precision core trains, performance and
hydrodynamic accuracy remain unverified. Native DAOE explicitly rejects
non-dense backends. Existing unnormalized compression diagnostics scale as
`2**(L/2)` and can overflow on very long chains; a normalized/log-scale report
API remains separate work.

The next useful implementation is a local Heisenberg gate split that limits
bond growth to the acted-on bond, followed by a schedule with explicit gate,
filter and compression intervals and independent convergence checks.
