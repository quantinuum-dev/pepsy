# 2026-10-01 — Resolve the seven sampling test failures

- Scope: user requested fixing the seven failures reported during TreeSampler
  finalization.
- Branch / baseline: `develop` / `5558857`; working tree was clean on resumption.
  The earlier TreeSampler work is committed in `3bfaaa9` and included in the merge.
- Commit status: these fixes are working-tree edits; nothing staged, committed,
  or published in this session.

## Changes and findings

The three MPS failures came from supported-mode parametrizations still listing
`dmrg1`. The [current API](../docs/api/optimizers/mps.md) and implementation
intentionally reject that mode; `dmrg` supplies one-site FIT. Removed the stale
entries from [MPI interface tests](../tests/test_mpi.py) and
[trajectory tests](../tests/test_trajectory_noise.py), retaining `dmrg` and the
existing constructor/set-mode/run/shot rejection coverage in
[compression-mode tests](../tests/test_mps_compression_modes.py). The same stale
entry was present in the [real MPI matrix](../tests/test_mpi_integration.py),
where a missing optional dependency masked it; corrected that matrix too.

The four PEPS failures came from [4×4 exact-reference tests](../tests/test_peps_sampler_4x4.py)
requesting boundary amplitudes. The
[proposal-default migration](2026-09-30-peps-proposal-default.md) made those
requests explicit without rerunning this suite. Boundary amplitudes inherit
`chi_prime` and the cutoff and can be approximate; these tests require exact
original-ket amplitudes for their dense oracle and importance-weight checks.
They now explicitly use `amplitude_mode="exact"`, including refresh references.
Proposal caps, cutoff settings, all numerical tolerances, and statistical
assertions are preserved. Added checks that exact-mode batches declare exact
weights. The separate [boundary amplitude suite](../tests/test_peps_boundary_amplitudes.py)
continues to check approximate caps, convergence, metadata, and the absence of
full-network exact plans.

Classification: **adopt** the documented API contracts in stale tests. No
numerical implementation, public API, dependency, or default behavior is changed
by this fix. Concurrent edits to `src/pepsy/sampling/tree.py` and the new
`tests/test_tree_sampler_validity.py` appeared during validation and were
preserved; their validation is outside this test-contract correction.
The installed Quimb/Autoray/Cotengra/Cotengrust/Symmray/NumPy/Torch versions
remain those recorded in the [earlier audit](../docs/development/notes/2026-10-01-tree-sampling-vectors.md).

## Validation

- Smoke: **93 passed**, two compatibility warnings, 33.48 seconds.
- Focused suites: **234 passed**, three gate-conversion warnings, 371.01 seconds.
  Ran `tests/test_mpi.py`, `tests/test_trajectory_noise.py`, the four
  `test_removed_dmrg1_is_rejected` cases, `tests/test_peps_sampler_4x4.py`, and
  `tests/test_peps_boundary_amplitudes.py`, with `-q -ra -o addopts=''`.
  All four formerly failing PEPS cases passed; the obsolete MPS success cases
  were removed and supported-mode/rejection coverage passed.
- Broader sampler/API suites: **513 passed, 16 skipped**, two compatibility
  warnings, 299.26 seconds. Ran `tests/test_peps_sampler.py`,
  `tests/test_peps_sampler_efficiency.py`, `tests/test_peps_sampler_chunks.py`,
  `tests/test_peps_sampler_backend_audit.py`, `tests/test_sampler.py`,
  `tests/test_public_api.py`, and `tests/test_package_layout.py`, with
  `-q -ra -o addopts=''`. Two skips require two logical JAX devices; fourteen
  require CUDA/CuPy. NumPy, Torch CPU, and JAX CPU were exercised.
- These runs activated the existing Python 3.12 environment and used
  `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
  NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig`.
- Final Ruff (`python -m ruff check src tests`), `git diff --check`, and handoff
  file-link checks: passed.
- Two-rank real-MPI attempt: **unverified**. `mpi4py` is absent, so the selected
  test cannot be collected; no environment packages were installed or modified.

The earlier [full-suite evidence](2026-10-01-tree-factor-sampling-integration.md#finalization--2026-10-01)
is historical: 6,493 passed, 287 skipped, seven failed. It is not a full-suite
pass for these edits. The full suite has not been rerun in this session.
