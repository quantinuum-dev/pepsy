# 2026-11-03 — Publish pending bounded CPU work

- Scope: user requested committing and pushing all pending Pepsy changes.
- Branch / baseline: `develop` at `9220fd6`; fetched `origin/develop` matched.
- Commit status at handoff: bounded CPU implementation, tests, API guides,
  changelog and October 27 evidence committed as `123032a`. The two existing
  October 29 journals and this validation handoff form a separate follow-up
  commit. Push is authorized; publication is verified after committing.

The implementation is described in the
[bounded CPU note](../docs/development/notes/2026-10-27-bounded-exact-cpu.md).
This session reviewed and validated the existing changes without modifying
numerical behavior or sibling repositories. The October 29 journals describe
earlier sibling work; their inclusion does not publish that sibling's files.

## New validation

Using the existing `py312` environment with four OpenMP/MKL threads and Agg:

- `python -m pytest -q -ra -o addopts='' tests/test_mps_exact_batch.py
  tests/test_sampler.py tests/test_public_api.py tests/test_package_layout.py`
  → **203 passed**, three warnings.
- `python -m ruff check src tests`, `git diff --check`, and local links in
  the four previously untracked documents → passed.
- Full suite attempted with `python -m pytest -q -ra -o addopts=''`;
  interrupted after **369 passed, six failed, two skipped**. MPI collection
  skips report unavailable `mpi4py`. This is not a completed full-suite run.
- All six failures reproduced against an isolated archive of unchanged
  baseline `9220fd6`, using its source and tests in the same environment:
  **six failed**. These preexisting failures remain unresolved.

All failures are in `tests/test_bp_symmray.py`:

- `test_native_reduced_loop_compression_uses_graded_svd_adapter`:
  reduced tensor shape mismatch.
- `test_fermionic_long_range_hopping_sign_survives_su_and_bp_gauges` and
  `test_spinful_long_range_hopping_sign_survives_su_and_bp_gauges[U1]` /
  `[U1U1]`: zero hopping expectation versus nonzero exact reference.
- `test_fermionic_open_rho_sweep_supports_native_symmetries[U1-2-None]` /
  `[U1U1-4-_site_charge]`: empty array concatenation.

Installed versions: Torch `2.14.1+cu126`, Quimb
`1.15.1.dev101+g1b64353d9`, Autoray `0.11.1.dev14+g014a3f69a`,
Cotengra `0.8.3`, Symmray `0.4.1.dev25+gdc377f033`, Numba `0.67.0`.
Inspected `TensorNetwork.copy(self, virtual=False, deep=False)` and confirmed
a writable NumPy view of untracked Torch CPU complex64 data. These version
checks supersede neither the dated October performance evidence nor its
recorded environment. No new production benchmark was run.

Temporary logs and baseline source are under `/tmp/pepsy-20261103-*`;
the results and failure names above retain the durable evidence.
