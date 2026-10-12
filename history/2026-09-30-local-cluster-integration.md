# 2026-09-30 — Commit all pending Pepsy work and integrate locally

- Scope: user explicitly approved all current Pepsy changes, a local
  `develop` into `main` merge, no push, and return to `develop`.
- Baselines: `develop` at `c1ff0f8`; local `main` at `a233f9e`.
- This handoff accompanies the approved package commit. The subsequent
  merge is recorded separately in Git; no remote publication is authorized.

## Included work

The approved set comprised 113 existing modified/untracked Pepsy files:
cluster MPO/PEPO channel preparation and replay, native cluster support,
constructed traces, numerical compression and automaton/QR integration,
focused tests, API documentation, benchmark evidence and session histories.
This handoff and the status-ledger update record their local integration.
Sibling repositories and device-local environment overrides are excluded.
Earlier dated handoffs describing uncommitted work remain historical records.

## Validation

- Ruff (`src tests`), staged whitespace, and skill catalog checks passed.
- The broad cluster/MPO/PEPO plus public API/layout selection ran 690 tests:
  **682 passed, 8 failed**, with two existing alias deprecation warnings.
- Failures were JAX float32 GPU value/gradient comparisons in
  `test_cluster_jit_gradients.py` (seven cases) and
  `test_cluster_trace.py::test_joint_two_site_trace_jax_jit_gradients[pepo]`.
  This environment selects a CUDA device, x64 is disabled, and the default
  matrix-multiplication precision setting is unset.
- Re-running both complete affected modules with
  `JAX_DEFAULT_MATMUL_PRECISION=highest` passed **24 tests**, including all
  eight failures. No implementation, tolerance, or global environment setting
  was changed for this rerun. Default GPU precision remains a known failure
  mode for those numerical checks; this is not a clean default-precision
  full-suite result.
- The earlier package API selection passed 297 tests; see its
  [integration record](2026-09-30-automaton-qr-package-api.md).
- A test edit arrived during staging. Its completed decorator correction
  was included before integration; the final channel/automaton/structure,
  integrated API, public API and layout selection passed **146 tests** with
  explicit highest JAX matrix-multiplication precision. Ruff and staged
  whitespace passed again. The local content commit was amended before
  merging; no intermediate revision was pushed.
- No full repository suite was run. The local merge preview had no conflicts;
  local `main` differs from its merge base only by merge ancestry.

The broad command selected `test_cluster*.py`, `test_mpo_cluster*.py`,
`test_mpo_delinearize.py`, `test_native_cluster*.py`, PEPO trace/routing,
graph autodiff, joint parity, square-plan, public API and layout tests.
Both runs used the existing Python 3.12 environment and one BLAS/OpenMP thread.
