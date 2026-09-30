# 2026-09-30 — 3x3 PEPS BP and simple-update validation

- Scope: user requested testing of Pepsy BP norms against exact 3x3 PEPS
  contractions and verification of the roughening SU/BP package integration.
- Pepsy branch / baseline: develop / 1b4ca92; initially clean, ahead one commit.
- Examples branch / baseline: main / 40cadd7; preserved unrelated dirty files.
- Commit status: new tests and this note are uncommitted; nothing staged.
- Production implementation was not changed in this testing pass.

## Package routing

The roughening engine imports `pepsy.operators.gate_simple` and
`pepsy.bp.two_norm_bp`. Evolution retains external gauges and `renorm=False`.
The runner's C=4 correction currently calls `contract_gloop_expand` on the
underlying BP object. Its output was compared with the public
`pepsy.bp.loop_cluster_expand(..., messages=bp.snapshot(), run_bp=False)`
and agrees on the tested snapshots. This does not claim the runner directly
calls the Pepsy cluster wrapper.

## Added checks

- [Package reference tests](../tests/test_bp_peps_norm_reference.py): scaled
  product and entangled tree states on a 3x3 geometry; random complex128 D=2
  PEPS with seeds 0, 1, 2 on NumPy and seed 0 on Torch CPU; unchanged source
  tensors; complex global scaling; C=0/BP agreement; exact C=9 limit.
- Examples `benchmark/tests/test_peps.py` now includes 3x3 D=2, dt=0.25,
  t=4 and t=6 checks on NumPy/Torch CPU for both gauge intervals 0 and 1.
  It compares first-step amplitudes against dense gates and checkpoint norms,
  local Z, imbalance, and source-state isolation against dense amplitudes.
- The exact oracle contracts the single-layer ket to all 512 amplitudes and
  evaluates their squared norm. It does not use BP or boundary truncation.

## Results

All reported norms here are squared norms, `<psi|psi>`.

| Random D=2 seed | C=0 relative error | C=4 relative error | C=9 relative error |
| --- | ---: | ---: | ---: |
| 0 | 2.96625% | 8.68128% | 1.0e-15 |
| 1 | 2.13758% | 2.24288% | 5.6e-16 |
| 2 | 6.02228% | 0.702485% | 1.2e-15 |

BP converged in every random-state check. Product/tree BP is exact within
the test tolerances. C=4 is an approximation and need not improve on C=0.

For SU with `--peps-gauge-every 1`, D=2, dt=0.25, complex128 NumPy:

| t | Exact squared norm | BP squared norm | C=4 squared norm | C=9 squared norm |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0.11874136506051862 | 0.11675724199521433 | 0.1202398782938727 | 0.11874136506051869 |
| 6 | 0.10234523702013144 | 0.08272223867672758 | 0.09178976066323888 | 0.10234523702013144 |

These are contractions of the evolved, truncated SU state, not a claim that
SU evolution equals untruncated exact time evolution. Local Z and imbalance
of that same state match the dense oracle to the 1e-10 test tolerance on
NumPy and Torch CPU. The first, untruncated Trotter step's maximum amplitude
error was 1.23e-15 in the separate NumPy probe.

## Unresolved default SU instability

With default `peps_gauge_every=0`, keeping `renorm=False`, the 3x3 SU
representation develops tiny gauge scales and huge core tensors before
overflow. This happens before norm measurement, and directly calling Quimb
`gate_simple_` instead of Pepsy's wrapper reproduces the D=2 failure.

- NumPy D=2, dt=0.25: overflow at step 13 (t=3.25), for cutoff 0 and 1e-12.
- NumPy D=4, dt=0.25, cutoff 1e-12: overflow at step 16 (t=4).
- NumPy D=2, dt=0.05, cutoff 1e-12: overflow at step 26 (t=1.3).
- Torch CPU also fails the default D=2 long-evolution regression.
- Existing explicit gauge equilibration every step reaches t=6 and passes
  the measurement checks on both backends. This is a tested setting, not a
  default change or a general stability guarantee. GPU/5x6 was not tested.

The two default-path regression cases remain failing deliberately: no xfail,
loosened tolerance, or production normalization change hides this finding.
Any later stabilization must preserve physical scale and `renorm=False`.

## Validation and evidence

- New package reference module: 6 passed (3.05 s).
- Existing `test_bp_relay.py` and `test_simple_update_gen.py`: 28 passed.
  Earlier combined run with the first five new tests: 33 passed.
- Existing examples PEPS cases: 27 passed. New four-case 3x3 selection:
  2 passed (gauge interval 1), 2 failed (default interval 0), 27 deselected.
- Pepsy `ruff check src tests`: passed before the final tree-test addition;
  changed reference module subsequently passed Ruff. Examples changed test
  passes Ruff. Full examples Ruff retains the same six unrelated E402/E731
  errors in effective/correlations and effective-model tests.
- Both repositories pass `git diff --check`. No full repository test suite.
- Detailed values and gauge traces:
  `/tmp/pepsy_bp_3x3_validation_20260930_zm4ro7sv/results.json`.
- Activated shared py312 environment, local Pepsy source first on PYTHONPATH,
  CPU-only test processes with CUDA hidden; active production jobs untouched.
- Installed Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
  Cotengra 0.8.3.dev7+g1d7fd333f, Symmray 0.4.1.dev7+g83fb22865,
  NumPy 2.5.2, Torch 2.6.0+cu124. Pepsy distribution metadata says 0.4.0,
  while the checked-out pyproject says 0.5.0; inspected imports resolve to
  this repository's `src/pepsy`, not a different installed implementation.
