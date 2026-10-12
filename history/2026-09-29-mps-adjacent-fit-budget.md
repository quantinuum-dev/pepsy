# 2026-09-29 — Separate adjacent-window DMRG sweep cap

- Scope: user requested up to five sweeps for two-site adjacent windows,
  eight for longer windows, convergence-based early stopping, and an explicit
  user-configurable MpsOptimizer option defaulting to five.
- Branch / baseline: Pepsy `develop` / `a233a40`; examples `main` / `efb7696`.
- Commit status: working-tree edits only; no commit or push. Existing cluster
  operator/API work and the unrelated examples notebook edit were preserved.

## Implementation

`MpsOptimizer.run(fit_single_pair_n_iter=5)` caps DMRG windows spanning exactly
two MPS sites at `min(n_iter, fit_single_pair_n_iter)`. Longer windows retain
`n_iter`, which defaults to eight. The cap is positive-integer-or-None and is
validated before replay. It applies to single gate/sub-MPO windows, complete
batch spans, Pauli measurement windows, and ordinary/shot replay. It is ignored
outside DMRG. Tensor contraction, SVD, target, normalization, and FIT update
kernels were not changed.

A non-None cap suppresses the named DMRG2 automatic adjacent-pair shortcut.
Explicit `fit_single_pair_fast_path=True` still requests one update; a None
cap restores the legacy mode schedule. `fit_rtol="auto"` retains convergence
stopping, while None requests the effective fixed budget. FIT timing records
now include `requested_sweeps` for that window.

Roughening defaults to `n_iter=8`, `fit_single_pair_n_iter=5`, and disabled
single-pair fast path. Shared/KZ parsers expose `--fit-single-pair-n-iter`;
configuration records, checkpoints, metadata, and timing summaries persist it.
Resume rejects changed pair caps. This supersedes the intermediate examples
change that applied five sweeps to all windows. Other frontends retain their
existing general iteration and shortcut defaults.

## Evidence and validation

The separate-budget regression suite checks all four DMRG schedules, layered
and materialized targets, adjacent/non-adjacent gates, same-pair and wider
batches, actual iteration counts against exact-state references, user overrides,
early convergence on NumPy complex64 and Torch CPU complex128, invalid options,
measurement probabilities, and shot replay overrides.

Environment remains the shared Python 3.12 environment: NumPy 2.5.2,
Torch 2.6.0+cu124, Quimb 1.15.1.dev66+ge927f06e1,
Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f,
Symmray 0.4.1.dev7+g83fb22865. Tests use one numerical CPU thread with CUDA
hidden, keeping production jobs undisturbed.

- New budget tests plus public API/layout checks: 99 passed; one existing
  failure, `test_package_version_matches_installed_distribution`, because
  installed metadata reports 0.4.0 while source declares 0.5.0. No environment
  reinstall was performed.
- Initial downstream focused selection: 290 passed, 2 CUDA-only skips.
- Pepsy `ruff check src tests` passes. Changed downstream Python files pass;
  broad downstream Ruff still reports the six existing effective-model
  findings documented in the examples handoff. Both repositories pass
  `git diff --check`.
- Full downstream suite: **463 passed, 6 skipped** in 72.51 s. All skips
  require CUDA, deliberately hidden for CPU validation. Log:
  `/tmp/roughening-window-budget-full-20260929.log`.
- Expanded MPS/MPO/native domain suite: **1288 passed, 38 skipped** in
  413.83 s. This includes native Symmray/fermion cases, their marked slow
  3x4 Hubbard stress tests, and the new window-budget regressions. Skips
  require CUDA, CuPy GPU, or Metal; CPU NumPy and Torch checks ran. Log:
  `/tmp/pepsy-window-budget-domain-20260929.log`.

Expanded domain command, from the Pepsy repository root:

```bash
source ~/envs/py312/bin/activate
PYTHONPATH=/home/reza.haghshenas@quantinuum.com/pepsy/src:. \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
MPLBACKEND=Agg python -m pytest -q -ra -o addopts='' \
  tests/test_optimize_mps.py tests/test_mps_*.py \
  tests/test_optimize_mpo.py tests/test_symmetric_tensors.py
```

Reviewed the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray repository](https://github.com/jcmgray/symmray). The Symmray array
documentation URL returned an internal error, so installed source/signatures
and its official repository were used. Classification: defer upstream changes;
this is a Pepsy replay-budget policy addition and needs no dependency shim.

Earlier audit: [roughening/MPS alignment](2026-09-29-roughening-mps-alignment-audit.md).
No production processes or existing datasets were modified.
