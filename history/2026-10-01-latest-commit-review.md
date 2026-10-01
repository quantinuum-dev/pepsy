# 2026-10-01 — Latest Pepsy commit review

- Scope: user requested pulling and reviewing the latest Pepsy commit.
- Branch / reviewed commit: `develop` / `7773d2e`.
- Pull: `git pull --ff-only` reported already up to date.
- Commit status: review only; this handoff is uncommitted. Existing untracked
  September 30 TreeSampler review documents were preserved.

## Finding

P2: delegated sweep constructor normalization does not honor the separate
normalization cap. `PepsOptimizer._optimize_with_sweep` supplies the intended
cap through `normalize_kwargs`, but `SweepOptimizer.__init__` calls
`normalize(chi=self.chi, ...)`, overriding that stored cap with the optimizer
environment cap. Later sweep normalization uses the stored cap correctly.
This leaves the commit's independent normalization-cap contract incomplete
when normalization and environment caps differ. The underlying constructor
precedence predates this commit; the newly forwarded automatic cap does not
resolve it.

Fresh probe: a NumPy complex128 2×2 PEPS, `chi=1`, `boundary_chi=2`,
`normalize_chi=8`, `boundary_engine="quimb-mps"`, greedy contraction.
Wrapped the real sweep normalization metric to record requested caps and
replaced only `SweepOptimizer.run` with one `_normalize_state` call, avoiding
a numerical optimizer. Stored normalization cap was 8; actual constructor
and subsequent normalization caps were `[2, 8]`. This establishes routing,
not a measured fidelity error. Existing cap-routing tests mock the sweep
constructor; the real sweep tests use equal default environment/metric caps.

Suggested fix, subject to a separate implementation request: preserve the
normalization cap during constructor renormalization, with a regression that
exercises an actual sweep constructor and unequal caps.

## Fresh validation

Activated `/home/reza.haghshenas@quantinuum.com/envs/py312` in each Python shell.

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -o addopts=''
  tests/test_optimize_peps.py tests/test_prepare_boundary_inputs.py
  tests/test_public_api.py tests/test_package_layout.py`: **339 passed**,
  five warnings, no skips, 21.45 seconds.
- `python -m ruff check src tests`: passed.
- `git diff --check`: passed before the handoff; rechecked afterward.
- No full-suite run, fixes, dependency changes, commits, or pushes.

The final commit uses `(4*D, 5*D)` norm/overlap caps, superseding intermediate
D² descriptions in its dated journal. Installed Pepsy metadata reports 0.5.0;
the historical metadata mismatch did not recur.
