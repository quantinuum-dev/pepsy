# 2026-10-01 — Extended PEPS optimizer review

- Scope: user requested more review after the initial normalization-cap fix.
- Branch / baseline: `develop` / `7773d2e`, with the prior fix still uncommitted.
- Status: additional review only; no additional implementation/test changes,
  staging, commits or pushes. This handoff is uncommitted.

## Findings

### P2: named diagnostic caps lose to stored mapping caps

`PepsOptimizer._normalize_state` and `estimate_infidelity` use `setdefault`
after merging stored mapping options. Consequently a stored
`normalize_kwargs={"chi": 2}` defeats `normalize(normalize_chi=8)`;
`infidelity_kwargs={"chi": 3}` defeats
`estimate_infidelity(..., evaluation_chi=9)`. A fresh public-method routing
probe, intercepting only the metric calls, recorded actual caps 2 and 3.
This conflicts with the documented temporary override behavior. Run records
also resolve the named caps separately and can report a cap different from
the actual mapping-selected contraction cap. The precedence predates the
latest commit and is separate from the repaired sweep constructor issue.

Suggested correction: define one explicit cap precedence across metric calls,
delegated cleanup and records; test named per-call overrides with stored chi
mapping entries, preserving intentionally higher-priority per-call mappings.

### P2: final gate chi reaches the exact target and crashes for PEPS

`_target_gate_options` protects `cutoff`, `max_bond` and `path_compress`, but
leaves final whole-network `chi` intact from `gate_kwargs` or
`target_gate_kwargs`. `gate` consequently calls `_apply_chi_compression` after
building the target. That helper selects `tn.compress(form="left", ...)`
when a PEPS exposes `compress`, forwarding an MPS-only option into its 2D
compression path. The installed Quimb split raises
`TypeError: svd_truncated_numpy() got an unexpected keyword argument 'form'`.

Fresh public-run probe: NumPy float64 2×2 |0000> product PEPS, gate
`cos(0.3) I - i sin(0.3) X⊗X` on (0,0)/(0,1), optimizer chi=1,
quimb-mps boundary engine and greedy contraction. Without the extra option,
the target has bond two and the run correctly uses a warm start. Both
`gate_kwargs={"chi": 1}` and `target_gate_kwargs={"chi": 1}` crash.
Target protection and the generic compression dispatch both predate the
latest commit. A working final-compression route would still defeat the
exact-target contract unless the optimizer removes/rejects that target cap.
No silent truncation was observed in this installed stack: it raises first.

Suggested correction: block output-compression controls from exact targets
and route standalone final gate compression according to TN dimensionality;
test dense target reconstruction and the public PEPS gate final-chi route.

## Fresh checks

- Activated existing `envs/py312`; installed dependencies unchanged.
- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -o addopts=''
  tests/test_optimize_global.py tests/test_optimize_peps.py`: **135 passed**,
  14 warnings, no skips, 14.79 seconds. Passing tests do not cover the probes.
- Eight complex64 product-state normalization/self-infidelity probes passed;
  no roundoff correctness finding is claimed from those cases.
- `git diff --check`: passed. Prior fix diff preserved.
- Full suite and new Ruff run not performed; the preceding fix's Ruff and
  343-test validation remain earlier results, not new results in this review.
