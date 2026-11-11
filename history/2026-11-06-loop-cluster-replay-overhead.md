# 2026-11-06 — Remove redundant loop-cluster replay work

- Scope: user requested fixes to the measured snapshot and diagnostic-norm
  overhead, using the existing Pepsy algorithms in Eli's runner.
- Baselines: Pepsy `develop` at `a76aafd`, examples `main` at `fa2c63c`.
- Commit status: source, tests, API/changelog, ledger and this handoff are
  included in this commit. The examples integration is committed separately.
  Nothing pushed; no production simulation, frozen source or saved result changed.

## Result

Public gate/prepared APIs accept `materialize_physical=False` to omit their
separate physical-network snapshots; defaults retain existing behavior.
`ReducedALSSolution.target_norm` exposes the computed solve-time scalar.
Eli selects omitted snapshots and reuses this norm for prepared/reference
replay, preserving all local diagnostics, checkpoint/readout reconstruction
and numerical policy. See the
[implementation and measurements](../docs/development/notes/2026-11-06-loop-cluster-replay-overhead.md).

The saved D6 CUDA pair has identical output cores, gauges, exponent, costs,
iteration counts and fidelity with the two controls. Target-norm differences
are at most 1.67e-16 relative. Two CPU-pinned fifteen-pair timing repeats show
median paired ratios 1.04624/1.04702; raw timings vary substantially and the
earlier unpinned run was slower. No full-trajectory speedup is claimed.

## Fresh validation

- Affected BP/compression, Torch/batch/prepared, public API/layout selection:
  **313 passed**, including CPU/CUDA derivatives and ordinary reference replay.
- Native Symmray local-gauge and reduced-compression selection: **3 passed**.
- Examples loop-cluster/global-cost selection: **98 passed**. A regression
  rejects norm recomputation in Torch logging; disk reuse and dense full
  operator/prefix/backend checks still pass.
- Full suite with `-x`: **326 passed, 2 skipped, 1 failed** in 51.40 s.
  `test_bp_symmray.py::test_fermionic_long_range_hopping_sign_survives_su_and_bp_gauges`
  again returns 0 versus the expected -0.24383291443216362 in upstream exact
  readout. This matches the independently recorded pre-existing failure in
  the [prepared-replay handoff](2026-11-05-prepared-cluster-replay.md).
  The full suite is not green; this unrelated failure was not changed.
- Full Pepsy Ruff, affected examples Ruff, documentation relative-link targets
  and whitespace checks pass. Shared Python 3.12 with local source imports and
  one numerical thread. Dependency inspection and unavailable upstream-page
  fallbacks are recorded in the implementation note.

Automatic batching, global preparation-copy reduction and GPU scalar-check
changes remain separate optimization targets; this change does not relax
conditioning, stopping, gradients or amplitude preservation.
