# 2026-10-06 — Include transferred roughening runs in the comparison notebooks

- Scope: add the seven user-supplied MPS/tree and PEPS exports to the existing
  rough-exact comparison plots in the sibling examples repository.
- Branch / baseline: examples `main`, `501ba11`; Pepsy `develop` (unchanged
  implementation). Existing examples edits and untracked results were preserved.
- Commit status: working-tree edits only; nothing staged, committed, or published.

## Changes and measured coverage

- Updated [rough_exact_mps.ipynb](../../pepsy_examples/experiments/mps_magnetization/benchmark/plots/rough_exact_mps.ipynb)
  to prefer the explicitly named transfers, retaining per-run overrides and
  older-root fallbacks. Added Tree Direct χ=512 and Tree DMRG χ=2046 registry
  entries. χ=2046 remains distinct from χ=2048. A run key has one source;
  archives from different runs are not combined or counted twice.
- The transfers provide 12 completed angles each for Tree Direct χ=512 and
  Tree DMRG χ=512/1024, five for Tree DMRG χ=2046, and two for MPS DMRG χ=4096.
  The selected MPS/tree comparison now has 195/240 completed angles at t=6,
  versus the October 5 selection's 159/216. Exact references remain separate.
- Updated [run_exact_peps.ipynb](../../pepsy_examples/experiments/mps_magnetization/benchmark/plots/run_exact_peps.ipynb)
  and its loader to read D=10/25 runner NPZ checkpoints, alongside the older
  custom JSON schema. New runs use CUDA, no initial gauge equilibration,
  automatic cutoffs, and parallel BP with DIIS. Labels/prose distinguish these
  from the CPU data; this is not a controlled D-only convergence comparison.
- Explicit comparison opt-in permits differing execution settings and norm
  schedules while preserving physical checks. Retry replacement stays strict.
  Raw norms, convergence flags, clipping flags, and source paths are retained.
  At t=6, D=10 has five saved angles, three converged; D=25 has eight saved
  angles, five converged. Unconverged or absent observations remain gaps.
- Documented transfer roots and overrides in the plotting guide. No result
  archive was moved or modified; no simulation was launched.

## New validation

- Plot-helper and PEPS norm suites: **94 passed, 4 skipped**. Skips concern
  unavailable optional wall reconstruction. Regressions cover transfer/override
  precedence, χ=2046 identity, partial runner archives, convergence, source
  provenance, schedule preservation, and rejection of mismatched physics.
- Changed Python files pass Ruff. The required benchmark-wide Ruff check
  reports six existing issues in `magnetization/effective/correlations.py`,
  `tests/test_roughening_eff_correlations.py`, and `tests/test_roughening_eff_theta.py`.
- Executed MPS/tree setup, coverage, and all nine scalar/norm-survival figures;
  executed PEPS setup and both 5×6 norm stacks. Copies and PNG/PDF/CSV exports
  are under `/tmp/pepsy_examples_executed/roughening_20261006/`; log is
  `/tmp/render_roughening_20261006.log`. Visually reviewed the combined MPS/tree
  norm-survival plot and the six-panel PEPS angle stack.
- Refreshed only those source notebook outputs via patches after execution
  under `/tmp`. Notebook schemas and preservation of untouched outputs pass;
  `git diff --check` passes. Optional wall cells and 3×3 PEPS plots were not
  rerun. No full numerical-suite validation is claimed.

Saved sweep statuses describe the transferred snapshot, not remote process
liveness. Refresh notebooks after further exports arrive.
