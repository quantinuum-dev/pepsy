# 2026-09-25 — Review of new exact roughening and MPS sweeps

- Scope: inspect the user's new 4×4/4×5 sweeps and MPS χ=512 results used by
  `pepsy_examples/experiments/mps_magnetization/benchmark/plots/rough_exact.ipynb`.
- Branch / baseline: Pepsy `develop`, `80f451a`; examples `main`, `301159a`.
  Both repositories already contained unrelated and task-related working-tree edits.
- Commit status: this handoff is uncommitted; no source, notebook, or simulation
  output was changed, and nothing was staged, committed, or published.

## Findings

- All six new `exact_batch_{4x4,4x5}_dtheta12_dt{005,025,040}_depth*_samples8192`
  sweeps under `/tmp/pepsy_examples_runs` completed all 12 angles through t=10.
  They use 8192 Z shots and record middle-chain-cut entropy in bits.
- `mps_direct_5x6_chi512_dtheta12_dt005_depth100` completed all 12 angles
  through t=5 with 4096 Z shots. This new χ=512 run is 5×6.
- The current examples plotting helper defaults still select older t=5
  small-lattice runs and χ≤256 at 5×6. New t=10 data, dt=0.4, entropy, and
  5×6 χ=512 are not included in the default notebook plots. The examples root
  README also still describes the older 9×10 MPS comparison.
- dt=0.4 has retained shots at t=4,6,8,10; t=5,7,9 do not lie on its time
  grid. Use t=4 to compare all these datasets without interpolating.
- At t=5, the RMS sampled imbalance difference from the matching 5×6 dense
  run across 12 angles is 0.062798 at χ=256 and 0.048230 at χ=512. χ=512
  improves agreement but does not establish convergence. Its largest absolute
  imbalance difference is 0.081462. The recorded compression norm survival
  spans 0.171131–0.993159; this is not exact-state overlap.
- At t=10, maximum middle-cut entropy differences from dt=0.05 are
  0.17826/0.21259 bits (4×4/4×5) for dt=0.25, and 0.48939/0.84027 bits for
  dt=0.4. These compare Trotter circuits, not a continuous-time reference.

## Validation performed in this session

- Inspected completion status, zero child exit codes, checkpoint depths,
  archives, finite primary observables, shot counts, model parameters, angle
  grids, and the variance identity for 144 cases (new runs plus 5×6 references).
- Independently reconstructed five magnetization/imbalance moments from
  3,637,248 retained bitstrings across 456 snapshots in the 84 new cases.
  Maximum disagreement was 3.46e-8, within float32 sampling-reduction accuracy.
- Generated and visually inspected timestep and χ comparison plots under
  `/tmp/roughening_review_20260925`, with CSV and JSON summaries. Diagnostic
  scripts are `/tmp/review_roughening_20260925.py` and
  `/tmp/check_roughening_shots_20260925.py`; these temporary files may disappear.
- No simulations, numerical package tests, or full suite were run. This was a
  saved-data review; no numerical implementation was modified.

## Limitations

Sample error bars exclude truncation and timestep errors. No saved exact-state
overlap was measured. Agreement between two time steps does not establish
continuous-time convergence. Notebook integration remains a separate change.
