# 2026-11-19 — DMT warning policy and L=64 stability

- Scope: make preservation verification optional when costly; user explicitly
  selected **warn and continue, record errors** for finite check failures.
- Branch / baseline: `develop`, `f5d31006d02821028c890a04b17e4b15b9683d0b`.
- Commit status: uncommitted working-tree edits. Earlier Ising/Fig. 7/8 work
  remains present. Nothing staged, committed or published.

## Implemented

- Added `preservation_check={raise,warn,off}` across shared DMT, Pauli methods,
  persistent Pauli driver and ordinary MPO compression/replay; replay uses
  `dmt_preservation_check`. Library default remains strict for compatibility.
- Warning mode retains measured failures even without diagnostics, deduplicates
  notifications, and clears stale reports. Off skips outer verification while
  retaining the protected construction and computational feasibility checks.
- Ising CLIs default to warning mode, append each error with step/gate/bond to
  JSONL, expose supported Torch CPU SVD selection and full-region check skipping.
- Added a reproducible overhead benchmark and optional Fig. 7 continuation plot.

## Evidence and limits

See the [numerical record](../docs/development/notes/2026-11-19-dmt-preservation-policy.md)
for the protocol, measurements and validation. Continuing the L=64 chi=16
run exposes actual trace/local-preservation loss and large negative local
eigenvalues. Warn/off trajectories coincide. GESVD and real Pauli storage do
not establish a stable solution. No mathematical stability fix was adopted.

The API/domain selection has 333 passes, 1 skip and one pre-existing installed
version metadata mismatch (0.5.0 versus project/HEAD 0.6.0). Lattice/large
selection: 145 passes, 1 expected failure. Adapter/replay/policy/example
selection: 115 passes. Ruff and whitespace checks pass. No full-suite claim.
Final focused policy tests: 21 passes; policy/example selection: 37 passes.

The expensive full-check ordinary MPO continuation was stopped after measuring
its cost; a separate continuation skips full-region verification while
retaining bond checks. This is not equivalent evidence of local preservation.
Earlier strict guard failures remain recorded in their original notes.
The fast ordinary MPO continuation completes through figure time 40 with
trace drift 8.32e-5 and substantial Hermiticity/nonpositivity defects. A
four-step operator-variant probe also retains order-1e-6 preservation errors.
The note links saved PNG/SVG, plotted CSV, aggregate JSON with timings and
all 21 native complex128 Pauli error records. Plot layout was inspected.
Final timing after the long trajectories finished: ordinary MPO 2.465 s/step
with full-region checks versus 0.538 s with bond checks only; Pauli warn/off
0.385/0.362 s. Timed work excludes diagnostics and readout.

## Follow-up: requested L=64 paper comparison

The user corrected the response focus: compare with the paper's L=64 results.
Created a [focused Fig. 7 comparison](../docs/development/notes/2026-11-19-dmt-l64-paper-comparison.md)
using the completed trajectories and extracted published curves. No kernels
changed and no evolution reruns were needed. Saved PNG/SVG, CSV and JSON.
At t=40: published DMT E=0.30248171; native Pauli DMT 0.30248211;
ordinary MPO real energy 0.30249049. Published/Pepsy MPS give
1.35252676/1.35136771. Whole-curve DMT RMSE is 1.1593e-4 (Pauli) and
9.3456e-5 (ordinary MPO). The plotted imaginary MPO energy is separately
recorded in CSV/JSON and qualified in the note. Fig. 8 has L=16,20,24,
not L=64. Verified common times and inspected the focused plot. Uncommitted.
