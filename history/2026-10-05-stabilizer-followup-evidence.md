# 2026-10-05 — Stabilizer trajectory validation and performance follow-ups

- Scope: user requested the remaining follow-ups and further learning.
- Branch / baseline: `develop`, `c649aaa` (local priority-1 commit).
- Commit status: included in the reviewed local MPI/benchmark/profile commit
  (see Git history). No push requested or performed.

Real MPI execution found a stale removed `dmrg1` success-case parameter,
invisible to the single-rank full suite. Removed it from the supported-mode
matrix, which already contains `dmrg`. Added NumPy/Torch CPU mixed STN
trajectories compared with MPI-self using global shot IDs, conditional states,
measurement/noise histories and importance weights.

Expanded actual integration checks pass **26 tests per rank** with two and
three ranks. Focused MPI unit/fingerprint, noisy dense-reference and trajectory
execution checks pass **109 tests, 2 CUDA skips**. Ruff and whitespace pass.
The prior complete suite belongs to the baseline, not these new edits.

Added a reproducible CPU measurement-routing benchmark and recorded its raw
observations. Native selection yields roughly 4–5x on the small Clifford case
and 1.5–1.7x on the mixed case, with matching outcomes and final states. Eight
archived 52-qubit Helix shots finish on NumPy and Torch CPU with coefficient
bond one and all 8,640 collapse events routed through Stim.

See [detailed evidence and limitations](../docs/development/notes/2026-10-05-stabilizer-followup-evidence.md).
No production numerical algorithms changed. GPU batching, general entangled
region recognition and the missing original noisy Helix experiment remain
unimplemented/unvalidated by this follow-up. The user subsequently narrowed
the active scope to mixed-circuit profiling and committing these reviewed
changes, explicitly deprioritizing broader region recognition.

## Mixed magic/noise profiling and commit review

Added a reproducible profile for six-qubit measurement-heavy and twelve-qubit
entangling/noisy workloads on NumPy/Torch CPU, independent/coalesced replay.
One warm-up and three uninstrumented repetitions distinguish throughput from
profiling overhead. Nested temporary wall timers independently verify stage
attribution because cProfile's outer cumulative times are incomplete in this
installation. Instrumented outcomes/weights/counts match baseline runs; norms
and terminal physical readout agree. No production algorithms changed.

Entangling coefficient MPO application/compression occupies 70–72% of the
independent run, while Kraus probabilities occupy about 2%. Coalescing speeds
these cases up 1.87–1.91x, but yields no meaningful gain for the measurement-
heavy case. Whole-MPS branch summation is 25–32% there; a guarded mapped
one-site application is the first concrete implementation candidate. Existing
certificate rechecks add measurable coalesced overhead; expanding recognition
is not needed to investigate operation-scoped reuse.

See [profile, caveats and proposals](../docs/development/notes/2026-10-05-stabilizer-mixed-profile.md).
The final reviewed MPI tests passed again with two/three ranks (26 cases per
rank); focused checks gave 109 passed/2 CUDA skips. Ruff, whitespace, new report
links and JSON validation passed. The existing benchmark reproducer and both
16-shot profile workloads were executed. The final profile CLI was also checked
at one shot after adding profiler-warning metadata. The complete numerical
suite is the preceding baseline result, not a fresh result for these test/
example-only edits.

Temporary logs: `/tmp/stn-followup-mpi-{2,3}-expanded.log`,
`/tmp/stn-followup-focused.log`, `/tmp/stn-followup-benchmark.log`,
`/tmp/stn-followup-helix.log`, `/tmp/stn-followup-helix-torch.log`.
