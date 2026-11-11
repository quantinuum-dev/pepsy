# Automatic independent-cluster scheduling

`batch_size="auto"` discovers ready clusters from the gate stream and lattice,
then chooses compatible widths from current ranks, costs and memory estimates.
The default remains sequential. This extends the
[consecutive batch prototype](2026-11-05-independent-cluster-batches.md).

## Dependency rule

Each gate writes its active site tensors, A. A pair reads its entire direct
cluster, C; a one-site action reads/writes its one tensor. Selected pair
clusters must be disjoint. A later candidate j may cross preceding skipped
or pending actions S only when

$$
{\color{purple}A_j}\cap\bigcup_{i\in S}{\color{blue}C_i}=\varnothing,
\qquad
{\color{blue}C_j}\cap\bigcup_{i\in S}{\color{purple}A_i}=\varnothing.
$$

Both dependencies matter: a gate can change a spectator used by another
gate's environment even when their active bonds differ. Every skipped action
contributes its read/write footprint, including shape mismatches and one-site
gates. Active-gauge changes are covered by their endpoint write sets.

Ready fits can be prepared ahead of independent intervening actions. Their
endpoint/gauge/exponent mutations, trackers and callbacks are committed in
original order. A previously prepared result is excluded from later batches
but still constrains their dependencies. Other ready groups may be evaluated
while that result waits for its commit position. Every `apply(start, stop)`
boundary is a hard barrier. Callbacks must remain observational.

This preserves the original local approximation, conditioning, rank and
stopping rules. It does not replace dependent updates with stale environments,
change target gates, fuse gates or infer independence from physical gate
commutativity alone. Geometry-only serial windows are rejected before making
expensive numerical pair copies.

## Choosing the width

The initial ceiling is the minimum of 16, the number of scheduled pair actions,
and the site-tensor count divided by the smallest scheduled patch. This is a
conservative disjoint-patch capacity estimate, not a packing optimum. Search
examines at most `max(64, 8 * site_tensor_count)` actions within the current
apply interval. Actual tensor layouts, program graphs and reduced dimensions
must agree within each group.

The current preparation routine gauge-inserts a private **full network** for
each pair. Width selection therefore includes its lattice-dependent storage,
in addition to Cotengra intermediates, environment/operand stacks, targets and
ALS Gram/Cholesky/inverse buffers. Snapshot allowance is larger when the
prepared arrays carry gradients. Ignoring these full-network copies would
underestimate large-lattice batches.

Auto reserves at most 2 GiB and one quarter of currently free CUDA memory, or
512 MiB and one eighth of available host memory. It refreshes the allowance
while earlier fits remain pending. Explicit contraction FLOP and peak limits
can shrink the width further. When only one update fits, replay stays serial;
a single contraction that exceeds its own explicit budget still raises.
These are working-memory estimates, not an allocator/OOM guarantee or a
performance-optimal batch-size search. Explicit integer caps keep their
consecutive grouping and explicit combined-budget checks. Eager is required
for automatic or integer batching above one.

Inspect `plan.info`: `batch_size`, `auto_batch_limit`, `auto_workspace_bytes`,
`auto_estimated_member_bytes`, `cluster_batches`, `batched_pair_updates`,
`lookahead_pair_updates`, `largest_batch` and `batch_histogram`. No numerical
state is retained in these diagnostics.

## Real-stream geometry scan

Read the complete homogeneous depth20 Eli middle-out stream at T=6/dt=.05.
Preserve its source-slice barriers. The following is a geometry-only scheduling
projection: it performs no tensor evolution, rank/shape checks or memory fit.

| Geometry | Pair actions | Consecutive cap4 batched actions | Auto batched actions | Auto ceiling |
| --- | ---: | ---: | ---: | ---: |
| 5x5 corner-removed | 5,040 | 940 (18.7%) | 1,360 (27.0%) | 6 |
| 10x10 corner-removed | 24,640 | 4,720 (19.2%) | 6,200 (25.2%) | 16 |

The 5x5 scan finds 680 auto groups of two. The 10x10 scan finds 2,420 groups
of two, 140 of three, 160 of four and 60 of five. Current shapes and memory
can change the actual groups; these counts do not establish throughput gains.
The bounded greedy heuristic does not claim a globally optimal schedule.

## Bounded timing and numerical checks

Synthetic complex128 PEPOs on 3x14, seed 931, with six disjoint interior
six-site patches and five intervening one-site actions at (0,0), outside all
six patches. An integer consecutive policy cannot group across these actions;
auto chooses width six. Same one-thread/greedy/native-Torch/`gesvd` controls as
the earlier batch record; cap6/tolerance1e-8 ALS and local fidelity enabled.
Alternate sequential/auto over four repetitions, discard one cold repetition
and report medians of three. Include input copies, preparation, contractions,
ALS, diagnostics and writeback; synchronize CUDA. Plan construction is excluded.

| Device / D | Sequential ms | Auto ms | Ratio |
| --- | ---: | ---: | ---: |
| CPU / 2 | 115.583 | 81.160 | 1.424x |
| CUDA / 4 | 226.838 | 161.172 | 1.407x |
| CUDA / 6 | 353.241 | 286.057 | 1.235x |
| CUDA / 8 | 943.191 | 898.010 | 1.050x |

All pair iteration counts, normalized errors and local fidelities match.
A directly contracted physical matrix element agrees within relative 1e-14.
These seeded states are not evolved high-D circuit results. Hardware is the
shared Xeon W-3375 / RTX A5000 workstation, so timings remain workload-specific.

A separate read-only check applies the first 96 real scheduled actions to
the saved corrected 23-spin homogeneous depth20 chi6 final operator. It is an
additional diagnostic, not a new T=6 run or the original circuit's final cost.
This window has 36 pair actions and no compatible independent batch. Auto
falls back to serial, with median 2422.434 ms versus 2400.584 ms sequential
(about 0.9% overhead). Pair diagnostics match, and the separate MPS-cap18
diagnostic readout is identical at 0.15328406188101884. No speedup is claimed
for this window.

## Validation and evidence

The 35-case batch suite covers automatic and fixed widths, CPU/CUDA, both
complex precisions, Hermitian/PSD environments, physical reconstruction,
independent stopping, rank growth, source/one-site barriers, shape skips,
transitive conflicts, nested pending batches, lattice-dependent widths,
workspace/cost shrinking and full-network snapshot memory. Gate and gauge
derivatives match sequential replay and finite differences with native and
stabilized linalg. Final integrated validation is recorded in the
[handoff](../../../history/2026-11-05-auto-cluster-batches.md).

Classification remains **prototype**. The earlier active task's
[upstream audit and installed versions](2026-11-05-independent-cluster-batches.md#upstream-and-validation)
remain unchanged; no dependency, installed source or global registration
changed. The new path adds only topology/resource scheduling around the
already checked dense batch kernels. Native symmetry remains on its original
route, and the previously recorded optional-path defects remain separate.

Reproduction/evidence under `/tmp`:

- `pepsy-cluster-auto-bench-20261105-final.py` and matching JSON/log: synthetic
  six-patch execution with independent intervening one-site actions.
- `pepsy-cluster-auto-stream-20261105-final.py` and matching JSON/log: bounded
  saved-state real-schedule replay and MPS diagnostic.
- `pepsy-cluster-auto-schedule-scan-20261105.py` and
  `pepsy-cluster-auto-schedules-final-20261105.json`: complete stream geometry.
- `pepsy-cluster-auto-final-tests.log`: affected regression selection.

Activate the selected Python environment and put local `pepsy/src` first on
`PYTHONPATH`. The real-stream scripts also read the existing sibling workflow,
Gaugy source and archived checkpoint. They write only temporary diagnostics;
no production job, saved state or frozen source snapshot was modified.
