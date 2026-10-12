# Stabilizer trajectory follow-up evidence — 2026-10-05

Baseline: `develop`, `c649aaa`. This follow-up adds validation and benchmarks;
it does not change production simulation, compression or sampling rules.

## Real MPI execution

The first two-rank run found one stale success-case parameter, `dmrg1`, which
ordinary MPS has removed. The mode matrix already included its supported
replacement `dmrg`; the removed spelling is now excluded from the success
matrix. Existing ordinary-MPS tests still cover rejection of that spelling.
After this correction the existing suite passed 24 cases on each rank with
both two and three processes.

Added NumPy and Torch CPU complex128 mixed-trajectory cases: Clifford+T,
amplitude damping with uniform importance proposals, measurement, feedback
and reset. Seven shots are partitioned across actual ranks and compared with
`MPI.COMM_SELF` using global shot IDs. Conditional state fidelity, normalized
norms, measurement outcomes, noise labels, physical probabilities and weights
agree. The initial certified measurement also records native Stim routing.

The expanded integration module passes **26 tests on every rank** under both
`mpiexec -n 2` and `mpiexec -n 3`. These are two runs of the same 26 cases;
do not add rank counts as independent coverage. MPI unit/fingerprint and
STN noisy-reference/execution suites pass **109 tests, 2 CUDA skips**.

Open MPI 5.0.10 is installed. Its local sockets require execution outside the
filesystem sandbox. No remote cluster or GPU MPI execution was tested.

## Measurement routing benchmark

Reproducer: [stabilizer_measurement_benchmark.py](../../../examples/stabilizer_measurement_benchmark.py).
Run with Pepsy's stabilizer and Torch extras installed:

```bash
python examples/stabilizer_measurement_benchmark.py
```

Six qubits, 16 independent shots, one CPU worker, seed 37, complex128,
`mode="direct"`, `chi=64`, cutoff zero, exact cooling disabled. Each row is
the median of three full runs after one warm-up; construction/compilation
is outside run timing. The fixed comparison supplies explicit `False` for
visible measurement basis absorption; resets keep their normal routing.
Both variants have identical recorded outcomes and final state fidelity
within 1e-11. The cap exceeds the maximum six-qubit Schmidt rank.

| Circuit | Backend | Native selection (s) | Fixed measurements (s) | Ratio |
| --- | --- | ---: | ---: | ---: |
| Repeated Clifford GHZ, measure, reset | NumPy | 0.1294 | 0.5158 | 3.99 |
| Repeated Clifford GHZ, measure, reset | Torch CPU | 0.1615 | 0.7716 | 4.78 |
| Repeated magic/entangling rotations, measure, reset | NumPy | 0.0986 | 0.1518 | 1.54 |
| Repeated magic/entangling rotations, measure, reset | Torch CPU | 0.1151 | 0.1994 | 1.73 |

In the Clifford case all 576 logical collapse events use Stim. The coefficient
bond remains one, versus eight for the fixed comparison. In the mixed case
96 of 192 events use a separated Stim region and 96 fall back to MPS; both
variants finish with coefficient bond two. These are small CPU workload
measurements, not an MpsOptimizer/STN throughput comparison or a universal
speedup claim. Raw observations are in the [benchmark record](2026-10-05-stabilizer-followup-benchmark.json).

## Archived Helix replay

Recovered gate-only input SHA-256:
`18c6f2bbc9ad59bbf2e7d805c07b21e3c8be3ef974ada74ab26f25c850067149`.
52 qubits, 550 visible measurements and 530 reset operations per shot.
One Stim compilation and stream extraction; Torch payloads are explicitly
prepared once before installing the stream. Eight shots, automatic strategy
and workers, `chi=1024`, direct mode, retained histories:

- NumPy: 23.84 seconds; Torch CPU complex128: 25.72 seconds.
- Both select coalescing with one worker and retain eight leaves.
- All 8,640 logical collapse events use Stim, with no MPS fallback.
- Every shot retains 550 visible measurements; coefficient bond remains one.

These are single elapsed-time observations without layout optimization or a
performance baseline. The original noisy builders, detector annotations and
logical-observable annotations are absent. This does not validate the original
experiment's logical error rate.

## What this supports next

- **Validated:** real CPU MPI partition/replay parity for the mixed STN path;
  observed native-routing benefit and a larger archived gate replay.
- **Not available:** CUDA and Torch MPS devices in this environment. No GPU
  performance or cross-parent frame batching result follows from CPU checks.
- **Measured:** the [mixed magic/noise profile](2026-10-05-stabilizer-mixed-profile.md)
  identifies coefficient application/compression as the leading cost in the
  entangling workload. Ordinary MPS local Kraus/gate batching cannot simply be applied to
  physical STN gates whose frame support is nonlocal and branch-dependent.
- **Deferred by user:** broader exact stabilizer-region certification. The current
  conservative dimension-one/Pauli-product certificate deliberately rejects
  uncertified entanglement; these timings do not justify tolerance-based
  stabilizer recognition or ignoring external coefficient edits.

Installed dependency versions are unchanged from the
[preceding audit](2026-10-05-stabilizer-trajectory-parity.md#upstream-audit).
The earlier full result, 7170 passed/525 skipped, belongs to `c649aaa`.
No complete single-process rerun was needed for this test-only follow-up;
real multi-process execution and focused checks above are the new evidence.
