# Mixed magic/noise replay profile — 2026-10-05

Scope: profile mixed replay and commit the preceding MPI/benchmark work.
Broader stabilizer-region recognition is explicitly not a current priority.
Baseline: `develop`, `c649aaa`; no production algorithms changed.

## Reproduce

With Pepsy stabilizer and Torch extras installed:

```bash
python examples/stabilizer_trajectory_profile.py --output /tmp/stn-profile.json
```

The [reproducer](../../../examples/stabilizer_trajectory_profile.py) compiles
each stream once, warms up once, and measures three uninstrumented runs.
It separately collects cProfile observations and temporary nested wall timers.
Temporary wrappers are restored after replay, and profiling never changes
compression settings or enables simulator diagnostics. Worker count is one;
these timers target synchronous CPU replay, not threaded or GPU execution.

Two synthetic workloads use Clifford+T, non-Clifford XX rotations, amplitude
damping and bit-flip noise. The six-qubit workload interleaves measurement,
feedback and reset over four rounds. The twelve-qubit workload has four
entangling layers and terminal Z measurements. Both use complex128, 16 shots,
seed 37, direct mode, chi=32, cutoff 1e-12 and exact cooling disabled.
Recorded peak coefficient bonds are respectively two and 32.

## Uninstrumented timings

| Workload | Backend | Independent median (s) | Coalesced median (s) | Ratio |
| --- | --- | ---: | ---: | ---: |
| Measurement-heavy, 6 qubits | NumPy | 0.2606 | 0.2624 | 0.99 |
| Measurement-heavy, 6 qubits | Torch CPU | 0.3473 | 0.3401 | 1.02 |
| Entangling, 12 qubits | NumPy | 1.7875 | 0.9558 | 1.87 |
| Entangling, 12 qubits | Torch CPU | 2.6252 | 1.3724 | 1.91 |

Sharing saves substantial earlier evolution in the entangling circuit.
It gives no meaningful gain in the small measurement-heavy case despite
reducing sampled Kraus applications. Each case still retains 16 leaves;
the benefit comes from intermediate shared work. These are single-machine,
small-workload observations without hardware performance guarantees.

## Where time goes

Percentages below come from the separate nested wall-timed run. Inclusive
time contains child stages and **must not be summed**. The record also gives
exclusive time, subtracting timed children, and validates that its sum fits
inside the measured replay time. Costs inside untimed children remain with
their nearest timed parent; this is a stage attribution, not an allocation
or kernel profiler.

- **Entangling independent replay:** coefficient `gate_with_submpo_` application
  and compression consumes 70.1% on NumPy and 72.2% on Torch. It is called 965
  times. Coalescing reduces these calls to 401/399 and this stage to about
  62% of its shorter run. Kraus probability evaluation consumes only 1.8%/1.6%
  independently, around 1% after coalescing.
- **Measurement-heavy independent replay:** `_apply_operator_sum` consumes
  24.6%/31.9%. Source review shows that dense gates mapped to zero or one
  coefficient-site support currently reach the balanced whole-MPS branch sum,
  which copies/adds/compresses MPS branches. Sparse multisite sums already use
  an exact sub-MPO. Kraus probabilities are 6.2%/5.1%, and Pauli expectations
  are 17.0%/13.1% (these expectation costs can be nested in other stages).
- **Measurement-heavy coalescing:** Kraus probability calls fall from 64 to 37,
  but Pauli expectation calls rise from 280 to 398 and exact product-state
  checks from 1,004 to 1,810. These checks consume 17.2%/20.0% of replay;
  expectation evaluation consumes 28.3%/22.1%. Source review shows separate
  measurement probability probes and forced collapse both recheck eligibility.
  Sharing fewer Kraus applications therefore does not imply a faster total run.

The [machine-readable record](2026-10-05-stabilizer-mixed-profile.json) preserves
all uninstrumented repetitions and inclusive/exclusive stage counts/times.

## Profiler caveat

The installed CPython 3.12.14 cProfile reports some outer replay calls with zero
primitive calls and zero cumulative time; its summed self time can exceed the
measured wall interval. A separate two-qubit probe reproduced the zero outer
`run` cumulative time. Do not interpret those cumulative totals as a time budget.
The script flags zero-primitive-call observations and preserves them for audit;
the percentages above use independent wall timers instead. A similar symptom
is documented in [CPython issue 106152](https://github.com/python/cpython/issues/106152),
but that historical issue does not establish the cause in this installation.

## Validation and next implementation candidates

Instrumented and uninstrumented runs agree in recorded outcomes, noise labels,
weights and coalesced counts for each backend/strategy. All retained states
normalize to one and their final physical Z expectations agree with terminal
readout. This checks that instrumentation is observational; it is not a
cutoff-zero ideal-circuit fidelity check of the capped twelve-qubit workload.
NumPy and Torch need not produce identical coalesced branch histories at
floating-point probability boundaries.

1. **Small mapped supports:** evaluate a zero-/one-site coefficient operator
   sum as a scalar/local matrix instead of copying and compressing the whole
   MPS. Guard this by exact mapped support and preserve dtype, device, gradients,
   phases, the norm ledger and rare Kraus support. This remains a proposal.
2. **Entangling replay:** profile reusable sub-MPO construction and canonical
   moves before adding GPU batches. Reusing compatible work must retain the
   requested compression method/cutoff and branch-local state.
3. **Coalesced measurement:** consider operation-scoped probability/certificate
   reuse while the state is unchanged. Do not cache across arbitrary external
   tensor edits or callbacks. This reduces repeated work on the existing
   certificate, without expanding recognition.

No optimization was implemented by this profile. GPU timing remains unavailable.
Versions are unchanged from the [preceding audit](2026-10-05-stabilizer-trajectory-parity.md#upstream-audit).
