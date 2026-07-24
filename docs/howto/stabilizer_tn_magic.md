# Simulate magic circuits with STN

`MpsStabOptimizer` represents a state as `|psi> = C |nu>`: a Clifford tableau
`C` plus a coefficient MPS `|nu>`. Clifford gates change `C` and do not grow
`|nu>`. The choices below control the work caused by non-Clifford rotations.

## Start with exact constructive cooling

Keep `exact_cooling=True`, which is the default. Before a multi-site
non-Clifford Pauli rotation would build and apply its bond-dimension-two MPO,
the optimizer checks for a simple exact identity. If the frame image has a
separable stabilizer pivot, it performs the required local coefficient rotation
and moves the controlled-Pauli part into the tableau. That rotation therefore
does not increase the coefficient-MPS bond.

This is a deterministic applicability check, not an iterative optimization and
not a collection of SVD trials. It is cheap enough to leave on in ordinary
simulation. It simply does nothing when the required pivot is absent.

```python
from pepsy import MpsStabOptimizer, run_stabilizer_mps_stream

sim = MpsStabOptimizer(n_qubits, chi=64, exact_cooling=True)
sim.apply(circuit)
print(sim.exact_cooling_events)  # Empty when no rotation met the exact condition.
```

Set `exact_cooling=False` only to test the regular MPO rotation path or to
separate its effect from another benchmark.

## Use greedy cooling only at checkpoints

`disentangle_cliffords` has a different purpose. It examines nearby MPS bonds,
uses local Schmidt/SVD data to score two-qubit Clifford changes of basis, and
keeps an improving change. It can reduce a bond that has already grown, but it
does real local SVD work.

Call it after a sizeable non-Clifford block, a layer, or another natural
checkpoint. Do not call it after every T gate.

```python
sim.apply(first_block)
sim.disentangle_cliffords(sweeps=1)
sim.apply(second_block)
```

The operation preserves the represented physical state up to its chosen MPS
cutoff. `exact_cooling` prevents a special next update from growing; greedy
cooling tries to shrink an existing state. They complement one another rather
than being competing `auto` modes.

## Choose a magic-injection schedule

Injection handles `("t", q)`, `("tdg", q)`, and non-Clifford
`("rz", k * pi / 4, q)` entries. Other `rz` angles stay direct because preparing
their magic resource has the same non-Clifford cost as applying the rotation.

| Mode | Ancillas | When projection occurs | Best use |
| --- | --- | --- | --- |
| Direct | none | not applicable | Reference behavior or arbitrary rotations |
| Immediate injection | one reusable ancilla is enough | At every injected gate | Normal low-ancilla, steady-throughput simulation |
| Deferred MAST | one fresh ancilla per injected gate | One final projection phase | Separate low-bond replay from projection cost |

### Ask the Pepsy stream for settings advice

The broad advisor works directly on a Pepsy stream, independent of Stim:

```python
advice = MpsStabOptimizer.recommend_settings(
    circuit,
    n_qubits=n_qubits,
    ancilla_budget=1,
    goal="run",
)
print(advice.message)
print(advice.settings)
```

The returned `StabilizerMpsSettingsAdvice` contains a typed
`StreamAnalysisRecord`, the recommended execution method, constructor settings,
ancilla requirements, warnings, and the narrower magic-injection strategy used
inside the recommendation. It is still only advice; it never rewrites or runs
the circuit for you.

For a correctness smoke run before benchmarking, use the explicit runner:

```python
result = run_stabilizer_mps_stream(
    circuit,
    n_qubits=n_qubits,
    mode="direct",  # or "immediate", "deferred", or explicit "recommended"
)
print(result.final_bond, result.peak_bond, result.norm_diagnostics)
```

The default mode is direct. `mode="recommended"` is an explicit opt-in to the
mode from `recommend_settings`; the result records what actually ran, including
settings, replay/projection time, measurements, and injection reports.

### Ask only for the injection schedule

Use the gate stream to make the first decision without running the circuit:

```python
advice = MpsStabOptimizer.recommend_magic_strategy(
    circuit,
    ancilla_budget=1,
    prioritize_peak_bond=False,
)
print(advice["message"])
```

The report counts injectable T-family rotations, other named non-Clifford
rotations, and the small physical Clifford matrices emitted by Stim. Larger or
non-unitary matrices and coefficient-MPO entries remain opaque. Its conservative
default for a Clifford+T-like stream is `immediate`. It recommends `deferred`
only when `prioritize_peak_bond=True` and the supplied `ancilla_budget` is at
least the number of injectable gates. No budget of zero recommends `direct`.

This is not an `auto` mode. It never executes or rewrites the circuit. The user
still chooses `apply`, `with_injection`, or `with_deferred_injection` explicitly.

When the gate stream comes from Stim, ask after conversion and before `run()`.
Stim is just a parsing/onboarding adapter here; the queued Pepsy stream is what
the advisor inspects:

```python
sim = MpsStabOptimizer.from_stim(
    stim_circuit,
    stream_transform=lambda stream: [*stream, ("t", 0)],
)
print(sim.queued_recommend_settings(goal="run").message)
result = sim.run_queued_stream(mode="direct", goal="validate")
```

Native Stim circuits are Clifford/noise circuits, so T-family gates normally
arrive through a converted or `stream_transform`-augmented Pepsy stream.

### Immediate injection

This is the normal injection path. It measures each magic ancilla as soon as
its gate is teleported, resets it before reuse, and returns used ancillas to
`|0>` at the end, so one ancilla can serve the full circuit.

```python
sim = MpsStabOptimizer.with_injection(
    n_data,
    circuit,
    n_ancilla=1,
    chi=64,
    seed=7,
)
print(sim.last_immediate_injection_report)
```

Use more ancillas only when their physical placement helps the localizer; the
immediate scheduler chooses a nearby clean ancilla. Reserved ancillas are
validated before replay: they must be unique, in range, initially clean physical
`|0>` qubits, and ordinary stream entries must not touch them. The typed
`ImmediateInjectionReport` and `ImmediateProjectionRecord` objects are still
mapping-compatible, so both `report.projection_elapsed_s` and
`report["projection_elapsed_s"]` work.

### Deferred MAST injection

Deferred MAST keeps one magic ancilla for every injectable gate. During replay,
it prepares the magic state, performs the Clifford gadget, and places the known
branch correction at that gate's original location. It then performs the
physical basis-updating magic-register projections only after the circuit has
finished.

```python
sim = MpsStabOptimizer.with_deferred_injection(
    n_data,
    circuit,
    chi=64,
    seed=7,
    projection_order="middle_out",
)
report = sim.last_deferred_injection_report
print(report.pre_projection_peak_bond, report.projection_elapsed_s)
```

When `n_ancilla` is omitted, the constructor counts the injectable entries and
allocates exactly that many trailing ancillas. Passing `n_ancilla` is allowed,
but it must be at least that count. The circuit must not act on this reserved
magic register, and deferred ancillas are never recycled. The same unique,
in-range, initially-clean, ordinary-entry isolation checks are performed before
any replay mutation.

With `chi=None`, projection order changes the final cost, not the simulated
circuit. With a finite `chi`, it can also change where projection truncation is
incurred, which is another reason to compare the recorded projection data:

- `middle_out` is the default static center-out order in coefficient-MPS site
  order.
- `input` uses injection order.
- `min_span` is a greedy projection planner that repeatedly selects the current
  shortest tableau-frame MPS span.
- A sequence containing every used ancilla exactly once selects an explicit
  order.

`min_span` is unrelated to the greedy Clifford cooler. It plans only the final
measurement order and does not run disentangling SVDs.

## Compare the modes

The same deterministic T-doped Clifford circuit can be replayed through
``run_stabilizer_mps_stream`` in all three modes. Scaling harnesses are kept
outside the installed package so the distribution contains only supported API
examples and tests.

Read the output as follows:

- `peak` is the maximum coefficient-MPS bond reached anywhere in the run.
- `proj-bond` is the maximum bond during an injection projection phase.
- `proj[s]` is direct mode's zero projection time, immediate mode's accumulated
  in-stream projection time, or deferred mode's final projection time.
- `replay[s]` and `total[s]` show where deferred MAST moves work instead of
  pretending it disappears.

Run once with the default exact cooling for production behavior and once with
`--no-exact-cooling` when studying injection or MAST alone. A deferred run wins
when its lower replay-phase bond and the ability to schedule projections are
worth the extra ancillas and final projection work; it is not automatically
faster for every circuit.
