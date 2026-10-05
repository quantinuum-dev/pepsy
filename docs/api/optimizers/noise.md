# `pepsy.optimizers.noise`

Pepsy's native noise design is **stream-local**: put stochastic entries directly
where the hardware schedule says the channel acts, then choose trajectory
sampling settings (`shots`, `seed`, independent/coalesced replay, and
`run_kwargs`) at the runner.

## Choose the workflow and interpret its result

| Need | Entry point | Retained state unit |
| --- | --- | --- |
| Stochastic entries or custom channels | `run_trajectory_shots` | One state per shot for independent replay, or one per coalesced leaf |
| Uniform post-gate Pauli faults | `run_noisy_shots` with `PauliErrorModel` | Independent shots |
| Stim circuit input | `compile_stim_circuit`, `stim_plan_to_gate_stream`, then `engine.compile` / `engine.run` | Unsampled independent Pauli noise in a reusable Pepsy stream |
| Native correlated/heralded Stim noise | `compile_stim_circuit`, then `run_stim_shots` | Native Stim shot records |
| Shared noisy prefixes | `run_coalesced_trajectory_shots` / `run_coalesced_stim_shots` | One state per leaf, with a shot count |
| Existing prepared optimizer | Shot-aware `optimizer.run(shots=...)` | `NoisyResult` wraps independent or coalesced storage |

These functions are available from `pepsy.optimizers.noise`. Compile a Stim
plan once and reuse it across runs. Compilation translates circuit syntax;
the runner makes random choices and evolves optimizer state. Stim is an
optional dependency loaded when compiling circuit input.

With retained results, `estimate(values)` expects one real scalar per stored
state: per shot for independent replay, per leaf for coalesced replay.
For independent trajectories it computes `sum(weight * value) / shots`;
for coalesced leaves it computes `sum(count * weight * value) / shots`.
The denominator is the represented shot count, not the sum of weights.
Weights are target/proposal likelihood ratios, equal to one for ordinary
unbiased draws. `effective_sample_size` accounts for those ratios and leaf
counts; it is not an error bar. An empty estimate returns NaN.

`NoisyResult.optimizers`, `counts`, and `weights` are aligned by retained
state. `shots` can exceed their length when branches are coalesced, and
retention settings can omit data. Use `retain="all"` when inspecting replay
history. Terminal bit samples have a separate row per sampled shot; their
`leaf_indices` identify the source branch.

Jump to [MPS shots](#shot-aware-mpsoptimizer-api),
[coalesced ensembles](#exact-coalesced-ensembles-for-rare-noise),
[MPI](#mpi-shot-ensembles), or
[custom channels](#user-defined-quantum-trajectories).

```python
stream = [
    ("h", 0),
    ("x_error", 1e-4, 0),
    ("cnot", 0, 1),
    ("depolarize2", 1e-3, 0, 1),
    ("t", 0),
    ("pauli_channel1", {"z": 2e-4}, 0),
]

result = pepsy.run_coalesced_trajectory_shots(
    lambda: pepsy.StabilizerMpsSimulator(2, chi=64),
    stream,
    shots=10_000,
    seed=7,
)
```

Equivalently, select the sampling strategy on the trajectory runner:

```python
result = pepsy.run_trajectory_shots(
    lambda: pepsy.StabilizerMpsSimulator(2, chi=64),
    stream,
    shots=10_000,
    seed=7,
    strategy="coalesced",   # or "independent" / "auto"
    max_branches=256,
)
```

Supported stochastic entries are:

- `("x_error", p, q)`, `("y_error", p, q)`, `("z_error", p, q)`
- `("depolarize1", p, q)`, `("depolarize2", p, q0, q1)`
- `("pauli_channel1", probs, q)`, where `probs` is `(p_x, p_y, p_z)` or a mapping
- `("pauli_channel2", probs, q0, q1)`, using Stim's 15 non-identity two-qubit Pauli labels
- `("amplitude_damping", gamma, q)`, sampled with the state-dependent trajectory runner

Stateful leakage entries are also Pepsy-native trajectory events:

- `("leakage", p, q)` or `("leak", p, q)` marks `q` leaked with probability `p`
- `("leakage_return", p, q)`, `("seepage", p, q)`, or `("unleak", p, q)`
  returns an already leaked qubit to a random computational-basis branch with
  probability `p`
- `("measure_leaked", q)` records a ternary PECOS/Selene-style result:
  `0` or `1` for a normal computational-basis measurement, and `2` when the
  trajectory state knows the qubit is leaked
- `("leak2depolar", enabled)` makes later `("leakage", p, q)` events use a
  full one-qubit depolarizing replacement instead of marking leakage
- `("leakage_depolarize", p, q)` applies that depolarizing replacement for a
  single event regardless of the current `leak2depolar` mode

Leakage state is carried per shot or coalesced leaf outside the qubit MPS. While
a qubit is leaked, ordinary gates touching it are suppressed. `reset` and
`measure_reset` clear the leakage flag; `measure_reset` first records the
leaked-qubit measurement as bit `1`. The sampled diagnostics live in
`TrajectoryShotResult.leakage_records` or
`CoalescedTrajectoryLeaf.leakage_records` as `LeakageRecord` objects. Leakage
streams support independent and coalesced replay; coalescing branches only when
the classical leakage outcome changes the represented state.

An MPS `cap` is a structural boundary in both strategies. It always removes
the selected site, including a leaked site's placeholder, then removes that
site's leakage flag and shifts higher logical labels down by one. Subsequent
gate suppression, reset, and measurement use those updated labels. Conditional
caps apply this update only to the selected branches; persistent layouts still
reject caps, while `perm` maintains its shortened logical mapping.

`PauliErrorModel` remains a convenience macro for clean deterministic streams.
It samples independent **physical Pauli trajectories**, not a density matrix.
Each non-identity X/Y/Z fault is inserted into a concrete gate stream after every
target of an ordinary gate. The resulting stream can be replayed by either
`MpsOptimizer` or `StabilizerMpsSimulator`; for STN, every sampled fault is a Clifford
that is absorbed by the Stim tableau. Do not mix this macro with stream-local
stochastic entries; use `run_trajectory_shots(...)` or
`run_coalesced_trajectory_shots(...)` when the stream already contains noise.

```python
import pepsy

noise = pepsy.PauliErrorModel.depolarizing(1e-3)
result = pepsy.run_noisy_shots(
    lambda: pepsy.StabilizerMpsSimulator(6, chi=32),
    gates,
    noise,
    shots=1_000,
    seed=7,
)

# Each trajectory can be measured/read independently.
samples = [sim.sample_bits(100, seed=shot) for shot, sim in enumerate(result.optimizers)]
```

For ordinary MPS replay, use the same function with a factory that constructs a
fresh state for every shot:

```python
result = pepsy.run_noisy_shots(
    lambda: pepsy.MpsOptimizer(initial_mps, chi=64, mode="direct"),
    gates,
    pepsy.PauliErrorModel.bit_flip(0.01),
    shots=100,
    seed=7,
)
```

`result.gate_streams` holds the sampled, replayable streams and `result.faults`
holds concise `(gate_index, site, pauli)` records. Use
`sample_noisy_gate_stream(...)` or `sample_noisy_gate_streams(...)` when only
stream construction is needed.

## Shot-aware `MpsOptimizer` API

`MpsOptimizer` now owns the ordinary and noisy replay APIs. Pass the initial
MPS and gate stream once; `run()` keeps the existing single-state behavior for
ordinary streams, while a stream-local noisy event or `shots > 1` dispatches to
the trajectory runner:

```python
simulator = pepsy.MpsOptimizer(
    initial_mps,
    gate_stream,
    chi=64,
    mode="direct",  # default compression algorithm
)
result = simulator.run(
    shots=10_000,
    strategy="auto",
    seed=7,
)
```

With `error_model=None`, stream-local stochastic entries use trajectory replay.
For the legacy clean-stream Pauli model, pass
`error_model=pepsy.PauliErrorModel.depolarizing(1e-3)` to `run`. The result is a
`NoisyResult` with `.optimizers`, `.counts`, `.gate_streams`, `.weights`,
`.shots`, and `.branches`; `.coalesced` identifies count-coalesced storage and
`.raw` retains the original runner result. `strategy="auto"` shares exact
prefixes under the `max_branches` safety cap. It performs a conservative
branch-cap preflight when possible, avoiding a partial coalesced replay that is
guaranteed to restart; the exact branch cap remains a hard safety limit.

The `retain` option is available on `MpsOptimizer`, `TreeNoisy`,
and the low-level shot runners. `retain="all"` keeps states and replay
metadata, `retain="final"` keeps only final states and weights, and
`retain="none"` keeps no final optimizer states. Use the last form for runs
whose outputs are consumed during execution rather than inspected afterward.

For repeated custom-runner use, `compile_trajectory_stream(gates)` returns a
backend-neutral `TrajectoryStreamPlan`. It parses stochastic entries once and
records ordinary-segment boundaries; live optimizers still perform their own
device/backend conversion.

The low-level factory-based functions remain available for custom optimizer
classes or non-MPS backends.

`TreeNoisy` exposes the same API for `TreeOptimizer` and accepts either an
entangled `TreeTensorNetwork` or a product MPS as its initial state. Its
`tree_settings` mapping contains `TreeOptimizer` options such as `chi`,
`layout`, `tree`, and `max_arity`; the gate stream is used when constructing an
automatic tree layout:

```python
simulator = pepsy.TreeNoisy(
    initial_state,
    gate_stream,
    tree_settings={"chi": 64, "layout_objective": "congestion"},
)
result = simulator.run(shots=10_000, strategy="auto", seed=7)
```

Tree and MPS optimizers share the logical feed-forward form
`("if", record, bit, action)`. Tree replay resolves the measurement record
before applying the action, including in count-coalesced branches. `NoisyResult`
is the generic result-facade name for all noisy replay backends.

## Exact coalesced ensembles for rare noise

When the total fault rate is small, avoid replaying the same no-error prefix
once per shot. `run_coalesced_noisy_shots(...)` holds one optimizer state per
distinct sampled branch and its number of represented shots. It runs an ideal
prefix once, samples exact multinomial branch counts at each Pauli channel,
and copies an MPS only when two nonempty branches genuinely diverge:

```python
result = pepsy.run_coalesced_noisy_shots(
    lambda: pepsy.StabilizerMpsSimulator(6, chi=32),
    gates,
    pepsy.PauliErrorModel.depolarizing(1e-3),
    shots=100_000,
    seed=7,
)

assert sum(result.counts) == 100_000
for leaf in result.leaves:
    print(leaf.count, leaf.faults)
```

The represented samples are still independent draws; only their identical
state evolution is shared. `run_coalesced_trajectory_shots(...)` provides the
same exact tree for `TrajectoryEvent` mixtures, state-dependent Kraus channels,
leakage, and mid-circuit controls. It branches `measure`, `reset`, and
`measure_reset` with exact binomial counts when a hidden measurement outcome
can change the pure state. Product-state resets use a one-leaf fast path; leaf
`measurements` records selected projective outcomes.

For ordinary MPS states, the reset fast path requires dimension-one bonds on
both sides of the logical target (after layout mapping). A numerical purity
tolerance cannot discard rare entangled outcomes. A product state with larger,
redundant bonds may therefore retain separate equivalent leaves. Leakage of an
entangled site also branches its hidden reset outcome before marking the
placeholder leaked. Branch limits count all live leaves, including retained
siblings and parents awaiting processing.

Dense MPS Kraus probabilities use projected amplitudes with the optimizer's
tracked center, preserving rare positive outcomes without cancellation in a
Gram expectation. Nonadjacent supports are gathered on a private MPS using
untruncated swaps; only the selected physical legs are fused, in the channel's
declared order. The normalized amplitude block is prepared once per channel,
and compatible outcomes are applied in bounded batches through Autoray on the
state's backend (including Torch CUDA and CuPy). Converted operator batches
are cached by channel, backend, dtype and device. State tensors and projected
amplitudes remain on that backend; only the outcome norm vector is transferred
for host sampling, in addition to the existing base-norm scalar check.
Scaled reductions and host float64 squaring preserve very small probabilities
without promoting the state arrays. Native symmetric states retain their
existing Gram route.
Control norms use the tracked center and the stored
exponent; exact and simple-update states retain their general norm paths.
Ordinary tree Kraus probabilities likewise use exact local expectations of
`K.conj().T @ K` on a private TTN wrapper. Probability evaluation does not
compress trial branches or apply unitary stabilization, and the common stored
exponent cancels from the ratios. Only the selected branch is replayed with
the requested direct/DMRG settings, then physically normalized, including
very small positive branches selected by importance sampling.

Every trajectory result exposes a lightweight `diagnostics` summary:

```python
print(result.diagnostics.max_kraus_probability_residual)
print(result.diagnostics.used_kraus_copy_fallback)
```

`max_kraus_probability_residual` is the largest deviation of the raw Kraus
branch probabilities from one before the sampler normalizes them. A small
residual can arise from floating-point contraction error; a large residual
indicates that the channel or probability calculation should be checked.
The exact local MPS/tree probability paths do not include trial-branch
truncation in these weights.

This is normally more useful than `torch.vmap` for rare faults: after a fault
or collapse, states have different tensor data and often different bond
profiles, while a coalesced no-error group stays one ordinary MPS/STN replay.
For terminal readout, call `result.sample_bits(...)` (or
`sample_coalesced_bits(result, ...)`). It invokes one batched `MpsSampler`
call per ordinary-MPS leaf and the STN tree sampler per STN leaf, returning
only terminal rows plus the source `leaf_indices`—never one optimizer per row:

```python
samples = result.sample_bits(seed=8)
assert samples.shots == result.shots
# samples.configs: (shots, n) computational-basis rows
# samples.leaf_indices: source coalesced leaf for each row
```

Conditional caps can leave different register lengths across leaves. In that
case `samples.configs` has width equal to the longest surviving register and
pads shorter rows on the right with `-1`. `samples.lengths[row]` gives the valid
prefix length, so use `samples.configs[row, :samples.lengths[row]]` for measured
bits. Columns use each leaf's current logical numbering after its caps, not
the original register labels. Lengths, leaf indices, and probabilities remain
aligned when rows are shuffled. Uniform-length batches keep their previous
configuration values and shape, and also expose `lengths`.

### Conservative automatic strategy

`run_noisy_shots(...)` keeps its backward-compatible independent replay by
default. For ordinary Pauli gate streams, `strategy="auto"` selects exact
count coalescing only when the expected per-shot number of non-identity faults
is small:

```python
result = pepsy.run_noisy_shots(
    factory,
    gates,
    pepsy.PauliErrorModel.depolarizing(1e-3),
    shots=512,
    strategy="auto",
    max_branches=128,
)
```

The automatic threshold is `lambda = (# noisy gate targets) *
(p_x + p_y + p_z) <= 0.1`. This is deliberately conservative: coalescing is
strongest when most shots take the no-fault path. An unforced `measure`,
`reset`, or `measure_reset` makes the policy choose independent trajectories,
because its physical collapse branches can dominate even when noise is rare.

The live-leaf cap is exact safety control, not truncation. If a selected
coalesced run would retain more than `max_branches`, automatic mode discards
the partial tree and restarts the whole ensemble independently. Pass
`strategy="coalesced"` to request coalescing explicitly; its same cap raises
instead of silently changing strategy. `auto_max_expected_faults` can tune the
default `0.1` threshold when profiling a different workload.

## Rare-event importance sampling

For a logical event much rarer than the physical noise rate, bias the proposal
distribution toward the relevant branches and retain an unbiased likelihood
ratio. The physical probabilities remain the `TrajectoryChannel` or
`PauliErrorModel` probabilities; only the sampling proposal changes:

```python
proposal = pepsy.ImportanceSamplingPolicy({
    12: {"I": 0.5, "X": 0.5},  # event 12: proposal, not physical probability
})
result = pepsy.run_trajectory_shots(
    factory,
    noisy_stream,
    shots=100_000,
    seed=7,
    importance_sampling=proposal,
    max_branches=256,
    max_branch_factor=4,
)
logical_error = [is_logical_error(sim) for sim in result.optimizers]
estimate = result.estimate(logical_error)
print(estimate, result.effective_sample_size)
```

The policy mapping can be label-based for every event, event-index keyed, or a
callable `(event_index, labels, target_probabilities, optimizer)`. Stateful
trajectory replay passes the current optimizer to the callback, including each
parent state in coalesced mixtures. Stream-only sampling has no optimizer state.
Every positive target branch must have positive proposal probability, including
arbitrarily rare outcomes. Impossible target branches must have zero proposal
probability; invalid support raises before applying a branch. `TrajectoryRecord`
exposes both
`probability` (physical) and `proposal_probability`, plus `likelihood_ratio`.
Coalesced leaves carry the product ratio in `leaf.weight`, and
`CoalescedTrajectoryResult.estimate(...)` includes leaf multiplicities. For the
Pauli convenience API, pass a proposal `PauliErrorModel` as
`importance_sampling` to `run_noisy_shots(...)` or
`run_coalesced_noisy_shots(...)`.

MPS Kraus probability contractions stay on the state's array backend, dtype,
and device; only scalar probabilities are read out for the host sampler.
Outcomes reuse the channel's base state norm. With `retain="none"`, independent
replay aggregates quality diagnostics without storing a snapshot per shot.

For coalesced dense Torch CUDA/CuPy states, compatible one-site channel parents
also share a bounded probability batch. Shape, dtype and device are checked
before stacking, and outcome weights keep the scaled-amplitude/host-float64
rule. Sampling order, branch budgets and importance ratios remain unchanged.
Callable proposals use the per-parent path. See the [MPS API](mps.md) for the
workspace scope and `max_kraus_parent_batch` diagnostic.

Library-generated NumPy operators with exactly zero imaginary components keep
a real MPS dtype, including real Torch states with bit-flip or amplitude-damping
noise. A genuinely complex operator on a real non-NumPy state raises a clear
error: initialize that state with `complex64` or `complex128` to run it.
Dense JAX MPS replay, canonicalization, readout and probability contractions
use scoped `highest` matmul precision. State dtype/device and the caller's JAX
precision setting are preserved when the operation returns or raises.

Both `exact` and `exact-batch` support coalesced measurement/reset and `auto`
selection. Their contracted states are rebuilt without truncation before
coalesced control probabilities are evaluated. Exact Kraus probabilities act
on full-state amplitudes and keep exact-mode canonical metadata separate.

To check the complete small-system ensemble and distinguish compression bias
from sampling error, run the integration references from the repository root
in your activated development environment:

```bash
python -m pytest -q -o addopts='' tests/test_mps_trajectory_density_reference.py
```

These compare all two-qubit Pauli observables with explicit density-matrix
evolution through successive channels, measurement and feed-forward. Ordinary
and importance-sampled trajectories use fixed seeds and six standard errors
computed from an independently enumerated proposal law. A four-qubit reference
varies `chi`, cutoff and shots, reconstructing the compressed model's ensemble
from branch probabilities separately from the random shot counts. This checks
specific small circuits; convergence on a larger circuit still requires its
own bond-dimension and sampling study.

`max_branches` bounds live coalesced states and `max_branch_factor` bounds the
number of nonempty children created by any one stochastic event. These are hard
safety budgets: a bounded coalesced run raises (or `strategy="auto"` restarts
independently) rather than pruning probability mass.

For stream-local `strategy="auto"`, preflight uses a structural history bound.
Fixed mixtures without importance sampling can additionally use a rare-event
estimate: one dominant history plus the expected number of shots departing
from it, bounded by the sum of per-event non-dominant probabilities. Coalescing
is attempted when that bound uses at most half the total branch budget.
Kraus channels, dynamic branching and proposal policies retain conservative
structural selection. The per-event budget is checked even when
`max_branches=None`; runtime caps remain authoritative. Planning never samples
a state or invokes an importance proposal. The Pauli `error_model` convenience
path retains its `auto_max_expected_faults` rule.

Local `MpsOptimizer.run` also selects workers from backend and workload
metadata; see [automatic MPS execution](mps.md). Explicit lower-level
`parallel_workers` retains its existing behavior.

## Deterministic parallel trajectories

Use `parallel_workers` directly on `run_trajectory_shots(...)` or
`run_noisy_shots(...)`, or call the explicit
`run_parallel_trajectory_shots(...)` / `run_parallel_noisy_shots(...)` helpers.
Independent shots receive their channel and optimizer child seeds before worker
dispatch, so changing the worker count preserves shot order and outcomes.
Coalesced execution keeps one deterministic branch-splitting stream and runs
independent live leaves concurrently:

```python
result = pepsy.run_trajectory_shots(
    factory,
    noisy_stream,
    shots=100_000,
    seed=7,
    strategy="coalesced",
    parallel_workers=8,
    parallel_backend="thread",
)
```

`parallel_backend="gpu"` also uses threads, intentionally keeping
Torch/CuPy/JAX objects in one process. It is a scheduling hint, not a device
selector: place the initial MPS on the desired device once (for example with
`mps.apply_to_arrays(py.build_backend(device="cuda"))`). `MpsOptimizer` then
lazily coerces ordinary dense gates to the live MPS backend, and converts
sub-MPO tensor payloads through `apply_to_arrays`; pre-converting gates is still
preferable when avoiding per-shot conversion overhead matters. This is
concurrent trajectory execution, not an unsafe shared mutable optimizer or an
automatic choice of Torch versus CuPy, CUDA device, or dtype.
The high-level `run_noisy_shots` and `run_trajectory_shots` helpers resolve
`strategy="auto"` consistently before local parallel dispatch. The lower-level
`run_parallel_noisy_shots` and `run_parallel_trajectory_shots` helpers still
expect an explicit `"independent"` or `"coalesced"` strategy.

## MPI shot ensembles

Use `MPIShotRunner` when the shot ensemble should be distributed across MPI
processes. It is an orchestration layer rather than another optimizer, so the
same factory works for `MpsOptimizer`, `StabilizerMpsSimulator`, `TreeOptimizer`,
and `StabilizerTreeSimulator`:

```python
import pepsy

runner = pepsy.MPIShotRunner(
    lambda: pepsy.StabilizerMpsSimulator(32, chi=64),
    noisy_stream,
)
result = runner.run(
    shots=1_000_000,
    seed=7,
    retain="final",
)
```

For a single ensemble, `run_mpi_shots` is the concise equivalent:

```python
result = pepsy.run_mpi_shots(
    lambda: pepsy.StabilizerMpsSimulator(32, chi=64),
    noisy_stream,
    shots=1_000_000,
    seed=7,
    retain="final",
)
```

Launch the program with `mpiexec` or `mpirun` after installing the optional
MPI profile (`pip install -e ".[mpi]"`). Every rank must construct and call
the runner collectively. Each rank owns complete local optimizer states; MPI
distributes global shot IDs, not pieces of an MPS or tree tensor network.
All ranks perform a synchronized preflight for runner arguments before
entering the shot collectives. Invalid input therefore raises an
`MPIShotError` on every rank; callers must still provide the same valid run
configuration on every rank.

MPI supports independent and rank-local coalesced execution. With
`strategy="independent"`, the global shot ID is part of the trajectory seed,
so changing the number of ranks does not change a shot's stochastic stream.
With `strategy="coalesced"`, each rank coalesces only its local batch; this is
useful for rare faults but is not rank-count invariant. Use `retain="none"`
when no post-run state is needed, or `retain="final"`/`"all"` before reducing
an observable:

Independent MPI execution supports all four optimizer families. Coalesced
execution additionally requires the backend's trajectory-copy contract; the
current coalesced backends are `MpsOptimizer`, `StabilizerMpsSimulator`, and
`TreeOptimizer`. Use independent MPI execution for `StabilizerTreeSimulator`.

The same orchestration is available directly from `MpsOptimizer.run`,
`StabilizerMpsSimulator.run`, `TreeOptimizer.run`, and `StabilizerTreeSimulator.run` by
passing `shots=...` and `mpi=...`. Direct calls create fresh per-shot copies
from the current optimizer state and leave the caller's state and queued
stream unchanged. Use `MPIShotRunner` when the factory/stream needs to be
shared across optimizer types or when constructing a reusable runner.

```python
def observable(optimizer):
    # Define this for the optimizer backend you are using.
    return measure_observable(optimizer)

estimate = result.reduce_mean(observable)
```

`reduce_mean` requires retained final states (`"final"` or `"all"`). With
`retain="all"`, `result.gather_records(root=0)` gathers trajectory records in
global shot order for independent runs; optimizer states are never gathered
automatically. For million-shot runs, evaluate an observable in bounded
chunks instead:

```python
streamed = runner.run(
    shots=1_000_000,
    seed=7,
    observable=measure_observable,
    chunk_size=2_048,
)
estimate = streamed.reduce_mean()
```

The callback is evaluated on each temporary optimizer and those states are
released after each chunk. `retain="none"` is required in this mode.
The callback may return a scalar or a numeric array, provided its shape is
consistent across ranks. `result.reduce_mean(...)` uses the same shot-count
denominator as the underlying result estimator while combining rank-local
multiplicities and importance weights. `result.reduce_sum(value)` combines
already-computed local scalars or arrays. The runner materializes the gate
stream once, so it can be reused for multiple collective runs. MPI is the
outer process-level parallelism; `local_workers` can optionally enable
the existing thread/GPU runner inside each rank. The direct runner defaults to
one local worker to avoid oversubscription; pass `local_workers="auto"` to
divide the host CPU allowance among ranks. `progress=True` reports one
rank-zero aggregate bar for independent ordinary, streaming, and checkpointed
runs; coalesced runs intentionally suppress shot-level progress because their
work is branch-based rather than one optimizer per shot.

For rank-scaling measurements, call `MPIShotRunner.run` from the workload you
intend to measure and vary only the MPI process count between runs. Record the
slowest-rank wall time and global completed-shot count. Use
`local_workers=1` for a process-only baseline and compare independent and local
coalesced execution separately; coalescing is not rank-count invariant.

On a scheduler, launch that workload with the site's supported MPI transport
and ensure Pepsy and its MPI-enabled Python environment are available on every
node. The repository's multi-process integration tests validate the API
contract without prescribing a cluster-specific launcher.

### Resuming a streaming run

For long bounded-memory observable runs, pass a checkpoint prefix. Each rank
atomically writes its own progress file after every completed chunk:

```python
checkpoint = "/scratch/pepsy/shots"
result = runner.run(
    shots=1_000_000,
    seed=7,
    observable=measure_observable,
    chunk_size=2_048,
    checkpoint_path=checkpoint,
)
```

If a rank fails, rerun the same collective call with `resume=True` and the
same checkpoint prefix, seed, shot count, strategy, chunk size, retention mode,
and MPI process count:

```python
result = runner.run(
    shots=1_000_000,
    seed=7,
    observable=measure_observable,
    chunk_size=2_048,
    checkpoint_path=checkpoint,
    resume=True,
)
estimate = result.reduce_mean()
```

`checkpoint_keep` controls how many historical per-rank snapshots are retained
in addition to the atomically updated latest file; the default is `2`. If the
latest file is unreadable, resume searches the retained snapshots from newest
to oldest. An existing checkpoint is never overwritten by a fresh run; use
`resume=True` or choose a new prefix.

Checkpointing also supports independent optimizer-state runs when the result
must retain states:

```python
retained = runner.run(
    shots=100_000,
    seed=7,
    retain="final",
    chunk_size=2_048,
    checkpoint_path="/scratch/pepsy/retained",
    checkpoint_keep=3,
)
```

This mode serializes each completed raw shot-result chunk plus a small index in
trusted per-rank checkpoint files, then merges the chunks when resuming. It requires
`strategy="independent"`, `retain="final"` or
`"all"`, and pickle-compatible optimizer states. Coalesced optimizer-state
checkpoints are intentionally rejected until branch identity and count merges
have a durable protocol. Checkpoint files must live on a reliable shared
filesystem, or on rank-local storage with the same path visible to each rank.
When a custom optimizer factory or observable callback changes independently of
the gate stream, pass the same stable `checkpoint_id` on every run to bind the
checkpoint to the application-level semantics.
A resumed result exposes `resumed=True`, keeps the prefix in
`result.checkpoint_path`, and publishes one `MPIRankDiagnostics` record per
rank through `result.rank_diagnostics` with shot ownership and elapsed time.
Set `checkpoint_sync=False` only when an external filesystem policy provides
the required durability; set `collect_diagnostics=False` to skip profiling
clock reads and the final diagnostics gather on very large communicators.
The `MpsOptimizer.run` facade defaults this option to `False`; enable it
explicitly to obtain the rank summaries described above.
After a successful run, call `result.cleanup_checkpoints()` collectively on
all ranks when the files are no longer needed.

## User-defined quantum trajectories

`TrajectoryEvent` is the general independent noise-simulation interface. Put
one directly inside an ordinary gate stream and run independently sampled shots
with `MpsOptimizer`, `TreeOptimizer`, or `StabilizerMpsSimulator`. It does not require
Stim or a density matrix.

Use a `mixture` for a user-defined random-unitary channel. Its outcomes have
explicit probabilities, so `sample_trajectory_stream(...)` can make a concrete
noisy stream without an optimizer:

```python
import numpy as np
import pepsy

x = np.array([[0, 1], [1, 0]], dtype=complex)
bit_flip = pepsy.TrajectoryChannel.mixture([
    ("I", 0.99, np.eye(2)),
    ("X", 0.01, x),
])
stream = [
    (pepsy.h(), 0),
    pepsy.TrajectoryEvent(bit_flip, 0),
]
sample = pepsy.sample_trajectory_stream(stream, seed=7)
```

Use `kraus` when the branch probability must be computed from the evolving
state. Each selected branch is normalized before the later stream entries run;
this supports non-Pauli channels such as amplitude damping:

```python
stream = [
    (pepsy.x(), 0),
    pepsy.TrajectoryEvent(pepsy.TrajectoryChannel.amplitude_damping(0.02), 0),
    (pepsy.h(), 0),
]
result = pepsy.run_trajectory_shots(
    lambda: pepsy.StabilizerMpsSimulator(1, chi=32),
    stream,
    shots=10_000,
    seed=7,
)

# One named result per noise event in each shot.
print(result.records[0])
```

`TrajectoryChannel.kraus([("no_jump", K0), ("jump", K1)])` accepts any
complete local qubit channel (`sum(K.conj().T @ K) == I`) on the corresponding
one- or multi-qubit `TrajectoryEvent` support. For ordinary MPS or TTN replay,
replace the factory above with a fresh `MpsOptimizer(initial_mps, ...)` or
`TreeOptimizer(...)` and pass its usual options through `run_kwargs`.

Tree FIT controls such as `fit_n_iter` are constructor settings, rather than
`run_kwargs` entries. The default is at most four iterations per fitted gate
window, each with two directional passes; see [tree FIT controls](tree_fit.md)
for early stopping and the single-node shortcut. Read the actual count with
`optimizer.get_fit_diagnostics()["iterations"]` after a fitted update.

For ordinary `MpsOptimizer`, Kraus normalization is tracked automatically in
the optimizer's norm-survival ledger. The selected branch event retains its
Born `branch_probability` and is marked as a `physical_boundary`; the expected
norm includes that probability, so physical renormalization is not reported as
compression infidelity. Inspect `optimizer.norm_diagnostics()` and
`optimizer.get_norm_events()` after independent or coalesced replay. Fidelity
tracking is automatic for `StabilizerMpsSimulator`; no tracking flag is needed.

For `StabilizerMpsSimulator`, a selected Kraus outcome is a
normalized trajectory boundary, just like a measurement/reset: its Born weight
is retained in the trajectory record but is not treated as compression loss.
`sim.norm_diagnostics()["norm"]` is the square root of the product of all
completed/current segment survivals, so it remains meaningful after state
renormalization. The older `total_norm_proxy` key remains as a compatibility
alias.

## Reading a Stim circuit

### Translate, analyze, prepare, run

For independent Pauli/depolarizing channels, keep one **unsampled** Pepsy
stream and let the simulator own all shots:

```python
from pepsy.optimizers import (
    StabilizerMpsSimulator, compile_stim_circuit, stim_plan_to_gate_stream,
    stim_readout_parities,
)

stim_plan = compile_stim_circuit("""
R 0 1
H 0
CX 0 1
X_ERROR(0.02) 1
M 0 1
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
""")
gate_stream = stim_plan_to_gate_stream(stim_plan)

# Optional analysis: no state mutation, execution or random draws.
analysis = StabilizerMpsSimulator.analyze_stream(
    gate_stream, n_qubits=stim_plan.num_qubits,
)
engine = StabilizerMpsSimulator(stim_plan.num_qubits, chi=128, mode="direct")
engine.compile(gate_stream)
result = engine.run(shots=100, strategy="auto", workers=1, seed=7)
detectors, observables = stim_readout_parities(stim_plan, result.measurements)
counts = result.counts
```

The stages have distinct responsibilities:

- `compile_stim_circuit` expands repeat blocks and parses Stim quantum
  operations, noise and readout annotations into a `StimCircuitPlan`. Passing
  an existing plan returns that same object. It does not sample or simulate.
- `stim_plan_to_gate_stream` accepts that plan and returns an ordinary tuple
  of Pepsy entries. Gates, resets, measurements and feed-forward retain their
  order; independent noise remains stochastic. It does not draw faults.
- `analyze_stream` is optional inspection. Its `trajectory_entries` and
  `clifford_trajectory_entries` counts distinguish known stochastic channels
  from opaque entries, including their touched qubits.
- `engine.compile` prepares the execution queue and validates matrix
  backend/dtype/device, returning the engine. `set_gates` remains equivalent.
  For reuse across engines, first call `compile_trajectory_stream(gate_stream)`
  and pass that `TrajectoryStreamPlan` to `engine.compile`; the plan is reused
  by identity and is available as `engine.compiled_stream`. Threaded workers
  reuse it too. This preparation does not mutate the quantum state or RNG.
- `engine.run(shots=..., strategy=..., workers=...)` samples and executes
  trajectories internally. Coalesced rows carry multiplicities in `counts`;
  their count sum is the number of shots, not necessarily their row count.

Translation generates NumPy complex128 ideal matrices. Explicitly convert
matrix payloads, including nested feed-forward actions, before installing a
stream on a different backend/dtype/device; preparation checks rather than
silently converts user-provided streams.

`stim_readout_parities` returns raw uint8 arrays with one row per retained
shot/leaf. It excludes hidden reset records and XOR-combines repeated
`OBSERVABLE_INCLUDE` entries into columns indexed by logical observable ID.
These are raw parities, without Stim reference-sample subtraction or decoder
correction. Use `result.counts` when averaging coalesced rows.

The unsampled translation supports `X_ERROR`, `Y_ERROR`, `Z_ERROR`,
`DEPOLARIZE1`, `DEPOLARIZE2`, `PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`, and identity
noise. Correlated-error chains and heralded channels raise explicitly at this
translation stage; use the native Stim runners below for those channels.
The compiler rejects nonzero measurement readout-error arguments, inverted
measurement results and `MPAD`, whose classical semantics the current replay
engines cannot represent. These features are never silently discarded.
Empty detector/observable annotations are retained as zero parities, and
grouped record-controlled Pauli gates are lowered pair by pair.

### Native Stim shot runners

`compile_stim_circuit(...)` accepts `stim.Circuit` or Stim source text. It
compiles one- and two-qubit Clifford gates, Pauli measurements/resets, and
**every native Stim stochastic error instruction**, then reuses that plan for
all shots:

```python
circuit = """
H 0
CX 0 1
PAULI_CHANNEL_2(0,0,0, 0,0.01,0,0, 0,0,0,0, 0,0,0,0) 0 1
HERALDED_PAULI_CHANNEL_1(0, 0, 0, 0.02) 0
"""

result = pepsy.run_stim_shots(
    lambda: pepsy.StabilizerMpsSimulator(2), circuit, shots=10_000, seed=7,
)
print(result.faults[0])
print(result.heralds[0])
```

`run_coalesced_stim_shots(...)` has the same output shape as the coalesced
ordinary trajectory runner and supports the complete compiled native Stim
noise set, including two-qubit, heralded, and `E`/`ELSE_CORRELATED_ERROR`
chains. It shares all ideal segments and records per-leaf Pauli faults and
herald bits. `TrajectoryShotResult.measurements` and
`StimShotResult.measurements` expose structured Pauli outcomes with event
metadata. Detector and logical-observable annotations are compiled into the
plan and resolved as `result.syndromes` and `result.observables`; coalesced
Stim results expose the same records once per leaf, alongside each leaf's
count.

Measurement-record feed-forward is also supported: `CX/CY/CZ rec[k] q` is
lowered to `("if", k, bit, action)`, and the ordinary MPS/STN stream form is
available directly. In a hand-written stream, `("if", record, bit, action)`
uses computational bits (`+1 -> 0`, `-1 -> 1`), with negative records counting
back from the latest measurement. Independent and coalesced trajectory replay
resolve the predicate separately for every shot/leaf before applying `action`.
Selected actions inherit the configured replay/FIT settings. Trajectory
runners retain the concrete executed action rather than both an action and its
conditional wrapper; nested conditional controls use normal measurement,
reset, and cap handling, including coalesced branching and leakage updates.
The `PauliErrorModel` convenience macro treats the conditional wrapper as a
control event and automatically samples ordinary named or matrix gate actions
only on the branch where the predicate is true. Conditional measurements,
resets, nested controls, and sub-MPO actions are left unchanged; use explicit
stream-local noise entries when those operations need a particular noisy
channel. `k=-1` means the latest measurement; general record-to-record
arithmetic is intentionally not lowered.

Supported Stim error channels are `X_ERROR`, `Y_ERROR`, `Z_ERROR`,
`DEPOLARIZE1`, `DEPOLARIZE2`, `PAULI_CHANNEL_1`, `PAULI_CHANNEL_2`,
`CORRELATED_ERROR`/`E`, `ELSE_CORRELATED_ERROR`, `HERALDED_ERASE`,
`HERALDED_PAULI_CHANNEL_1`, `I_ERROR`, and `II_ERROR`. Stim itself only
represents Pauli noise, so amplitude damping is not a missing Stim channel.

## Coherent crosstalk and truncation studies

`CoherentCrosstalkModel` inserts coherent nearest-neighbour `ZZ` rotations
after selected two-qubit gates. The emitted `rzz` angle follows Pepsy's
`exp(-i theta P / 2)` convention; `sign_mode="random_sign"` provides a
reproducible random-sign comparison when a seed is supplied:

```python
model = pepsy.CoherentCrosstalkModel(
    theta=0.01,
    adjacency={0: (1,), 1: (0, 2), 2: (1,)},
    sign_mode="random_sign",
)
noisy_stream = model.transform(gates, seed=7)
```

For coherent-noise and QEC studies, both STN frontends provide
`StabilizerMpsSimulator.truncation_convergence(...)` and
`StabilizerTreeSimulator.truncation_convergence(...)`. They replay the same stream
at several `chi` values and report peak bond, norm diagnostics, and an
optional observable. `chi=None` is the lossless reference up to the configured
cutoff.

For an end-to-end repeated-check validation, use the public
`compile_stim_circuit`, `run_stim_shots`, and
`run_stabilizer_tree_stream` APIs directly. Keep performance experiments in
the external benchmark workspace so the package remains focused on reusable
simulation APIs.
