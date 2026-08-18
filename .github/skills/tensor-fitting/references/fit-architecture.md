# FIT architecture and review checklist

## File map

- `src/pepsy/fitting/local.py`: `FIT`, cached environments, one-/two-site
  active-window sweeps, native split, convergence, and timing.
- `src/pepsy/optimizers/mps/optimizer.py`: public FIT controls, target/layer
  construction, transaction fallback, infidelity, stabilization, and profiling.
- `src/pepsy/optimizers/mpo/optimizer.py`: MPO use of the shared FIT class.
- `src/pepsy/sampling/samplers.py`: full-chain `run_eff` consumer.
- `docs/api/optimizers/mps.md`: public behavior and algorithm choices.
- `tests/test_optimize_mps.py`: dense correctness, timing, growth, and
  complex64 regression coverage.
- `tests/test_symmetric_tensors.py`: native Z2/U1/U1U1 and fermionic coverage.

## Public controls

- `fit_block_size=2`: recommended two-site local wavefunction and native SVD.
- `fit_block_size=1`: fixed-rank compatibility update.
- `fit_sweep_sequence="RL"`: alternating left-to-right/right-to-left sweeps.
- `fit_min_iter`, `fit_rtol`, `fit_patience`: mode-neutral adaptive stopping
  controls for DMRG and mixed DMRG. Patience counts same-phase norm samples,
  so the default value 2 represents one stable comparison; legacy
  `mix_fit_*` names are deprecated.
- `fit_layer_size=N`: number of circuit gates in one target block; compatibility
  alias `k_2q_batch` remains.
- `target_cutoff=0.0`: target construction cutoff.
- `fit_target_strategy={"auto", "layered", "mps"}`: lazy exact gate layers
  for ordinary arrays versus a materialized/native-routed target MPS.
- `fit_single_pair_fast_path=True`: one update for an adjacent active pair.
- `cutoff`, `cutoff_mode`, `chi`: output split/truncation controls.
- `stabilize_unitary=True`: restore raw norm after recording compression loss
  for DMRG/FIT, mixed MPO compression, and standalone MPO/swap/perm/SVD modes,
  preventing deep complex64 underflow. Sampling and stabilization are
  independent controls. `fit_stabilize_unitary` remains a deprecated alias.
- `environment_strategy={"auto", "mps-direct", "symmray-native", "generic"}`
  on `FIT`: dense MPS specialization, native Symmray chain contraction, or the
  general conservative route.
- `timing_sync_device=True`: opt-in accelerator barriers for kernel-complete
  profiling; two-site timings include effective/SVD/writeback/environment.
  Resolve the accelerator once, wait on actual JAX stage results, and keep
  timing independent of `collect_split_diagnostics`.
- `local_norm_trace` stores one terminal retained-center scalar per completed
  sweep. With `finite_check=True`, active backend leaves are reduced natively
  and transferred with the optional rtol norm in one compact vector.

## Algorithm map

For local circuit compression, `MpsOptimizer` builds an exact target,
constructs `FIT(target, p=current, range_int=[xmin, xmax], inplace=True,
copy_target=False)`, then calls `run_gate`. Ordinary dense targets default to
small spatially split gate tensors layered over one owned MPS copy. If an
active block is still below its attainable chi, the optimizer also constructs
a separate exact MPS `target_support` source. FIT uses that source only for
local two-/three-site subspace expansion; it never installs it as `fit.p` and
never performs a global rank-chi target warm start. This avoids intermediate
target-MPS rank growth in the normal objective path and repeated full-state
copies. Symmray
uses its native auto-swap MPS target until graded layered targets have an
independently validated tag/phase contract. Left/right overlap environments
project the target onto the fixed
outside MPS. A two-site update contracts both target site tensors with those
environments, yielding the two physical groups and two outer virtual legs.
`Tensor.split` truncates only the middle bond and absorbs singular values in
the sweep direction.

During local expansion, FIT first performs the ordinary variational effective
tensor and SVD. When the active bond is below its target rank and `max_bond`,
the separate target-support factors provide the missing local Schmidt sectors,
with the bond capped at `max_bond`. The old current state remains the FIT
initial state, and once the bond reaches its rank ceiling subsequent sweeps are
ordinary fixed-rank rotations/refinements. Native Symmray inputs use their
graded local sector rules instead of dense support factors.

Fresh gate sweeps build fixed environments only beyond the first active
block. Completed block sweeps retain the minimal cumulative boundaries needed
by an equal-size reversed sweep. Immediately before a reversed one-site
transition, FIT extends that cache through one terminal tensor for a two-site
producer or two terminal tensors for a three-site producer; it never rebuilds
the complete fixed side. A terminal single-pair fast path needs no
active-window environment. Layered targets
cache boundary index names discovered from neighboring site tensors rather
than scanning the global target index map; this cache owns no tensor data.
The private `_SweepEnvironmentCache` keeps each completed sweep's boundary
mapping, direction, and block size together. It retains the mapping by
reference and performs compatibility checks once per sweep; update kernels
continue to use direct dictionary lookups with no wrapper in their hot loops.

An active interval containing one adjacent pair reaches its complete local
optimum after that split. MpsOptimizer enables the single-pair fast path even
when tolerance stopping is disabled. Thus `dmrg1`, `dmrg2`, and `dmrg3` each
perform one two-site update for a two-site window, skip one-site refinement,
and advance to the next gate regardless of the remaining `n_iter` budget.
After any final sweep, FIT's retained norm and center tensor are authoritative
for infidelity and unitary stabilization; recanonicalizing the interval is
redundant. Non-unitary scale control likewise normalizes that singleton center
in place when it remains inside the active interval; it must not move a valid
left endpoint to the right endpoint merely to extract the same norm.

For `dmrg1`, inspect the active and full-chain attainable rank targets before
starting FIT. An already-capped window starts with one-site updates. An
under-capacity non-adjacent window requires at least three requested sweeps:
two two-site growth sweeps followed by at least one one-site refinement sweep.
The two-site phase is bounded at two sweeps; it does not extend because of
rank stagnation. Once every full-chain bond reaches its physical/``chi``
ceiling, the optimizer latches one-site updates for later windows in the same
replay. Named `dmrg2` and `dmrg3` are fixed warm-up schedules: they perform
exactly `fit_adaptive_sweeps` two- or three-site sweeps (two by default), then
spend the remaining `n_iter` budget on one-site refinement subject to
`fit_rtol`. Generic `dmrg` remains available for rank-adaptive block
scheduling.

`run_eff` is a separate global full-chain fit used by boundary/sampling code.
Do not substitute it for the gate-window solver.
`run` and `run_eff` retain fixed-sweep behavior; PEPS boundary diagnostics
describe them as `fixed_sweeps` and use only coarse opt-in elapsed timing.

FIT timing records contain both compatibility totals and their named subsets.
`canonicalization_seconds` includes preparation and moving canonicalization;
legacy `environment_seconds` includes the complete post-writeback phase. Do
not sum every timing field. MpsOptimizer owns its temporary FIT instances, so
it moves their records into the replay collector and copies only at the public
getter boundary.

## Native tensor rule

Quimb and Symmray own contraction order, dual indices, fusion metadata, dummy
modes, graded signs, and block SVD. Do not convert native arrays with
`np.asarray`, `ar.to_numpy`, or `.to_dense()` in the solver. Host conversion is
allowed only for bounded diagnostics after native scalar reduction.

Validate at least one spinful `U1U1FermionicArray` result against a native MPO
reference, not merely for finite values.

## Literature boundary

- Stable FIT basis: Ayral et al., PRX Quantum 4, 020304 (2023),
  <https://doi.org/10.1103/PRXQuantum.4.020304>.

## Focused validation commands

Run with `source /Users/rezah/envs/genpy/bin/activate` first.

```bash
python -m pytest -q -m '' tests/test_optimize_mps.py
python -m pytest -q -m '' tests/test_symmetric_tensors.py -k 'mps_optimizer and (dmrg or two_site_fit)'
python -m pytest -q -m '' tests/test_optimize_mpo.py
python -m ruff check src/pepsy/fitting src/pepsy/optimizers/mps tests/test_optimize_mps.py tests/test_symmetric_tensors.py
```

For performance work, compare `environment_strategy="mps-direct"` with
`"generic"` on identical inputs and verify numerical equivalence before
claiming speedup. Compare `fit_target_strategy="layered"` and `"mps"` by exact
dense equality on small ordinary problems. Use `run(timing=True)` and inspect
`dmrg.target`, `dmrg.fit`, `dmrg.stabilize`, FIT call/record indices,
directions, pair phases, and failed partial sweeps. On an asynchronous GPU,
repeat with `timing_sync_device=True` before attributing time to a stage.
