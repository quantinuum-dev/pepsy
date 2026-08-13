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
- `stabilize_unitary=True`: restore raw norm after recording compression
  loss for FIT and mixed-mode MPO compression, preventing deep complex64
  underflow. `fit_stabilize_unitary` remains a deprecated compatibility alias.
- `environment_strategy={"auto", "mps-direct", "generic"}` on `FIT`: dense
  MPS specialization versus general/native-safe contraction.
- `timing_sync_device=True`: opt-in accelerator barriers for kernel-complete
  profiling; two-site timings include effective/SVD/writeback/environment.

## Algorithm map

For local circuit compression, `MpsOptimizer` builds an exact target,
constructs `FIT(target, p=current, range_int=[xmin, xmax], inplace=True,
copy_target=False)`, then calls `run_gate`. Ordinary dense targets default to
small spatially split gate tensors layered over one owned MPS copy. This avoids
intermediate target-MPS rank growth and repeated full-state copies. Symmray
uses its native auto-swap MPS target until graded layered targets have an
independently validated tag/phase contract. Left/right overlap environments
project the target onto the fixed
outside MPS. A two-site update contracts both target site tensors with those
environments, yielding the two physical groups and two outer virtual legs.
`Tensor.split` truncates only the middle bond and absorbs singular values in
the sweep direction.

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
when tolerance stopping is disabled. Thus `dmrg1` and `dmrg2` each perform one
two-site update for a two-site window, skip one-site refinement, and advance
to the next gate regardless of the remaining `n_iter` budget. After any final
sweep, FIT's retained norm and center tensor are authoritative for infidelity
and unitary stabilization; recanonicalizing the interval is redundant.

For `dmrg1`, inspect the active attainable rank targets before starting FIT.
An already-capped window starts with one-site updates. An under-capacity
non-adjacent window requires at least three requested sweeps: two two-site
growth sweeps followed by at least one one-site refinement sweep. Reaching all
targets during growth switches the remaining budget to one-site updates;
rank stagnation below a target does not impersonate reaching the target.

`run_eff` is a separate global full-chain fit used by boundary/sampling code.
Do not substitute it for the gate-window solver.
`run` and `run_eff` retain fixed-sweep behavior; PEPS boundary diagnostics
describe them as `fixed_sweeps` and use only coarse opt-in elapsed timing.

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
- pTEBD/IPMC parallel compression: Phys. Rev. B 110, 085149 (2024),
  <https://doi.org/10.1103/PhysRevB.110.085149>. Do not label sequential FIT as
  this method.
- Local-TDVP circuit compression: 2025 preprint,
  <https://arxiv.org/abs/2508.10096>. Keep experimental until independently
  implemented and validated.

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
