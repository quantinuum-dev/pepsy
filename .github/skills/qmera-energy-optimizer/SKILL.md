---
name: qmera-energy-optimizer
description: "Design, implement, review, or extend Pepsy qMERA energy optimization in src/pepsy/optimizers/qmera, including QMeraGeometry, QMeraBuilder schedules, qMERA RG-layout search and scoring, parameterized gate registries, reverse-lightcone energy chunks, compiled JAX/Torch losses, Symmray-native fermion gates/terms, schematics, and QMeraEnergyOptimizer APIs."
---

# qMERA Energy Optimizer in Pepsy

Use this skill for qMERA/QMERA-B work in Pepsy. The current implementation is
Pepsy-owned and schedule-first: Pepsy
defines geometry, RG blocking, gate placement, parameter dictionaries, local
lightcone chunks, and optimizer shells; quimb/cotengra provide tensor-network
storage and contraction.

## Read First

- Repository rules: `AGENTS.md`.
- Design reference: [references/design.md](references/design.md).
- Current source: `src/pepsy/optimizers/qmera/`.
- Focused tests: `tests/test_optimize_qmera.py`.
- Public exports: `src/pepsy/optimizers/__init__.py`,
  `src/pepsy/__init__.py`, `tests/test_public_api.py`, and
  `tests/test_package_layout.py`.
- Pepsy backend helpers: `src/pepsy/backends/` and public helpers such as
  `pepsy.backend_jax(...)`, `pepsy.backend_torch(...)`,
  `pepsy.set_default_array_backend(...)`, and
  `pepsy.set_default_grad_backend(...)`.
- Pepsy operator/tensor surfaces: `src/pepsy/operators/` and
  `src/pepsy/tensors/symmetric.py`.
- Optional prototype, if present: `/home/reza.haghshenas@quantinuum.com/mera/`.

Use quimb MERA, quimb circuit docs, and the `~/mera` prototype as design
references only. Do not rebuild Pepsy qMERA around `qtn.Circuit`, `PTensor`, or
copied prototype scripts.

## Implementation Map

- `terms.py`: `LocalTerm` normalization and backend conversion for local
  Hamiltonian inputs.
- `geometry.py`: `QMeraGeometry` with explicit lattice labels, boundary,
  optional `OneDMap`, register ordering, and per-site mode expansion such as
  `(site, "up")`, `(site, "down")`.
- `schedules.py`: `QMeraBlockSpec`, `QMeraGatePlacement`,
  `QMeraLayerSpec`, and `QMeraSchedule`. Schedules are built bottom-to-top:
  non-overlapping isometry blocks cover the active register, boundary
  disentanglers connect neighboring blocks, then the layer coarse-grains.
- `gates.py`: `GateSpec`, `GateRegistry`, `UserGateFamily`, and default
  parameterized spin gates. Context-aware gates use
  `GateSpec.matrix_for_placement(...)`.
- `builders.py`: `QMeraBuilder` and `QMeraAnsatz`. The builder owns schedule
  construction, parameter initialization/casting, debug direct-gate TN building,
  schedule-only local chunks, compiled chunks, and parameter-dict optimizer
  creation.
- `lightcones.py`: schedule-first local energy logic. Use
  `build_qmera_parametric_lightcone_chunks(...)`,
  `qmera_parametric_lightcone_tn(...)`, and
  `contract_qmera_lightcone_tn(...)` to rebuild and contract only the scheduled
  local cone for each Hamiltonian term.
- `compiled.py`: `cotengra.array_contract_expression` wrappers for dense and
  native graded qMERA local cones. Native Symmray expressions keep their
  product-state/operator constants as Symmray arrays and freeze only the
  contraction topology, so Torch/JAX parameter dictionaries remain
  differentiable without dropping charge or fermionic-order metadata.
- `parametric.py`: `QMeraEnergyOptimizer`, a `GradientOptimizer`-based shell
  for parameter dictionaries, including compiled-loss runs. The old
  `QMeraParametricEnergyOptimizer` name remains only as a compatibility alias.
- `layout.py`: `QMeraLayoutFinder` and immutable candidate/score/report objects
  for structural pre-ranking of RG architectures.
- `prototype.py`: a loader and stream-level scorer for serialized
  `~/mera/U_q3_l*` placement streams; prototype streams remain diagnostics,
  not Pepsy schedules.
- `fermions.py`: Symmray-native fermion helpers, including
  `QMeraSymmrayFermionBackend`, `qmera_symmray_fermi_hubbard_terms(...)`, and
  `symmray_fermion_gate_registry(...)`.
- `schematics.py`: `qmera_schematic_blocks(...)` and
  `draw_qmera_schedule(...)` for user-visible layer/block drawings.

## Core Rules

- Keep qMERA schedule-first. The parametric loss must not rely on a prebuilt
  full direct-gate tensor network except for debugging or exact comparison
  tests.
- Preserve the MERA RG structure. At each scale, isometry blocks are
  non-overlapping covering blocks; disentanglers are shifted boundary windows
  between those blocks; the schedule proceeds from bottom to top.
- Let users define gate families. The builder decides where gates go; the
  registry decides how a placement and parameter vector become a tensor.
- Keep parameters explicit as dictionaries keyed by placement `param_key`.
  This mirrors Gaugy-style parameter maps and keeps JAX/Torch optimization
  stable.
- Keep reverse-lightcone selection tied to `QMeraSchedule` placements. Generic
  tensor tags are useful diagnostics, but the energy path should follow
  `schedule.reverse_lightcone_placements(...)`.
- Expose explicit lightcone TNs when useful. The intended local energy path is
  still ordinary tensor-network contraction with `contract(all,
  optimize=cotengra_optimizer)`.
- For JAX JIT, freeze geometry, schedules, chunks, term supports, and compiled
  contraction expressions outside the traced function. Dynamic values should be
  only backend-native arrays in the parameter dictionary.
- Use Pepsy backend helpers and autoray-compatible arrays. Do not introduce a
  qMERA-specific backend abstraction.
- Keep qMERA gate unitarity and symmetry in the gate families. Do not add a
  separate dense tensor projection or normalization path to the qMERA
  optimizer.
- Keep optional dependencies optional. Symmray-specific tests must use
  `pytest.importorskip("symmray")`.

## Fermion Rules

- Treat native Symmray fermions as a graded tensor-network path, not as a
  Jordan-Wigner qubit encoding. Do not insert JW/parity strings in the native
  Symmray path.
- Validate mode labels and charge maps. `symmray-fsim` and hopping terms are
  neutral only for like modes with matching charge maps; cross-spin operations
  should raise unless a deliberate spin-changing gate is designed.
- Use `QMeraGeometry(site_modes=("up", "down"))` for Fermi-Hubbard style
  registers. `mode_order="mode-major"` is often useful when same-spin nearest
  neighbors should be adjacent in the active register.
- Use `QMeraSymmrayFermionBackend.product_state(...)` through the builder's
  `product_state_factory` when contracting native Symmray lightcones.
- Native Symmray compilation is supported when the builder receives a graded
  `product_state_factory`, normally
  `QMeraSymmrayFermionBackend.product_state`. The compiler must use Symmray's
  autoray dispatch for pairwise contractions; never densify the frozen
  constants or runtime gate blocks. `QMeraCompiledLightconeChunk.is_graded`,
  `.symmetry`, and `.contraction_backend` expose this choice.
- Two-dimensional multi-mode schedules are supported when geometry and block
  shapes are explicit. Preserve mode labels and same-flavor pairing; true
  rectangular isometries remain unsupported and must not be mislabeled as
  unitary completions.

## Remaining Work

- Add optional actual cotengra FLOP/peak-memory estimates to layout search
  after its cheap structural pre-ranking, reusing the path cache.
- Extend compiled graded-array coverage to larger later-scale schedules and
  optional device-specific benchmarks while retaining direct native cones as
  the correctness oracle.
- Extend the prototype adapter with level-to-scale inference only when the
  serialized stream format is formally specified; do not infer RG semantics
  from a flat placement list by guesswork.
- Add broader later-scale direct-versus-lightcone comparisons and validate any
  future true-isometry implementation separately from unitary completion.

## qMERA RG Layout Finder

The layout finder searches immutable RG architecture candidates and
return a reproducible `scales` plan that can be passed directly to
`QMeraBuilder`. It should not replace the schedule-first energy path or mutate
the builder while searching.

The public objects are `QMeraLayoutCandidate`, `QMeraLayoutScore`,
`QMeraLayoutReport`, and `QMeraLayoutFinder`. A candidate records, for
every RG scale:

- isometry block shape and orientation;
- disentangler placement (`boundary-faces`, `boundary-square`, or
  `within-block`), corner policy, periodic wrapping, and executable rounds;
- internal circuit depths, gate families, and parameter-sharing policy;
- the resulting `QMeraScaleSpec` values and a stable candidate id.

Candidate generation must validate the existing qMERA invariants before
scoring:

- isometry blocks form a non-overlapping covering partition;
- concurrent disentangler supports are disjoint, while sequential circuit
  rounds may reuse a block support;
- boundary disentanglers connect neighboring isometry regions and cover the
  relevant interaction boundaries;
- reverse lightcones are finite, reproducible, and compatible with periodic
  geometry, explicit modes, and the selected fermion symmetry;
- native Symmray candidates preserve mode labels, charge maps, and graded
  contraction semantics without adding Jordan-Wigner strings.

The score must expose components rather than hiding all decisions in one
opaque number. The initial components should include:

- structural cost: gate count, circuit depth, number of placements, and
  maximum/mean local-cone width;
- contraction cost: cotengra estimated FLOPs, peak intermediate size, and
  path-search cost for representative local-cone topologies;
- interaction coverage: weighted coverage of Hamiltonian supports and
  important interaction boundaries by the candidate's blocks and
  disentanglers;
- optional entanglement coverage: weighted coverage of a user-supplied
  mutual-information, entropy, correlation, or covariance map.

Without state-derived data, report interaction coverage as a Hamiltonian proxy;
do not call it measured physical entanglement. For expensive searches, use a
cheap structural pre-ranking followed by actual cotengra path estimates for
the top candidates, reusing `QMeraContractionPathCache`. Return both a scalar
weighted score and a Pareto front so users can inspect cost-versus-coverage
tradeoffs.

The intended workflow is:

1. generate and structurally validate candidate scale plans;
2. pre-rank candidates using cone width, gate count, depth, and interaction
   coverage;
3. evaluate cached contraction estimates for the top candidates;
4. optionally run a short pilot optimization and rescore with measured
   entanglement coverage;
5. return the best plan, Pareto candidates, component scores, and schematic
metadata without silently changing mode order or fermion conventions.

For comparison with the research prototype, call
`load_qmera_prototype_layout(...)` and
`finder.score_prototype_layout(...)`. That path reports flat-stream gate
count/depth and support coverage separately; it never converts `U_q3_l*`
into a `QMeraScaleSpec` without an explicit RG mapping.

The finder should integrate with `QMeraBuilder`, `QMeraSchedule`, and
`draw_schematic` so a selected architecture can be inspected at every
`rg_step`. Add focused tests for candidate validity, deterministic ranking,
bounded lightcones, contraction-cost reporting, OBC/PBC layouts, and native
Fermi-Hubbard mode/symmetry preservation before exposing top-level exports.

## Validation

Run focused validation after qMERA edits:

```bash
env NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig PYTHONPYCACHEPREFIX=/tmp \
  /home/reza.haghshenas@quantinuum.com/envs/py312/bin/python -m pytest -q tests/test_optimize_qmera.py
```

For API/export changes, also run:

```bash
env NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig PYTHONPYCACHEPREFIX=/tmp \
  /home/reza.haghshenas@quantinuum.com/envs/py312/bin/python -m pytest -q tests/test_public_api.py tests/test_package_layout.py
```

For syntax-only checks:

```bash
/home/reza.haghshenas@quantinuum.com/envs/py312/bin/python -m pyflakes src/pepsy/optimizers/qmera tests/test_optimize_qmera.py
```

The focused suite also covers independent 2D PBC Jordan-Wigner Fock-space
checks for every native Hubbard term, compiled native Symmray Hubbard
lightcones and Torch gradients, prototype-stream loading/scoring,
and canonical `pepsy.optimizers.qmera` imports with the temporary `mera`
compatibility alias.
