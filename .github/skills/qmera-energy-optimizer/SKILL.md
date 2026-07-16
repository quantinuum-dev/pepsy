---
name: qmera-energy-optimizer
description: "Design, implement, review, or extend Pepsy MERA/qMERA energy optimization in src/pepsy/optimizers/mera, including QMeraGeometry, QMeraBuilder schedules, parameterized two-qubit gate registries, reverse-lightcone energy chunks, compiled JAX/Torch losses, Symmray-native fermion gates/terms, schematics, and MeraEnergyOptimizer APIs."
---

# qMERA Energy Optimizer in Pepsy

Use this skill for qMERA/QMERA-B, dense MERA, and `MeraEnergyOptimizer` work in
Pepsy. The current implementation is Pepsy-owned and schedule-first: Pepsy
defines geometry, RG blocking, gate placement, parameter dictionaries, local
lightcone chunks, and optimizer shells; quimb/cotengra provide tensor-network
storage and contraction.

## Read First

- Repository rules: `AGENTS.md`.
- Design reference: [references/design.md](references/design.md).
- Current source: `src/pepsy/optimizers/mera/`.
- Focused tests: `tests/test_optimize_mera.py`.
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

- `optimizer.py`: dense/isometric `MeraEnergyOptimizer` over existing MERA-like
  tensor networks, with local lightcone energy chunks and quimb `TNOptimizer`
  integration.
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
- `compiled.py`: `cotengra.array_contract_expression` wrappers for dense
  qMERA local cones. These freeze contraction topology so Torch/JAX see a pure
  array loss over parameter dictionaries.
- `parametric.py`: `QMeraParametricEnergyOptimizer`, a
  `GradientOptimizer`-based shell for parameter dictionaries, including
  compiled-loss runs.
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
- Keep dense MERA projection and explicit qMERA circuit unitarity separate.
  Dense MERA uses isometric tensor projection; qMERA uses unitary or
  symmetry-preserving gate families.
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
- The compiled dense path in `compiled.py` is not automatically a Symmray
  compiled path. Keep native Symmray contractions explicit until a tested
  compiled graded-array route exists.
- Two-dimensional multi-mode schedules are intentionally guarded until the RG
  design is explicit; do not silently flatten this case as if it were a normal
  spin schedule.

## Next Useful Work

- Add user-facing docs/examples for `QMeraBuilder`, schematics, parametric
  lightcone losses, compiled JAX/Torch loss usage, and Symmray-native
  Fermi-Hubbard terms.
- Add local-cone grouping or reusable path-cache helpers around cotengra once
  one-term-per-chunk correctness remains stable.
- Extend tests that compare direct full-qMERA TN energy to schedule-only
  lightcone energy for more schedules and boundary conditions.
- Design the explicit 2D multi-mode/Fermi-Hubbard RG blocking before enabling
  it in `build_qmera_schedule(...)`.
- Decide whether any qMERA symbols should become top-level `pepsy.*` exports;
  if yes, update docs and public API tests in the same patch.

## Validation

Run focused validation after qMERA edits:

```bash
env NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig PYTHONPYCACHEPREFIX=/tmp \
  /home/reza.haghshenas@quantinuum.com/envs/py312/bin/python -m pytest -q tests/test_optimize_mera.py
```

For API/export changes, also run:

```bash
env NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig PYTHONPYCACHEPREFIX=/tmp \
  /home/reza.haghshenas@quantinuum.com/envs/py312/bin/python -m pytest -q tests/test_public_api.py tests/test_package_layout.py
```

For syntax-only checks:

```bash
/home/reza.haghshenas@quantinuum.com/envs/py312/bin/python -m pyflakes src/pepsy/optimizers/mera tests/test_optimize_mera.py
```
