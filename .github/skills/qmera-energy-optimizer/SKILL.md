---
name: qmera-energy-optimizer
description: 'Design, implement, review, or extend Pepsy qMERA/MERA energy optimization, especially a future `MeraEnergyOptimizer`, quantum-circuit MERA/QMERA ansatz builders, causal-cone local-energy contraction, isometric/unitary projection, and migration of the `~/mera` prototype into Pepsy without copying notebook code.'
---

# qMERA Energy Optimizer in pepsy

Use this skill for qMERA/QMERA-B, dense MERA, and `MeraEnergyOptimizer` work in
Pepsy. The target is a clean Pepsy wrapper around quimb MERA/isometric tensor
network machinery, not a direct import of the exploratory `~/mera` scripts.

## Read First

- Repository rules: `AGENTS.md`.
- Public optimizer surface: `src/pepsy/optimizers/__init__.py`,
  `src/pepsy/optimizers/README.md`, `src/pepsy/optimizers/energy/peps.py`.
- Pepsy operator/tensor surfaces: `src/pepsy/operators/`, especially gate
  builders, and `src/pepsy/tensors/symmetric.py` for `SymHamiltonian`,
  `SymGateStream`, Fermi-Hubbard U1U1 helpers, and Jordan-Wigner conventions.
- Pepsy backend surface: `src/pepsy/backends/`,
  `src/pepsy/tensors/core.py`, and public helpers such as
  `pepsy.backend_jax(...)`, `pepsy.backend_torch(...)`,
  `pepsy.set_default_array_backend(...)`, and
  `pepsy.set_default_grad_backend(...)`.
- Existing fermion notes: `docs/api/tensors/symmetric.md` and
  `docs/development/fermi_hubbard_u1u1_mpo_notes.md`.
- Public API tests before exporting a symbol: `tests/test_public_api.py` and
  `tests/test_package_layout.py`.
- quimb substrate: `qtn.MERA`, `qtn.TNOptimizer`, `Tensor.isometrize`, and,
  only behind a local adapter, `quimb.experimental.merabuilder.TensorNetworkGenIso`.
  The quimb circuit guide is useful only for ideas about tags and reverse
  lightcones; do not build the Pepsy implementation around `qtn.Circuit` or
  `PTensor`.
- JAX design reference:
  https://quimb.readthedocs.io/en/latest/examples/ex_quimb_within_jax_flax_optax.html
- Prototype, if present: `/home/reza.haghshenas@quantinuum.com/mera/cg.py`,
  `/home/reza.haghshenas@quantinuum.com/mera/performance.py`, and
  `/home/reza.haghshenas@quantinuum.com/mera/run_p.py`.

Read [references/design.md](references/design.md) before making the first code
change or API decision.

## Design Direction

- Add a focused `src/pepsy/optimizers/mera/` package rather than expanding the
  PEPS or MPS optimizer modules.
- Treat qMERA design as a pipeline:
  geometry/Hamiltonian -> qMERA layer schedule -> parametrized gate registry ->
  lightcone/tag cache -> local energy chunks -> Torch/JAX autodiff and JIT.
- Make the pipeline Pepsy-first. Use Pepsy for geometry mapping, Hamiltonian
  metadata, public gate builders, symmetric/Fermi-Hubbard helpers, backend
  conversion, optimizer wrappers, tests, and docs. Use quimb as the tensor
  network substrate where Pepsy intentionally wraps it.
- Do not invent a separate qMERA backend layer. Accept Pepsy backend casters
  such as `pepsy.backend_jax(...)` and `pepsy.backend_torch(...)`, infer backend
  from representative tensor data where practical, and keep gate/parameter
  arrays on compatible Pepsy-supported backends.
- Do not use quimb `Circuit` or `PTensor` as the core qMERA implementation.
  Build Pepsy qMERA schedules and explicit parameter dictionaries, then create
  quimb tensor networks/tensor skeletons from those parameters.
- Make `MeraEnergyOptimizer` the first public optimizer class, but do not make
  it responsible for inventing the ansatz. Model its public shape on
  `PepsEnergyOptimizer`: store `state`, `terms`, `loss_kwargs`, expose
  `loss()`, `energy()`, `make_tn_optimizer()`, and `optimize()`.
- Reuse `EnergyEstimate` unless MERA-specific metadata genuinely needs a new
  dataclass.
- Keep ansatz construction separate from energy optimization. A later
  `QMeraBuilder` or `QMeraAnsatz` should build qMERA/QMERA-B states, while
  `MeraEnergyOptimizer` evaluates and optimizes any compatible MERA-like state.
- Start every implementation by making geometry and Hamiltonian explicit:
  lattice shape, boundary conditions, site labels, optional 1D mapper, and local
  terms. Accept `{where_tuple: operator}` or iterable `(where_tuple, operator)`
  pairs as the canonical first term format.
- Design disentangler and isometry blocks separately and in RG order. At each
  bottom-to-top scale, isometry blocks form a non-overlapping covering
  partition of the active lattice/register, while disentangler blocks are
  shifted boundary windows connecting neighboring isometry blocks before
  coarse-graining.
- Make circuit gate families explicit and fully parametrized. Spin gates from
  Pepsy/quimb (`U3`, `SU4`, `RXX`, `RYY`, `RZZ`, etc.), Pepsy symmetric
  Fermi-Hubbard streams, and user-defined two-qubit gates should flow through a
  registry with arity, parameter count, generator, fermion convention, and
  backend conversion.
- Let users supply the local two-qubit gate family; the MERA builder should
  decide only where each gate is placed from the geometry and block schedule.
- Tagging is core behavior, not decoration. Precompute reverse-lightcone tags
  for every Hamiltonian support and expose diagnostics for causal-cone width.
  For fixed qMERA architecture the selected lightcone width should remain
  bounded rather than growing with total system size.
- Provide schematic views of the explicit qMERA schedule. Users should be able
  to inspect disentangler and isometry blocking at every layer before running
  optimization.
- Compute energy by contracting only each Hamiltonian term's lightcone chunk, or
  groups of compatible term chunks, not the whole state. Use reusable
  cotengra/quimb contraction optimizers and cache term selectors.
- Enforce MERA constraints with quimb isometric projection, usually
  `state.isometrize(method="exp")` for differentiable global optimization.
  Avoid new local dense linear algebra unless quimb/autoray cannot express it.
- Make Torch and JAX first-class. The same static contraction graph should run
  under torch autodiff and JAX autodiff; JAX JIT should be enabled for the
  frozen lightcone/chunk structure with only parameter arrays treated as dynamic.

## Prototype Mapping

Useful ideas from `~/mera`:

- `find_tags_where(...)`: precompute causal-cone selectors for each local term.
- `local_expectation_mera(...)` and `energy_f_qmera(...)`: minimal local-energy
  objective structure.
- `build_contraction_order()`: use `ctg.ReusableHyperOptimizer` for repeated
  causal-cone contractions.
- `mera_2d(...)` and `tree_2d(...)`: schedule examples for experimental
  `TensorNetworkGenIso.layer_gate_fill_fn`.
- `circ_qmera(...)`, `draw_params_torch(...)`, and `loss_fn(...)`: qMERA circuit
  parameterization experiments. Treat these as design notes, not source code to
  paste into Pepsy.
- quimb `Circuit` ideas to preserve conceptually: stable gate ids, gate-family
  tags, physical site tags, and reverse lightcone queries. Recreate these in
  Pepsy helpers instead of depending on `qtn.Circuit` or `PTensor`.
- quimb-with-JAX ideas to preserve: use a fixed tensor-network skeleton,
  `qtn.pack`/`qtn.unpack` where appropriate, and a pure JAX loss over parameter
  dictionaries so Pepsy's `GradientOptimizer` JAX solvers can JIT the update.

## API Sketch

Preferred first user flow:

```python
import cotengra as ctg
import pepsy

opt = pepsy.optimizers.MeraEnergyOptimizer(
    state=mera,
    hamiltonian=terms,
    contraction_opt=ctg.ReusableHyperOptimizer(progbar=False),
    isometrize_method="exp",
    energy_per_site=True,
)

estimate = opt.energy()
mera_opt = opt.optimize(n=200, autodiff_backend="torch", optimizer="L-BFGS-B")
```

Preferred builder separation for later qMERA work:

```python
from pepsy.optimizers.mera import QMeraBuilder, MeraEnergyOptimizer

ansatz = QMeraBuilder(shape=(8, 8), q=3, depth=4, structure="brickwall").build()
opt = MeraEnergyOptimizer(ansatz.state, terms)
```

More explicit builder configuration should be supported as the design matures:

```python
ansatz = QMeraBuilder(
    geometry={"shape": (8, 8), "boundary": "periodic", "mapper": "snake"},
    disentangler={"block_size": 2, "depth": 3, "structure": "brickwall"},
    isometry={"block_size": 2, "depth": 2, "structure": "brickwall"},
    gate_family=user_two_qubit_gate_family,
    backend="jax",
).build()
```

Fermi-Hubbard should start from Pepsy's existing model helpers:

```python
from pepsy.tensors import OneDMap, SymHamiltonian

mapper = OneDMap(Lx, Ly, mode="snake")
ham = SymHamiltonian.from_edges(
    "fermi_hubbard_u1u1",
    "U1U1",
    edges,
    t=1.0,
    U=8.0,
)

ansatz = QMeraBuilder(
    geometry={"shape": (Lx, Ly), "mapper": mapper},
    gate_family=fermionic_two_qubit_gate_family,
    fermion_convention="jw-nearest-neighbor",
).build()
```

## Implementation Guardrails

- Preserve Pepsy public API rules. If `MeraEnergyOptimizer` becomes top-level,
  update `src/pepsy/optimizers/__init__.py`, `src/pepsy/__init__.py`, docs, and
  public API/layout tests in the same change.
- Do not add old flat modules such as `pepsy.optimize_mera`.
- Do not add Gaugy or Tensy code, notebooks, or imports here.
- Use `source ~/envs/py312/bin/activate` for local commands.
- Prefer `isometrize` over deprecated `unitize` names in new code, while
  accepting that quimb examples may still show `unitize`.
- Support both old and experimental quimb MERA tag conventions when selecting
  metadata: old `"_UNI"`, `"_ISO"`, `"_LAYER{i}"`; experimental `"UNI"`,
  `"ISO"`, `"CAP"`, `"LAYER{i}"`.
- Keep site tags and gate/block tags stable enough to support reverse
  lightcone lookup, chunk caching, and repeated JIT compilation.
- For qMERA circuits, preserve quimb-compatible tags such as physical `I*`,
  `GATE_*`, gate-family tags, and round/layer tags so Pepsy can reuse quimb's
  lightcone semantics while adding Pepsy diagnostics. Implement the lookup in
  Pepsy; do not require a `qtn.Circuit` object.
- For Fermi-Hubbard, keep the fermion convention explicit: native Symmray
  fermionic gates, bosonic Jordan-Wigner nearest-neighbor two-site gates, or
  caller-provided dense two-qubit gates are distinct paths.
- Keep optional experimental quimb imports lazy and wrapped so ordinary Pepsy
  imports still work if the experimental module changes.

## Validation

Start with focused tests in a new `tests/test_optimize_mera.py`:

- small 1D `qtn.MERA.rand(...)` energy for nearest-neighbor terms contracts and
  returns a real scalar;
- local causal-cone expectation matches a full exact contraction on `L <= 8`;
- `isometrize_method="exp"` optimizer construction works with torch autodiff;
- invalid term/site formats raise clear errors;
- qMERA builder tests are marked or skipped if they depend on experimental
  quimb APIs.
- JAX smoke tests either run a one-step JIT-compiled loss or skip cleanly when
  JAX is unavailable.
- Pepsy `GradientOptimizer(..., solver="jax-adam"|"jax-adamw")` smoke tests
  should cover the explicit-parameter qMERA loss once that path exists.

If exporting the class, also run:

```bash
source ~/envs/py312/bin/activate
pytest -q tests/test_public_api.py tests/test_package_layout.py
```

For optimizer behavior, add the focused MERA tests first, then run nearby
energy optimizer tests if they exist.
