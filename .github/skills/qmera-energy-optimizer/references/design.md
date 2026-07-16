# qMERA / MERA Energy Optimizer Design Notes

## Research Anchor

The motivating paper is Haghshenas, Gray, Potter, and Chan, "Variational Power
of Quantum Circuit Tensor Networks", Phys. Rev. X 12, 011047 (2022),
doi:10.1103/PhysRevX.12.011047.

Key design points for Pepsy:

- Dense MERA uses unitary disentanglers and isometric coarse-graining tensors.
- qMERA/QMERA-B keeps the MERA global structure but decomposes dense unitary and
  isometric blocks into finite-depth local quantum circuits.
- The useful controls are `q` or local block width, internal circuit depth, gate
  count, and local circuit structure such as brickwall.
- The energy objective used in the paper is exact contraction of local
  Hamiltonian terms, with global gradients supplied by automatic
  differentiation through quimb tensor networks.
- Unitary/isometric constraints should be enforced by differentiable projection
  from parameters to constrained tensors or by explicit circuit gates.

## quimb Anchors

quimb has two relevant layers:

- `quimb.tensor.MERA`: stable 1D MERA class used in the public MERA example.
  It supports causal-cone selection with site tags and works with
  `qtn.TNOptimizer`.
- `quimb.experimental.merabuilder.TensorNetworkGenIso`: arbitrary-geometry
  isometric builder with `layer_gate_fill_fn(operation="uni"|"iso"|"cap")`.
  It is useful for 2D/qMERA prototypes but should remain behind a Pepsy adapter.

The quimb example computes local terms by:

1. find site tags for `where`;
2. select tensors with `which="any"` to get the causal cone;
3. apply the local gate/operator to the selected ket;
4. join with the conjugate selected cone;
5. contract exactly with a reusable contraction optimizer.

For constrained global optimization, quimb uses a `norm_fn` that projects the
state with `unitize(method="exp")`; in new Pepsy code prefer the current
`isometrize(method="exp")` spelling.

For qMERA circuits, use the quimb quantum-circuit guide as conceptual guidance
only:

- use the ideas of stable gate ids, physical site tags (`I*`), gate tags
  (`GATE_*`), and reverse lightcone queries;
- do not use `qtn.Circuit`, `PTensor`, or quimb's circuit parameter machinery
  as the Pepsy qMERA core;
- implement Pepsy schedule objects and explicit parameter dictionaries instead;
- build ordinary quimb `TensorNetwork` objects or skeletons from those Pepsy
  schedules when contraction is needed.

For JAX implementation, use the quimb-with-JAX/Flax/Optax example as the
closer model: freeze a tensor-network skeleton, use `qtn.pack`/`qtn.unpack`
where appropriate, define a pure loss over JAX arrays, and run
`jax.value_and_grad`/Optax under `jax.jit`. Pepsy can use the same structure
through `GradientOptimizer`'s `jax-*` solvers.

## Pepsy-First Policy

Use Pepsy as the public owner of the qMERA workflow:

- geometry/mapping: `pepsy.tensors.OneDMap` where a 1D register is needed;
- Hamiltonian metadata: Pepsy local-term payloads and `SymHamiltonian` for
  Fermi-Hubbard when possible;
- gate arrays: Pepsy `operators` wrappers, `SymGateStream`, and user-provided
  gate-family callables;
- backend conversion: Pepsy `backends` helpers such as
  `pepsy.backend_jax(...)`, `pepsy.backend_torch(...)`, default backend setters,
  and autoray;
- optimization shell: Pepsy optimizer classes and tests;
- docs/API: Pepsy docs, examples, and public export rules.

Use quimb for tensor-network mechanics that Pepsy already treats as upstream
substrate: ordinary tensor networks, MERA/isometric tensor networks, local
selection, contraction, `qtn.pack`/`qtn.unpack`, and `TNOptimizer` when useful.
Avoid quimb `Circuit` and `PTensor` in the qMERA implementation path.

## Pepsy Backend Model

Do not create a new qMERA-specific backend abstraction. The optimizer and
builder should use Pepsy's existing backend surface:

- user-facing array casters: `pepsy.backend_numpy(...)`,
  `pepsy.backend_torch(...)`, `pepsy.backend_jax(...)`, and
  `pepsy.backend_cupy(...)`;
- package defaults: `pepsy.set_default_array_backend(...)` and
  `pepsy.set_default_grad_backend(...)`;
- backend inference/conversion helpers in `pepsy.backends.convert`;
- autoray operations inside backend-generic tensor code.

The first implementation should accept an optional `to_backend` or
`array_backend` callable, matching nearby Pepsy patterns. For JAX qMERA, this
means parameters and gate tensors are created as JAX arrays through Pepsy's
backend helper before entering the loss. For Torch qMERA, trainable parameters
should be Torch tensors created through the same Pepsy path. User-provided gate
tensors are not silently coerced; diagnostics should report incompatible
backends early.

For JAX JIT, the static objects are geometry, schedule, term selectors,
lightcone chunks, and contraction trees. The dynamic objects are only the
backend-native arrays in the parameter dictionary.

## Proposed Pepsy Package

Add:

```text
src/pepsy/optimizers/mera/
  __init__.py
  optimizer.py
  terms.py
  lightcones.py
  gates.py           # only when qMERA construction is implemented
  builders.py        # only when qMERA construction is implemented
  schematics.py      # manual layer/block drawings from schedules
  fermions.py        # only when FH/JW/native fermion helpers are implemented
  adapters.py        # only if wrapping experimental quimb merabuilder
```

First milestone:

- `optimizer.py`: `MeraEnergyOptimizer`
- `terms.py`: geometry/Hamiltonian term normalization
- `lightcones.py`: reverse-lightcone tag caching and chunk formation
- backend plumbing: use Pepsy backend casters/defaults for state, gates,
  parameters, and scalar conversion
- tests in `tests/test_optimize_mera.py`

Second milestone:

- `gates.py`: parametrized spin/fermion gate registry
- `builders.py`: `QMeraBuilder` for small 1D/2D brickwall qMERA schedules
- `schematics.py`: quimb-schematic visualizations of per-layer
  disentangler/isometry blocking
- `fermions.py`: Fermi-Hubbard conventions, two-site gate-family adapters, and
  consistency checks against `SymHamiltonian`
- optional prototype parity tests against `~/mera` examples if the prototype is
  available on the local machine

## Design Data Flow

Build the implementation around six explicit stages:

1. Geometry and Hamiltonian.
2. Disentangler/isometry block layout.
3. Explicit parameter dictionary plus user/Pepsy gate family.
4. Tags and reverse-lightcone cache.
5. Local energy chunk contraction.
6. Torch/JAX autodiff, with JAX JIT for frozen contraction structure.

Do not let a later stage infer hidden defaults from global variables or cached
prototype files. Every schedule and term should be reproducible from explicit
configuration.

## Geometry and Hamiltonian

Represent geometry before constructing the qMERA:

```python
QMeraGeometry(
    shape=(Lx, Ly),          # or L for 1D
    boundary="open" | "periodic",
    site_labels=((0, 0), ...),
    mapper=OneDMap(...) | "snake" | None,
)
```

The Hamiltonian layer should normalize all inputs to local terms:

```python
LocalTerm(where=((0, 0), (1, 0)), operator=H2, weight=1.0)
```

Start with nearest-neighbor spin Hamiltonians and keep the format general
enough for:

- 1D chains and 2D lattices;
- periodic or open boundaries;
- coordinate labels or mapped integer labels;
- spin local operators;
- fermionic local terms after a caller-selected encoding such as Jordan-Wigner
  or a fermionic gate-aware ansatz.

`MeraEnergyOptimizer` should not build the Hamiltonian from model names. That
belongs in separate helpers or examples. The optimizer consumes normalized
local terms.

For Fermi-Hubbard, prefer Pepsy's existing symbolic/symmetric layer:

```python
ham = SymHamiltonian.from_edges(
    "fermi_hubbard_u1u1",
    "U1U1",
    edges,
    t=t,
    U=U,
    mu=mu,
)
```

Then choose one explicit local representation:

- `native-fermion`: use Symmray/Pepsy native fermionic arrays and preserve their
  fermionic ordering metadata. This is correct for fermionic tensor networks
  but not automatically a dense qubit-circuit representation.
- `jw-nearest-neighbor`: use Pepsy's Jordan-Wigner gate/energy conventions for
  mapped nearest-neighbor bonds. This gives ordinary two-site bosonic gates
  only when the mapped bond is adjacent.
- `user-dense-two-qubit`: user supplies dense local gates/operators and is
  responsible for the fermion encoding. Pepsy still owns placement, tags,
  lightcones, backend conversion, and energy chunks.

Do not silently convert between these paths. Report the chosen fermion
convention in ansatz metadata and energy diagnostics.

## Disentangler and Isometry Blocks

Design disentanglers and isometries as separate schedule families in the actual
RG order. The schedule is not a generic brickwall circuit over all active
sites. At each scale:

1. partition active sites into non-overlapping isometry blocks;
2. form shifted boundary windows between neighboring isometry blocks;
3. place disentangler circuit layers on those boundary windows;
4. place isometry circuit layers inside each isometry block;
5. ascend one representative or output site per isometry block.

Thus isometry blocks cover the active lattice/register, while disentangler
blocks connect the boundaries of those blocks. This is the structural reason
local lightcones remain bounded.

```python
QMeraBlockSpec(
    kind="disentangler" | "isometry",
    block_size=2,
    circuit_depth=3,
    structure="brickwall",
    gate_family="spin-su4",
)
```

Default assumptions:

- gates inside a block are two-qubit gates;
- optional one-qubit rotations may be inserted before/after two-qubit layers;
- disentanglers preserve the active site set and straddle isometry-block
  boundaries;
- isometry blocks are non-overlapping and coarse-grain active sites to fewer
  upper sites;
- each MERA scale has boundary disentangler stages followed by covering
  isometry stages;
- the builder repeats scales until a cap/top tensor is reached.

The depth of the local circuit inside a disentangler block and the depth inside
an isometry block should be independent parameters. For example, a user should
be able to set `disentangler_depth=4` and `isometry_depth=2`.

For 1D binary/q-ary MERA, the number of RG scales should be
`ceil(log_q(L / top_size))`, up to finite-size edge handling. For 2D qMERA, the
scale count should follow the spatial side-length reduction, e.g.
`O(log(Lx))` for square systems, with x/y or block-local coarse-graining
substeps at each scale. The prototype in `~/mera` uses this style: functions
like `iso_(...)` form covering blocks, `uni_(...)` forms shifted/wrapped
boundary blocks, and 2D builders alternate horizontal/vertical unitary and
isometry sublayers before capping.

For dense MERA, block tensors can be direct quimb isometries. For qMERA, dense
blocks are replaced by local parameterized circuit layers whose contraction
acts as the block tensor. Keep those modes explicit.

## Parametrized Gate Registry

Use a small registry rather than hardcoded `if gate == ...` logic:

```python
GateSpec(
    name="SU4",
    arity=2,
    num_params=15,
    family="spin",
    generator=callable,
    supports_backend=("numpy", "torch", "jax"),
    convention="spin" | "native-fermion" | "jw" | "user",
)
```

Initial spin gates:

- one-qubit: Pepsy/quimb `RX`, `RY`, `RZ`, `U3`;
- two-qubit: Pepsy/quimb `RXX`, `RYY`, `RZZ`, `FSIM`, `SU4`;
- fixed gates only when explicitly requested, such as `H` or `CNOT`.

Initial fermionic gate direction:

- Pepsy native `fermi_hubbard_u1u1_*_gate_stream(...)` for Symmray fermionic
  tensors when the ansatz/backend supports them;
- Pepsy `fermi_hubbard_u1u1_jw_*_gate_stream(...)` or
  `SymHamiltonian.jw_trotter_gates(...)` for the bosonic Jordan-Wigner
  nearest-neighbor path;
- Pepsy/quimb gate matrix generators such as `FSIM`, `GIVENS`, `GIVENS2`,
  `XXPLUSYY`, and `XXMINUSYY` when they match the chosen convention;
- user-provided dense two-qubit gate families when they are the research object.

Every variational gate should produce backend-native arrays through Pepsy,
autoray, or direct quimb gate matrix generators. The JAX path must avoid
Python-side mutation during the compiled loss.

User-defined gate families should be first-class:

```python
UserGateFamily(
    name="fh-hop-var",
    arity=2,
    num_params=...,
    generator=lambda params, backend: ...,
    convention="jw-nearest-neighbor",
    default_tags=("FH_HOP",),
)
```

The qMERA builder consumes the gate family and the block schedule. It should not
know the internal formula for a hopping or interaction gate.

Do not represent these gates as quimb `PTensor`s. Store variational parameters
in a Pepsy-owned mapping such as:

```python
params = {
    "L0_DIS_000": array(...),
    "L0_ISO_000": array(...),
}
```

Then generate backend-native gate tensors with the selected `GateSpec`. This
keeps the same path usable for NumPy, Torch, and JAX and makes Pepsy's gradient
solvers the natural optimization layer.

## JAX and Pepsy Gradient Route

There are two useful JAX patterns:

1. Dense/isometric MERA tensor optimization:
   - create a quimb MERA-like tensor network;
   - call `qtn.pack(state)` once to get `(params, skeleton)`;
   - define `loss_fn(params)` as `state = qtn.unpack(params, skeleton)`, optional
     `state.isometrize(method=...)`, then local energy contraction;
   - run either external Flax/Optax or Pepsy `GradientOptimizer` with a
     `jax-*` solver.

2. qMERA gate-parameter optimization:
   - create a Pepsy `QMeraSchedule` from geometry/disentangler/isometry specs;
   - initialize a JAX pytree/dict of gate parameters;
   - define `loss_fn(params)` by generating all gate tensors from params,
     assembling the fixed qMERA tensor skeleton/chunks, and contracting local
     lightcone chunks;
   - run `GradientOptimizer(params, loss_fn, solver="jax-adam"|"jax-adamw")`
     once the loss is pure and JAX-compatible.

Pepsy's `GradientOptimizer` already JITs its JAX update step with
`jax.value_and_grad` and Optax. Prefer this route for Pepsy integration unless
there is a specific reason to expose a Flax module.

Implementation rule: JAX tracing should see a static schedule and static
contraction/chunk topology. Dynamic inputs should be arrays in `params`, not
new Python tensor-network objects, changing tag sets, or file-loaded schedules.

## `MeraEnergyOptimizer` Shape

Mirror `PepsEnergyOptimizer` where possible:

```python
class MeraEnergyOptimizer:
    def __init__(
        self,
        state,
        hamiltonian,
        *,
        normalized=True,
        energy_per_site=True,
        real=True,
        isometrize_method="exp",
        contraction_opt="auto-hq",
        backend="auto",
        jit=False,
        compute_kwargs=None,
        loss_kwargs=None,
    ): ...

    def loss(self, state=None, *, hamiltonian=None, terms=None, **kwargs): ...
    def energy(self, state=None, *, hamiltonian=None, terms=None, **kwargs): ...
    def make_tn_optimizer(...): ...
    def optimize(...): ...
```

Candidate loss kwargs:

- `normalized`: whether to divide by local norm if state is not guaranteed
  isometric. Default can be `True` only if implemented cheaply and tested.
- `energy_per_site`: divide by inferred number of physical sites.
- `real`: return `autoray.real(...)`.
- `isometrize_method`: method used by `norm_fn`, commonly `"exp"`.
- `contraction_opt`: exact contraction optimizer for causal cones.
- `precompute_tags`: cache causal-cone tags for fixed term supports.
- `chunk_terms`: group compatible local terms by identical or similar
  lightcone selectors.
- `jit`: request backend JIT when the loss graph is static, especially JAX.
- `solver`: Pepsy solver name such as `"jax-adam"`, `"jax-adamw"`,
  `"torch-adam"`, or `"torch-lbfgs"` when optimizing explicit parameter dicts.
- `simplify`: optional local simplification such as `full_simplify(seq="R")`.

`make_tn_optimizer()` should pass:

- `loss_fn`: static adapter around `_loss_state`.
- `norm_fn`: `lambda state: state.isometrize(method=isometrize_method)` when
  using dense isometric tensors.
- `loss_constants`: normalized terms and any cached tag selectors.
- `loss_kwargs`: contraction and scalar-format options.

For qMERA circuit builders with explicit unitary gates, `norm_fn` might be
unneeded because circuit gates already enforce unitarity. Keep this a builder
or ansatz property rather than auto-detecting silently.

## Term Normalization

Support these inputs first:

```python
{(i, j): H2, (j, k): H2}
[((i, j), H2), ((j, k), H2)]
```

Keep later support possible for:

- `qtn.LocalHam1D` / `LocalHam2D` `.terms`
- Pepsy `ham_tn` payloads where local terms can be extracted
- prototype-style `(list_sites, list_inter)` pairs

Each normalized term should carry:

- `where`: tuple of hashable site labels
- `operator`: array-like local operator
- `tags`: optional cached selector, usually `[state.site_tag(site) for site in where]`

Do not assume sites are integers. The experimental 2D builder uses coordinate
labels such as `(x, y)`.

## Tags and Reverse Lightcones

Tagging must support reverse lightcone lookup:

- physical site tags: stable `I{site}` or equivalent hash-safe encoding;
- layer tags: `LAYER{scale}` plus optional stage tags;
- operation tags: `DISENTANGLER`, `ISOMETRY`, `CAP`, and gate-family tags;
- quimb-inspired gate tags: `GATE_*`, gate names, and optional
  `ROUND_*`/block tags;
- Hamiltonian term tags or cached term ids for diagnostics.

For every normalized local term, precompute:

```python
LightconeTerm(
    term=local_term,
    tags=frozenset(...),
    tensor_ids=frozenset(...),   # optional after state construction
    width=max_active_sites,
)
```

The whole point of MERA/qMERA is that the causal-cone width for local
observables is bounded by architecture parameters rather than system size. Add
diagnostics such as `max_lightcone_width`, `num_tensors_by_term`, and
`num_indices_by_term`; tests should check that increasing `L` with the same
architecture does not increase the local cone width beyond the expected small
constant.

For qMERA, implement Pepsy-owned reverse lightcone helpers inspired by quimb's
circuit methods:

```python
tags = reverse_lightcone_tags(ansatz, where=term.where)
psi_chunk = select_reverse_lightcone(ansatz.state, where=term.where)
```

For comparison/debugging against quimb circuits, `Circuit.get_reverse_lightcone_tags`
is useful, but the public Pepsy helper should not require a circuit object.

## Causal-Cone Expectation Kernel

Pseudocode:

```python
def local_expectation(state, term, *, optimize, real=True, simplify=False):
    tags = term.tags or tuple(state.site_tag(site) for site in term.where)
    ket = state.select(tags, which="any")
    ket_g = ket.gate(term.operator, term.where)
    expec = ket.H & ket_g
    if simplify:
        expec = expec.full_simplify(seq="R", inplace=False)
    value = expec.contract(all, optimize=optimize)
    return ar.do("real", value) if real else value
```

Watch the order of `where` and operator dimensions. For multi-site terms, the
operator shape must match quimb's `gate` expectations.

## Energy Chunks

The energy is a sum over local Hamiltonian chunks:

```python
E = sum(contract_lightcone_chunk(state, chunk) for chunk in chunks)
```

A chunk can be one Hamiltonian term or a small group of terms whose lightcones
share the same selected subnetwork. Start with one term per chunk, then add
grouping once correctness is locked down.

Chunk metadata should include:

- selected tags;
- term ids;
- local operators and supports;
- contraction optimizer key;
- output scalar dtype/backend;
- optional simplified tensor-network template for JAX JIT.

For JIT, freeze the chunk topology and term structure outside the compiled
function. The dynamic inputs should be parameter arrays, not Python lists of
tensors or changing tag sets.

## qMERA Builder Sketch

Keep qMERA construction out of the energy optimizer. A builder should expose:

- lattice shape or number of sites;
- boundary condition and mapper;
- physical dimension;
- disentangler block size and depth;
- isometry block size and depth;
- structure: `"brickwall"` first, then `"ladder"` or `"tree"` if needed;
- non-overlapping isometry block partitions per scale;
- boundary disentangler windows derived from adjacent isometry blocks;
- gate set/family: spin or fermionic, fully parameterized;
- user-defined two-qubit gate family hooks;
- fermion convention and validation policy;
- tags for layer, operation, and site lightcones;
- schematic views of disentangler and isometry blocks at every layer;
- reverse-lightcone cache construction;
- explicit parameter initialization and packing;
- reproducible initialization.

Builder outputs should be ordinary quimb tensor networks and small metadata:

```python
QMeraAnsatz(state=tn, schedule=schedule, parameters=..., metadata=...)
```

Do not store global mutable schedules in module-level files like the prototype
`U_q3_l*` artifacts. If schedules are useful, generate them deterministically
from configuration or load them explicitly from a user-provided path.

The builder should produce a quimb-compatible tensor network, but Pepsy should
own the configuration schema and parameter mapping. A good mental split is:

- Pepsy decides the lattice, active sites at each scale, block schedule, gate
  family, fermion convention, and tags.
- quimb stores and contracts the actual tensor network.
- user code supplies custom local gate formulas when Pepsy's existing gates are
  not enough.

## Torch, JAX, and JIT

Torch and JAX should be peers:

- use autoray/quimb conversion helpers to put state tensors, gate tensors, and
  Hamiltonian terms on the requested backend;
- install existing Pepsy SVD/JAX/Torch registrations only when the chosen path
  uses differentiable decompositions;
- keep Python object mutation out of compiled losses;
- freeze term supports, selected tags, contraction paths, and chunk topology
  before JAX tracing;
- expose `jit=True` for JAX first. For explicit parameter optimization, route
  through Pepsy `GradientOptimizer` JAX solvers where possible;
- provide a non-JIT fallback with the same numerical result.

JAX JIT is plausible here because each lightcone chunk is a fixed tensor
contraction once the qMERA architecture, term supports, and contraction path are
fixed. The implementation should make that static structure explicit rather
than rebuilding tensor networks inside a traced function.

## Prototype Lessons

`~/mera/cg.py` shows:

- ITF 2D local terms can be represented as local two-site operators with a
  coordinate-to-1D map;
- precomputing reverse lightcone tags matters for repeated term evaluation;
- `ctg.ReusableHyperOptimizer` with a directory cache is useful for repeated
  contractions;
- 2D schedules can be expressed with `TensorNetworkGenIso.layer_gate_fill_fn`;
- `circ_qmera(...)` is a reminder to keep gate schedules separate from
  parameter arrays and loss evaluation;
- `find_tags_where(...)` should become a Pepsy lightcone helper independent of
  quimb `Circuit`;
- qMERA circuit experiments currently mix schedule construction, parameter
  extraction, MPS reconstruction, and loss evaluation. Split these concerns
  before moving anything into Pepsy.

## Risks

- quimb has both public and experimental MERA paths with different tag names.
  Avoid depending on private tag names except in a small helper.
- `unitize` appears in examples but is deprecated locally; use `isometrize`.
- Exact local contractions are differentiable but can be path-search heavy.
  Use reusable cotengra optimizers and cache tags/paths.
- JAX JIT can fail if tensor-network objects are rebuilt or mutated inside the
  traced function. Keep compiled kernels as pure contractions over arrays.
- quimb `Circuit` and `PTensor` are useful references but should not become
  Pepsy qMERA dependencies.
- Fermionic gates and fermionic Hamiltonian terms need an explicit convention;
  never assume a spin gate schedule represents a fermionic ansatz.
- Jordan-Wigner Fermi-Hubbard hopping is a two-site gate only for adjacent
  mapped bonds. For non-adjacent 2D mapped bonds, use the Pepsy MPO/term path or
  an explicitly provided long-string gate representation.
- Dense MERA projection and explicit qMERA circuit unitarity are different
  constraint mechanisms. Do not mix them silently in one code path.
- A DMRG-like local gate optimizer from the paper is a later milestone. Start
  with global autodiff since it matches existing Pepsy energy optimizer style.

## Validation Milestones

1. Dense 1D MERA smoke:
   - `qtn.MERA.rand(L=8, max_bond=4)` or current quimb signature equivalent.
   - nearest-neighbor Heisenberg terms.
   - `loss()` returns finite real scalar.

2. Causal-cone correctness:
   - compare a local expectation from selected tags with a full exact
     contraction for `L <= 8`.
   - assert lightcone width diagnostics remain bounded when increasing system
     size at fixed architecture.

3. Optimizer construction:
   - `make_tn_optimizer(autodiff_backend="torch", optimizer="adam", n=1)` runs
     or cleanly skips if torch is unavailable.
   - JAX loss and `jax.jit(loss)` agree for a tiny frozen chunk or skip cleanly
     if JAX is unavailable.
   - Pepsy `GradientOptimizer` with `solver="jax-adam"` runs one tiny
     explicit-parameter qMERA step or skips cleanly if JAX/Optax is unavailable.

4. Public API:
   - if exported, update public tests and docs in the same patch.

5. qMERA builder:
   - start with tiny deterministic schedules and assert the output is an
     isometric/circuit tensor network with expected site tags and finite local
     energy.
   - user-defined two-qubit gate family gets placed correctly and participates
     in reverse-lightcone tagging.
   - Fermi-Hubbard U1U1 nearest-neighbor JW path agrees with
     `SymHamiltonian.jw_energy(...)` or skips with a clear explanation when the
     mapped bonds are not nearest-neighbor.
