# `pepsy.optimizers.qmera`

Pepsy's qMERA API places parameterized gate families on a static RG
schedule. It evaluates local Hamiltonian terms through their reverse
lightcones, without building the full state for each energy call.

## 1D spin qMERA

`QMeraBuilder(system_size=N)` uses `N` spatial sites and the retained-register
1D circuit. The older `shape=N` spelling remains accepted; multidimensional
geometries use `shape=(Lx, Ly)` or `geometry=`. Each isometry block covers
adjacent child registers; a boundary disentangler connects the
edges of neighboring blocks. With the default block length 2, an odd final
group has three children, so no one-site block is left behind. At each scale,
`bond_qubits` wires are retained for the next coarser register (or all wires
when the block is smaller). The default `retention="right"` selects the
rightmost wires in block order. `"left"`, `"balanced"`, and `"explicit"`
are also available; explicit retention uses a mapping from `(scale, block)`
to an ordered tuple of child wire indices. The schedule records those choices
in `layer.retained_registers` and executes gates from coarse to fine.

```python
from pepsy.optimizers.qmera import QMeraBuilder

N = 7
builder = QMeraBuilder(
    system_size=N,
    boundary="periodic",
    isometry={"block_size": 2, "circuit_depth": 1},
    disentangler={"block_size": 2, "circuit_depth": 1},
    bond_qubits=2,
    retention="right",
    spin_symmetry="z2",
    initial_state="zero",
)
schedule = builder.build_schedule()
assert schedule.layers[0].isometry_blocks[-1] == (5, 6, 0)
assert schedule.layers[0].retained_registers[-1] == (6, 0)
```

### Pair circuit structure and depth

`isometry=` and `disentangler=` each accept `structure="brickwall"` (the
default) or `structure="ladder"` for unmoded spin 1D. The structure orders
**pair placements within each block**; `pair_ansatz=` determines the
rotations applied at each placement.

For ordered block wires `(0, 1, 2)`, one `brickwall` sweep applies pairs
`(0,1) → (1,2)`. One `ladder` sweep applies
`(0,1) → (1,2) → (2,0)`. On a longer block, ladder walks all adjacent
pairs and then closes from the last wire to the first. A two-wire ladder uses
its one pair once. This closure is **inside the block even when the whole
chain has open boundaries**. `circuit_depth=2` repeats the entire chosen
sweep twice, with separate parameters per placement by default; it does not
add an RG scale. Isometry and disentangler stages may choose their own
structure and depth.

```python
builder = QMeraBuilder(
    system_size=3,
    isometry={"structure": "ladder", "circuit_depth": 2},
    disentangler={"circuit_depth": 0},
)
schedule = builder.build_schedule()
assert [gate.where for gate in schedule.placements] == [
    (0, 1), (1, 2), (2, 0), (0, 1), (1, 2), (2, 0),
]
```

The ml4mb reference uses brickwall pair rounds; ladder is a Pepsy
extension. Existing 2D and explicit-mode fermion schedules support only
`brickwall`; they reject `ladder` rather than silently using a different
gate order.

### Pair ansatz and symmetry

`pair_ansatz=` selects the ordered rotation template used on every scheduled
pair in the 1D spin circuit. `spin_symmetry=` chooses or checks its global-X
symmetry class. Both apply only to unmoded spin 1D geometries.

| `spin_symmetry` | Default `pair_ansatz` | Circuit guarantee |
| --- | --- | --- |
| `"z2"` (default) | `"z2_zz_yy_rx"` | Every pair gate commutes with global `X⊗X⊗…⊗X` |
| `"unrestricted"` | `"all_paulis"` | No global-X symmetry guarantee |

The preferred names say which Pauli rotations are present. The older ml4mb
names remain accepted aliases; they retain their original gate-family and
metadata names when used explicitly.

| Preferred `pair_ansatz` | Old alias | Logical rotations in time order | Angles per pair |
| --- | --- | --- | ---: |
| `"z2_zz_yy_rx"` | `"minimal"` | `RZZ(0,1) → RYY(0,1) → RX(0) → RX(1)` | 4 |
| `"z2_rx_zz_yy_xx_rx"` | `"extended"` | `RX(0) → RX(1) → RZZ(0,1) → RYY(0,1) → RXX(0,1) → RX(0) → RX(1)` | 7 |
| `"z2_zz_rx"` | `"ising"` | `RZZ(0,1) → RX(0) → RX(1)` | 3 |
| `"z2_yz_zy"` | `"real_z2"` | `RYZ(0,1) → RZY(0,1)` | 2 |
| `"z2_all_paulis"` | `"pauli_z2"` | `RX(0) → RX(1) → RXX → RYY → RYZ → RZY → RZZ` (two-wire gates on `0,1`) | 7 |
| `"all_paulis"` | `"unrestricted"` | `RX(0), RX(1), RY(0), RY(1), RZ(0), RZ(1)`, then `RXX, RXY, RXZ, RYX, RYY, RYZ, RZX, RZY, RZZ` on `0,1` | 15 |

Wires `0` and `1` mean the first and second wires of each scheduled pair.
Every listed rotation has its own trainable angle, including both occurrences
of `RX` on the same wire in the seven-angle family. The first five families
preserve global-X Z₂; `"all_paulis"` can break it. `"z2_all_paulis"` contains
all seven nonidentity two-qubit Pauli words commuting with `X⊗X`.

`RYZ` and `RZY` are logical Pauli rotations. ml4mb lowers each to a fixed
`RX(+π/2)`, a trainable `RZZ`, and a fixed `RX(-π/2)` on the relevant wire.
Pepsy composes the listed rotations into one differentiable pair tensor per
placement; its schedule records one pair gate, not each physical rotation.
Mixed-Pauli rotations in `"all_paulis"` are also composed directly.

```python
from pepsy.optimizers.qmera import (
    QMeraBuilder,
    QMeraPairSpec,
    available_qmera_pair_ansatzes,
    get_qmera_pair_ansatz,
)

names = available_qmera_pair_ansatzes()       # six preferred names
all_names = available_qmera_pair_ansatzes(include_aliases=True)
extended = get_qmera_pair_ansatz("z2_rx_zz_yy_xx_rx")
assert extended.rotation_sequence == (
    ("RX", (0,)), ("RX", (1,)), ("RZZ", (0, 1)),
    ("RYY", (0, 1)), ("RXX", (0, 1)), ("RX", (0,)), ("RX", (1,)),
)
assert extended.num_params == 7

z2_builder = QMeraBuilder(
    system_size=8, spin_symmetry="z2", pair_ansatz="z2_rx_zz_yy_xx_rx",
    initial_state="plus",
)
general_builder = QMeraBuilder(
    system_size=8, spin_symmetry="unrestricted", initial_state="zero"
)
assert z2_builder.spin_symmetry == "global-X-Z2"
assert general_builder.spin_symmetry == "unrestricted"

repeated = get_qmera_pair_ansatz("z2_rx_zz_yy_xx_rx", repetitions=2)
repeated_builder = QMeraBuilder(system_size=8, pair_ansatz=repeated)
custom = QMeraPairSpec(
    ("ZI", "XX", "IZ"), name="my-pair", symmetry="unrestricted"
)
custom_builder = QMeraBuilder(system_size=8, pair_ansatz=custom)
```

`ansatz=` remains an alias for `pair_ansatz=`; supply only one. The older
`available_qmera_pair_ansatze()` function continues to list the legacy names;
`available_qmera_pair_ansatzes()` lists the preferred names. Supplying both
`spin_symmetry` and a pair
ansatz checks that they agree; Pepsy does not infer symmetry from the
Hamiltonian. Each occurrence of a Pauli word has an independent trainable
angle. The `"z2_zz_yy_rx"` template starts its four angles at one shared
random value, matching ml4mb's initialization policy while keeping them
separately trainable. Other templates start their angles independently. A
custom `QMeraPairSpec` defaults to `symmetry="global-X-Z2"` and rejects
generators that break it; declare `symmetry="unrestricted"` to allow them.
Explicit `spin_symmetry=` or `pair_ansatz=` controls both isometry and boundary
pair gates and cannot be combined with conflicting `gate_family` overrides.
The lower-level gate registry remains available for other spin gate families;
`builder.spin_symmetry` is `None` when overrides prevent Pepsy from certifying
a single pair-ansatz contract.

The input is a **separate choice**: `initial_state="zero"` (the default)
starts from `|0…0⟩`, which is not an eigenstate of global X.
`initial_state="plus"` starts from `|+…+⟩`, the +1 global-X sector, while
leaving the gate ansatz unchanged. A Z₂-preserving circuit keeps a definite
sector only when its input is in one. `initial_hadamards=True` remains an
alias for the plus input, and a `product_state_factory` can supply another
input state.

Explicit-mode fermion geometries keep their native Symmray gate and state
conventions; retained-register spin layouts and initial Hadamards do not
apply to them. The default 2D layout remains the existing site-retention
schedule. An opt-in 2D spin hierarchy is described below.

```python
import numpy as np

zz = np.diag([1.0, -1.0])
h2 = np.kron(zz, zz).reshape(2, 2, 2, 2)
from pepsy.optimizers import QMeraBuilder, build_qmera_contraction_optimizer

builder = QMeraBuilder(
    system_size=8,
    gate_family="rxx",
    isometry_gate_family="rzz",
    seed=2,
    param_scale=0.03,
)
schedule = builder.build_schedule()
params = builder.initialize_parameters(schedule)
chunks = builder.parametric_lightcone_chunks({(0, 1): h2}, schedule)

optimize = build_qmera_contraction_optimizer(directory=False, max_repeats=16)
energy = builder.parametric_loss(
    params,
    schedule=schedule,
    chunks=chunks,
    contraction_opt=optimize,
)
```

For repeated optimization, estimate the forward cost before running and
reuse its searched paths with a matching backend and optimizer:

```python
report = builder.estimate_contraction_cost(
    schedule=schedule, chunks=chunks,
    contraction_opt=optimize, normalized=False, element_bytes=16,
)
print(report.summary())
compiled = builder.compile_parametric_lightcones(
    schedule=schedule, chunks=chunks,
    contraction_opt=report.contraction_opt, path_cache=report.path_cache,
)
loss_fn = builder.compiled_parametric_loss_fn(
    schedule=schedule, compiled_chunks=compiled, normalized=False,
)
energy = loss_fn(params)
```

`report.log10_flops` is the base-10 logarithm of complex-arithmetic
work summed over the evaluated local cones. `report.log2_peak_bytes` is the largest estimated live tensor size in
one cone, using `element_bytes` per element. These forward estimates exclude
autodiff and optimizer storage. Only evaluated paths are planned: denominator contractions are included when
`normalized=True`. Unsliced Cotengra paths are primed for compilation; native
Symmray block costs need a separate estimator.

For dense spin gates, `torch_fullgraph=True` freezes Cotengra paths and uses
Torch operations to build each scheduled gate once per energy evaluation. Pass
Torch parameters; use `array_backend=backend_torch(dtype=torch.complex128)`
when preparing the cones. The built-in qMERA pair ansatzes and `rxx`, `ryy`,
`rzz` carry the required `GateSpec.torch_pauli_words` metadata:

```python
import torch
from pepsy.backends import backend_torch

torch_array = backend_torch(dtype=torch.complex128)
torch_params = builder.cast_params(params, backend="torch", dtype=torch.float64)
torch_cones = builder.compile_parametric_lightcones(
    {(0, 1): h2}, schedule=schedule, array_backend=torch_array,
)
torch_loss = builder.compiled_parametric_loss_fn(
    schedule=schedule, compiled_chunks=torch_cones,
    array_backend=torch_array, torch_fullgraph=True, normalized=False,
)
fullgraph_loss = torch.compile(torch_loss, backend="aot_eager", fullgraph=True)
energy = fullgraph_loss(torch_params)
```

The returned callable is also a faster eager Torch loss for repeated local
terms. Native Symmray arrays and gate families without Pauli-rotation metadata
continue to use the standard compiled path. AOT eager checks complete graph
capture; Inductor performance depends on the local compiler installation.

Native Symmray qMERA uses the same compiled API, but the builder must receive
the graded product-state factory. The compiled object then reports
`contraction_backend="symmray"` and `is_graded=True`; its frozen constants and
runtime gate blocks stay block-sparse Symmray arrays, so fermionic signs,
charge maps, duals, and Torch autodiff are retained:

```python
from pepsy.optimizers.qmera import (
    QMeraBuilder,
    QMeraGeometry,
    QMeraSymmrayFermionBackend,
    symmray_fermion_gate_registry,
)

backend = QMeraSymmrayFermionBackend(symmetry="U1U1")
builder = QMeraBuilder(
    geometry=QMeraGeometry(
        shape=(4, 4),
        boundary="periodic",
        site_modes=("up", "down"),
        mode_order="mode-major",
    ),
    gate_registry=symmray_fermion_gate_registry(backend=backend),
    gate_family="symmray-fsim",
    product_state_factory=backend.product_state,
)
compiled = builder.compile_parametric_lightcones(
    terms,
    convert_terms=False,
    path_cache=builder.contraction_path_cache(max_repeats=16),
)
assert compiled[0].is_graded
```

`path_cache` reuses topology-specific cotengra searches while each compiled
expression reuses its frozen contraction tree on every parameter evaluation.
Use `builder.parametric_loss(...)` or `builder.direct_parametric_loss(...)` as
the explicit native correctness oracle when validating a new periodic layout.

`QMeraEnergyOptimizer` routes the same compiled loss through Pepsy's gradient
solvers. Its built-in spin and fermion gate families are parameterized and
unitary/symmetry-preserving by construction, so normalization is disabled by
default. Pass `normalized=True` for a custom non-unitary gate family:

```python
param_opt = builder.parametric_optimizer(
    {(0, 1): h2},
    schedule=schedule,
    chunks=chunks,
    parameters=params,
    energy_per_site=False,
)
result = param_opt.run(solver="torch-adam", n_steps=10, compiled=True)
```

For dense spin gates, the same optimizer exposes the Torch-only cost through
`param_opt.compiled_loss_fn(torch_fullgraph=True)`. Its
`run(solver="torch-adam", compiled=True, torch_fullgraph=True)` path uses that
cost for training. The callable can also be passed to
`torch.compile(..., backend="aot_eager", fullgraph=True)`.

## qMERA schematics

`QMeraSchedule.draw_schematic()` uses Quimb's manual `schematic.Drawing`
primitives. The default `style="clean"` follows the schedule's gate direction.
Retained-register circuits read **coarse → W → D → fine**; site-retention
schedules read **fine → D → W → coarse**. `rg_step=0` selects the finest RG
interface even when all retained layers are drawn in preparation order:

```python
drawing = schedule.draw_schematic(
    rg_step=0,
    style="clean",            # or "register" for the older wiring view
    figsize=(10, 8),
    label_sites=True,
    label_blocks=True,
    scale_figsize=False,
)
```

In 1D, green brackets show covering isometry blocks; colored patches show
actual pair-gate positions in each round. A long periodic seam gate is drawn
as an arc, so it does not appear to cover the wires between its endpoints.
Green coarse wires are retained from the preceding scale; gray wires enter in
the product state. In 2D, colored windows show covering blocks, and dark links show the exact
scheduled pair gates in each round. Long pairs, including periodic seams,
curve around intervening sites. The coarse panel groups retained wires by
parent register (`R0`, `R1`, …); arrows point in the schedule's gate direction.
The drawings show structure and support, while `schedule.placements` gives the
exact gate sequence. The older `style="register"` view is a structural
fine-to-coarse wiring view, including for preparation schedules. Omit
`rg_step` to draw all scales. `layer=` remains an alias, and
`schedule.schematic_blocks()` remains the machine-readable block audit.

For an odd 2D retained hierarchy, the same API shows boundary disentanglers,
covering isometry cells, and retained parent registers:

```python
odd_grid = QMeraBuilder(
    shape=(3, 4), hierarchy="retained", bond_qubits=2,
    isometry={"block_size": (2, 2)},
)
odd_grid.draw_schematic(rg_step=0, style="clean", figsize=(16, 5))
```

## qMERA layout search and prototype comparison

`QMeraLayoutFinder` searches valid immutable RG scale plans and ranks them by
gate count, executable depth, reverse-lightcone width, and Hamiltonian-support
coverage. The result can be passed directly to `QMeraBuilder`:

```python
from pepsy.optimizers.qmera import QMeraGeometry, QMeraLayoutFinder

geometry = QMeraGeometry(shape=(6, 6), boundary="periodic")
report = QMeraLayoutFinder(geometry, max_layers=3).search({(0, 1): h2})
builder = QMeraBuilder(geometry=geometry, scales=report.best.scales)
```

The research prototype's serialized `U_q3_l*` files are flat gate-placement
streams, not RG schedules. Load them only for structural comparison:

```python
from pepsy.optimizers.qmera import load_qmera_prototype_layout

prototype = load_qmera_prototype_layout("/home/.../mera/U_q3_l1")
prototype_score = QMeraLayoutFinder(
    QMeraGeometry(shape=prototype.num_sites)
).score_prototype_layout(prototype, {(0, 1): h2})
```

This adapter deliberately does not infer qMERA scales from the prototype
stream.

Native Symmray fermion helpers are available under this module, but the
fermion convention is explicit. The `Fermion` helper can now be supplied to
`QMeraBuilder`; it infers the canonical spinful mode pair and mode-major
register order, then extracts qMERA mode terms internally. Do not mix this
native graded-array path silently with dense spin or Jordan-Wigner local
operators. The same model can supply both site-native MPS terms and qMERA mode
terms:

```python
import pepsy
from pepsy.optimizers.qmera import QMeraGeometry

fermion = pepsy.Fermion(
    spinful=True,
    symmetry="U1U1",
)
edges = ((0, 1), (1, 2))

site_terms = fermion.local_terms(edges, t=1.0, U=8.0)
gate_stream = fermion.gate_stream(
    edges, dt=0.01, sites=range(3), t=1.0, U=8.0
)

geometry = QMeraGeometry(shape=3, site_modes=("up", "down"))
qmera_terms = fermion.local_terms(
    geometry, layout="qmera", t=1.0, U=8.0
)
```

For the normal spinful Hubbard workflow, let the builder own the mode
expansion and conversion:

```python
from pepsy.optimizers.qmera import (
    QMeraBuilder,
    QMeraSymmrayFermionBackend,
    symmray_fermion_gate_registry,
)

backend = QMeraSymmrayFermionBackend.from_fermion(fermion)
builder = QMeraBuilder(
    shape=3,
    fermion=fermion,
    gate_registry=symmray_fermion_gate_registry(backend=backend),
    gate_family="symmray-fsim",
    product_state_factory=backend.product_state,
)
geometry = builder.geometry                 # inferred from the model
terms = builder.fermion_terms(t=1.0, U=8.0)  # explicit physical couplings
optimizer = builder.fermion_parametric_optimizer(
    energy_per_site=False,
    term_params={"t": 1.0, "U": 8.0},
)
```

Passing `site_modes` or `mode_order` remains useful when testing a custom
register convention. A site-layout object such as
`fermion.hamiltonian(edges, t=..., U=...)` is still a valid native MPS/PEPS Hamiltonian, but
it does not by itself specify qMERA's explicit mode registers or RG schedule;
use `fermion.local_terms(geometry, layout="qmera")` or the builder shortcut
above for that conversion.

For a larger native Torch workflow, use the corresponding examples maintained
in the separate `pepsy_examples` repository; the package API is demonstrated
by the builder flow above.

## 2D spin retained hierarchy

Set `hierarchy="retained"` for an unmoded spin lattice. Each scale tiles the
current *coarse grid*, applies two-qubit gates within each covering isometry
block, and retains up to `bond_qubits` physical wires as the next cell's
register. Boundary disentanglers pair matching register wires across both x
and y block faces. With odd dimensions, a final one-cell tail joins the
preceding block along that axis: a length 5 axis blocked by 2 becomes
`2 + 3`; length 7 blocked by 3 becomes `3 + 4`.

```python
from pepsy.optimizers.qmera import QMeraBuilder

builder = QMeraBuilder(
    shape=(5, 7),
    hierarchy="retained",
    isometry={"block_size": (2, 3), "circuit_depth": 1},
    disentangler={"block_size": 2, "circuit_depth": 1},
    bond_qubits=2,
    retention="balanced",
    pair_ansatz="z2_zz_yy_rx",
)
schedule = builder.build_schedule()
assert [(layer.input_grid_shape, layer.output_grid_shape)
        for layer in schedule.layers] == [
    ((5, 7), (2, 2)), ((2, 2), (1, 1)),
]
assert len(schedule.top_sites) == 2
```

`(3, 3)` and `(4, 4)` covering blocks use the same grid rule.
`layer.isometry_cell_blocks` records child-cell coordinates at that scale;
`layer.isometry_blocks` records their physical wire indices; and
`layer.retained_registers` records the selected parent wires. The
`"balanced"` policy spreads retained wires across each block's ordered child
registers; explicit retention uses the same `(scale, block)` keys as 1D.
State preparation runs
coarse to fine, applying each block's unitary-completion circuit before the
boundary disentanglers. Its adjoint gives the fine-to-coarse disentangler,
then isometry, order. The compiled local-energy, Torch, and JAX paths use the
same schedule.

This option currently supports spin qubits, brickwall isometry circuits, and
`placement="boundary-faces"` with `disentangler.block_size=2` on both
axes. A covering
block is a circuit of local two-qubit gates, not a dense multi-qubit unitary.
The default `hierarchy="site"` 2D path and explicit-mode fermion schedules
also absorb odd one-cell tails, while preserving their one-site-per-block
retention.

## 2D multimode RG schedules

For a two-dimensional geometry, `shape` describes physical lattice sites and
`site_modes` expands each site into explicit register modes. For example,
`site_modes=("up", "down")` gives two modes per site. The RG schedule blocks
physical sites first, then retains every mode on the representative coarse
site. Spatial gates are grouped by mode, so the schedule never silently turns
an `up`/`down` pair into a spatial fermion gate. `mode_order="mode-major"`
selects registers as
`((site_0, "up"), ..., (site_n, "up"), (site_0, "down"), ...)`; the default
`"site-major"` interleaves modes at each site.

Use `QMeraScaleSpec` when the RG geometry changes from one scale to the next.
For example, this generic 6x6 periodic plan reduces 6x6 to 3x3 with 2x2
covering blocks, then reduces 3x3 to one site with a 3x3 covering block and
vertical 3-site internal disentangler strips:

```python
from pepsy.optimizers.qmera import (
    QMeraBuilder,
    QMeraDisentanglerSpec,
    QMeraIsometrySpec,
    QMeraScaleSpec,
)

scale_plan = (
    QMeraScaleSpec(
        isometry=QMeraIsometrySpec(block_shape=(2, 2)),
        disentangler=QMeraDisentanglerSpec(
            block_shape=(2, 2),
            placement="boundary-square",
        ),
    ),
    QMeraScaleSpec(
        isometry=QMeraIsometrySpec(block_shape=(3, 3)),
        disentangler=QMeraDisentanglerSpec(
            block_shape=3,
            orientation="vertical",
            placement="within-block",
            circuit_depth=3,
        ),
    ),
)
builder = QMeraBuilder(
    shape=(6, 6),
    boundary="periodic",
    scales=scale_plan,
)
schedule = builder.build_schedule()  # 36 -> 9 -> 1 active sites
```

`orientation="vertical"` resolves an integer strip length to a 1x3 block in
the geometry's `(x, y)` convention. The three circuit rounds cover the three
nearest-neighbor edges of an odd periodic 3-site line without overlapping
gates. `placement="within-block"` is for internal disentanglers; the default
`"boundary-faces"` and `"boundary-square"` placements remain the
inter-isometry-boundary choices.

The schedule is independent of the operator representation. A native
spinful Fermi--Hubbard workflow therefore uses `U1U1` and the Symmray FSIM
registry:

```python
from pepsy.optimizers.qmera import (
    QMeraBuilder,
    QMeraGeometry,
    QMeraSymmrayFermionBackend,
    symmray_fermion_gate_registry,
)

geometry = QMeraGeometry(
    shape=(4, 4),
    site_modes=("up", "down"),
    mode_order="mode-major",
)
backend = QMeraSymmrayFermionBackend(symmetry="U1U1")
builder = QMeraBuilder(
    geometry=geometry,
    gate_registry=symmray_fermion_gate_registry(backend=backend),
    gate_family="symmray-fsim",
    isometry={"block_size": (2, 2), "gate_family": "symmray-fsim"},
    product_state_factory=backend.product_state,
)
```

For a 1D multimode geometry, use `mode_order="mode-major"` when spatial
brickwall gates should connect equal flavors. The scheduler partitions every
1D block by explicit mode before pairing, so native `U1U1` gates cannot
silently become spin-changing `up`--`down` gates. A one-site two-mode block
therefore receives no invalid gate rather than raising during gate materialization.

For the explicit 4x4 periodic construction, use a 2x2 square disentangler
around every inter-block face and a 2x2 covering unitary for each RG block:

```python
from pepsy.optimizers.qmera import (
    QMeraBuilder,
    QMeraDisentanglerSpec,
    QMeraGeometry,
    QMeraIsometrySpec,
    QMeraSymmrayFermionBackend,
    QMeraUnitarySpec,
    symmray_fermion_gate_registry,
)

geometry = QMeraGeometry(
    shape=(4, 4),
    boundary="periodic",
    site_modes=("up", "down"),
    mode_order="mode-major",
)
backend = QMeraSymmrayFermionBackend(
    symmetry="U1U1",
    site_modes=("up", "down"),
    mode_order="mode-major",
)
hubbard_unitary = QMeraUnitarySpec(
    gate_family="symmray-hubbard",
    family="fermion",
    arity_kind="mode",
    symmetry="U1U1",
    preserves_parity=True,
    metadata={"model": "fermi-hubbard", "term": "hopping"},
)
builder = QMeraBuilder(
    geometry=geometry,
    gate_registry=symmray_fermion_gate_registry(backend=backend),
    disentangler=QMeraDisentanglerSpec(
        block_shape=(2, 2),
        unitary=hubbard_unitary,
        placement="boundary-square",
        circuit_depth=2,
        periodic_wrap=True,
    ),
    isometry=QMeraIsometrySpec(
        block_shape=(2, 2),
        unitary=hubbard_unitary,
        circuit_depth=2,
        implementation="unitary-completion",
    ),
    max_layers=2,
)
schedule = builder.build_schedule()  # 4x4 -> 2x2 -> 1, including PBC wraps
```

Set `parameter_sharing` on `QMeraUnitarySpec` to choose the parameter scope:

- `"per-placement"`: every scheduled gate has independent parameters.
- `"per-block"`: gates in one RG block share parameters across brickwall
  rounds; different blocks remain independent.
- `"per-scale"`: all placements of one stage at one RG scale share parameters.
- `"per-axis"`: one parameter set per scale and spatial axis.
- `"shared"`: one parameter set is shared across all scales for that stage.

The default is `"per-placement"`.

`QMeraIsometrySpec` currently means a square local unitary circuit whose
output representative is retained by the RG schedule; it does not yet build a
rectangular tensor with an exact isometry constraint. Set
`implementation="true-isometry"` only when that backend is implemented.
`symmray-hubbard` is the native two-mode, number-conserving Hubbard hopping
layer used by this schedule (`symmray-fsim` is its lower-level alias). The
onsite `U n_up n_down - mu n` terms remain part of the
Fermi--Hubbard Hamiltonian supplied through `Fermion.local_terms(...)`; this
keeps gate topology and Hamiltonian terms separate while preserving the native
Symmray `U1U1` grading.

Use `convert_terms=False` when the Hamiltonian terms are already native
Symmray arrays. This preserves their graded contraction behavior.

## Grouped cones and direct validation

`builder.parametric_loss(...)` groups terms with the same input support and
scheduled gate topology. `builder.contraction_path_cache(...)` creates a lazy
cache with one reusable contraction optimizer per topology:

```python
path_cache = builder.contraction_path_cache(max_repeats=16)
energy = builder.parametric_loss(
    params,
    terms,
    schedule=schedule,
    convert_terms=False,
    path_cache=path_cache,
)
```

For debugging a new schedule, compare this local-cone result with
`builder.direct_parametric_loss(...)`. The latter constructs the complete
direct-gate qMERA tensor network and is intentionally a validation oracle,
not the optimization path. Agreement should be checked for representative
1D, 2D multimode, native Hubbard, and native Majorana cases.

## Majorana and pairing convention

The implemented true-Majorana convention is one spinless complex mode per
physical site with native `Z2` fermion parity:

```python
import pepsy as py
from pepsy.optimizers.qmera import (
    QMeraGeometry,
    qmera_symmray_majorana_terms,
    symmray_majorana_gate_registry,
)

fermion = py.Fermion(spinful=False, symmetry="Z2")
gamma_x = fermion.majorana_operator("x", site=0)  # c + c^dag
gamma_y = fermion.majorana_operator("y", site=0)  # -i (c - c^dag)
pairing = fermion.pairing_operator((0, 1), phase=0.2)
geometry = QMeraGeometry(shape=(2, 2), site_modes=("mode",))
terms = qmera_symmray_majorana_terms(
    geometry,
    fermion=fermion,
    coupling=0.4,
    pairing=0.2,
)
registry = symmray_majorana_gate_registry()
```

Individual Majoranas are parity odd (`charge=1`), while Majorana bilinears,
pairing operators, and their gates are neutral (`charge=0`). This is why the
Majorana path uses `Z2`: a single Majorana is not homogeneous under particle-
number `U1`, and generic pairing is not compatible with the charge-conserving
`U1U1` route. `Z2Z2` remains the natural future extension for two explicit
flavors with separately tracked parity, but it is not currently presented as
the default Majorana convention.

The implementation does not introduce independent real Majorana sites or a
separate BdG/Nambu symmetry. Nambu doubling is useful as a quadratic-model
calculation basis, but here the physical representation remains one complex
mode per site and the native Symmray `Z2` graded algebra carries the signs.
This keeps operator construction, parity-preserving gates, mode ordering, and
2D fermionic sign validation on one representation path.

Runnable versions of these workflows are in
`examples/qmera_fermion_hubbard_2d.py`,
`examples/qmera_fermion_hubbard_4x4_pbc.py`, and
`examples/qmera_majorana_2d.py`. For a first Torch energy comparison against
the U1U1 SymDMRG2 reference, see
`examples/qmera_fermion_hubbard_vs_symdmrg.py`. The comparison is variational:
the shallow qMERA energy is expected to remain above the better-converged DMRG
energy, while both calculations use the same physical Hubbard terms and
particle-number sector.


> API details are maintained as handwritten Markdown in this page.
