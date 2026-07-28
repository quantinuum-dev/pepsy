# `pepsy.optimizers.mera`

MERA and qMERA energy helpers evaluate local Hamiltonian terms through reverse
lightcones rather than by contracting a full state for every term.

The dense MERA path wraps an existing MERA-like tensor network:

```python
import numpy as np
import quimb.tensor as qtn

from pepsy.optimizers import MeraEnergyOptimizer

zz = np.diag([1.0, -1.0])
h2 = np.kron(zz, zz).reshape(2, 2, 2, 2)
mera = qtn.MERA.rand(L=8, max_bond=2, dtype="complex128", seed=1)

opt = MeraEnergyOptimizer(mera, {(0, 1): h2, (2, 3): h2})
estimate = opt.energy()
```

The qMERA path is schedule-first. `QMeraBuilder` creates the geometry, RG
schedule, parameter dictionary, and local-cone chunks. The loss then rebuilds
only the selected lightcone for each local term.

```python
from pepsy.optimizers import QMeraBuilder, build_qmera_contraction_optimizer

builder = QMeraBuilder(
    shape=8,
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

For a fixed MERA-like state, `lightcone_energy(...)` exposes the same local
contraction primitive directly. It selects only the reverse cone, applies each
local operator with `.gate`, and can reuse one Pepsy/cotengra path per cone
topology:

```python
from pepsy.optimizers.mera import lightcone_energy

path_cache = builder.contraction_path_cache(max_repeats=16)
energy = lightcone_energy(
    mera,
    {(0, 1): h2, (2, 3): h2},
    energy_per_site=False,
    path_cache=path_cache,
)
```

This is the fixed-state analogue of `builder.parametric_loss(...)`; the latter
rebuilds the qMERA cone from a parameter dictionary on every evaluation.

For repeated optimization, compile static local-cone contractions once and
reuse them from NumPy, Torch, or JAX-compatible parameter dictionaries:

```python
compiled = builder.compile_parametric_lightcones(
    schedule=schedule,
    chunks=chunks,
    contraction_opt=optimize,
)
loss_fn = builder.compiled_parametric_loss_fn(
    schedule=schedule,
    compiled_chunks=compiled,
)
energy = loss_fn(params)
```

`QMeraParametricEnergyOptimizer` routes the same compiled loss through Pepsy's
gradient solvers:

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

### qMERA schematics

`QMeraSchedule.draw_schematic()` uses Quimb's manual `schematic.Drawing`
primitives. The default `style="clean"` view separates the input sites,
disentangler (`D`), isometry (`W`), and coarse-output stages, with colored
patches and arrows for the RG flow:

```python
drawing = schedule.draw_schematic(
    style="clean",             # or "register" for the low-level wiring view
    figsize=(14, 5),
    label_sites=True,
    label_blocks=True,
    scale_figsize=False,
)
```

The clean view is intended for explaining a schedule or a fermionic block
layout; `schedule.schematic_blocks()` remains the machine-readable placement
audit.

Native Symmray fermion helpers are available under this module, but the
fermion convention is explicit. The `Fermion` helper can now be supplied to
`QMeraBuilder`; it infers the canonical spinful mode pair and mode-major
register order, then extracts qMERA mode terms internally. Do not mix this
native graded-array path silently with dense spin or Jordan-Wigner local
operators. The same model can supply both site-native MPS terms and qMERA mode
terms:

```python
import pepsy
from pepsy.optimizers.mera import QMeraGeometry

fermion = pepsy.Fermion(
    spinful=True,
    symmetry="U1U1",
    t=1.0,
    U=8.0,
)
edges = ((0, 1), (1, 2))

site_terms = fermion.local_terms(edges)
gate_stream = fermion.gate_stream(edges, dt=0.01, sites=range(3))

geometry = QMeraGeometry(shape=3, site_modes=("up", "down"))
qmera_terms = fermion.local_terms(geometry, layout="qmera")
```

For the normal spinful Hubbard workflow, let the builder own the mode
expansion and conversion:

```python
from pepsy.optimizers.mera import (
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
terms = builder.fermion_terms()             # inferred from the model
optimizer = builder.fermion_parametric_optimizer(
    energy_per_site=False,
)
```

Passing `site_modes` or `mode_order` remains useful when testing a custom
register convention. A site-layout object such as
`fermion.hamiltonian(edges)` is still a valid native MPS/PEPS Hamiltonian, but
it does not by itself specify qMERA's explicit mode registers or RG schedule;
use `fermion.local_terms(geometry, layout="qmera")` or the builder shortcut
above for that conversion.

For a larger native Torch workflow, use the corresponding examples maintained
in the separate `pepsy_examples` repository; the package API is demonstrated
by the builder flow above.

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
from pepsy.optimizers.mera import (
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
from pepsy.optimizers.mera import (
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
from pepsy.optimizers.mera import (
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
from pepsy.optimizers.mera import (
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
