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

Native Symmray fermion helpers are available under this module, but the
fermion convention is explicit. Use `QMeraGeometry(site_modes=("up", "down"))`
and `qmera_symmray_fermi_hubbard_terms(...)` for the native graded-array path;
do not mix it silently with dense spin or Jordan-Wigner local operators.

```{eval-rst}
.. automodule:: pepsy.optimizers.mera
   :members:
```
