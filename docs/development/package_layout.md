# Package Layout Migration

The package has moved from a flat module layout to responsibility-based
namespaces. The implementation now lives under the new namespaces, and the old
root-level module aliases have been removed. Top-level convenience symbols such
as `pepsy.SweepOptimizer` and `pepsy.rx` still work, but submodule imports
should use the new package layout.

## Target Namespaces

```text
pepsy.backends      backend selection, conversion, and linalg registration
pepsy.tensors       tensor-network maps, constructors, contraction helpers
pepsy.operators     gates, gate application, MPO/PEPO builders, Hamiltonians
pepsy.boundary      boundary-MPS states, sweeps, metrics
pepsy.solvers       gradient and finite-difference parameter solvers
pepsy.fitting       local tensor fitting routines
pepsy.optimizers    high-level MPS, MPO, PEPS, and energy optimizers
pepsy.sampling      MPS, vector, and PEPS samplers
pepsy._internal     private formatting and utility helpers
```

## Import Rules

Use the clearer namespaces for submodule imports:

```python
from pepsy.boundary import BdyMPS, contract_boundary
from pepsy.optimizers import SweepOptimizer, MpsOptimizer, PepsOptimizer
from pepsy.operators import rx, rzz, gate
from pepsy.tensors import haar_random_state, ps_to_peps, ps_to_3dpeps, ps_to_mps
```

When a leaf module is needed, import the new implementation path directly:

```python
from pepsy.optimizers.sweep import SweepOptimizer
from pepsy.boundary.states import BdyMPS
from pepsy.operators.gates import rx, gate
```

## Migration Order

1. Keep new packages green with public API tests.
2. Keep backend conversion and tensor validation under the new namespaces.
3. Keep the boundary subsystem together: states, sweeps, metrics.
4. Keep optimizer implementations under `pepsy.optimizers`.
5. Split `pepsy.tensors.core` into maps, constructors, contractions, and observables.
6. Split `pepsy.operators.gates` last because it mixes primitives, routing, builders, and
   1D/2D/3D application logic.

After each phase, run:

```bash
pytest -q tests/test_public_api.py
pytest -q tests/test_package_layout.py
pytest -q tests/test_prepare_boundary_inputs.py
pytest -q tests/test_gate.py
```
