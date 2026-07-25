# Package Layout Migration

The package uses responsibility-based namespaces. Implementations live under
these namespaces, while old flat module paths remain warning-emitting
compatibility facades during the deprecation window. Top-level convenience
symbols such as `pepsy.SweepOptimizer` and `pepsy.rx` still work, but new
submodule imports should use the canonical layout.

## Target Namespaces

```text
pepsy.backends      backend selection, conversion, and linalg registration
pepsy.tensors       tensor-network maps, constructors, contraction helpers
pepsy.operators     gates, gate application, MPO/PEPO builders, Hamiltonians
pepsy.boundary      boundary-MPS states, sweeps, metrics
pepsy.solvers       gradient and finite-difference parameter solvers
pepsy.fitting       local tensor fitting routines
pepsy.optimizers    high-level MPS, MPO, PEPS, sweep, and global optimizers
pepsy.sampling      MPS, vector, and PEPS samplers
pepsy.vmc           optional Torch and NetKet/JAX VMC adapters
pepsy.bp            belief propagation, loop expansions, and PNE methods
pepsy.experimental  explicit lazy entry points for advanced domains
pepsy.extensions    compatibility entry points for optional integrations
pepsy._internal     private formatting and utility helpers
```

## Import Rules

Use the clearer namespaces for submodule imports:

```python
from pepsy.boundary import BdyMPS, contract_boundary
from pepsy.optimizers import SweepOptimizer, MpsOptimizer, PepsOptimizer, SimpleUpdateGen
from pepsy.operators import rx, rzz, gate
from pepsy.tensors import (
    haar_random_state,
    ps_to_peps,
    ps_to_3dpeps,
    ps_to_mps,
    ps_to_ttn,
)

# Optional or advanced domains can be made explicit at the call site.
from pepsy.experimental import bp, symmetry, vmc
from pepsy.vmc import TorchVMCDriver
```

When a leaf module is needed, import the new implementation path directly:

```python
from pepsy.optimizers.sweep.optimizer import SweepOptimizer
from pepsy.boundary.states import BdyMPS
from pepsy.operators.gates import rx, gate
```

Optimizer implementations are split into subpackages. Prefer public class
imports from `pepsy.optimizers`; use leaf paths such as
`pepsy.optimizers.mps.optimizer` only for implementation-level tests or
internal development.

## Migration Order

1. Keep new packages green with public API tests.
2. Keep backend conversion and tensor validation under the new namespaces.
3. Keep the boundary subsystem together: states, sweeps, metrics.
4. Keep optimizer implementations under `pepsy.optimizers`.
5. Split `pepsy.tensors.core` into maps, constructors, contractions, and observables.
6. Keep standard gate primitives separate from routing and tensor-network application.
7. Keep optional/advanced integrations behind `pepsy.extensions` and lazy imports.

After each phase, run:

```bash
pytest -q tests/test_public_api.py
pytest -q tests/test_package_layout.py
pytest -q tests/test_prepare_boundary_inputs.py
pytest -q tests/test_gate.py
```
