# Package API map

Pepsy has two layers:

- **Core namespaces** are the normal application API.
- **Advanced namespaces** contain optional backends, research workflows, and
  domain-specific optimizers.

Use the namespace import as the canonical path. The top-level `pepsy` module
retains many convenience aliases for compatibility, but it is not the best
place to discover the whole API.

## Core namespaces

| Area | Canonical import | Use it for |
| --- | --- | --- |
| Backends | `pepsy.backends` | Backend selection, conversion, and linalg registration |
| Tensors | `pepsy.tensors` | Tensor-network constructors, maps, contractions, and observables |
| Operators | `pepsy.operators` | Gates, gate application, MPO/PEPO builders, and Hamiltonians |
| Boundary | `pepsy.boundary` | PEPS norm, overlap, boundary states, and sweeps |
| Solvers | `pepsy.solvers` | Gradient and finite-difference parameter solvers |
| Fitting | `pepsy.fitting` | Local tensor fitting |
| Optimizers | `pepsy.optimizers` | MPS, MPO, PEPS, sweep, and global optimization workflows |
| Sampling | `pepsy.sampling` | MPS, PEPS, vector, and tree sampling |

Typical imports:

```python
from pepsy.boundary import BdyMPS, build_bra_ket, contract_boundary
from pepsy.operators import gate, rx, rzz
from pepsy.optimizers import MpsOptimizer, PepsOptimizer, SweepOptimizer
from pepsy.sampling import MpsSampler
from pepsy.tensors import OneDMap, ps_to_mps, ps_to_peps, tn_norm
```

For a shared backend contract across tensor-network classes, use
`pepsy.backend_infer(value)`. It accepts an array or an MPS/TTN and returns
`backend`, `dtype`, and `device`; Symmray inputs also report the underlying
`array_backend` used by their charge-sector blocks.

## Advanced namespaces

| Area | Canonical import | Notes |
| --- | --- | --- |
| Belief propagation | `pepsy.bp` | BP, relay gauges, loop corrections, and PNE |
| VMC | `pepsy.vmc` | Torch and NetKet/JAX variational Monte Carlo |
| qMERA | `pepsy.optimizers.qmera` | qMERA geometry, gates, and energy optimization |
| Stabilizer TN | `pepsy.optimizers.stabilizer_tn` | Stim tableau plus coefficient-MPS simulation |
| Tree TN | `pepsy.optimizers.tree` | Tree layout and circuit replay |
| Tree stabilizer | `pepsy.optimizers.tree_stabilizer` | Tableau plus tree-coefficient simulation |
| Symmetry | `pepsy.tensors.symmetric` | Symmray and fermionic tensor workflows |

Advanced domains can also be discovered through the explicit lazy namespace:

```python
from pepsy.experimental import bp, symmetry, stabilizer, tree, vmc
from pepsy.vmc import TorchVMCDriver
```

## Top-level convenience aliases

These common shortcuts remain supported:

```python
import pepsy

pepsy.BdyMPS
pepsy.rx
pepsy.SweepOptimizer
pepsy.ps_to_mps
pepsy.backend_infer
```

For new code, prefer the canonical namespace imports above. They make the
responsibility and optional-dependency boundary clear.

## Removed compatibility modules

Old flat paths such as `pepsy.core`, `pepsy.gates`, and `pepsy.optimize_mps`
were removed in the 0.4 package-layout cleanup. Use
`pepsy.tensors`, `pepsy.operators`, and `pepsy.optimizers.mps` instead.
