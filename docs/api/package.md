# Package API map

Import from the namespace that owns a feature. The top-level `pepsy` aliases
remain available for compatibility. Advanced domains have their own APIs;
optional dependencies alone do not determine their stability.

## Core namespaces

| Area | Canonical import | Use it for |
| --- | --- | --- |
| Backends | `pepsy.backends` | Backend selection, conversion, and linalg registration |
| Tensors | `pepsy.tensors` | Tensor-network constructors, maps, contractions, and observables |
| Operators | `pepsy.operators` | Gates, gate application, MPO/PEPO builders, and Hamiltonians |
| Boundary | `pepsy.boundary` | PEPS norm, overlap, boundary states, and sweeps |
| Solvers | `pepsy.solvers` | Gradient and finite-difference parameter solvers |
| Fitting | `pepsy.fitting` | Local tensor fitting |
| Interoperability | `pepsy.interop` | Adapters for external circuit and tensor-network representations |
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
`pepsy.backends.backend_infer(value)`. It accepts an array or an MPS/TTN and returns
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
import pepsy.experimental

dir(pepsy.experimental)  # Lists domain names without loading their implementations.
```

Prefer the owning namespaces in the table for actual imports. Optional
installation and API stability are separate concerns; see the
[stability policy](../stability.md).

## Lazy namespace discovery

`dir(pepsy)` and `dir()` on its lazy entry namespaces (including `optimizers`,
`tensors`, `boundary`, and `vmc`) include their advertised exports before
those objects are loaded. Listing names neither resolves deprecated aliases
nor imports the corresponding numerical implementations. Accessing a symbol
still loads its implementation and may require the relevant optional extra.

The optimizer entry packages `mps`, `mpo`, `peps`, `sweep`, `tree`, `tree_peps`,
`energy`, and `qmera` also defer their implementation imports. This allows
geometry-only use such as:

```python
from pepsy.optimizers.qmera import QMeraGeometry

geometry = QMeraGeometry((2, 2))
```

This geometry construction does not load the numerical stack. Requesting
`QMeraBuilder` or an optimizer class loads the dependencies needed by that
implementation. Public names and their defining modules remain unchanged.

## Backend compatibility

Use `pepsy.backends.TorchLinalgConfig` to configure Torch SVD/QR policy.
See the [MPS](optimizers/mps.md) and [PEPS](optimizers/peps.md) guides for
optimizer-specific usage.

`pepsy.backends.backend_cupy(...)` uses upstream Autoray namespace handling
when it supports unhashable device objects. On affected older versions,
Pepsy bypasses the failing namespace cache while keeping the original device
object for array creation. No dependency upgrade is required for this fix.

Random FIT initialization inherits the template's backend, device, and dtype
unless a dtype is explicitly supplied. Complex normal samples use total
variance `scale**2`, with half in each of the real and imaginary components,
including the older-Autoray fallback. This corrects that fallback's previous
factor-of-two variance; existing seeded complex fallback samples therefore
change in magnitude. Seeds are reproducible within a route, but do not
promise identical samples across backends or dependency versions.

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
