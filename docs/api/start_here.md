# API starting points

Choose the task you want to perform. Each guide explains the main API and
its options; the <a href="reference/index.html">generated reference</a> lists
full signatures and source locations in the built documentation.

## Choose by task

| Task | Entry point | Guide |
| --- | --- | --- |
| Build an MPS or PEPS | `pepsy.tensors` constructors | [Tensor constructors](tensors/constructors.md) |
| Contract a PEPS norm or overlap | `build_bra_ket`, `BdyMPS`, `contract_boundary` | [Boundary metrics](boundary/metrics.md) |
| Apply gates or build operators | `pepsy.operators` | [Gates](operators/gates.md), [Hamiltonians](operators/hamiltonians.md) |
| Replay an MPS circuit | `MpsOptimizer` | [MPS](optimizers/mps.md) |
| Prepare a thermal state | `GibbsMps` | [Gibbs MPS](optimizers/gibbs_mps.md) |
| Evolve a PEPS | `PepsOptimizer` | [PEPS](optimizers/peps.md) |
| Optimize PEPS boundaries | `SweepOptimizer` | [Sweeps](optimizers/sweep.md) |
| Replay a circuit on a tree | `TreeOptimizer` | [Tree networks](optimizers/tree.md) |
| Sample a tensor-network state | `pepsy.sampling` | [MPS/PEPS sampling](sampling/samplers.md), [tree sampling](sampling/tree.md) |
| Use symmetric or fermionic tensors | `SymMPS`, `SymPEPS`, `Fermion` | [Symmetric tensors](tensors/symmetric.md) |
| Run belief propagation | `pepsy.bp` | [BP](bp.md) |
| Run variational Monte Carlo | `pepsy.vmc` | [VMC](vmc.md) |
| Adapt an external circuit | `pepsy.interop` | [Circuit adapters](interop.md) |

## Import from the owning namespace

```python
from pepsy.boundary import BdyMPS, build_bra_ket, contract_boundary
from pepsy.optimizers import MpsOptimizer, PepsOptimizer
from pepsy.sampling import MpsSampler
from pepsy.tensors import ps_to_mps, ps_to_peps
```

Use the [package map](package.md) for namespace ownership, lazy discovery, and
backend compatibility. Existing top-level `pepsy` aliases remain available.

Install the [optional features](../installation.md#optional-features) needed
by your workflow. Optional dependencies and API stability are separate
concerns; see the [stability policy](../stability.md).
