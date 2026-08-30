# API starting points

This page is the short, task-oriented map of PePsY's public API. Start here
when you know what you want to accomplish but not which class owns it. The
linked pages explain the workflow; the generated
<a href="reference/index.html">API reference</a> contains the complete
signatures, members, and source links.

## Choose by task

| Task | Canonical entry point | Detailed guide |
| --- | --- | --- |
| Build an MPS or PEPS from a product state | `pepsy.tensors` constructors | [Tensor constructors](tensors/constructors.md) |
| Prepare and contract a PEPS norm | `pepsy.boundary` | [Boundary metrics](boundary/metrics.md), [boundary states](boundary/states.md) |
| Apply gates and build operators | `pepsy.operators` | [Gates](operators/gates.md), [Hamiltonians](operators/hamiltonians.md) |
| Evolve an MPS circuit | `pepsy.optimizers.mps` | [MPS optimization](optimizers/mps.md) |
| Evolve or clean up a PEPS | `pepsy.optimizers.peps` | [PEPS optimization](optimizers/peps.md) |
| Run boundary sweeps | `pepsy.optimizers.sweep` | [Sweep optimization](optimizers/sweep.md) |
| Replay a circuit on a tree | `pepsy.optimizers.tree` | [Tree optimization](optimizers/tree.md) |
| Sample MPS, PEPS, vector, or tree states | `pepsy.sampling` | [Sampling](sampling/samplers.md) |
| Use symmetry-aware or fermionic tensors | `pepsy.tensors.symmetric` | [Symmetric tensors](tensors/symmetric.md) |
| Run belief propagation | `pepsy.bp` | [Belief propagation](bp.md) |
| Run variational Monte Carlo | `pepsy.vmc` | [VMC](vmc.md) |

## Key classes and functions

### Construct and contract tensor networks

- {py:class}`OneDMap <pepsy.tensors.maps.OneDMap>`,
  {py:func}`ps_to_mps <pepsy.tensors.constructors.ps_to_mps>`, and
  {py:func}`ps_to_peps <pepsy.tensors.constructors.ps_to_peps>` create states from product-state
  data. See [tensor constructors](tensors/constructors.md).
- {py:func}`build_bra_ket <pepsy.boundary.metrics.build_bra_ket>` prepares a
  tagged ket and double-layer norm network.
- {py:class}`BdyMPS <pepsy.boundary.states.BdyMPS>` stores PEPS boundary states,
  while {py:func}`contract_boundary <pepsy.boundary.metrics.contract_boundary>`
  performs the boundary contraction and returns diagnostics. See the
  [boundary API](boundary/metrics.md).

### Optimize states

- {py:class}`MpsOptimizer <pepsy.optimizers.mps.optimizer.MpsOptimizer>` replays
  one-dimensional gate streams with exact, MPO, FIT,
  DMRG, and mixed modes. See [MPS optimization](optimizers/mps.md).
- {py:class}`PepsOptimizer <pepsy.optimizers.peps.optimizer.PepsOptimizer>`
  applies gates to PEPS and optionally performs boundary or
  global cleanup. See [PEPS optimization](optimizers/peps.md).
- {py:class}`SweepOptimizer <pepsy.optimizers.sweep.optimizer.SweepOptimizer>`
  owns PEPS boundary environments and local/global sweep
  solvers. See [sweep optimization](optimizers/sweep.md).
- {py:class}`TreeOptimizer <pepsy.optimizers.tree.optimizer.TreeOptimizer>`
  replays gates on a rooted tree; {py:class}`TreeTensorNetwork
  <pepsy.optimizers.tree.ttn.TreeTensorNetwork>` is the underlying state
  container. See [tree optimization](optimizers/tree.md).

### Operators, sampling, and specialized domains

- `gate`, `rx`, and `rzz` are convenient operator constructors; the
  [operators guide](operators/gates.md) explains the broader gate and MPO API.
- {py:class}`MpsSampler <pepsy.sampling.samplers.MpsSampler>` and
  {py:class}`PepsSampler <pepsy.sampling.samplers.PepsSampler>` provide state
  sampling, with vector and
  stabilizer variants documented in [sampling](sampling/samplers.md).
- {py:class}`Fermion <pepsy.tensors.symmetric.Fermion>` is the entry point for
  symmetry-aware fermionic operators and
  states. See [symmetric tensors](tensors/symmetric.md).
- {py:class}`TorchLinalgConfig <pepsy.backends.config.TorchLinalgConfig>` is the
  single configuration point for PePsY's Torch
  SVD/QR policy. See the [MPS](optimizers/mps.md) and [PEPS](optimizers/peps.md)
  guides for usage in optimization workflows.

## Canonical import rule

Prefer responsibility-based namespace imports in application code:

```python
from pepsy.boundary import BdyMPS, build_bra_ket, contract_boundary
from pepsy.optimizers import MpsOptimizer, PepsOptimizer
from pepsy.sampling import MpsSampler
from pepsy.tensors import ps_to_mps, ps_to_peps
```

The top-level `pepsy` module keeps useful convenience aliases, but the owning
namespace is the stable place to discover and import a feature. See the
[package API map](package.md) for the complete namespace ownership table.

## Optional integrations

The core tensor, operator, boundary, optimizer, and sampling paths use the
base installation. Install only the integration you need:

| Feature | Extra |
| --- | --- |
| Torch autodiff and Torch-backed optimization | `.[torch]` |
| SciPy/NLopt solvers | `.[solvers]` |
| Tree layout search | `.[layout]` |
| Symmray symmetry and fermions | `.[symmetry]` |
| Stabilizer tensor networks | `.[stabilizer]` |
| NetKet/JAX and Torch VMC | `.[vmc]` |
