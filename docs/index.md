# PePsY documentation

PePsY is a tensor-network library for contraction, simulation, optimization,
sampling, symmetric tensors, and variational Monte Carlo workflows.

**PePsY** is a stylized shorthand for “PEPS in Python.” The public project
name uses `PePsY`; the Python package and import name remain `pepsy`.

The documentation is organized around complete workflows first and API
discovery second. If you already know the class or function you need, start
with the [API starting points](api/start_here.md); if you are learning PePsY,
follow one of the paths below.

## Start with a working result

- [Installation](installation.md)
- [Getting started](getting_started.md)
- [Quickstart](quickstart.md)

## Choose a workflow

| I want to... | Start here | Main API |
| --- | --- | --- |
| Contract a PEPS norm or overlap | [Contract a PEPS norm](tutorials/contract_norm.md) | `build_bra_ket`, `BdyMPS`, `contract_boundary` |
| Evolve or fit an MPS | [MPS API guide](api/optimizers/mps.md) | `MpsOptimizer` |
| Evolve or optimize a PEPS | [PEPS API guide](api/optimizers/peps.md) | `PepsOptimizer` |
| Optimize PEPS boundaries and sweeps | [Sweep API guide](api/optimizers/sweep.md) | `SweepOptimizer` |
| Replay a circuit on a tree | [Tree API guide](api/optimizers/tree.md) | `TreeOptimizer`, `TreeTensorNetwork` |
| Sample tensor-network states | [Sampling API guide](api/sampling/samplers.md) | `MpsSampler`, `PepsSampler` |
| Work with fermionic or symmetric tensors | [Symmetric tensor API](api/tensors/symmetric.md) | `Fermion` |
| Run variational Monte Carlo | [VMC API guide](api/vmc.md) | `pepsy.vmc` |

## User guides

- [Tutorials](tutorials/index.md)
- [How-to guides](howto/index.md)
- [Examples](examples.md)

## Reference

- [API starting points](api/start_here.md)
- [API reference and namespace map](api/index.md)
- [API stability policy](stability.md)
- [Package layout](development/package_layout.md)
- [Development notes and plans](development/README.md)
- [Fermi-Hubbard notes](development/fermi_hubbard_u1u1_mpo_notes.md)

```{toctree}
:hidden:

installation
getting_started
quickstart
tutorials/index
howto/index
examples
api/index
stability
development/package_layout
development/README
development/fermi_hubbard_u1u1_mpo_notes
```
