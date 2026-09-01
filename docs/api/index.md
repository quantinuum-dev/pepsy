# API reference

Start with the [API starting points](start_here.md) if you are choosing a
workflow or looking for the main class to use. The [package API map](package.md)
explains the core versus advanced namespaces and shows the recommended imports.
The pages below contain the detailed functions and classes for each area.

When the optional Sphinx build is used, the navigation also contains a
generated reference for the canonical source modules. It complements these
curated pages by listing signatures, parameters, class members, and direct
source links for each public object. The complete generated catalog is
available as the <a href="reference/index.html">generated API reference</a> in
the built site.

## Find the right entry point

Use the curated [API starting points](start_here.md) to move from a task to a
canonical namespace, main object, and detailed guide. Use the generated
reference when you need every method, property, overload, or source location.

## Core

- [Package exports](package.md)
- [Boundary metrics](boundary/metrics.md)
- [Boundary states](boundary/states.md)
- [Boundary sweeps](boundary/sweeps.md)
- [Tensor maps](tensors/maps.md)
- [Tensor constructors](tensors/constructors.md)
- [Tensor contractions](tensors/contractions.md)
- [Tensor observables](tensors/observables.md)
- [Symmetric tensors](tensors/symmetric.md)
- [Fitting](fitting/local.md)
- [Operators](operators/gates.md)
- [Explicit MPO automata](operators/automaton.md)
- [Higher-order MPO foundation](operators/higher_order_mpo.md)
- [MPO cluster expansion](operators/mpo_cluster.md)
- [Pauli-basis MPO](operators/pauli_mpo.md)
- [PEPO cluster expansion](operators/cluster_expansion.md)
- [Unified MPO/PEPO exponential API](operators/exponentials.md)
- [External circuit adapters](interop.md)
- [Hamiltonians](operators/hamiltonians.md)
- [Solvers](solvers/gradient.md), [finite differences](solvers/finite_difference.md)
- [Sampling](sampling/samplers.md), [tree sampling](sampling/tree.md)

## Advanced domains

- [Belief propagation](bp.md)
- [MPS optimization](optimizers/mps.md)
- [Gibbs purified MPS](optimizers/gibbs_mps.md)
- [MPO optimization](optimizers/mpo.md)
- [PEPS optimization](optimizers/peps.md)
- [Sweep optimization](optimizers/sweep.md)
- [Global optimization](optimizers/global_opt.md)
- [Energy optimization](optimizers/energy.md)
- [qMERA](optimizers/qmera.md)
- [Noise and trajectories](optimizers/noise.md)
- [Simulator planning](optimizers/planning.md)
- [Stabilizer tensor networks](optimizers/stabilizer_tn.md)
- [Symmetry-aware DMRG](optimizers/sym_dmrg.md)
- [Tree tensor networks](optimizers/tree.md)
- [Tree PEPS states](optimizers/tree_peps.md)
- [Tree stabilizer optimization](optimizers/tree_stabilizer.md)
- [VMC](vmc.md)

```{toctree}
:hidden:

start_here
package
boundary/metrics
boundary/states
boundary/sweeps
tensors/maps
tensors/constructors
tensors/contractions
tensors/observables
tensors/core
tensors/symmetric
fitting/local
operators/gates
operators/automaton
operators/higher_order_mpo
operators/mpo_cluster
operators/pauli_mpo
operators/cluster_expansion
operators/exponentials
interop
operators/hamiltonians
solvers/gradient
solvers/finite_difference
sampling/samplers
sampling/stabilizer
sampling/tree
bp
optimizers/mps
optimizers/gibbs_mps
optimizers/mpo
optimizers/peps
optimizers/sweep
optimizers/global_opt
optimizers/energy
optimizers/qmera
optimizers/noise
optimizers/planning
optimizers/stabilizer_tn
optimizers/sym_dmrg
optimizers/tree
optimizers/tree_peps
optimizers/tree_stabilizer
vmc
```
