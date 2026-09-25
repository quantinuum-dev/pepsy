# `pepsy.sampling.tree`

`TreeSampler` is the tree-tensor-network analogue of `MpsSampler`: it draws
exact Born samples from a `TreeTensorNetwork` (or the live state of a
`TreeOptimizer`) with the same public surface and batched efficiency.

```python
from pepsy.optimizers import TreeOptimizer
from pepsy.sampling import TreeSampler

opt = TreeOptimizer(gate_stream, n=nqubits, chi=chi)
sampler = TreeSampler(opt, seed=0)

batch = sampler.sample_batch(n_samples=4096, seed=0)
configs = batch.configs        # (n_samples, nqubits) int array
probs = batch.probs            # (n_samples,) Born probabilities
```

For dense NumPy, Torch, or CuPy trees, `TreeSampler` preserves the live array
backend by default. `backend="native"` makes that choice explicit, while
`backend="numpy"` requests a host copy. Explicit `backend="torch"` or
`backend="cupy"` requires the live tree tensors to already use that backend;
prepare gate/operator payloads with `TreeOptimizer.to_backend(...)`, and use
the resulting converter with `TreeTensorNetwork.apply_to_arrays(...)` when
moving the live TTN tensors before constructing the sampler. Native batch
results and raw arrays remain on the state device, and `batch.to_numpy()` or
`sample_arrays(..., to_numpy=True)` performs an explicit host conversion:

```python
sampler = TreeSampler(torch_tree_optimizer, backend="native")
batch = sampler.sample_batch(n_samples=4096, seed=0)
torch_configs = batch.configs
numpy_configs = batch.to_numpy().configs
```

Native Symmray trees can use `backend="symmray"` (or
`backend="native"`). The sampler keeps the canonical tree block-sparse and
uses native projected-norm contractions for sampling, amplitudes, and
probabilities. The default `backend="auto"` remains the fast dense batched
compatibility path for Symmray trees; select `symmray` when avoiding dense
materialization is more important than maximum throughput. The
`physical_code_maps` property exposes each source physical code's
`(charge, sector_offset)` pair for both ordinary Abelian and fermionic trees.

`amplitudes(..., to_numpy=False)` and `probabilities(..., to_numpy=False)`
likewise return arrays on the resolved native backend. The legacy `sample()`
method continues to return host Python lists.

The source object is never mutated: the sampler copies the tree, moves the
orthogonality centre onto the root, normalizes, and caches the per-node arrays.
After the source state changes, call `sampler.refresh()` before sampling again.

`sample_arrays(...)` returns the raw `(configs, probs)` tuple, and `sample(...)`
returns the list-based `TreeSampleResult`. To score existing configurations:

```python
amps = sampler.amplitudes(configs)        # <config|psi>
probs = sampler.probabilities(configs)    # |<config|psi>|**2
```

The sampler canonicalizes once with the centre on the root, so every non-root
node is isometric toward its parent bond. Sampling then walks the tree
depth-first carrying a per-sample reduced density matrix on the active parent
bond; unvisited sibling subtrees telescope to the identity, keeping the density
transfer bounded by the bond dimension squared. All samples share the cached
arrays and advance together through batched contractions, and each returned
probability is the exact product of that shot's conditional Born probabilities.

## Fermionic tree states

Native Symmray fermionic trees (for example a spinful `phys_dim=4` Fermi-Hubbard
tree) are sampled through the native path without calling `to_dense`. The
canonical tree is projected one physical code at a time and the exact native
branch norm supplies the conditional Born probability. This is a general
fallback for Abelian tree geometries; the existing dense path remains
available for high-throughput batches.

Sampled physical codes retain the source Symmray basis order. For supported
fermionic sectors, the batched and list results carry a
`FermionConfigurationEncoding` so the codes decode to `(n_up, n_down)`
occupations:

```python
from pepsy.tensors import Fermion
from pepsy.sampling import TreeSampler

# psi_tree: a native Symmray fermionic TreeTensorNetwork / TreeOptimizer state.
sampler = TreeSampler(psi_tree, fermion=Fermion(spinful=True, symmetry="U1U1"))
batch = sampler.sample_batch(n_samples=4096, seed=0)

codes = batch.configs            # (n_samples, nqubits) dense-basis codes
occ = batch.occupations()        # (n_samples, nqubits, 2) in (n_up, n_down)
```

The fermionic state is detected automatically, so passing `fermion=` is
optional; it only pins the recorded `symmetry`/`spinful` labels. Signed
`amplitudes(...)` follow the same dense basis convention and may differ from the
graded amplitude ordering by a per-configuration sign, whereas
`probabilities(...)` are exact.


> API details are maintained as handwritten Markdown in this page.
