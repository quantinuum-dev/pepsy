# `pepsy.sampling.tree`

`TreeSampler` is the tree-tensor-network analogue of `MpsSampler`: it draws
exact Born samples from a `TreeTensorNetwork` (or the live state of a
`TreeOptimizer`) with the same public surface and batched efficiency.

```python
from pepsy import TreeOptimizer, TreeSampler

opt = TreeOptimizer(gate_stream, n=nqubits, chi=chi)
sampler = TreeSampler(opt, seed=0)

batch = sampler.sample_batch(n_samples=4096, seed=0)
configs = batch.configs        # (n_samples, nqubits) int array
probs = batch.probs            # (n_samples,) Born probabilities
```

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

```{eval-rst}
.. automodule:: pepsy.sampling.tree
   :members:
   :undoc-members:
   :show-inheritance:
```
