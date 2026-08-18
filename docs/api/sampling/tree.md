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
tree) are sampled with the same `O(L)` sweep. The graded-canonical tensors are
densified once; because every Born probability contracts a tensor with its own
conjugate over the shared indices, the fermionic exchange signs enter squared
and cancel, so the plain dense sweep reproduces the exact graded probabilities
and marginals (validated to machine precision against the doubled-network
contraction). No per-conditional graded contraction is required, so fermionic
sampling runs at the same speed as the dense path.

Sampled physical codes follow Symmray's dense basis order — `empty, up, down,
up-down` for spinful `phys_dim=4` and `empty, occupied` for spinless
`phys_dim=2`. The batched and list results carry a
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
