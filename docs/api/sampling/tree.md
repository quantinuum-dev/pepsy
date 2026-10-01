# `pepsy.sampling.tree`

`TreeSampler` is the tree-tensor-network analogue of `MpsSampler`: it draws
exact Born samples from a `TreeTensorNetwork` (or the live state of a
`TreeOptimizer`) with the same public surface and batched efficiency.

CPU threads are not capped by default (`threads=None`), matching TreeOptimizer.
Pass a positive `threads` value to explicitly limit BLAS/OpenMP around sampling.

Dense sampling can optionally use `TreeSampler(..., chunk_size=1000)` to
process at most 1,000 shots per contraction batch while returning the requested
total sample count. The default `chunk_size=None` sets no explicit shot limit;
the factor strategy can still reduce batches to meet its workspace target.
Chunking also tiles the first-child density environment, avoiding the complete
quartic bond tensor and its transpose copy. Its internal environment target is
1 GiB per tile (or one child-index slice when larger); this is not a bound on
total memory, which also includes state tensors and per-shot intermediates.
Chunk results fill one preallocated output instead of retaining every chunk
and concatenating a second full copy. A request fitting in one chunk reuses
its output directly. The full output and uniform-draw arrays still scale with
the total requested samples. Arrays remain on the selected backend/device;
CuPy sampling and scoring use the captured tree device even if the caller
changes the current CUDA device. Uniform draws retain the same
seed ordering across chunk sizes; floating-point rounding can still change
outcomes exactly at a probability threshold. Native Symmray already draws one
shot at a time and accepts this option without changing that algorithm.

Dense NumPy, Torch and CuPy trees can opt into the experimental, exact
`strategy="factor"`. The default remains `strategy="standard"`. Both use
structural pure vectors; the factor strategy additionally retains compact
mixed factors, groups identical measurement prefixes and keeps large branch
remainders grouped. It groups collapsed messages by their own subtree codes,
while conditional densities use the entire measured prefix. Each shot keeps
its own uniform draws. No bond/rank truncation or dtype reduction is applied.

```python
from pepsy.sampling import TreeSampler

sampler = TreeSampler(tree, strategy="factor", chunk_size=2048)
configs, probabilities = sampler.sample_arrays(8192, seed=0)
```

`cache_bytes` defaults to 128 MiB and bounds retained density keys and values
within one sampling call. Zero disables cross-chunk caching; within-chunk
grouping remains enabled. Cached hits and already-computed misses are reused
even when admission is refused. Caches are cleared on success or failure and
never survive a call or `refresh()`.

`workspace_bytes` defaults to 512 MiB. It controls factor projections,
remainder gathers and density-transfer tiles, and can reduce effective chunk
sizes. If a grouped remainder exceeds the target, sampling retries smaller
chunks with the same uniform draws. This target is not a total-memory limit:
source tensors, simultaneous buffers, cache growth/sorting copies, outputs and
Torch autograd graphs require additional memory, and a minimum one-shot or
one-slice tile can exceed the target. Signed 64-bit key overflow falls back to
exact row grouping without cross-chunk caching.

Torch source gradients, dtype/device preservation and native result conversion
are supported. Sharing thresholds are performance heuristics, so speedups
depend on bond dimensions and repeated configurations. Native Symmray trees
retain their existing block-sparse sampling algorithm with either strategy.

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
Existing canonical-region metadata limits this move to the required region/path;
a source already centered at the root requires no new gauging. Sampling batches
do not move the canonical center. Live tensor `left_inds` identify each proven
isometry's direction, allowing redundant lossless QR moves to be skipped.
After the source state changes, call
`sampler.refresh()` before sampling again. An immediate refresh after constructing
the sampler is unnecessary.

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

Within each dense batch, the incoming environment is shared until physical
conditioning distinguishes the shots. The root begins with an amplitude
vector. When only trivial unmeasured sibling bonds remain, an incoming vector
yields a pure child environment, carried as a vector instead of a squared
bond density. Unmeasured siblings can make it mixed, so those branches retain
the density route. Purity follows from the traversal, without a numerical
rank test or approximation. Shared first-child density transfers contract
the density with the node tensor directly, avoiding both repeated per-shot
work and a quartic transfer environment. This applies with and without
chunking; subsequent shot-dependent densities use the existing batched path.
The optimization preserves the uniform-draw order and introduces no
truncation. The standard strategy keeps densities local to a batch; the factor
strategy can reuse them across chunks within one call and clears its cache on
return. Unlike MPS prefix vectors, tree densities can be mixed because sibling
branches remain unmeasured.

Sampling and dense scoring release their recursive traversal closures on both
success and failure. Discarded sampler snapshots and shot buffers therefore do
not wait for cyclic garbage collection to release their CPU/GPU arrays.

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
