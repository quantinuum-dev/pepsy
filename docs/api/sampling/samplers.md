# `pepsy.sampling.samplers`

## MPS sampler quick API

`MpsSampler.sample_batch(...)` is the preferred batched interface for new code:

```python
sampler = MpsSampler(psi, one_d_to_two_d, backend="native")
batch = sampler.sample_batch(n_samples=4096, seed=0)

configs = batch.configs
probs = batch.probs
```

With dense Torch or CuPy MPS tensors, `backend="native"` keeps `configs` and
`probs` on the tensor device. Use `batch.to_numpy()` to copy to CPU NumPy, or
`batch.to_sample_result()` when the legacy list/grid `MpsSampleResult` is
needed. `sample_arrays(...)` remains available for direct tuple unpacking, and
`sample(...)` preserves the original `MpsSampleResult` behavior.
The native path builds backend-native right environments once, so it does not
require quimb to canonicalize Torch or CuPy tensors before sampling.
It caches those environments for the current MPS tensors: after modifying the
MPS, call `sampler.refresh()` before sampling again.

For a Symmray MPS, `MpsSampler` detects the block-sparse tensor data and
selects its native symmetry-aware route automatically. It caches a
right-canonical copy, preserves the source physical-code/charge map, then
samples by projecting one physical sector at a time and absorbing it into a
block-sparse boundary. No MPS tensor is converted with `to_dense()`. The
physical-code reconstruction is generic over Symmray's abelian charge maps,
including non-fermionic and fermionic `Z2`, `U1`, `U1U1`, and `Z2Z2` states.
In particular, a degenerate physical sector such as spinful `U1` remains a
selected sparse block rather than forcing the MPS dense. The sampler does not
need a Fermion object: it preserves the generic source basis map for every
site:

```python
sampler = MpsSampler(psi)  # Symmray MPS
print(sampler.physical_code_maps)
# ({0: (charge_0, 0), 1: (charge_1, 0), ...}, ...)
```

Each map is `physical_code -> (Symmray charge, offset within that charge
sector)`. It includes source sectors with zero amplitude that canonicalization
pruned, so it remains suitable for interpreting user configurations.
The symmetry-aware route currently requires an open-boundary MPS; periodic
MPS sampling needs a separate cyclic conditional-environment algorithm.

For a batch, all shots that have the same sampled prefix share its normalized
block-sparse boundary and local conditional distribution. This makes product,
low-entanglement, and charge-restricted states substantially cheaper to sample
while retaining the source MPS and its symmetry metadata unchanged. This
supports fermionic U1U1 starts directly:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
psi = pepsy.ps_to_mps(8, fermion=fermion)
sampler = MpsSampler(psi)  # resolved_backend == "symmray"
configs, probs = sampler.sample_arrays(256, seed=0)
```

Use `backend="symmray"` to require this route explicitly. If Symmray blocks
are backed by Torch or CuPy, contractions and local probability vectors remain
on that device; only the final discrete code choice is synchronized with the
Python control flow.

### Fermionic configuration codes

`configs` are always physical-index codes, not universal occupation labels.
Bind the `Fermion` definition when constructing the sampler to attach an
explicit, symmetry-aware code map to every batch before using it in a VMC or
local-estimator workflow:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="Z2")
sampler = MpsSampler(psi, backend="symmray", fermion=fermion)
batch = sampler.sample_batch(4096, seed=0)

physical_configs = batch.configs       # shape (batch, n_sites)
occupations = batch.occupations()      # shape (batch, n_sites, 2), (n_up, n_down)
encoding = batch.configuration_encoding
assert np.array_equal(encoding.encode(occupations), physical_configs)
```

Passing `fermion=...` directly to `sample_batch` remains supported when one
sampler is intentionally shared across compatible workflows. A bound Fermion
also lets `fermion_configuration_encoding()` and the diagonal-observable
helpers omit their repeated Fermion argument.

This is essential for collapsed sectors: spinful Symmray `Z2` uses physical
codes in `empty, double, up, down` order, while resolved `U1`, `U1U1`, and
`Z2Z2` use their sector-derived code order. The
`FermionConfigurationEncoding` object is immutable, site-aware, and rejects
invalid codes instead of silently interpreting them with a VMC-specific
default.

For predictable memory on high-entropy states, choose the prefix policy when
constructing the sampler:

```python
sampler = MpsSampler(
    psi,
    backend="symmray",
    prefix_strategy="auto",  # "prefix" or "serial" are also available
    max_prefix_groups=256,
)
configs, probs = sampler.sample_arrays(4096, seed=0)
print(sampler.symmray_sampling_stats)
```

`"auto"` shares equal prefixes while they still amortize a boundary: singleton
prefixes are completed serially, and retained groups obey both the active-group
cap and a per-level block-storage budget. `"prefix"` keeps every group allowed
by `max_prefix_groups`, including singletons; `"serial"` retains one boundary
at a time. The statistics report distinct conditional distributions,
candidate contractions, charge-pruned branches, peak active groups, and
serial/adaptive fallbacks. Set `max_prefix_groups=None` to remove the hard
group cap while retaining the `"auto"` reuse decision.
Use `python benchmarks/mps_symmray_sampling.py --help` for comparable
product/entangled, fermionic/non-fermionic sampling measurements. By default,
it runs the sparse Symmray, dense-native, and Quimb sampler variants in
separate processes and reports median throughput, setup peak RSS, and
post-setup resident RSS for each variant.
Use `--variants symmray` when comparing prefix policies only.
For fermionic `Z2`/`Z2Z2` inputs it reports the dense variants as unsupported:
naively expanding their graded virtual legs is not a state-preserving dense MPS
conversion, so only the Symmray result is a valid comparison.

Torch sampling is inference-only by default, avoiding an autograd graph for
the discrete draw and its sampled probabilities. Use `track_grad=True` only
when those sampled Born probabilities need gradients:

```python
configs, probs = sampler.sample_arrays(1024, track_grad=True)
```

For repeated, device-resident, unseeded Torch batches, construct with
`torch_compile=True` to opt into `torch.compile`. If the installed compiler
cannot support the workload, the sampler automatically uses its eager path.
Seeded sampling and calls that convert results to NumPy always use eager mode.

For evaluating existing configurations, use:

```python
amps = sampler.amplitudes(configs, to_numpy=False)
probs = sampler.probabilities(configs, to_numpy=False)
```

Both methods evaluate the whole batch in one backend-native sweep.

### Fermionic diagonal observables

`MpsSampler` can estimate diagonal observables of a
`pepsy.tensors.Fermion` directly from Born samples. This safely handles the
block-ordering of spinful Symmray `Z2`, `U1`, and `U1U1` physical indices:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
sampler = MpsSampler(psi, backend="symmray", fermion=fermion)
estimate = sampler.estimate_fermion_diagonal(
    "doublon",             # average n_up n_down per site
    n_samples=16_384,
    seed=0,
)
print(estimate.mean, estimate.standard_error)

density_corr = sampler.estimate_fermion_diagonal(
    "density_correlation", # average n_i n_j over the listed pairs
    pairs=((0, 1), (2, 3)),
    n_samples=16_384,
    seed=1,
)
```

Supported names are `"occupation"` (mean occupation on `sites`),
`"total_charge"` (occupation sum), `"doublon"`, and
`"density_correlation"`. `fermion_diagonal_values(configs, fermion, ...)`
evaluates the same observable on an existing batch, which is useful for exact
short-chain probability sums. The returned `MpsDiagonalEstimate` contains the
sample mean and a standard error based on the unbiased sample variance.
Hopping, pairing, and spin-flip
operators are not diagonal and require a separate fermionic local-estimator
calculation.


> API details are maintained as handwritten Markdown in this page.
