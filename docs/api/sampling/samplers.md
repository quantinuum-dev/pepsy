# `pepsy.sampling.samplers`

Prefer public imports from `pepsy.sampling`. Implementations are organized in
`sampling.mps`, `sampling.vector`, `sampling.peps`, and `sampling.bp`, with
shared records in `sampling.results`. Historical `sampling.samplers` imports
and serialized class references remain compatible.

## Direct PEPS sampler

`PepsSampler` has an exact reference mode and a compressed boundary-MPS mode.
The exact mode is:

```python
from pepsy.sampling import PepsSampler

sampler = PepsSampler(peps)
result = sampler.sample(samples=16, seed=0)
```

The boundary-MPS mode separates the conditioned ket boundary from the future
double-layer environment:

```python
sampler = PepsSampler(
    peps,
    chi=64,                     # χ: future double-layer environment
    chi_prime=32,               # χ′: conditioned single-layer ket
    boundary_engine="dmrg",     # or "quimb-mps"; "auto" picks DMRG here
    ket_compression="quimb",    # or "fit", or None
    cutoff="auto",              # Resolve from the working tensor dtype
    cutoff_mode="auto",         # Relative discarded squared weight (rsum2)
)
```

### Mode selection and compatible names

| Construction | Selected behavior |
| --- | --- |
| `PepsSampler(peps)` | Exact full-network conditionals |
| `PepsSampler(peps, chi=64, chi_prime=32)` | DMRG future boundaries, Quimb ket compression |
| `PepsSampler(peps, chi_prime=32)` | Conditioned ket compression with identity future caps |
| `PepsSampler(peps, chi=64, ket_compression=None)` | DMRG future boundaries, uncompressed conditioned ket |
| `PepsSampler(peps, boundary_engine="quimb-mps", ket_compression=None)` | Identity future caps and uncompressed ket |

The default `boundary_engine="auto"` (also `None`) selects DMRG when either
positive bond cap is supplied, otherwise exact contraction. A positive `chi`
alone therefore requires `chi_prime` or explicit `ket_compression=None`.
Explicit `boundary_engine="exact"` rejects positive bond caps. A zero `chi`
disables future environments in boundary mode; it does not by itself select
boundary sampling. `chi_prime` must be positive when supplied.

Existing names remain supported: `marginal_chi` is an alias for `chi`, and
`sample_chi` is an alias for `chi_prime`. If both spellings are non-None,
they must agree; conflicting values raise `ValueError`. A `None` value lets
the other spelling supply the value. Read-only `sampler.chi` and
`sampler.chi_prime` expose the resolved caps. Construct a new sampler to
change contraction options; `refresh()` rebuilds it after source tensors change.

### Automatic truncation policy

`cutoff="auto"` and `cutoff_mode="auto"` are the defaults, following ordinary
`MpsOptimizer` state compression:

| Working tensor dtype | Resolved cutoff | Resolved mode |
| --- | --- | --- |
| `complex64` / `float32` | `1e-6` | `rsum2` |
| `complex128` / `float64` | `1e-12` | `rsum2` |

The shared policy uses `1e-3` for 16-bit data, although dense linear algebra
support for those dtypes depends on the selected backend. Resolution happens
**after `to_backend` conversion** and repeats on `refresh()`. Inspect
`sampler.cutoff` and `sampler.cutoff_mode` for resolved values.

`rsum2` thresholds relative discarded squared singular-value weight. It is a
local truncation rule, not a bound on the total sampling or observable error.
`cutoff_mode=None` also selects `rsum2`. Explicit modes `rel`, `abs`, `sum1`,
`sum2`, `rsum1`, and `rsum2` override the policy. A numeric `cutoff` stays fixed
across dtype changes; use `cutoff=0.0` to disable cutoff-driven rank reduction.
The independent χ/χ′ bond caps still apply.

Both options reach Quimb future-environment compression and the conditioned
ket compressor, including the compressed initial guess used by `"fit"`.
They are also forwarded to the DMRG future provider. Its current one-site FIT
updates retain fixed bond dimensions: χ controls that rank and there is no
singular-value thresholding during those one-site updates. Exact contraction
and `ket_compression=None` perform no ket truncation.

### Sequential sweep and the two cutoffs

Boundary mode defaults to the simple conditioned-boundary sweep:

1. Cache only future double-layer environments, from the last row toward the
   first, using `chi` (**χ**).
2. Combine the current row, its cached future, and the previously sampled
   single-layer ket boundary.
3. Form a local `d × d` conditional rho, sample its normalized real diagonal,
   and fix that physical value on both ket and bra. Continue across the row.
4. After the whole row is fixed, absorb its **single ket layer** into the
   conditioned boundary and compress with `chi_prime` (**χ′**).
5. Continue to the next row. Accumulate the conditional log probabilities and
   contract the original private ket for the final sampled amplitude.

| Quantity | Dimension / role |
| --- | --- |
| Current physical rho | `d × d` (`2 × 2` for qubits) |
| Conditioned ket boundary | One outgoing PEPS bond per site; MPS bond capped by χ′ when compression is enabled |
| Future double-layer boundary | Ket and bra outgoing bonds; compressed environment bonds controlled by χ |
| Joint row probability | Product of site conditionals given the preceding sampled rows |

Sampling sites successively implements a joint row draw without materializing
its exponentially large density matrix. The current indexing fixes `y` and
advances `x` within a row, then advances `y`. Describing these slices as columns
is the same construction after exchanging axes. Future environments are cached
in the direction opposite to sampling. They are reused unchanged across samples;
the conditioned ket boundary is different for different sampled prefixes.

`ket_compression=None` leaves the conditioned boundary uncompressed, so χ′
does not cap its represented bonds. `chi=0` or `None` uses identity
future caps, an explicitly different approximate proposal. Quimb can retain the
outermost future row as exact factored PEPS tensors before its first compression;
χ caps the compressed bonds, not every original bond in that factored row.
The DMRG provider builds only the requested future boundary MPSs lazily.

The final row needs no further boundary update. Temporary boundary normalization
does not change the original PEPS or its returned amplitudes. This sampler uses
Pepsy's standard `X*`, `Y*`, and `I*` lattice tags; incompatible custom tag schemes
are rejected by the norm-network builder. Boundary mode rejects periodic edges;
the full-contraction exact mode can handle them.

### Array backend and conversion

By default, the sampler infers the backend, dtype, and device from the PEPS
arrays. NumPy, Torch (CPU or CUDA), and JAX arrays use their own contractions,
identity caps, local density matrices, conditional probabilities, and random
draws. The public `sampler.backend` reports the inferred array backend;
`boundary_engine` independently selects the contraction algorithm.

Supply a callable `to_backend` to convert a private copy explicitly:

```python
import torch
from pepsy.backends import backend_torch
from pepsy.sampling import PepsSampler

sampler = PepsSampler(torch_peps)  # infer the existing dtype and device

sampler = PepsSampler(
    numpy_peps,
    to_backend=backend_torch(device="cuda:0", dtype=torch.complex128),
    chi_prime=32,
    chi=64,
    boundary_engine="dmrg",
)

# For a NumPy PEPS, a JAX converter can be supplied similarly:
# sampler = PepsSampler(numpy_peps, to_backend=jax.numpy.asarray)
```

The source PEPS is preserved, including when a converter modifies its argument
in place. `refresh()` repeats an explicit conversion, or re-infers the source
backend if no override was supplied. Mixed backends/devices and incompatible
dtypes require a converter that makes the tensors consistent. The sampler
currently requires dense floating or complex tensors. Integer arrays require
an explicit converter to a floating or complex dtype.

Seeded draws are reproducible for a given backend, device, and sampling method.
Sequences can differ between NumPy, Torch, JAX, serial sampling, and grouped
sampling. Grouped sampling uses native uniform draws and inverse cumulative
probabilities; its seeded sequences can differ from earlier releases that
called categorical sampling separately for each prefix group. Local-rho
validation uses the real dtype's precision and clips only roundoff-scale
negative diagonals; larger negative values and invalid traces raise an error.
Hermiticity diagnostics scale the matrix before taking norms, avoiding
complex64 overflow while preserving `norm(rho - rho.H) / max(norm(rho), 1)`.

Quimb projection and prefix grouping still use Python/NumPy indices. In
`sample_batch`, all active prefix groups at a site share one native probability
validation and draw operation, with one validation-scalar read and one integer
choice-array transfer per site. Serial sampling validates each conditional
separately. The existing result lists contain host scalars. `rho_diagnostics`
converts its cached scalar diagnostics when accessed. This is an eager sampler;
it does not provide a fully compiled JAX or Torch sampling loop, and contractions
and boundary compression still run separately for distinct prefix groups.

### Boundary and result semantics

`chi_prime` caps the conditioned single-layer ket boundary. `chi`
caps the optional future double-layer environment. The `dmrg` engine prepares
future boundaries with Pepsy `BdyMPS`/`CompBdy`; `quimb-mps` uses Quimb's MPS
environment cache. Ket compression is performed separately with Quimb MPS
compression or Pepsy `FIT`.

`result.configs` contains row-major physical-index configurations, while
`result.omegas` and `result.ps` contain proposal probabilities and projected
PEPS amplitudes as `(mantissas, exponents)`, representing `m * 10**e`. For multiple shots,
`sample_batch(...)` shares a local Quimb conditional network until shot
prefixes diverge:

```python
batch = sampler.sample_batch(samples=256, seed=0)
log_q = batch.log_probabilities        # natural log of proposal q(S)
log_abs_psi = batch.log_abs_amplitudes # natural log of |Psi(S)|
log_w = batch.log_weights             # 2 * log_abs_psi - log_q
print(sampler.batch_stats)
print(sampler.rho_diagnostics)
```

These log properties return one-dimensional NumPy arrays, computed directly
from the scaled pairs without forming `10**e`. They copy backend scalars to
the host when accessed and preserve the existing `configs`, `omegas`, and
`ps` fields. Weights are unnormalized and use the original PEPS amplitudes.
A zero amplitude has `log_abs_amplitudes = -inf` and, for positive proposal
probability, `log_weights = -inf`. Sampling produces positive-probability
configurations; manually constructed zero-probability results retain ordinary
logarithmic division semantics (infinity or undefined `0/0`).

This uses prefix groups rather than adding a shared batch index to PEPS
tensors, because a repeated Quimb index would be contracted as an ordinary
bond. The batch contracts each distinct final configuration's amplitude once from
the private PEPS for importance weights.

An optional optimization for compact boundary centers uses a Quimb transfer cache: its
right suffixes are contracted once, while the conditioned left prefix is
updated immediately after each sampled site. Traced transfers reuse the local
column contraction. Row bonds and identity future caps are reused until
`refresh()`.

`row_cache_max_bytes` defaults to **0**, selecting the simple sweep above.
Supply a positive budget, for example 64 MiB (`64 * 2**20`), to opt into dense
transfers. Before allocation, the sampler estimates row storage and workspace from the actual
PEPS/future bonds, conditioned-boundary bond bounds, dtype, and maximum number
of live prefix groups. Estimates above the budget use the existing local-center
contractions. Setting the budget to zero disables dense row caches. Larger
centers with collapsed future MPSs and highly fragmented batches can also use
the reference route even when they fit the budget. This option bounds the
estimated cache allocation; it is **not** a cap on total process/device memory,
contraction-planner workspace, or all retained boundary states.

```python
sampler = PepsSampler(
    peps, chi_prime=32, chi=64, boundary_engine="dmrg",
    row_cache_max_bytes=64 * 2**20,
)
batch = sampler.sample_batch(samples=64, seed=0)
print(sampler.row_cache_stats)
print(sampler.batch_stats["conditional_batches"])
```

`row_cache_stats` reports the selected `mode`, suffix builds, prefix updates,
`estimated_cache_bytes` (`None` when caching is disabled), `cache_budget_bytes`, and `cache_decision`
(`within-budget`, `memory-budget`, `disabled`, `prefix-count`, or `large-future`).
`batch_stats["conditional_batches"]` counts the site-level validation/draw
batches. Amplitudes still come from the original private PEPS, and reported
proposal probabilities still follow the selected boundary approximation.

### Likelihoods, zero branches, and truncation limits

The sampler accumulates **sums of log conditional probabilities**. Internally
it uses base-10 logs for the result's mantissa/exponent format. The public
`log_probability(config)` returns the **natural log**:

```python
log_q = sampler.log_probability(config)
q = sampler.probability(config)  # exp(log_q), if representable as a float
```

Exact and default boundary likelihood queries use exponent-stripped local
contractions. This protects rare-configuration likelihoods against intermediate
underflow; the common rho scale cancels from each normalized conditional.
`rho_diagnostics` describes the rho actually evaluated, so traces from these
queries are scaled. Sampling diagnostics retain their usual contraction scale.
An opt-in dense transfer cache still materializes ordinary tensors and does not
provide the same scaling protection. Log bookkeeping also does not guarantee
that arbitrary ill-scaled input contractions or final raw amplitudes cannot
overflow/underflow.

A genuine zero branch returns `log_q = -inf` and `q = 0`. A finite log likelihood
can also produce `q = 0` when the final value underflows a Python float. All
configuration entries are validated before an early zero-branch return.
Non-finite traces and materially negative diagonals remain errors. Missing
requested future boundaries raise an error instead of silently using identities.

With boundary truncation, `q(S)` can differ from the PEPS Born probability.
Importance weights `abs(Psi(S))**2 / q(S)` require nonzero proposal probability
wherever the PEPS amplitude is nonzero. A small χ′ can remove this support:
a tested two-row state with two equally likely target configurations gives
proposal probabilities `(1, 0)` at χ′ = 1 and `(1/2, 1/2)` at χ′ = 2.
Reweighting cannot restore a configuration that is never sampled. Check cutoff
convergence and support on tractable reference cases; no probability floor or
uniform mixture is inserted implicitly.

Run the small [sampling example](../../../examples/peps_sampling.py) with
`python examples/peps_sampling.py`. It demonstrates χ/χ′, batch log results,
and a comparison to exact proposal probabilities on a 2×3 PEPS.

## MPS sampler quick API

`MpsSampler.sample_batch(...)` is the preferred batched interface for new code:

```python
sampler = MpsSampler(psi, one_d_to_two_d, backend="native")
batch = sampler.sample_batch(n_samples=4096, seed=0)

configs = batch.configs
probs = batch.probs
```

For a stabilizer tensor-network state `|psi> = C|nu>`, use
[`StabilizerMpsSampler`](stabilizer.md). It keeps the same batch/result shape while
using frame-mapped Pauli projectors, so X/Y/Z product-basis sampling remains
scalable without forming the dense physical statevector. It is a separate
sampler from `StabilizerMpsSimulator`: pass an existing optimizer, or pass `(C, nu)`
with optimizer construction options such as `chi` and `mode`. Set
`disentangle=True` to use branch-local basis-updating measurements. The legacy
`absorb_basis=True` keyword remains accepted as an alias.

With dense Torch or CuPy MPS tensors, `backend="native"` keeps `configs` and
`probs` on the tensor device. Use `batch.to_numpy()` to copy to CPU NumPy, or
`batch.to_sample_result()` when the legacy list/grid `MpsSampleResult` is
needed. `sample_arrays(...)` remains available for direct tuple unpacking, and
`sample(...)` preserves the original `MpsSampleResult` behavior.
The native path builds backend-native right environments once, so it does not
require quimb to canonicalize Torch or CuPy tensors before sampling.
Complex right environments retain their bra-row/ket-column orientation when
forming Born weights. This corrects the earlier transposed contraction, which
could bias native sampling and probability evaluation for complex states.
It caches those environments for the current MPS tensors: after modifying the
MPS, call `sampler.refresh()` before sampling again.

`sampler.entanglement_entropy(cut=None)` measures the captured source MPS at
the requested bond (`None` selects the middle cut). It delegates to
`pepsy.tensors.mps_entanglement_entropy`, so the entropy diagnostic uses the
source array backend rather than the legacy Quimb sampling copy.

### Dense exact-vector sampler

`VecSampler` supports computational-basis and Pauli-basis sampling from a
dense state vector:

```python
sampler = pepsy.VecSampler(state, one_d_to_two_d)
batch = sampler.sample_batch(4096, seed=0, basis="random", chunk_size=1024)

batch.configs       # shape (4096, n_sites)
batch.basis         # one resolved X/Y/Z label per site
batch.probs         # p(outcome | resolved basis)
batch.weights       # p(basis, outcome)
batch.basis_probability
```

`VecSampler` preserves the backend of a dense NumPy, Torch, or CuPy state
vector for probability evaluation and `sample_batch` / `iter_samples`. Thus a
CuPy or Torch exact state produces device-resident configurations and
probabilities; no implicit state-vector copy to NumPy is made. Call
`batch.to_numpy()` when a host copy is explicitly needed. The legacy
`sample()` method still returns the list/grid compatibility result on the
host. It also exposes the common MPS sampler methods `sample_arrays`,
`amplitudes`, `probabilities`, and `refresh`; `Lx`, `Ly`, `L`,
`one_d_to_two_d`, and `resolved_backend` follow the same conventions.

Use `basis="X"`, `basis="Y"`, `basis="Z"`, a per-site string such as
`"XYZZ"`, or `basis="random"`. Random mode chooses each site's basis
uniformly and independently once per batch, so all shots in that batch share
the resolved pattern. `sample(...)` retains the legacy list/grid result. The
batch method avoids constructing a Python grid for every shot, and transformed
basis distributions are cached with a bounded cache. `probability_vector(...)`
returns the full conditional distribution for a requested resolved basis.
For true bounded-memory processing, consume `iter_samples(...)` instead:

```python
for batch in sampler.iter_samples(100_000, seed=0, chunk_size=4096):
    train_or_process(batch)
```

`sample_batch(...)` preallocates the final arrays and uses chunks only for
temporary draw memory; `iter_samples(...)` does not retain the complete shot
set.

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
    strategy="auto",  # "prefix", "serial", or "dense" are also available
    max_prefix_groups=256,
    dense_memory_limit="256MiB",
)
configs, probs = sampler.sample_arrays(4096, seed=0)
print(sampler.symmray_sampling_stats)
```

`"auto"` selects the fully batched dense kernel when the batch has at least
`dense_min_samples` shots and the estimated dense MPS view fits within
`dense_memory_limit`; otherwise it shares equal sparse prefixes. `"prefix"`
keeps every group allowed by `max_prefix_groups`, including singletons;
`"serial"` retains one boundary at a time. `"dense"` materializes a private
dense view of the source MPS and uses the backend-native batched conditional
kernel. It is supported for resolved fermionic U1/U1U1 states; parity-collapsed
Z2/Z2Z2 states remain on the sparse charge-aware route. Dense batching can be
substantially faster for moderate bond dimensions and high-entropy batches,
but uses more memory. The statistics report the requested and selected
strategy, dense-memory estimate, conditional distributions, candidate
contractions, charge-pruned branches, peak active groups, and serial/adaptive
fallbacks. Set `max_prefix_groups=None` to remove the hard group cap while
retaining sparse prefix sampling.

`prefix_strategy=` remains a backward-compatible alias for `strategy=`.
For comparable throughput measurements, call the public sampling APIs from an
external benchmark harness. For fermionic `Z2`/`Z2Z2` inputs, do not treat a
naive dense expansion of graded virtual legs as a state-preserving conversion;
only the native Symmray path is a valid comparison.

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

For binary dense-native MPSs, local-energy estimators can evaluate all
single-site-flip ratios without constructing a separate full MPS amplitude
sweep for every flip:

```python
ratios = sampler.single_site_flip_amplitude_ratios(
    configs,
    to_numpy=False,
)  # shape: (n_configs, n_sites)
```

The method uses bounded prefix/suffix workspaces and supports native NumPy,
Torch, and CuPy samplers. Symmray and legacy Quimb samplers should retain the
explicit connected-configuration path through `amplitudes(...)`.

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
