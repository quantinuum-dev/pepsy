# `pepsy.sampling.samplers`

Prefer public imports from `pepsy.sampling`. Implementations are organized in
`sampling.mps`, `sampling.vector`, `sampling.peps`, and `sampling.bp`, with
shared records in `sampling.results`. Historical `sampling.samplers` imports
and serialized class references remain compatible.

## Choose a sampler

| State | Entry point | Read next |
| --- | --- | --- |
| MPS | `MpsSampler` | [MPS batch API](#mps-sampler-quick-api) |
| Dense qubit vector | `VecSampler` | [Exact-vector API](#dense-exact-vector-sampler) |
| PEPS | `PepsSampler` | [Direct PEPS sampler](#direct-peps-sampler) |
| Tree tensor network | `TreeSampler` | [Tree sampling](tree.md) |
| Stabilizer frame with coefficient MPS | `StabilizerMpsSampler` | [Stabilizer sampling](stabilizer.md) |

Import these classes from `pepsy.sampling`. Result probabilities and weights
have sampler-specific meanings; use the contract for the selected engine.

## Direct PEPS sampler

`PepsSampler` has an exact reference mode and a compressed boundary-MPS mode.
The explicit exact reference mode is:

```python
from pepsy.sampling import PepsSampler

sampler = PepsSampler(peps, boundary_engine="exact", amplitude_mode="exact")
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
    amplitude_mode="boundary",  # Opt in to boundary-MPS amplitude correction
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

For relative modes (`rel`, `rsum1`, `rsum2`), private Quimb future inputs and
conditioned ket boundaries are rescaled before compression. Quimb's future
sweep also equalizes intermediate boundary tensor norms. This prevents large
finite complex64 singular values from overflowing when squared for `rsum2`.
The positive scalar cancels from normalized conditionals; the source PEPS and
returned physical amplitudes keep their original scale. Absolute modes (`abs`,
`sum1`, `sum2`) retain their original compression scale and threshold semantics.
This does not guarantee arbitrary-scale contractions or eliminate finite-χ/χ′
proposal error.

### Sequential sweep and the two cutoffs

Boundary mode defaults to a conditioned-boundary sweep with FIT-style numerical
row-environment caching:

1. Cache future double-layer environments, from the last row toward the
   first, using `chi` (**χ**).
2. Combine the current row, its cached future, and the previously sampled
   single-layer ket boundary.
3. Select local factors by site/column tags and build right suffix environments
   once. Form a local `d × d` rho from those factors, the right suffix, and the
   conditioned left prefix. Sample and fix that value on both ket and bra,
   update the left prefix, and continue across the row using `auto-hq` for
   these small contractions.
4. After the whole row is fixed, absorb its **single ket layer** into the
   conditioned boundary and compress with `chi_prime` (**χ′**).
5. Continue to the next row. Accumulate the conditional log probabilities and
   optionally contract the original private ket for the final sampled amplitude
   with the selected amplitude method (boundary MPS by default).

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
Construction prepares that future cache once. Repeated `sample`, `sample_batch`,
and probability queries reuse it; call `refresh()` after changing the source
PEPS to rebuild the cache explicitly.

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
arrays. NumPy, Torch (CPU or CUDA), CuPy, and JAX arrays use their own contractions,
identity caps, local density matrices, conditional probabilities, and random
draws. The public `sampler.backend` reports the inferred array backend;
`boundary_engine` independently selects the contraction algorithm.

The numerical core uses Autoray namespaces inferred from the actual arrays,
including their dtype/device (`ar.get_namespace(template)` and its real-valued
counterpart). This follows the [Autoray namespace API](https://autoray.readthedocs.io/en/latest/automatic_dispatch.html#namespace-api).
The same rule covers cached environment tensors, local rho construction,
conditional probabilities, RNG, and physical-slice scaling for amplitudes.
Backend support does not imply identical sequences, precision policy, or
identical solver convergence across libraries/devices.

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
validation uses the real dtype's precision. By default it clips only
roundoff-scale negative diagonals; larger negative values and invalid traces
raise an error. Optional positive repair is described below.
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
Result log arrays, normalized weights, and summary statistics are explicitly
NumPy host postprocessing. This does not move the PEPS, rho, boundary MPS, or
cached tensor environments to NumPy, and is not a claim that the entire Python
sampling loop or result analysis is device-resident.

### Hermitian and positive local proposals

Sampling uses `H = (rho + rho.H) / 2` at every site. The contraction is scaled
before this addition and before probability normalization. Hermitian
symmetrization alone leaves `real(diag(rho))` unchanged; it does not guarantee
positive eigenvalues or repair NaN/Inf.

`rho_positivity` controls an additional, explicit spectral repair:

| Value | Matrix whose diagonal defines the conditional |
| --- | --- |
| `None` (default) | `H`, with only roundoff-scale negative diagonal clipping |
| `"clip"` | `V diag(max(lambda, 0)) V.H`, where `H = V diag(lambda) V.H` |
| `"absolute"` | `V diag(abs(lambda)) V.H = sqrt(H.H @ H)` |

The square root is a **matrix** square root. For a non-Hermitian raw rho,
`sqrt(rho.H @ rho)` generally differs from this Hermitize-then-repair policy.
`"clip"` is the nearest positive semidefinite matrix to `H` in Frobenius norm
before trace normalization; `"absolute"` reflects negative eigenvalues to
positive values. See [Higham's PSD projection explanation](https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/)
and [polar decomposition](https://nhigham.com/2020/07/28/what-is-the-polar-decomposition/).

For the absolute-value repair requested in the sampling workflow:

```python
sampler = PepsSampler(
    peps, chi=64, chi_prime=32,
    rho_positivity="absolute",  # Optional; default is None
)
result = sampler.sample_batch(samples=128, seed=0)
```

Both repairs change the finite-cap proposal. All serial, grouped, cached, and
probability-query paths use the same repaired conditional, so reported
`log_probabilities` and importance weights use the actual sampling proposal.
PEPS amplitudes still come from the original private ket. These repairs do not
guarantee full target support or remove finite-χ/χ′ bias in the proposal.
NaN/Inf input and a zero repaired trace raise errors; no uniform fallback or
probability floor is introduced.

For qubits, a batched 2×2 spectral formula avoids a general eigensolver and
preserves small spectral weights using cancellation-resistant roots and
projector weights. Larger physical dimensions use native batched `eigh`.
Only the diagonal of the repaired matrix is computed; neither `H.H @ H` nor
a full positive matrix is formed. Array operations preserve backend, dtype,
and device. Prefix grouping still performs one explicit validation scalar
read per site; an upstream eigensolver for larger local dimensions can add
its own synchronization.

`rho_diagnostics` retains diagnostics of the **raw** contracted rho, including
its Hermiticity defect, trace, and negative diagonal mass. It also reports
`relative_positivity_correction = norm(f(H) - H) / norm(H)` and its per-site
maximum `max_relative_positivity_correction`. These correction fields are zero
when repair is disabled; they do not certify positivity in that mode.
`clipped_negative_mass` records the default diagonal roundoff clipping;
spectral repairs are recorded by the new correction fields.

### Boundary and result semantics

`chi_prime` caps the conditioned single-layer ket boundary. `chi`
caps the optional future double-layer environment. The `dmrg` engine prepares
future boundaries with Pepsy `BdyMPS`/`CompBdy`; `quimb-mps` uses Quimb's MPS
environment cache. Ket compression is performed separately with Quimb MPS
compression or Pepsy `FIT`. The Quimb future sweep can also truncate temporary
boundaries between ket and bra layers. A cap large enough for the final boundary
rank does not by itself establish exact sampling; check convergence against an
independent reference and account for the cutoff as well.

`result.configs` contains row-major physical-index configurations, while
`result.omegas` contains proposal probabilities as `(mantissas, exponents)`,
representing `m * 10**e`. `result.ps` is `None` by default; explicit boundary
or exact amplitude modes populate it with scaled PEPS amplitudes. For multiple shots,
`sample_batch(...)` shares a local Quimb conditional network until shot
prefixes diverge:

```python
sampler = PepsSampler(peps, chi=64, chi_prime=32, amplitude_mode="boundary")
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
configurations; weight access rejects manually constructed zero-probability
draws instead of returning infinity or an undefined `0/0`.

This uses prefix groups rather than adding a shared batch index to PEPS
tensors, because a repeated Quimb index would be contracted as an ordinary
bond. The batch contracts each distinct final configuration's amplitude once from
the private PEPS for importance weights. Repeated shots receive independent
configuration lists, so editing one returned shot cannot change another shot.
As usual, stored probabilities and amplitudes describe the original draws.

The default factored environment cache reuses actual numerical tensors: its
right suffixes are contracted once per row and incoming prefix group, while
the conditioned left prefix is updated after each sampled site that has a
successor in the row. Descendant groups share immutable suffixes but keep their
own conditioned prefixes. The initial row has no sampled predecessor, so its
transfer cache is reused across calls until `refresh()`. Later rows depend on
the sampled preceding rows and are rebuilt for each distinct incoming prefix;
they are released after that row. Local factors stay separate instead of
forming dense two-interface column transfers. The cache does not retain its
temporary center network. This is numerical environment reuse, distinct from
reusing Cotengra contraction plans.

Private transfer factors, suffixes, and prefixes are rescaled by positive
scalars. These common factors cancel from normalized conditionals, preventing
rare-prefix products from underflowing without changing the PEPS amplitudes or
boundary truncation policy. Row bonds, identity future caps, and the independent
future boundary MPS cache are also reused until `refresh()`. Call `refresh()`
after changing the source PEPS to invalidate all these caches together.

`row_cache_max_bytes` defaults to **64 MiB** (`64 * 2**20`) and
`row_cache_mode` defaults to **"factored"**. Before allocation, the sampler
estimates storage and workspace from PEPS/future bonds, conditioned-boundary
bond bounds, dtype, live prefix groups, and the retained initial row. An
estimate above the budget falls back to reference local-center contractions.
Set the budget to zero to explicitly disable row environments, or choose
`row_cache_mode="dense"` for the legacy materialized-transfer path. That dense
mode has additional prefix-count/large-future fallback rules. The budget is
**not** a cap on total process/device memory or exact contraction workspace.
Use `chunk_size` to keep live prefix groups small enough for the cache budget.

```python
sampler = PepsSampler(
    peps, chi_prime=32, chi=64, boundary_engine="dmrg",
    row_cache_max_bytes=64 * 2**20,
)
batch = sampler.sample_batch(samples=64, seed=0, chunk_size=4)
print(sampler.row_cache_stats)
print(sampler.batch_stats["conditional_batches"])
```

`row_cache_stats` reports the selected `mode`, actual suffix builds, nonterminal
prefix updates, initial-row reuse (`initial_row_cache_hits` in transfer mode),
`estimated_cache_bytes` (`None` when caching is disabled), `cache_budget_bytes`, and `cache_decision`
(`within-budget`, `memory-budget`, `disabled`, `prefix-count`, or `large-future`).
`batch_stats["conditional_batches"]` counts the site-level validation/draw
batches. Amplitudes still come from the original private PEPS, and reported
proposal probabilities still follow the selected boundary approximation.

### Reusable plans, cached rows, and amplitude limits

With `contraction_opt=None` (default), `PepsSampler` constructs one
`pepsy.tensors.build_optimizer(parallel=False)` for full-network contractions.
This is the maintained name for `build_contraction`. The reusable Cotengra
optimizer caches plans by network structure and dimensions, not tensor values.
It survives `refresh()`; physical boundaries, row environments, normalized
amplitude leaves, and the selected amplitude tree are rebuilt after refresh.

Cached-row contractions default to `row_contraction_opt="auto-hq"`, independently
of the full-network optimizer. Set a different row optimizer explicitly, or
pass `row_contraction_opt=None` to inherit the full optimizer. For an external
reusable plan cache:

```python
from pepsy.tensors import build_optimizer
from pepsy.sampling import PepsSampler

optimizer = build_optimizer(parallel=False, directory="/tmp/peps-paths")
sampler = PepsSampler(
    peps, chi=16, chi_prime=8,
    amplitude_mode="exact",  # Opt in to amplitudes and the exact-plan limits below.
    contraction_opt=optimizer, row_contraction_opt="auto-hq",
    row_cache_mode="factored", row_cache_max_bytes=64 * 2**20,
    amplitude_max_intermediate_bytes=512 * 2**20,
    amplitude_max_cost=5e10,
)
batch = sampler.sample_batch(8, chunk_size=2, seed=1)
print(sampler.amplitude_plan_info)
```

The row cache selects columns using their `X{x}` tags, builds right suffixes
once for a conditioned row, and updates the left prefix after each measured
site. The bottom ket boundary advances after the row and depends on the sampled
prefix. Tags identify regions; they do not make contraction values reusable
across different prefixes or evolved states. This is the same directional
reuse principle as FIT/DMRG environments, separate from Cotengra plan caching.

Optional amplitude limits check the estimated largest intermediate bytes and
Cotengra contraction cost **before** scaled-leaf allocation and exact execution,
and recheck an existing amplitude plan before reuse. A rejected plan remains
inspectable through `amplitude_plan_info`; no approximate amplitudes replace it.
Both limits default to `None`. They do not bound total process/GPU memory,
planner memory, all simultaneous inputs, or wall time. Better paths or exact
slicing can reduce intermediates; prefix chunking alone cannot. A rejected
public batch raises rather than returning partial samples.

### Bounded batches, amplitude evaluation, and weight diagnostics

The proposal probability is the product of the sampled conditional
probabilities: `q(s) = exp(sum(log(p_site)))`. This is sufficient to draw
configurations and compute ordinary proposal averages. It does not in general
equal the normalized Born probability of the original PEPS when the boundary
environments are truncated, and it contains no complex amplitude phase.

Amplitude evaluation is a separate choice for importance correction:

```python
sampler = PepsSampler(
    peps, chi=64, chi_prime=32,
    amplitude_mode="boundary",  # opt-in correction; default is "proposal"
    amplitude_chi=32,           # defaults to chi_prime
)
```

`amplitude_mode="boundary"` projects the original private ket onto each
configuration, contracts its rows through a boundary MPS with this cap and
the sampler's cutoff, and contracts the remaining one-dimensional boundary.
It rescales and caches physical slices before boundary contraction, preserving
phase and physical scale, including large PEPS exponents, without overflowing
intermediate input products. The cutoff acts on these internally rescaled
boundary tensors; relative cutoff modes are the scale-independent choice. It never
builds a full-network exact amplitude plan. Only open PEPS are supported.
Weights are then `abs(Psi_estimate)**2 / q`, so the correction itself is
approximate. `PEPSSampleResult.amplitude_mode` and
`weight_diagnostics["weights_are_approximate"]` make that explicit.
`amplitude_max_cost` and `amplitude_max_intermediate_bytes` apply only to exact
amplitude mode. Neither cap controls total memory.

To skip amplitude evaluation entirely and average the proposal draws directly:

```python
sampler = PepsSampler(peps, chi=64, chi_prime=32, amplitude_mode="proposal")
batch = sampler.sample_batch(4096, seed=17, chunk_size="auto")
q_logs = batch.log_probabilities  # sum of the selected site log conditionals
assert batch.ps is None
weights = batch.normalized_weights  # equal weights: 1 / number of samples
```

`amplitude_mode="none"` remains a compatibility alias for `"proposal"`.
Both spellings use the same algorithm; result and diagnostic metadata retain
the supplied spelling so existing saved-record consumers keep working.
The default is `"proposal"`: no separate amplitude contractions. Select
`"boundary"` or `"exact"` explicitly to obtain amplitudes and correction weights.

This mode never evaluates amplitudes, including in serial and streamed calls.
`log_weights` is zero, `weight_kind` is `"proposal"`, and `log_mean_weight` is
None because no PEPS norm estimate was made. Accessing `log_abs_amplitudes`
raises instead of inventing amplitudes or phases from q. ESS equals the number
of draws by construction; it does not diagnose proposal accuracy. Ordinary
averages use equal weights, **not q again**, because the configurations were
already drawn from q.

The product of selected conditional probabilities is q, not an importance
weight or a complex PEPS amplitude. `0.5 * batch.log_probabilities` gives
`log(sqrt(q))`, the log magnitude of a normalized positive proposal wavefunction.
The magnitude sqrt(q) approximates the normalized PEPS amplitude magnitude
only when the proposal converges, and contains neither the PEPS phase nor
its physical norm. Keep `ps=None` for this mode; use `amplitude_mode="boundary"` or
`"exact"` when separate amplitude estimates and correction weights are wanted.

Increasing χ and χ′ can recover exact Born probabilities as boundary
compression and solver errors vanish. A fixed nonzero cutoff or unconverged
variational boundary fit can leave residual error. q approaches the normalized
`abs(Psi)**2`, not the complex amplitude itself. Boundary amplitudes converge
separately as `amplitude_chi` (default χ′) increases and truncation vanishes;
an explicitly fixed `amplitude_chi` does not grow when χ′ changes. If both
amplitude_chi and χ′ are None, the amplitude sweep has no rank cap but still
uses the selected cutoff. No full exact amplitude mode is selected implicitly.
Proposal-engine selection is independent: omitting both χ and χ′ still selects
exact conditional contractions. Supply those caps for boundary sampling.

Use `chunk_size` to bound the number of live sample-prefix states while still
returning a complete result, or consume `iter_samples` to bound output storage:

```python
batch = sampler.sample_batch(8192, seed=17, chunk_size=64)
weights = batch.normalized_weights
print(batch.effective_sample_size)
print(batch.weight_diagnostics)
print(sampler.diagnostics)

for chunk in sampler.iter_samples(8192, chunk_size=64, seed=17):
    # Consume/write each chunk; do not retain it for bounded output storage.
    consume(chunk.configs, chunk.log_weights)
```

Both interfaces use one continuous backend random stream. The same seed and
chunk size reproduce the same configurations and probabilities; changing the
chunk size can change draw order. Omitting `chunk_size` preserves the original
single-batch proposal/draw order. Prefix groups never exceed the current chunk
size. Caches and exact-contraction workspace are additional memory; this is
not a hard process/GPU memory limit.

The `sample_batch` default is `chunk_size=None` (one group batch containing
all requested shots); `iter_samples` defaults to 128 shots per chunk.
`"auto"` is opt-in and does not change either default.

Use `chunk_size="auto"` to select at most 32 shots, reduced to fit the
estimated row-cache budget after accounting for the retained initial row.
This avoids disabling useful suffix caches merely because a fixed chunk is
too large. Exact proposals, disabled caches, or a budget too small for one
cached history select one shot at a time; the usual reference fallback still
applies when a single cached row cannot fit. The estimate is conservative,
not a measured peak-memory or speed guarantee. `batch_stats["chunk_size"]`
records the resolved size and `requested_chunk_size` records `"auto"`.
The policy does not change χ, χ′, cutoff, or amplitude mode. Fix an integer
chunk size when reproducing a draw sequence across different cache budgets.

For boundary proposals, the original PEPS and future boundaries remain shared.
Each distinct sampled history owns a conditioned boundary MPS, with at most
one group per shot in the current chunk. Within a row, sibling groups share their incoming boundary
and right suffixes, carrying separate left contractions. Unsplit reference
groups move their owned network forward without copying it. Completed group
networks are released before optional amplitude contractions; one last
boundary snapshot remains available for diagnostics. Larger chunks can still
retain more boundaries after histories diverge. Use `chunk_size=1` for a
single history at a time, while retaining row-cache reuse.

`sample_batch(..., chunk_size=...)` aggregates conditional counts and maximum
rho defects/repairs across chunks. `iter_samples` exposes statistics and rho
diagnostics for the most recently yielded chunk. `row_cache_stats` describes
the latest chunk. `batch_stats["final_prefix_groups"]` for a collected chunked
batch sums distinct configurations within each chunk, not globally deduplicated
configurations. `sampler.diagnostics` copies a compact summary to the host;
`amplitude_stats` reports amplitude contractions and exact plan builds since refresh.
A subsequent likelihood query replaces the rho diagnostics but leaves the last
batch counts intact. Serial `sample` rho diagnostics describe its last draw.
Each scaled result field must contain one mantissa and exponent per configuration;
log and weight accessors reject mismatched lengths instead of broadcasting them.
Proposal mantissas must be finite, real, and nonnegative, with finite
exponents. Weight access also rejects zero proposal probability for a saved
draw, including in proposal-only mode; equal weights must not conceal
malformed probability records. Very small positive probabilities remain
usable through their log/scaled representation without underflow.

`normalized_weights` computes `exp(log_w - max(log_w))` and normalizes the sum
(uniform weights in `amplitude_mode="proposal"`, also spelled `"none"`).
`effective_sample_size` is `1 / sum(normalized_weights**2)`.
`weight_diagnostics` adds ESS/N, maximum normalized weight, zero-weight count,
and the log mean unnormalized weight. With exact amplitudes and full proposal support, the latter
estimates the log of a Monte Carlo estimate of the squared PEPS norm; it is
not an unbiased estimator of the logarithm itself. Empty, all-zero, NaN, or
positive-infinite weight sets raise `ValueError` instead of yielding a uniform
fallback. Zero-amplitude shots receive zero weight. High observed ESS does not
establish proposal support or convergence.

Weights normalized separately in separate chunks are **not** globally
normalized weights. Combine chunks using their unnormalized log weights with
stable accumulated sums, or use the collected `sample_batch` result for global
normalization. Self-normalized observable estimates generally have finite-sample
bias. No generic statistical error bar is inferred from ESS alone.

Exact sampled amplitudes now share a Cotengra contraction plan until `refresh()`.
A private cache rescales every physical slice once, retains its logarithmic
scale, and uses exponent stripping for intermediate contractions. This keeps
phase and original PEPS scale, including amplitudes beyond the raw dtype range.
It retains at most one additional ket's array data plus small scale arrays and
contraction metadata. Result exponents may be fractional. No sampled amplitude
values or configurations are retained between batches. Full exact amplitude
contractions in `amplitude_mode="exact"` are not capped by χ or χ′; the stable path
adds arithmetic and can cost more than raw contraction on small networks.

The default factored suffix cache avoids forming dense column transfer tensors
with both horizontal interfaces open. Its settings can be made explicit:

```python
sampler = PepsSampler(
    peps, chi=16, chi_prime=8, boundary_engine="quimb-mps",
    row_cache_mode="factored", row_cache_max_bytes=64 * 2**20,
)
batch = sampler.sample_batch(256, seed=17, chunk_size=8)
```

This caches unmeasured row suffixes and updates the conditioned prefix after
every sampled site, with no additional truncation. It respects the estimated
memory budget and uses the reference path when oversized. The initial-row cache
is reused until refresh; later rows are conditioned on their incoming prefix.
It can reduce repeated contractions on wider lattices but be slower on small
lattices. The numerical-reuse policy is the default; zero budget explicitly
selects the reference path, and dense transfers require `row_cache_mode="dense"`.
`row_cache_stats` reports `cache_representation` and mode `factored` when used.

The [dated efficiency study](../../development/notes/peps_sampler_efficiency.md)
records earlier CPU/GPU measurements and their limits. Runnable usage examples
are listed in the [examples guide](../../examples.md).

### Likelihoods, zero branches, and truncation limits

The sampler accumulates **sums of log conditional probabilities**. Internally
it uses base-10 logs for the result's mantissa/exponent format. The public
`log_probability(config)` returns the **natural log**:

```python
log_q = sampler.log_probability(config)
q = sampler.probability(config)  # exp(log_q), if representable as a float
```

Exact and reference boundary likelihood queries use exponent-stripped local
contractions. This protects rare-configuration likelihoods against intermediate
underflow; the common rho scale cancels from each normalized conditional.
The optional transfer route instead rescales its cached factors and running
prefix/suffix tensors, so it also avoids multiplying full-prefix probabilities.
`rho_diagnostics` describes the rho actually evaluated, so traces from these
queries are scaled. Transfer-cache diagnostics also use the rescaled proposal
tensors. The cache materializes ordinary arrays, so scaling and log bookkeeping
do not guarantee that arbitrary ill-scaled input contractions cannot lose
precision or overflow/underflow. Sampled amplitudes use the separately scaled
contraction described above; reconstructing raw amplitudes from their scaled
pairs can still exceed the destination dtype range.

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

For small systems, compare `batch.log_probabilities` with values returned by
an exact `PepsSampler(peps)` before increasing χ and χ′. This checks proposal
support and truncation error without relying on a repository example script.

## MPS sampler quick API

`MpsSampler.sample_batch(...)` is the preferred batched interface for new code:

```python
from pepsy.sampling import MpsSampler

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
from pepsy.sampling import VecSampler

sampler = VecSampler(state, one_d_to_two_d)
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

Raw vectors contain ``2**L`` amplitudes with site 0 as the most significant
bit. The sampler normalizes a private vector, so amplitude queries return
normalized amplitudes even when the source is unnormalized. The source values
are preserved. After changing the source, call `refresh()`; pass a replacement
state to `refresh(state)` to reuse the site map with newly inferred backend
and dtype. The number of sites stays fixed.

For CuPy distributions and Torch distributions with more than `2**24` outcomes, inverse-CDF draws
use float64 accumulation and random numbers even for a complex64 state.
The CDF is normalized by its accumulated total; returned probabilities retain
their original dtype and device. This temporary CDF needs eight bytes per
outcome (8 GiB for 30 qubits), independently of the shot chunk size.
Seeded draws on this route differ from versions using a float32 CDF, which
could bias sampled bits. Previously saved shots require resampling from the
state; changing the plotting code cannot repair them. The CDF is rebuilt per
draw chunk without a persistent cache. CuPy accumulates directly into float64,
avoiding a separate promoted copy of its probability vector.

| Method | Result | Default array location |
| --- | --- | --- |
| `sample_batch(n_samples, ...)` | Batch record: `configs` `(n_samples, L)`, `probs` and `weights` `(n_samples,)` | State backend/device |
| `sample_arrays(n_samples, ...)` | Tuple `(configs, probs)` with the same shapes | State backend/device |
| `sample(n_samples, ...)` | Legacy lists and 2D grids | Host |
| `amplitudes(configs, ...)` | Normalized complex amplitudes `(batch,)`, always in the computational basis | NumPy |
| `probabilities(configs, basis=...)` | Conditional Born probabilities `(batch,)` | NumPy |
| `probability_vector(basis=...)` | Copy of the full distribution `(2**L,)` | State backend/device |

Use `to_numpy=False` on amplitude/probability queries to retain the native
backend. Torch amplitude and batch queries need `track_grad=True` to retain
their gradients; sampled integer configurations are not differentiable.
Converting a result to NumPy detaches it.

Use `basis="X"`, `basis="Y"`, `basis="Z"`, a per-site string such as
`"XYZZ"`, or `basis="random"`. Random mode chooses each site's basis
uniformly and independently once per batch, so all shots in that batch share
the resolved pattern. `sample(...)` retains the legacy list/grid result. The
batch method avoids constructing a Python grid for every shot, and transformed
basis distributions are cached with a bounded cache. `probability_vector(...)`
returns the full conditional distribution for a requested resolved basis.
To evaluate sampled configurations, use
`sampler.probabilities(batch.configs, basis=batch.basis)`. The probability
query's `basis="random"` uses its own fixed seed-0 pattern, so it need not
match a previous sample call. Batch `weights` are joint basis/outcome
probabilities, not importance ratios.

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
