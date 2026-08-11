# Plan: PEPS direct sampler

## Status

Phases 1–6 are implemented. Pepsy now provides a `PepsSampler` implementing the
direct-sampling construction from
[Direct sampling of projected entangled-pair states](https://arxiv.org/pdf/2109.07356).
The existing `PepsBpSampler` is a separate belief-propagation proposal
sampler.

The implementation targets finite, open-boundary PEPS with dense tensor data,
a row-major sweep, and independent samples. The default exact mode contracts
the full conditioned Quimb norm network at each site. The boundary mode keeps a
conditioned single-layer ket boundary, optionally compresses it to `sample_chi`,
and can attach cached future marginal environments truncated to `marginal_chi`.
Prefix-grouped batches are supported; native tensor batch axes and Symmray
support remain deferred.

## Contract

For a PEPS ket \(|\Psi\rangle\), the sampler must return configurations
sampled from an explicitly tracked proposal \(p_c(S)\), together with enough
information to evaluate the importance weight

\[
    w(S) = \frac{|\Psi(S)|^2}{p_c(S)}.
\]

The proposal is approximate; the amplitude and proposal probability must be
computed from the original PEPS so the estimator remains an importance
sampling estimator. Numerically, proposal probabilities and amplitudes should
use the same scaled mantissa/exponent convention already used by
`PEPSSampleResult`.

## Sampling algorithm

The default order is fixed `y` followed by increasing `x`:

```text
I0,0 -> I1,0 -> ... -> I(Lx-1),0
I0,1 -> I1,1 -> ... -> I(Lx-1),1
...
```

At the beginning of row `y`, maintain two distinct boundary objects:

1. `phi_{y-1}`: a single-layer ket boundary MPS conditioned on all sampled
   rows. Its maximum bond dimension is `chi_s`.
2. `E_m^{>y}`: an optional double-layer environment for the unmeasured rows
   above the current row. Its maximum bond dimension is `chi_m`.

For a current site `(x, y)`:

1. Attach the conditioned lower boundary, current row, and future environment.
2. Split the current bra physical index from the ket physical index.
3. Contract all tensors except those two physical indices to form
   `rho_{x,y}`.
4. Sample from its diagonal:

   ```text
   p(s | prefix) = real(rho[s, s]) / real(trace(rho))
   ```

5. Fix both physical indices to the selected value and absorb the fixed site
   into the active network.
6. Multiply the running proposal probability by the selected conditional
   probability.

After the row is complete, apply the projected row to `phi_{y-1}`, producing
`phi_y`, and compress it to `chi_s`. After the final row, evaluate the
projected ket amplitude \(\Psi(S)\) and return the configuration, proposal,
amplitude, and scaled importance-weight ingredients.

The baseline `chi_m=0` path uses an identity future environment. Increasing
`chi_m` improves the marginal proposal without changing the sampled PEPS
amplitude.

## Conditioning rule

Precomputed lower and upper environments must not be treated symmetrically
after sampling starts. A lower double-layer environment such as Quimb's
`ymin` environment sums over earlier rows and therefore loses the sampled
conditioning. The lower side must be the mutable single-layer `phi` boundary.

The future `ymax` environment is safe to precompute because rows above the
current row remain unmeasured. This is the central distinction between the
paper's proposal and simply sampling each row from an unconditioned PEPS norm.

Within a row, the boundary implementation also builds a Quimb transfer cache
when the center is compact enough: each column is contracted once into a
local ket/bra transfer, all right suffix transfers are cached, and the sampled
left prefix is updated immediately after each `isel_`. The suffix contraction
does not apply another cutoff, since `marginal_chi` has already controlled the
future double-layer boundary. This preserves the full-center serial proposal
on the supported compact paths. Large centers with an already-collapsed
future MPS adaptively use the reference local-center path because a dense
column transfer would otherwise be slower and more memory hungry.

## Pepsy and Quimb mapping

### Common network construction

- Use `pepsy.boundary.build_bra_ket` to create the tagged ket and double-layer
  norm networks.
- Use `ket.site_tag(x, y)` and `ket.site_ind(x, y)` rather than hard-coded
  physical-index names.
- Select the current row with the `Y{y}` tag.
- Split the bra physical index by selecting the tensors with
  `[site_tag, "BRA"]` and reindexing that physical index to a private bra
  name.

### Local density matrix

The primary Quimb operations are:

```python
center = lower | current_row | future
center.select([site_tag, "BRA"], "all").reindex_({ket_ind: bra_ind})
rho = center.contract(
    all,
    output_inds=(ket_ind, bra_ind),
    optimize="auto-hq",
)
```

`optimize="auto-hq"` is the Cotengra-backed high-quality contraction preset
in the supported Quimb environment. After sampling, use
`center.isel_({ket_ind: value, bra_ind: value})` to remove the selected
physical indices. `contract_tags(site_tag, which="all")` may then absorb the
fixed ket/bra site pair when that reduces the active network.

### Quimb MPS boundary engine

The Quimb route should use either the public Quimb method directly or
Pepsy's existing adapter:

```python
future_envs = norm.compute_ymax_environments(
    max_bond=chi_m,
    cutoff=cutoff,
    canonize=True,
    mode="mps",
    layer_tags=("KET", "BRA"),
)
future = future_envs["ymax", y]
```

For a reusable store, use
`pepsy.optimizers.sweep.environments.QuimbMpsBoundaryStore`. Its
`update_axis(norm, "y")` method fills the cached `envs` mapping, while the
existing `mps_b` mapping preserves the sweep optimizer compatibility keys.

### DMRG/FIT boundary engine

The dense Pepsy route is:

```text
build_bra_ket(...) -> BdyMPS(...) -> CompBdy.move_bdy(...)
```

For future double-layer environments, the implementation calls
`CompBdy.move_bdy(direction="y_right")`; the right-side entries
`BdyMPS.mps_b[f"Y{Ly - 2 - y}_r"]` then supply the cached future boundary for
row `y`.

The conditioned single-layer `phi` update is a separate stateful operation.
It is not confused with `BdyMPS`'s double-layer norm boundaries. The
`ket_compression="quimb"` option uses Quimb MPS compression, while
`ket_compression="fit"` uses `FIT` against the projected row target.

Truncated DMRG/FIT environments can acquire small negative diagonal
components in a local `rho`. The sampler validates the real trace and clips
only tiny negative diagonal probabilities; stronger Hermiticity diagnostics
and correction reporting remain a follow-up.

## Public API

The implemented boundary API is:

```python
sampler = PepsSampler(
    peps,
    sample_chi=chi_s,
    marginal_chi=chi_m,
    boundary_engine="dmrg",       # "exact", "quimb-mps", or "dmrg"
    ket_compression="quimb",      # "quimb", "fit", or None
    contraction_opt="auto-hq",
    cutoff=1.0e-12,
)
result = sampler.sample(samples=..., seed=...)
```

`boundary_engine="auto"` is accepted as an alias for the dense DMRG/FIT
future-environment path. `marginal_chi=None` or `0` disables future
environments and uses an identity future cap. `refresh()` rebuilds private
networks after a changed PEPS; sampling never mutates the caller's source
network. `rho_diagnostics` reports trace, Hermiticity defect, and any clipped
roundoff-scale negative diagonal mass. `sample_batch()` shares networks by
identical prefixes and exposes its group counts through `batch_stats`.
`row_cache_stats` reports suffix-cache construction and left-prefix update
counts for the most recent boundary sample. Highly fragmented large batches
use the reference prefix path rather than constructing one dense transfer
cache per singleton group.

The result preserves the existing PEPS sampling vocabulary: `configs`, proposal
probabilities (`omegas`), and amplitudes (`ps`). Boundary diagnostics and batch
group statistics remain sampler attributes so the result type stays compatible.

## Implementation phases

1. **Quimb dense prototype** — implemented as serial `PepsSampler` with exact
   local `rho` contractions, deterministic seeds, and full-amplitude checks.
2. **Conditioned boundary state** — implemented with explicit single-layer
   `phi`, projected rows, and `sample_chi` compression.
3. **Marginal environment** — implemented with cached future environments and
   `marginal_chi`, including the identity (`None`/`0`) path.
4. **DMRG/FIT provider** — implemented with `BdyMPS`/`CompBdy` for future
   environments and `FIT` as a conditioned-boundary compression option.
5. **Public API and result type** — implemented and documented.
6. **Performance extensions** — implemented row-template caching, compact-row
   right-suffix transfer caching, and prefix-grouped `sample_batch`; reusable
   `build_contraction` optimizers cache repeated Cotengra path searches.
   Adaptive reference fallbacks protect larger collapsed-boundary batches.
   Native tensor batch axes remain future work.

## Required validation

- On a small PEPS with no truncation, the product of conditionals must equal
  the exact configuration probability.
- The local `rho` must be Hermitian positive semidefinite up to the selected
  numerical policy.
- Fixing physical indices must agree with direct projected-ket amplitudes.
- Quimb and DMRG/FIT proposals should agree within their truncation error on
  small dense examples.
- Scaled probabilities must remain finite for large or tiny amplitudes.
- Repeated seeded calls must reproduce configurations and proposal traces.
- Sampling must leave the input PEPS and reusable environment caches unchanged.
- Existing `MpsSampler` and `PepsBpSampler` behavior and public exports must
  remain unchanged.

## Deferred features

- Symmray-native local density matrices and charge-aware sampling.
- Native tensor-axis batching and adaptive prefix-group limits.
- Adaptive sweep direction and arbitrary site orderings.
- Independent-proposal Metropolis correction using `p_c` as a transition
  kernel.
- Infinite PEPS and periodic-boundary sampling.
