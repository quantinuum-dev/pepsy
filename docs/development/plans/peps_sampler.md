# Plan: PEPS direct sampler

## Status

Proposed. Pepsy does not currently provide a `PepsSampler` implementing the
direct-sampling construction from
[Direct sampling of projected entangled-pair states](https://arxiv.org/pdf/2109.07356).
The existing `PepsBpSampler` is a separate belief-propagation proposal
sampler.

The first implementation targets finite, open-boundary PEPS with dense tensor
data, a row-major sweep, and independent importance samples. Symmray support,
batch prefix sharing, and alternative sweep orderings can follow after the
scalar dense path is validated.

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

For future double-layer environments, the relevant direction is the
`y_right` side. `BdyMPS.mps_b` then supplies the cached boundary objects that
can be attached to the current row.

The conditioned single-layer `phi` update is a separate stateful operation.
It should not be confused with `BdyMPS`'s double-layer norm boundaries. The
initial implementation may use Quimb MPS compression for `phi`; the FIT
alternative can use `FIT.run_eff` against the projected row target. A future
provider interface should hide this distinction from the sampling loop.

Truncated DMRG/FIT environments can acquire small non-Hermitian or negative
components in a local `rho`. The sampler must validate, symmetrize, and apply
a documented numerical policy before constructing probabilities, while
retaining diagnostics for the uncorrected `rho`.

## Draft public API

The names below are intentionally provisional until the first implementation
clarifies the result and cache lifetimes:

```python
sampler = PepsSampler(
    peps,
    sample_chi=chi_s,
    marginal_chi=chi_m,
    boundary_engine="auto",       # "dmrg" or "quimb-mps"
    contraction_opt="auto-hq",
    cutoff=1.0e-12,
)
result = sampler.sample(samples=..., seed=...)
```

`boundary_engine="auto"` should follow Pepsy's existing selector policy:
dense inputs use the DMRG/FIT path, while Symmray-looking inputs route to
Quimb MPS environments. The sampler should expose `refresh()` for a changed
PEPS and should never mutate the caller's source network during sampling.

The result should preserve the existing PEPS sampling vocabulary where
possible: `configs`, proposal probabilities (`omegas`), and amplitudes
(`ps`). A new result type is preferable if direct sampling needs additional
fields such as log weights, per-row proposal traces, boundary diagnostics, or
the selected engine.

## Implementation phases

1. **Quimb dense prototype** — implement one sample with `chi_m=0`, exact
   local `rho` contractions, deterministic seeds, and full-amplitude checks.
2. **Conditioned boundary state** — represent `phi` explicitly as a
   single-layer MPS, project each sampled row, and compress to `chi_s`.
3. **Marginal environment** — add cached `ymax` environments with `chi_m`,
   scaled contractions, and proposal diagnostics.
4. **DMRG/FIT provider** — adapt `BdyMPS`/`CompBdy` for the future environment
   path and add the FIT option for conditioned-boundary compression.
5. **Public API and result type** — export `PepsSampler`, settle parameter
   names, and integrate with the existing PEPS sampling documentation.
6. **Performance extensions** — cache row contractions, reduce repeated
   Cotengra path construction, then consider batched/prefix-shared samples.

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
- Batched samples sharing row or site prefixes.
- Adaptive sweep direction and arbitrary site orderings.
- Independent-proposal Metropolis correction using `p_c` as a transition
  kernel.
- Infinite PEPS and periodic-boundary sampling.
