---
name: peps-sampler
description: Implement, review, debug, and explain direct projected-entangled-pair-state sampling in the Pepsy repository, including Quimb conditional contractions, conditioned boundary MPS design, future environments, proposal probabilities, importance weights, and prefix-aware batching. Use for PepsSampler work, PEPS sampling questions, or changes involving the paper-based direct sampler; do not use for the separate PepsBpSampler belief-propagation proposal unless the task explicitly compares them.
---

# PEPS sampler

Use this skill when working on direct PEPS sampling in Pepsy, especially the
construction from [Direct sampling of projected-entangled-pair states](https://arxiv.org/abs/2109.07356).
Keep the implementation scientifically explicit: distinguish the fixed PEPS
from the sample-conditioned boundary state, track the proposal probability,
and validate small systems against exact contractions before optimizing.

## Current implementation

The reference implementation is `pepsy.PepsSampler` in
`src/pepsy/sampling/samplers.py`.

- It is dense, Quimb-based, and samples in increasing `y`, then increasing `x`
  order. `sample(...)` is serial; `sample_batch(...)` shares identical prefix
  groups without adding a batch index to PEPS tensors.
- Exact mode builds private ket and norm networks with
  `pepsy.boundary.build_bra_ket`, contracts a local density matrix `rho`, and
  fixes the selected ket/bra physical index in the conditioned norm network.
- Boundary mode keeps a conditioned single-layer ket boundary, projects each
  sampled row into it, and compresses it with `sample_chi` using either Quimb
  MPS compression or Pepsy `FIT`.
- `marginal_chi=None` or `0` uses an identity future cap. With a positive
  `marginal_chi`, `boundary_engine="quimb-mps"` uses the Quimb future cache and
  `boundary_engine="dmrg"` uses `BdyMPS`/`CompBdy` future boundaries.
- Compact boundary centers use a native Quimb row transfer cache: right
  suffix transfers are built once per row and the conditioned left prefix is
  updated immediately after each sampled `isel_`. Larger collapsed-boundary
  centers and fragmented large batches adaptively use the reference local
  center path when dense transfers would be slower.
- It contracts the original private ket after the full configuration is fixed
  to obtain the PEPS amplitude.
- It returns `PEPSSampleResult`: `configs`, scaled proposal probabilities in
  `omegas`, and scaled amplitudes in `ps`.
- `PepsSampler.probability(config)` evaluates the selected sequential proposal;
  in exact mode it is the small-system oracle, while boundary mode evaluates
  the compressed proposal.
- `refresh()` rebuilds private networks and future environments after the
  source PEPS changes.
- `rho_diagnostics` reports local trace, Hermiticity defect, and any clipped
  roundoff-scale negative diagonal mass. `batch_stats` reports prefix-group
  counts for `sample_batch(...)`; `row_cache_stats` reports row suffix and
  prefix-update counts.

The implementation remains dense and prefix-grouped rather than native
batch-axis. Symmray support and tensor-axis batching are not implemented; do
not describe a shared batch label as a valid per-shot Quimb contraction.

Read [implementation.md](references/implementation.md) for the source map,
contraction details, and the staged roadmap before changing this area.

## Conceptual model

For a configuration `S`, direct sampling builds an autoregressive proposal

```text
p_c(S) = product_y,x p_c(s[x,y] | earlier sampled sites)
```

and uses the importance weight

```text
w(S) = |Psi(S)|**2 / p_c(S).
```

The PEPS tensors are fixed. The lower boundary state is conditioned on the
sampled prefix and therefore changes from shot to shot. A future environment
is safe to cache because it contains only unmeasured rows. Never reuse an
unconditioned lower double-layer environment after sampling has begun: it
sums over the prefix and loses the conditional distribution.

The eventual optimized architecture is:

```text
fixed PEPS + conditioned boundary MPS + future environment
                         -> local rho
                         -> sample physical value
                         -> project/compress boundary MPS
```

## Implementation workflow

1. Read the repository instructions, `docs/development/plans/peps_sampler.md`,
   the current `PepsSampler`, and the closest sampler tests before editing.
2. Preserve a slow serial/reference path while adding optimizations. Add a
   focused exact test before changing contraction or boundary semantics.
3. For every local conditional, verify that the selected value is fixed on
   both ket and bra layers and that the proposal probability is accumulated.
4. For boundary-MPS work, keep the single-layer conditioned state separate
   from double-layer norm environments. Compress only with the requested
   sample cutoff and expose numerical diagnostics for truncation errors.
5. For batching, keep the PEPS tensors shared. A batch entry can be carried as
   a dangling index only when it appears once in the Quimb network. Different
   shots choose different physical values, so use shared-prefix groups or
   native array updates rather than assuming `isel_` can select per-shot
   values in one ordinary Quimb network.
6. Run the focused tests, existing sampler tests, public API tests, Ruff, and
   the repository smoke suite. Activate the Pepsy environment first:

```bash
source /Users/rezah/envs/genpy/bin/activate
```

## Non-negotiable checks

- The source PEPS must remain unchanged by sampling.
- The product of conditionals must equal exact Born probabilities on small
  untruncated networks.
- Local `rho` must have a valid positive trace; only tiny numerical negative
  diagonal components may be clipped, and truncation paths should report the
  correction.
- Seeded sampling must be reproducible.
- The direct sampler must remain distinct from `PepsBpSampler`, which samples
  a belief-propagation proposal and then contracts amplitudes for importance
  sampling.
- Do not add a batch axis to every PEPS tensor: a shared matching label is an
  ordinary Quimb bond and will be contracted, not treated as a special shot
  dimension.
