# Quimb opportunities — 2026-10-22

Prioritize upstream scale corrections, then prototype cluster-boundary
measurements and optional adaptive QR. This is a review, not an upgrade.

## Baselines and evidence

Pepsy `develop` at `5ff1fe6`. Installed Quimb is
`1.15.1.dev96+g5656a9bb2`; Cotengra is `0.8.3.dev8+g8954240f2`,
Autoray `0.11.1.dev14+g014a3f69a`, and Symmray `0.4.1.dev20+ge8c907157`.
Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html)
and upstream source at `1b64353d9e562529694e71abf40a2861ee8dcdfa`
(October 21), five commits after the installed Quimb revision. The new
features are listed under unreleased 1.16.0; many already exist locally.

Temporary source checkouts under `/tmp` supplied isolated import paths.
Cotengra upstream was `7a63d02b6d8d842bdee7de08fc1bd16b3e519765`.
No installed package was edited or upgraded. Source locations and Git SHAs
identify the probes: source-only Quimb imports reported the installed
distribution version through package metadata.

## Adoption priorities

| Classification | Opportunity | Pepsy relevance and next validation |
| --- | --- | --- |
| **Adopt** | Environment and contraction exponent fixes | `SymPEPS` caches plaquette environments; `compute_boundary_expectation` calls the affected boundary route. Validate unnormalized values and returned numerator/norm pairs, plus cached environments. |
| **Adopt, capability-gated** | Explicit `check_zero` for exact contractions | Useful for forbidden configurations and zero overlaps with exponent stripping. Requires compatible Cotengra as well as Quimb. Keep eager zero handling separate from JIT/autodiff and compressed contraction. |
| **Prototype** | `compute_local_expectation_cluster_boundary` | Add an explicit alternative to `compute_path_cluster_expectation`, using a spatial cluster and boundary-MPS compression. Test radius and bond-cap convergence, periodic seams, and native fermionic signs. |
| **Prototype** | `qr:auto` | Potential acceleration for tall QR reductions in FIT and PEPS updates. Benchmark representative shapes before adoption; preserve Pepsy's custom derivative policies and test rank-deficient/backend cases independently. |
| **Prototype** | Batched RDMs and streaming environments | Pepsy already supports `route='envs'`. The additional opportunity is reusing `compute_partial_traces` results across operators on the same support, with `callback_rhos` for incremental consumption. Cache invalidation must follow state changes. |
| **Prototype** | Edge-induced loop generator | Compare Quimb `NetworkPatch` output with Pepsy's explicit loop geometry before reducing duplicate enumeration code. Quimb `max_size` counts tensors; Pepsy's cutoff counts excited bonds. Join limits and periodic parallel bonds also require validation. |

No new compatibility shim is recommended by this review. Prefer the upstream
fixes after focused integration validation. **Defer** infinite-PEPS and new
run-management adapters to separately scoped work.

## Reproduced corrections

The [plaquette fix](https://github.com/jcmgray/quimb/commit/2f3ce403b52e9b32af11ddb7c776271a3ceac091)
and [contract exponent fix](https://github.com/jcmgray/quimb/commit/67ea9a878ed5875e34722cac7d825cbdfb5fe0ab)
corrected these CPU probes:

- A two-tensor contraction with network exponent 2 returned 1100 through
  `contract(all)` but 11 through `contract_tags(all)` locally. Upstream
  returned 1100 through both routes.
- For a seeded 3-by-4 network with exponent 2, recombining each single-site
  plaquette with its environment gave a ratio of 0.01 to the exact contraction
  with equalization disabled. Equalization produced other incorrect scales.
  Upstream ratios were within approximately `4e-15` of one in both cases.
- Through Pepsy `compute_boundary_expectation`, a seeded dense 3-by-3 PEPS
  with state exponent 1 and a central Z observable returned 0.01 times the
  exact unnormalized value, with either equalization setting. Upstream
  Quimb/Cotengra returned the reference within floating-point precision.
  This does not establish an error in locally normalized ratios, where a
  common scale can cancel.
- Explicit zero checks failed with `TypeError` in the installed Cotengra
  expression builder, including with newer Quimb alone. With both upstream
  checkouts, exact orthogonal-MPS contractions returned `(0.0, 0.0)` through
  both all-tensor and tag routes. See the
  [Cotengra 0.8.3 notes](https://github.com/jcmgray/cotengra/blob/7a63d02b6d8d842bdee7de08fc1bd16b3e519765/docs/changelog.md)
  and [dense-environment zero check](https://github.com/jcmgray/quimb/commit/4132cd12a58c0791b2fe3ea4519235adc042f103).
  These switches use data-dependent branching and do not certify JIT gradients.

## Feature probes and limits

With the installed build, one NumPy float64 matrix of shape `(4096, 64)`,
one warmup and five timed repetitions, and one BLAS/OMP thread:
`qr` median 13.23 ms; `qr:auto` 8.85 ms (about 1.50x).
Relative reconstruction errors were below `5e-16`; orthogonality errors
below `3e-15`. A zero final column produced finite adaptive-QR factors.
This is a microbenchmark, not a FIT/PEPS or GPU speedup claim. See
[adaptive QR source](https://github.com/jcmgray/quimb/blob/5656a9bb2aaf01b8160fd440761a1c784de1747a/quimb/tensor/decomp.py).

For one seeded dense 3-by-3 PEPS with bond dimension 2, a cluster covering
the whole lattice and an `envs` RDM both matched the exact central Z
expectation `-0.45495725841702234` within `6e-16` at cap 64 and zero cutoff.
No finite-radius accuracy, fermionic, GPU, or gradient claim follows.
The edge-induced loop generators agreed on all four loops of a chorded
square, using Quimb `max_size=4, num_joins=3` and Pepsy edge cutoff 5;
broader completeness and performance remain unverified.

Already adopted: SDC/SDCR/SRC compression modes, native Trotter schedules,
MPO auto-swap capability dispatch, ordinary direct product-state construction,
and the basic periodic `envs` measurement route. Avoid proposing these as
missing integrations.

Temporary reproducibility scripts: `/tmp/pepsy-quimb-review-probe.py` and
`/tmp/pepsy-quimb-feature-probe.py`; JSON results use the
`/tmp/pepsy-quimb-review-` prefix. No numerical regression suite was run for
this review; findings are isolated probes and source inspection.

Subsequent authorized work implemented the optional dense PEPS gauge-QR path;
see its [implementation and timing evidence](2026-10-22-peps-gauge-qr.md).
The broader QR opportunities above remain proposals.
