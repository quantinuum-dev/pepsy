# Boundary Quimb compression modes — 2026-09-11

## Scope

The finite-PEPS boundary API now exposes the installed Quimb 1D direct
compression family needed for mode-by-mode timing and accuracy plots. The
canonical internal spelling is Quimb's `srcmps`; Pepsy additionally accepts
`src-mps` and its oversampling spelling for readability.

## Upstream audit

In the activated `genpy` environment, the installed Quimb build reports
`tensor_network_1d_compress(..., method="dm", ..., inplace=False, **kwargs)`.
Its dispatch table contains `direct`, `dm`, `zipup`, `zipup-first`,
`zipup-oversample`, `sdc`, `sdc-oversample`, `sdcr`, `sdcr-oversample`, `src`,
`src-first`, `src-oversample`, `srcmps`, `srcmps-first`, and
`srcmps-oversample`. The installed Quimb, Autoray, Cotengra, and Symmray
versions were inspected before editing; the upstream documentation confirms
the `srcmps` and oversampling variants.

- **Adopt:** expose the existing Quimb direct compressors through the PEPS
  boundary selector, including `*-first` and `*-oversample` variants.
- **Compatibility shim:** accept `src-mps`, `src-mps-first`, and
  `src-mps-oversample`, canonicalizing them to Quimb's `srcmps` spellings.
- **Defer:** add a PEPS `dmrg3` mode. The existing PEPS `dmrg2` policy is a
  two-site warm-up followed by one-site refinement, while `dmrg1` is already
  the `eff`/one-site alias; inventing a third alias would misstate the
  algorithm.

The PEPS direct modes remain dense-only, preserve the existing sequential
layer policy, and do not change the default `fit_mode="eff"` FIT path.
