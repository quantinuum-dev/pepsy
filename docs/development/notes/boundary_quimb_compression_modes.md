# Boundary Quimb compression modes — 2026-09-11, updated 2026-09-14

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

## 2026-09-14 compatibility re-audit

The activated environment reports Quimb
`1.15.1.dev55+gd0591eb70`, Autoray `0.11.1.dev3+g1b476b305`, Cotengra
`0.8.3.dev7+g1d7fd333f`, and Symmray `0.3.2.dev8+g6c6dd34b5`.
`qtn.tensor_network_1d_compress` retains the public signature used here and
dispatches every mode listed above. Quimb's old
`quimb.tensor.tensor_1d_compress` compatibility module now emits a future
warning and its private registry import has moved to
`quimb.tensor.tn1d.compress`; Pepsy does not import either private registry.

- **Adopt:** continue calling the public `quimb.tensor` function. Direct PEPS
  compression now derives its MPS length from the boundary sweep geometry and
  avoids constructing or copying an unused FIT guess.
- **Compatibility shim:** retain the narrow `sdcr` cumulative-cutoff coercion
  and readable `src-mps` aliases. Current Quimb still rejects cumulative
  cutoff modes in randomized SVD while accepting the final exact cutoff for
  oversampled variants.
- **Defer:** no Autoray, Cotengra, or Symmray API change requires another
  boundary shim. Native Symmray direct compression remains capability-gated.

Focused validation covers direct compression without boundary-guess
materialization, sequential input/automatic layer order, all installed direct
mode selectors, and PEPS norm/infidelity contraction paths.
