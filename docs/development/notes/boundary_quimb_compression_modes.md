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
  compression now derives its MPS length from the lattice axis perpendicular
  to the boundary sweep, including for rectangular lattices, and avoids
  constructing or copying an unused FIT guess.
- **Compatibility shim:** retain the narrow `sdcr` cumulative-cutoff coercion
  and readable `src-mps` aliases. Current Quimb still rejects cumulative
  cutoff modes in randomized SVD while accepting the final exact cutoff for
  oversampled variants.
- **Defer:** no Autoray, Cotengra, or Symmray API change requires another
  boundary shim. Native Symmray direct compression remains capability-gated.

Focused validation covers direct compression without boundary-guess
materialization, sequential input/automatic layer order, all installed direct
mode selectors, and PEPS norm/infidelity contraction paths.

## 2026-09-14 finite-CTMRG mode audit

The same installed Quimb build exposes `contract_ctmrg(..., mode=...,` with
`canonize`, `canonize_opts`, `reduce_opts`, and `compress_opts`). Its 2D
boundary dispatcher provides the arbitrary-geometry `projector` and `l2bp`
compressors plus the explicit `projector2d` implementation. Projector
canonicalization accepts `True`, `False`, `"layered"`, and `"bp"`; the last
choice converges a fresh dense D2BP instance for each compressed boundary.
The public `tensor_network_1d_compress` function also accepts a callable
compression method; Quimb's unreleased 1.16 changelog explicitly records this
wrapper support. This makes a regional experiment possible without copying or
forking Quimb's CTMRG implementation.

- **Adopt:** expose `projector`, `projector2d`, and `l2bp` through dedicated
  `ctmrg_*` PEPS metric options while retaining `projector` plus simple
  gauging as the default.
- **Compatibility shim:** affected Quimb builds forward projector-only
  `canonize_opts`, `contract_opts`, `lazy`, and `reduce_opts` into the
  `projector2d` or `l2bp` implementation. Pepsy filters only the unsupported
  keys in a call-scoped adapter; it does not replace either algorithm.
- **Native policy:** retain the existing Symmray stabilization and live-network
  insertion safeguards for `projector` with simple or layered gauging.
  `projector2d` currently reaches a block shape mismatch on the native U1U1
  regression, while disabled gauging does not prepare a compatible projector
  environment. Those choices, dense D2BP-assisted projectors, and `l2bp`
  remain capability-gated as dense-only until native message charge alignment,
  projector shapes, and fermionic sign behavior are validated.
- **Prototype:** `ctmrg_projector_region=(2, 3)` wraps the public callable 1D
  compressor and enlarges every central two-site projector calculation with
  one neighboring site, alternating left and right extensions. Existing
  `layer_tags=("KET", "BRA")` semantics absorb bra and ket sequentially, and
  `ctmrg_canonize="bp"` supplies a newly converged D2BP environment to every
  projector compression. A dense 3x3 PEPS regression observes actual
  three-site regions and matches the untruncated exact contraction.
- **Defer:** message warm starts and transport between CTMRG steps, true
  generalized/Kikuchi BP region messages, periodic regional windows, native
  Symmray support, and MERA/disentangler optimization remain separate follow-up
  algorithms. The 2x3 selector does not imply any of them.

## 2026-09-14 flat directional and middle-out audit

The active environment contains Quimb `1.15.1.dev55+gd0591eb70`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, and Symmray
`0.3.2.dev8+g6c6dd34b5`. The installed
`TensorNetwork2D.contract_boundary(...)` and `contract_ctmrg(...)` signatures
both accept an explicit boundary `sequence` and an `around` region. Quimb's
shared interleaved dispatcher cycles that sequence one layer at a time, so
one-sided, two-opposing-sided, and four-sided outside-in schedules need no
Pepsy algorithm fork. With `around`, each selected direction stops immediately
outside the protected target bounding box and returns the reduced network when
`final_contract=False`. Autoray, Cotengra, and Symmray did not expose a
conflicting API change for this path.

- **Adopt:** `contract_flat(boundary_direction=...)` maps readable physical
  directions to Quimb's native `xmin`/`xmax`/`ymin`/`ymax` sequences for MPS
  and CTMRG methods. `compression_mode` is a readable alias for the existing
  `mode_` argument on the MPS route. Existing defaults and raw `sequence`
  control remain unchanged.
- **Adopt:** `middle-out-x` sets `sequence=("xmin", "xmax")`, protects the
  selected central row slab with `around`, and absorbs bottom/top boundaries
  towards it. `middle-out-y` analogously absorbs left/right boundaries towards
  a central column slab. The reduced core is then contracted exactly.
  `middle_slices` can target the geometric center or an explicit
  target/operator interface.
- **Direct and CTMRG:** the MPS route selects direct SVD compression by default
  for middle-out; CTMRG uses its selected projector/BP mode with the same
  target geometry. This keeps direction scheduling orthogonal to the boundary
  compressor.
- **Compatibility gate:** middle-out is 2D-only, requires two open boundaries
  on its selected axis, and relies on Quimb accepting `around` and
  `final_contract=False`. Ordinary boundary contraction remains available when
  it is not selected. A time-cyclic trace should contract left/right along y
  unless its x cycle is cut explicitly.
- **Validated:** odd/even x/y targets and left/right contraction on a
  time-cyclic x-axis network match exact contraction when `chi` is
  untruncated. A Torch float64 regression verifies that the raw mantissa
  retains gradients back to every input tensor.
- **Defer:** a genuinely center-first growing slab. It carries two open fronts
  and is a different, generally more expensive truncation topology than the
  requested outside-boundaries-to-middle contraction.
