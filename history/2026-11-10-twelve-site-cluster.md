# 2026-11-10 — Configurable prepared patches and 10×10 twelve-site default

- Scope: user requested rotated 4×3 clusters by default for the 10×10 Eli
  loop-cluster example and clarification of reduced-environment dimensions.
- Baseline: `develop`, `add66d9`; changes remain uncommitted, nothing published.

`PreparedLoopClusterUpdate` now accepts `cluster_mode`, `base_radius` and
`fillin`, retaining its six-site default. The selector controls environment
geometry, batch read footprints and problem/cache metadata together. Graph
distance one with fill-in selects a 3-row/4-column rectangle for a bulk
horizontal bond and its rotation for a vertical bond; open edges clip it.
The example runner defaults to this selection only for 10×10, with explicit
`3x2`/`4x3` overrides. Frozen sources and existing results were not changed.

QR on the left and LQ on the right expose dimensions rL/rR. The contracted
environment has shape `(rL,rR,rL,rR)`, is Hermitianized once per update and
reused during ALS. Physical identities are implicit. For qubit PEPOs and
uniform original bonds D, r ≤ min(D³,4D); it is not a D⁴-by-D⁴ matrix.
No solver, gauge inverse, amplitude or PSD policy changed.

Validation with shared Python 3.12, local sources and one numerical thread:

- BP prepared/batch/reduced/compression plus public API/layout: **214 passed**,
  including CPU/CUDA twelve-site environments, repeated-state refresh,
  truncation and operator matrix-element agreement with ordinary reference.
- Additional larger-patch batch-footprint regression: **1 passed**.
  It checks that updates disjoint at six sites are serialized when
  their twelve-site patches overlap.
- Example loop-cluster workflow: **38 passed**, including CPU/CUDA complete
  operators, both patch overrides, 10×10 default/orientation, preflight and
  cache checks. The broader initial global-cost selection had **94 passed,
  14 failed**; all failures use plain SU with unsupported upstream
  `gate_simple_(transpose=...)`. An independently loaded unchanged HEAD
  example core reproduces the same compatibility error. No upgrade or shim
  was introduced. No production replay or complete repository suite was run.

Upstream audit: inspected installed Quimb 1.15.0 `select_local` and prepared
signatures; coordinate-only 7×7 probes confirmed both orientations and edges.
Autoray 0.11.1.dev16+g14513fea6, Cotengra 0.8.4.dev7+g5d3558c4f,
Cotengrust 0.2.1, Symmray 0.5.0 and Torch 2.14.1+cu126 were unchanged.
Reviewed Quimb/Cotengra changelogs and Autoray/Symmray repositories. The
Symmray array-doc page was unavailable; installed public dispatch and the
repository were sufficient because no native-symmetry path changed.
Classification: adopt existing public `select_local`; no compatibility shim,
prototype or dependency modification. Twelve-site full-run cost/runtime and
convergence remain unmeasured. See the [API guide](../docs/api/bp.md).
