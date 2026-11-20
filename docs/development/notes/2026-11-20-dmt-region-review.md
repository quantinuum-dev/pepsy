# DMT region planning and contraction review

The protected-region construction is consistent with partial-trace inheritance;
this change shares repeated partial contractions without changing constraints.

## Scope and findings

Reviewed physical inventories, contained-region elimination, crossing-string
plans, minimum covers, numerical vector construction, environment invalidation,
and paper protected-rank handling on `develop` at `eaedf789`.

- A parent marginal implies every subset, including disconnected subsets.
  All open-square plaquettes contain all L-shaped triples, but not straight
  triples or joint marginals spanning plaquettes. Add uncovered supports.
- Persistent workspaces already plan crossing strings once. Numerical maps
  depend on current tensors and gauge; reusing them based only on shape is
  unsafe. Existing changed-range invalidation is retained.
- `min_cover` minimizes physical-string count, not numerical protected rank.
  `rank_aware` compares a bounded set of candidate covers using current ranks.
  Its candidate construction still repeats geometry work inside each call;
  caching immutable candidate plans is a possible later optimization, provided
  routed temporary orderings remain part of their identity.
- The paper budget $\chi\ge p+q$ is sufficient for its construction, not a
  universal minimum-rank requirement. Exact pass-through can report zero
  ranks because they were not measured. No rank policy or cutoff was changed.

**Adopt:** `_region_vectors` now shares normalized prefixes/suffixes across
physical strings during one call. Keys include the contraction endpoint and
the physical labels contracted so far. Identity gaps are included. No cached
array survives the call, so changes in tensors, gauges or autodiff graphs do
not create stale numerical maps. Memory increases by the distinct partial
vectors retained within that call. Ordering and per-path arithmetic are
unchanged; the tested old/new outputs are bitwise equal.

## Measurements

Complex128 NumPy cores, seed 77, open-chain virtual sizes shown below,
`min_cover`, warmed identity environments, one middle-cut vector assembly.
Twelve alternating old/new repetitions, medians, with
`VECLIB_MAXIMUM_THREADS=1`. These synthetic cores test this helper; they are
not a dynamics or positivity benchmark. The original function was captured
from baseline `eaedf789` before editing.

| Sites | Virtual size | Regions | Coordinates | Before / after ms | Partial contractions |
| --- | --- | --- | --- | --- | --- |
| 16 | 64 | 4×4 plaquettes | Pauli | 0.990 / 0.710 | 72 / 45 |
| 16 | 64 | 4×4 plaquettes | matrix units | 1.301 / 1.006 | 72 / 45 |
| 36 | 96 | 6×6 plaquettes | Pauli | 2.782 / 2.202 | 180 / 135 |
| 36 | 96 | 6×6 plaquettes | matrix units | 3.377 / 2.966 | 180 / 135 |
| 16 | 96 | separated triples | Pauli | 7.920 / 4.056 | 456 / 195 |
| 16 | 96 | separated triples | matrix units | 7.034 / 4.258 | 456 / 195 |

Separated triples request $(i,i+1,i+2,i+5,i+6,i+7)$ for every valid $i$.
The 4×4/6×6 helper gains are 1.14–1.39×; the separated-triple gains are
1.65–1.95×. Full-operation gains require separate measurement because QR/SVD
and protected-basis work remain unchanged.

An initial complete 4×4 compression comparison (complex128, positive mixture
of 96 products, Bloch strength 0.5, cap 64, `min_cover`, full-region scans
disabled) gave old/new medians of 116/123 ms in Pauli coordinates and
126/122 ms in matrix units. Four timed repetitions followed one warm-up in
alternating order. Outputs were bitwise equal and compression was lossy.
These small, concurrently measured differences do not establish an end-to-end
speedup. Numerical preservation checks were not removed.

## Validation and limits

The new focused module has 13 passing cases: independent full-boundary
contractions for plaquettes and separated triples, updates to both sides of
the environment, contraction-count reduction, Torch complex128 gradients
against independent finite differences, and an actual lossy six-spin joint
marginal check in both coordinate representations. The latter checks the
full 64×64 marginal and inherited triples/pair against the exact gated target.
It uses local exact gate contraction to avoid an inflated generic target.

Broader validation and publication status are recorded in the
[session handoff](../../../history/2026-11-20-dmt-region-review.md).
No positivity, long-time accuracy, GPU, or native loopy-PEPO guarantee is added.

## Dependency audit

Installed versions remain NumPy 2.5.2, SciPy 1.18.0, Quimb 1.15.0,
Autoray 0.8.4, Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1, and
Symmray 0.4.1.dev15+g0374aaa3c. Inspected `array_split` and `autoray.do`
signatures; no new dispatch, decomposition, or contraction API is required.

Checked the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The linked Symmray
`abelian_arrays.html` documentation page was inaccessible during this audit;
the repository and installed version were available. No upstream shim,
dependency upgrade, or dense Symmray support was introduced.
