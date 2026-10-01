# 2026-10-01 — Exact pure-vector TreeSampler propagation

User scope: inspect `left_inds` reuse and improve TreeSampler efficiency.
Implementation is in the working tree on `develop`, baseline `5eadd9f`;
the preceding solver/notebook-review changes were preserved separately.

## Canonicalization finding

`TreeSampler._resolve_ttn` copies the state and calls
`canonize_around_node_(root)`. This delegates to `canonize_subtree_`, which
reuses a known canonical region and gauges only its required connector/path.
`TreeTensorNetwork.can_skip_canonize` accepts a live `left_inds` isometry
proof, with additional native charge-map checks for fermionic arrays.
A root-centered source needs no edge gauging; an off-root center only traverses
its path to the root. Sampling performs no canonical moves and preserves the
source arrays, center and `left_inds`. New integration tests confirm this.

## Implemented exact changes

- Dense sampling starts with a singleton root amplitude vector. If the
  incoming environment is a vector and the remaining sibling bond space has
  dimension one, its child environment is also a vector. The physical Born
  weights are squared projected amplitudes instead of a density contraction.
- Tracing nontrivial unmeasured siblings can produce a mixed environment.
  Those branches retain the existing density representation and bounded
  first-child transfer. No numerical rank test, truncation, dtype reduction,
  physical ordering change or new persistent cache was introduced.
- Dense `_amplitudes` clears its recursive closure in `finally`, matching
  sampling. A discarded sampler/configuration snapshot is released without
  waiting for cyclic collection, including after an exception.

For a pure incoming environment, `rho[a,A] = v[a] conj(v[A])`, so transferring
`v` before forming a Gram matrix is algebraically identical to transferring
`rho`. When the future sibling space is trivial, that Gram matrix is never
needed. Purity is established by the traversal, not approximated from eigenvalues.

## Validation

- Activated the existing Python 3.12 environment; backend allocations and
  numerical checks succeeded on the available RTX A5000.
- `python -m pytest -q -o addopts='' tests/test_tree_sampler.py tests/test_tree_canonical_regions.py tests/test_public_api.py tests/test_package_layout.py`:
  **186 passed, one skipped** (requires two GPUs), two compatibility
  deprecation warnings. Coverage includes NumPy, Torch CPU/CUDA, CuPy,
  native Symmray, physical roots, chunk lifetimes, exact dense conditional
  samples and probabilities, source ownership and canonical-path reuse.
- New vector-route and snapshot-lifetime tests both fail against the frozen
  pre-change sampler for the intended reasons. Success/failure lifetime tests
  disable cyclic GC and check weak references to arrays, configs and samplers.
- New derivative checks compare sampled conditional probabilities to
  independently scored, explicitly normalized dense probabilities, for all
  free Torch source tensors and both physical-root/chunk modes. Sources are
  recanonicalized with differentiable QR after installing free parameters.
  Captured amplitude scoring uses a fixed root normalization scalar, whereas
  conditional sampling differentiates normalization; omitting the reference
  normalization produces the same mismatch before and after this change.
  A separate before/after probe found maximum derivative differences below
  `4e-16`. The normalized independent reference agreed below `4e-16` too.
- Ruff and whitespace checks passed; full repository tests were not run.

## Measurement

Serial before/after sampling comparisons use the same synthetic 30-site
balanced tree, three root children, complex128, state seed 19, sample seed 2,
and explicit `threads=1`. Node arrays are captured before timing. Both variants
warm up, and GPU timings synchronize before and after each sampling call.
CPU uses chi=32, 2,048 shots/chunk 256; GPU uses chi=256, 8,192 shots/chunk 2,048.
Configurations match exactly; all returned probabilities agree, and 16 shots
per result pass independent bottom-up scoring (`rtol=1e-10`, `atol=1e-22`).

| Backend | Before (seconds) | After (seconds) | Ratio of medians |
| --- | --- | --- | --- |
| NumPy, chi=32 | 16.374, 15.678, 11.698 | 8.589, 8.911, 8.675 | 1.81× |
| CuPy, chi=256 | 28.853, 29.005 | 17.647, 17.813 | 1.63× |

GPU median elapsed time fell from 28.929 to 17.730 seconds (38.7% less).
CPU median elapsed time fell from 15.678 to 8.675 seconds (44.7% less);
the CPU baseline has more timing variance than the optimized samples.
Temporary script, full timing log and JSON are
`/tmp/pepsy_tree_efficiency_benchmark.py`, `.log`, and `.json` respectively.
The production checkpoint was not supplied; these are synthetic measurements.

## Upstream audit and classification

Installed versions: Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Cotengrust `0.2.1`,
Symmray `0.4.1.dev7+g83fb22865`, NumPy `2.5.2`, Torch `2.6.0+cu124`,
CuPy CUDA12 `14.1.1`.

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The Symmray Abelian-array documentation URL returned an internal error;
the official repository and installed implementation are the fallback.
Inspected installed `tensor_canonize_bond`, `canonize_around_node_`, `ar.do`
signatures and the sampler's NumPy/Torch/CuPy einsum, sum and tensordot dispatch.

**Adopt:** structural vector propagation, closure cleanup, existing live
canonical-region/`left_inds` proofs and backend contractions.
**Defer:** the larger mixed-factor, prefix/subtree grouping and cross-chunk
cache prototypes from the September 30 reviews; these remain separate from
this focused improvement. No compatibility shim or dependency changes needed.
