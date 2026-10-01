# 2026-09-30 — TreeSampler correctness and repeated-work audit

Scope: investigate sampling a 5×6 system with chi=256, 8,192 samples,
chunk_size=2048, and native CuPy. This is an audit, not an algorithm change.
The production tree/checkpoint was not supplied; synthetic trees below have
30 physical qubits and explicitly specified geometry. They are not replicas
of the user's evolved 2D state.

## Findings

**Canonical metadata is used correctly on the inspected paths.**
[TreeSampler._resolve_ttn](../../../src/pepsy/sampling/tree.py) copies the
source and calls `canonize_around_node_(root)`. Despite that method's old
"from scratch" docstring, its implementation delegates to
`TreeTensorNetwork.canonize_subtree_`, which reuses `canonical_region`.
This is the tree counterpart of MPS center metadata; introducing a second
`info_c` dictionary would duplicate state ownership.

A 30-site D=8 probe counted canonicalization entry points:

| Source gauge | Constructor | 31 samples in chunks of 7 | Refresh |
| --- | --- | --- | --- |
| Root center | No edge gauging/full sweep | None | None |
| Leaf center, four edges from root | Four edge calls | None | Four edge calls |
| Unknown region | One full sweep | None | One full sweep |

The original state's center remained unchanged. A refresh deliberately starts
from the source, so an unchanged off-root source repeats that path. Construct
once, sample repeatedly, and refresh after source updates; an immediate
refresh after construction is redundant. Raw tensor changes still require
honest invalidation of source canonical metadata.

**CuPy contractions already use matrix multiplication.** The installed
CuPy `reduced_binary_einsum` lowers the sampler's binary contractions to
transpose/reshape plus `cupy.matmul`, unless a supported accelerator handles
them. Replacing the spelling `einsum` with `matmul` alone does not remove
the algorithm's dense density-transfer work. NumPy's current `einsum` calls
use its default `optimize=False`; that is a separate CPU tuning opportunity.

**TreeSampler does repeat shot-independent environment construction.**
`_sample_first_child_density` rebuilds `env[a,A,c,d]` for each node/tile in
each chunk, then releases it. Its cached node arrays are reused, but these
transfer environments are not cached between chunks or calls.
[MpsSampler](../../../src/pepsy/sampling/mps.py), in contrast, caches its
right environments and flattened site matrices until refresh.

Caching every tree transfer is not a safe direct translation: parent and
child dimensions both equal to 256 require 64 GiB for one full complex128
transfer, before layout copies. The 1 GiB tile target bounds an individual
environment tile, not all live allocations. Four chunks repeat the full
environment-construction sweep four times.

**The main measured cost is applying transfers to all shots, not rebuilding
environments or canonicalization.** For parent dimension P, first child C,
and remaining sibling space F, the present route forms a P²C² transfer with
O(P²C²F) arithmetic, then applies it with O(BP²C²) arithmetic for B shots.
Tiling limits memory without reducing that arithmetic. Later sibling
conditionals have separate contractions.

MPS native sampling instead carries a length-chi prefix vector per shot,
cached right environments, and matrix products with local site matrices.
A tree's unmeasured sibling branches generally require mixed parent-bond
densities; replacing them with MPS vectors indiscriminately is incorrect.
However, the current tree path also duplicates densities common to all shots
before the first physical draw: `rho_root` is explicitly allocated with B
copies. Exact sharing there, and use of pure/factored environments where
structurally justified, are candidates for a future performance change.

## Measured evidence

NVIDIA RTX A5000, CuPy complex128, 8,192 samples, four 2,048-shot chunks,
seed 2. Random states use seed 19, balanced binary branches and a root with
three children. Edge dimensions are capped by both chi and subtree Hilbert
space; actual maximum bonds were checked. Timings synchronize the device
around instrumented contractions, so they include instrumentation overhead
and are single measurements, not production throughput guarantees.

| Actual chi | Sampling | First-child transfer application | Transfer construction |
| --- | ---: | ---: | ---: |
| 128 | 9.54 s | 8.01 s | 0.17 s |
| 256 | 39.88 s | 32.03 s | 0.87 s |

At chi=256, applying first-child transfers accounts for approximately 80%
of elapsed sampling time. Removing all repeated transfer construction alone
would therefore have limited benefit in this geometry. The three large
non-root nodes have parent/child/sibling dimensions (256,32,32), not
(256,256,256). Their identical environment shapes were constructed 12 times
(three nodes × four chunks). The root has three 256-dimensional bonds.

For the chi=256 state, initial sampler capture took 0.012 s; three subsequent
refreshes from a root-centered source made no edge-gauging/full-sweep calls.
Eight returned probabilities agreed with separate bottom-up amplitude
contraction to maximum relative error 2.17e-14. This is independent scoring,
not a dense 2^30 statevector comparison.

Temporary scripts/data: `/tmp/pepsy_tree30_audit.py`,
`/tmp/pepsy_tree30_audit.json`, `/tmp/pepsy_tree_canonical_audit.py`, and
`/tmp/pepsy_tree_canonical_audit.json`. Earlier small synthetic timings in
`/tmp/pepsy_tree_sampling_audit.json` were exploratory; the 16-site D=256
request shrank to actual maximum bond 64 and must not be called a chi=256
benchmark.

An optional two-child-root chi=256 run was interrupted after exceeding seven
minutes of total audit-process runtime (including the earlier completed
runs); no completed sampling time or correctness result is claimed for it.
Its two large non-root tensors have bond dimensions (256,128,256), requiring
16 GiB for each untiled first-child transfer. This differs substantially from
the three-child-root geometry above; neither is confirmed to be the user's
actual tree. No production process was interrupted.

## Validation and dependency audit

An isolated prototype under `/tmp/pepsy_tree_shared_density_probe.py` keeps
the root density at batch size one until physical conditioning distinguishes
shots. For such shared densities it contracts the incoming density with the
node first, then its conjugate, avoiding a quartic transfer environment.
It broadcasts conditional probabilities before drawing the unchanged uniform
stream. No repository implementation was replaced.

Twelve CPU cases passed seeded configuration parity and independent dense
Born checks: NumPy/Torch, float64/complex64/complex128, ternary branches,
physical roots present/absent, 31 samples in seven-shot chunks. Maximum
absolute probability error was 4.98e-8 across these dtypes. This limited probe
does not establish full backend/device, gradient, or production readiness.

The same isolated prototype was then measured on the three-child-root
30-site chi=256 CuPy state, with the same seeds, dtype, sample/chunk counts,
and synchronized instrumentation. Sampling took **29.50 s**, versus
**39.88 s** above: approximately **26% less elapsed time** in these single
measurements. First-child transfer application fell from 32.03 s to 21.68 s.
Eight sampled probabilities agreed with independent bottom-up scoring to
1.20e-14 relative error. The large-run seeded configurations were not compared
against retained baseline outputs; seeded parity was checked in the 12 CPU
cases. Temporary GPU script/result: `/tmp/pepsy_tree_shared_density_gpu.py`
and `/tmp/pepsy_tree_shared_density_gpu.json`. This is evidence for an exact
sharing optimization, not an integrated fix or a production speed guarantee.

Activated the existing py312 environment, with local `src` first and one
BLAS/OpenMP thread. No dependency or shared-environment changes.

- `python -m pytest -q -ra -o addopts='' tests/test_tree_sampler.py`:
  **67 passed, 1 skipped**, including real Torch CUDA and CuPy execution.
  The skipped case requires two GPUs; only one was available.
- Existing tests cover independent small-state Born probabilities,
  empirical frequencies, source preservation, chunked seeded parity,
  native Symmray sampling, and backend/device behavior.
- Full-package tests were not run. No numerical implementation changed.

Installed Quimb 1.15.1.dev66+ge927f06e1, Autoray
0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f, Symmray
0.4.1.dev7+g83fb22865, NumPy 2.5.2, Torch 2.6.0+cu124, and
cupy-cuda12x 14.1.1. Inspected sampler/builder/einsum signatures,
Autoray matmul/einsum dispatch, and installed CuPy binary contraction source.

Consulted the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The [Symmray array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
was unavailable; installed source and repository information were available.

**Adopt/retain:** existing canonical-region ownership and native backend
dispatch. **Prototype only, not integrated:** exact sharing of shot-independent
densities, with the limited checks above. Further candidates are structurally
justified factored environments and a bounded cache for reusable small
transfers. Preserve RNG ordering,
refresh invalidation, native device/dtype, and independent Born checks.
**Defer:** dependency upgrades, production-run conclusions without its actual
tree, and blanket caching of quartic transfers. No compatibility shim needed.
