# 2026-09-28 — Hamiltonian-aware spatial reuse

Scope: user requested lattice/Hamiltonian translation and rotation awareness in
both Pepsy MPO and PEPO cluster constructors. Branch `develop`, baseline
`4e398e4`; implementation and this record are uncommitted and unpublished.
Earlier geometry/compile/API edits and unrelated PEPS/sampler work are preserved.
Gaugy and running simulation jobs were not changed.

## Implemented

- Shared private labeled-graph planner identifies equal local ordered targets
  under site permutations. Graph structure, edge multiplicity, operator/Pauli
  labels, immutable coefficients or parameter references, and factor order
  all participate. The plan is exact structural matching, without a numerical
  equivalence tolerance. Spatial permutations do not rotate physical X/Y/Z.
- The search groups sites by degree and checks at most 4096 permutations;
  larger searches use identity-only matching. This can miss valid reuse but
  cannot merge inequivalent targets. `search_fallbacks` reports occurrences.
- `spatial_reuse=True` defaults on in MPO interval/graph constructors,
  `MPOBasis` compilation helpers, cluster facades and `PauliPEPOBasis`.
  False selects unreduced local evaluation. MPO helper cache keys include the
  policy; ordered PEPO factors must agree on it.
- PEPO compilation prepares uniform and localized plans. Located default
  coefficients and independent coefficient-vector slots have separate plans;
  mixed product binding modes are added on demand. Sparse static slot support
  is cached once to avoid rescanning dense one-hot maps for every placement.
- Reused operators are transported by public Autoray reshape/transpose calls.
  Numerical target caches exist only inside one evaluation. Torch values and
  graphs remain fresh across calls. Residual subtraction and tensor insertion
  still run for each relevant placement; existing rank/collection policies
  are unchanged.
- MPO cache reports expose target/representative/reuse/search-fallback counts.
  PEPO reports expose compiled plan counts and last evaluated local-target count.
  The legacy dense `ClusterExpansionPlan` retains its existing TI/C4 behavior.

## Conservative limits

MPO matching currently accepts NumPy product operators and immutable numeric or
MPOParameter coefficients. Opaque coefficient callables, backend-connected
operator tensors, string operators and unfactorized dense local terms retain
independent evaluations. PEPO opaque coefficients retain slot identity; two
separate slots are not inferred equal from their current numbers. Uniform
PEPO slots also remain independent under vector overrides. Operator terms and
geometry should be treated as fixed after compilation, as with the existing
static embedding caches.

Graph equivalence can be broader than geometric C4: a uniform four-site path
has the same local Hamiltonian whether drawn straight or bent. All geometric
placements still enter the expansion. The automatic local-target policy does
not authorize the stronger existing `symmetry="C4"` PEPO block transport for
anisotropic models. Spin/lattice combined transformations and fermionic site
permutation reuse are deferred. No end-to-end performance or global accuracy
claim follows from local reuse.

## Upstream capability audit

Reviewed official Quimb changelog, Autoray repository, Cotengra documentation
and changelog, and Symmray repository. The Symmray abelian-array documentation
page could not be opened; official repository and installed source were used.

- [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html)
- [Autoray source](https://github.com/jcmgray/autoray)
- [Cotengra docs](https://cotengra.readthedocs.io/en/latest/)
- [Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html)
- [Symmray arrays, unavailable during audit](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
- [Symmray source](https://github.com/jcmgray/symmray)

Installed: Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Cotengrust
`0.2.1`, Symmray `0.4.1.dev7+g83fb22865`, Torch `2.6.0+cu124`.
`autoray.do(fn, *args, like=None, **kwargs)` and registered NumPy/Torch
reshape/transpose dispatch were inspected.

Classification: **adopt** public Autoray operations already used by Pepsy;
no compatibility shim or installed-library change. Native charge MPO output
and PEPO conversion were checked separately by focused tests. GPU/JAX
symmetry transport and fermionic permutation extensions remain **unverified /
defer**; no new support claim is made.

## Validation

Activated the device's existing Python 3.12 environment, with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for numerical checks.

- Combined focused gate:
  `python -m pytest -q -o addopts='' tests/test_cluster_spatial_reuse.py tests/test_mpo_cluster.py tests/test_cluster_expansion.py tests/test_public_api.py tests/test_package_layout.py`
  → **175 passed**, two existing compatibility deprecation warnings, 57.98 s.
- New suite: 15 checks cover complete square ordered targets, anisotropy,
  independent equal-valued parameters, interval and square Torch gradients
  over repeated calls, coefficient-vector overrides, uniform/located PEPOs,
  periodic bond multiplicity, native U1 MPO assembly, option/cache separation,
  and 5x6 snake-map counts. Existing dense exponential and native PEPO references
  also pass with reuse enabled.
- A new gradient probe initially used a residual weighted by entries 0..63;
  identical primal values had backward accumulation differences up to 5.2e-12.
  The test now uses a bounded 0..1 observable and keeps the 1e-12 absolute
  tolerance. Separate square permutation and full PEPO gradient comparisons
  check transport and accumulation. No production numerical tolerance changed.
- Full `python -m ruff check src tests` and `git diff --check` passed.
- The initial broader gate was interrupted after **1461 passed, 1 skipped,
  1 failed** (343.99 s). The failure was
  `test_mps_dynamic_controls.py::test_kraus_probabilities_use_tracked_center_without_global_norms[jax]`:
  JAX GPU allocation failed with CUDA out-of-memory / cuSolver initialization
  errors before cluster code. The same test failed with the same GPU error
  in a fresh process. Only this test runner was interrupted; no simulation
  process or shared backend configuration was changed.
- The replacement full gate used process-local `CUDA_VISIBLE_DEVICES=''`
  and `JAX_PLATFORMS=cpu`, plus the same numerical threading limits:
  `python -m pytest -q -o addopts='' --maxfail=1` → **5199 passed,
  105 skipped, 777 warnings in 1589.59 s (26:29)**. The earlier JAX MPS
  case passed on CPU. Skips include unavailable GPU/nondefault-device paths;
  this establishes a full CPU-suite pass, not GPU validation. Both test
  processes have exited; no background validation job is left running.
- Relevant documentation links checked: 19, no missing local targets.

Tracked edits use unified patches through `git apply` because the session's
`apply_patch` sandbox fails before execution. No work is staged or committed.

## Measured local construction work

One process, NumPy complex128, 5x6 open square, p=4, uniform
`0.2 sum X + 0.7 sum ZZ`, step `-0.01j`. MPO uses the square physical graph
with a snake chain map. The PEPO probe spells out located site/edge terms to
exercise occurrence-aware symmetry. Three evaluations per configuration;
medians below. Disabled policy was measured first; process-wide geometry
caches can therefore benefit the later compilation. These are small local
phase measurements on a shared machine, not full optimizer timings.

| Phase | Reuse disabled | Reuse enabled |
| --- | ---: | ---: |
| MPO compilation | 0.15542 s | 0.18877 s |
| MPO targets plus residual subtraction | 0.35230 s | 0.29262 s |
| Located PEPO compilation | 0.04250 s | 0.07773 s |
| Located PEPO local targets only | 0.10824 s | 0.00334 s |
| Placed targets / evaluated representatives | 492 / 492 | 492 / 6 |

All 492 placements remain (30 singles, 49 edges, 118 triples, 295 quadruples).
The six local graph types are one each at sizes 1, 2, 3 and three at size 4:
path, star, plaquette. The 19 oriented size-four shapes are unchanged. PEPO
residual contraction, factorization and insertion, MPO assembly/compression,
and optimizer/sampling costs are excluded from the respective measured phases.
The dominant remainder in the MPO phase is residual subtraction.
