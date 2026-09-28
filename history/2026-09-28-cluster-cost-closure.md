# 2026-09-28 — 5×6 cluster construction cost closure

- Scope: measure the remaining PEPO construction stages and peak memory,
  optimize the dominant stage, and measure numerical construction and accuracy
  of the 5×6 order-four graph MPO.
- Branch / baseline: `develop`, `4e398e4`.
- Commit status: working-tree edits only; nothing staged, committed or
  published. Earlier cluster work and independent user jobs were preserved.

## Changes and findings

- The [PEPO stage profile](../docs/development/notes/2026-09-28-pepo-stage-profile.md)
  records timed targets, lower-support contractions, Pauli expansion and
  sparse block insertion, plus whole-process RSS and traced per-call
  allocation peaks. The largest original stage was lower-support contraction.
  The verified local symmetry plan now caches its frozen lower result by
  representative during each order and permutes it to each placement.
  Calls fell from 462 to five, median stage time from 0.693 s to 0.014 s,
  and full construction from 1.313 s to 0.550 s. Fresh-process peak RSS
  remained about 0.223 GiB. The Pauli expansion is now the largest stage.
- The [graph MPO numerical record](../docs/development/notes/2026-09-28-graph-mpo-5x6-numerical.md)
  contains the actual order-four construction cost, memory and selected
  diagonal errors against an independent complete p=4 collection sum.
  A finite, ill-conditioned NumPy compression matrix caused the default thin
  SVD to fail. A narrow optional SciPy `gesvd` fallback now lets the
  construction continue without changing rank or collection policies.
- The public PEPO/MPO guides, module map, changelog and
  [status ledger](../docs/development/cluster_optimization_status.md) reflect
  the measured scope and numerical limits. No dependency was installed or
  upgraded.

## Validation

- CPU-only affected domain/API/layout gate:
  `python -m pytest -q -o addopts='' tests/test_cluster_spatial_reuse.py
  tests/test_cluster_correctness_review.py tests/test_cluster_fixed_factorization.py
  tests/test_cluster_jit_gradients.py tests/test_cluster_expansion.py
  tests/test_mpo_cluster_recursive.py tests/test_mpo_cluster.py
  tests/test_mpo_cluster_compression.py tests/test_mpo.py
  tests/test_public_api.py tests/test_package_layout.py`
  → **386 passed**, two existing deprecation warnings (132.82 s).
- The optimized 5×6 PEPO's 28,470 active blocks match the no-reuse route
  with maximum entry difference `3.19e-16`. A 2×2 deterministic test
  checks dense values and Torch coefficient/time gradients; a separate
  located JAX JIT test checks coefficient/time gradients.
- Independent graph collection scalar recurrence reproduces the two
  5×6 p=4 references; its small 2×3 p=2 form was checked against dense MPO
  entries. With batch four and 42,671 compression steps, χ=1 assembly took
  486.314 s and peaked at 0.810 GiB RSS; its two relative errors were
  44.3% and 80.7%. χ=2 assembly took 1237.625 s and peaked at 1.017 GiB;
  its errors were 13.6% and 42.9%. The χ=2 child completed and wrote its
  result after the monitor's initial 20-minute wall guard was extended.
  The resumed timeout wrapper exited 124 from its pending alarm; this was
  not a numerical-process failure.
- Full `python -m ruff check src tests` and `git diff --check` passed
  after implementation edits. Relative documentation links were checked.
  The earlier full-suite result predates this change; it is not a
  final-working-tree full-suite claim.

## Limits

- The complete p=4 collection is represented structurally; compression
  introduces a separate numerical approximation. Both tested bond caps are
  inaccurate for the selected diagonals. A larger-cap setting, global
  operator norm and cutoff convergence are unverified.
- Peak RSS is process-wide and includes Python, NumPy and static caches;
  traced allocation peaks are single-call values and do not sum.
- Measurements used one CPU thread while other workloads shared the host.
  Runtime and peak RSS may differ on another machine or compression policy.
