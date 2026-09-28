# 2026-09-28 — Cluster inventory and geometry caches

Scope: user requested clearer tree/loop counts and faster cluster caching,
with Pepsy first. Baseline is `develop` at `4e398e4`. These changes are in the
working tree, not committed or published. Gaugy was not edited.

## Implemented

- `ClusterExpansionPlan.cluster_inventory` and
  `PauliPEPOBasis.cluster_inventory` return fresh per-size counts for oriented
  shapes and C4 representatives, split into trees and loop-containing shapes.
  Four sites: 19 oriented shapes (18 trees, one plaquette), or seven C4
  representatives (six trees, one plaquette). These are geometry counts,
  independent of finite placements and numerical solve counts.
- Immutable shape levels have a bounded 18-entry cache, reused across cutoff
  changes. Aggregate inventories retain their existing eight-entry cache.
- Finite translated embeddings have a bounded 256-entry cache keyed by shape,
  lattice dimensions and boundary flags. Orientation and multiplicity are
  preserved. No numerical tensor, coefficient or autodiff graph is cached.
- The residual subtraction and tensor factorization algorithms are unchanged.
  Periodic embeddings are not reinterpreted as unique induced site subsets.

## New validation

- `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -o addopts='' tests/test_cluster_expansion.py tests/test_public_api.py tests/test_package_layout.py`:
  **120 passed**, two compatibility deprecation warnings, 35.76 seconds.
- Full `python -m ruff check src tests` and `git diff --check`: passed.
- Independent brute-force four-site connected-subset enumeration on 5x6 OBC:
  295 placements, including 20 plaquettes, with no duplicate placements.
- New tests cover inventory snapshots, both symmetry policies, incremental
  shape reuse, and cache separation across dimensions and boundary flags.
- Shape inventories equal the baseline through size seven for both rotation
  policies. The selected domain suite includes existing dense reconstruction
  and Torch autodiff checks. The full repository suite was not run.

## Measured geometry setup only

Compared old functions extracted from `4e398e4` with the new implementation in
one process, using the same Python 3.12 environment. Values are median wall
seconds over seven repetitions, clearing the corresponding geometry caches
before each repetition. Existing user simulations were running concurrently.

| Workload | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| Request oriented cutoffs 2, 3, 4, 5, 6, 7 in order | 0.03199 | 0.02845 | 1.12x |
| 30 passes of all 19 four-site shape embeddings on 5x6 OBC | 0.01706 | 0.000728 | 23.45x |

The second workload includes a cold first pass and 29 cached passes. These
small setup savings are not an end-to-end PEPO construction or simulation
speedup measurement. Exponentials, contractions and residual solves may still
dominate. No claim is made about memory in bytes; caches are bounded by entry
count and each placement entry grows with lattice size.

The normal apply_patch tool and its shell entry point both failed with the
sandbox startup error `mountinfo path is not absolute`. Tracked edits were
applied as unified patches with `git apply`; no Git metadata was changed.
