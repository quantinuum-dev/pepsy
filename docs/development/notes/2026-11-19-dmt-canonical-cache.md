# Persistent canonical DMT replay

**Adopt:** ordinary MPO DMT replay defaults to reusing its canonical workspace
across gates and matching `run()` calls. The protected DMT-SVD rule, paper
truncation default, dtype and preservation policy are unchanged.

## API and ownership

```python
info = opt.info_c
rho = opt.run(dmt_update="canonical", cutoff=0.)  # Default update policy.
rho = opt.run(cutoff=0.)
assert opt.info_c is info
center = info["cur_orthog"]
rho = opt.run(dmt_update="full", cutoff=0.)  # Every-gate reference sweep.
```

The first uncached gate establishes all caps with the existing full algorithm.
Thereafter adjacent gates split only their bond, distant gates use the existing
exact-forward/protected-return route, and one-site gates act exactly at their
canonical center. Moving the center to an arbitrary nonunitary one-site action
keeps outside isometries intact without a DMT split. Density mode still rejects
unresolved zero trace; operator mode accepts traceless/zero operators.

This is ordinary-MPO canonical reuse, not promotion of the separate
[real-Pauli interval prototype](2026-11-19-dmt-2d-window.md). No automatic
basis conversion or global variational solver is introduced. Compression order
can change unprotected correlations; full and canonical routes need not give
identical global approximants after actual loss.

- The workspace owns the center and lazily valid trace environments. Committed
  `info_c["cur_orthog"]` mirrors the actual gauge. The dictionary retains
  identity across commits, resets and rollback. Scratch targets use separate info.
- Reuse checks state/tensor/array objects, index order, isometry hints, shape,
  dtype/device, exponent and DMT options. Strong references prevent recycled
  IDs from passing these checks. Available Torch version counters detect
  ordinary tracked in-place writes.
- A valid warm-cache check performs no buffer hash, dense materialization or
  numerical isometry discovery. State replacement, manual canonicalization,
  synchronization, mode changes and rollback invalidate reuse. Changing only
  the gate queue does not. Matching `compress_dmt()` settings can seed the cache.
- **Raw NumPy buffer writes require `opt.sync_canonicalization()` before
  replay.** Identity cannot reveal these edits. A NumPy view of Torch storage
  likewise requires explicit synchronization after writes.
- Only successful finite candidates are committed and cached. Atomic failure
  restores the state/info contents and drops the cache; non-atomic failure
  retains the last successful commit/cache.

Trace environments invalidate only affected prefixes/suffixes. Projected
protection vectors and SVD support are never retained across tensor writes.
Full preparation remains the fallback for cache misses, dtype changes and
failed routed checks.

## Bounded benchmark

[The example](../../../examples/mpo_dmt_canonical_cache.py) uses a positive
mixture of 32 product states, L=16, chi=32, complex128, five 16-gate onsite
passes. Preparation is untimed for both policies; the requested Accelerate
limit is `VECLIB_MAXIMUM_THREADS=1`.

| Policy | Median seconds/pass | Timed full sweeps | Max triple-RDM oracle error |
| --- | ---: | ---: | ---: |
| `full` | 0.20784 | 80 | 8.07e-15 |
| `canonical` | 0.01870 | 0 | 1.14e-15 |

Observed speedup: **11.1× for this onsite workload**. Both retain occupied
bond dimension 32 and the same info dictionary. This is not an entangling
2D timing/accuracy claim. The independent oracle evolves local 2×2 factors
and sums triple densities, without a global density matrix.

```bash
VECLIB_MAXIMUM_THREADS=1 python examples/mpo_dmt_canonical_cache.py \
  --length 16 --chi 32 --repeats 5 --output /tmp/dmt-cache.json
```

[Raw repetitions and stage counts](2026-11-19-dmt-canonical-cache-results.json)
retain both policies. Prior unresolved chi192 native failures are not fixed
or reclassified by this cache change.

## Validation and upstream scope

Broad MPO selection: **319 passed**, eight skipped, four existing warnings.
Additional focused checks cover final lower-chi and non-DMT option cases.
Coverage includes actual loss/current-target marginals, canonical isometries,
stable info ownership, invalidation, rollback, Torch complex128 gradients and
JAX complex128 replay. This is not full-suite or GPU coverage.

The public API/package-layout check has 54 passes and the existing version
metadata failure: installed Pepsy reports 0.5.0 while the checkout declares
0.6.0. No installation metadata was changed as part of this task. Ruff,
skill catalog, local links and whitespace checks pass.

Installed: Quimb 1.15.0, Autoray 0.8.4, NumPy 2.5.2, SciPy 1.18.0,
Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1, Symmray
0.4.1.dev15+g0374aaa3c. Official upstream sources and installed canonicalization
and `array_split` signatures were checked. Symmray's array-doc URL was
unavailable; its repository remained available. No dependency upgrade,
linalg registration, native Symmray algorithm or compatibility shim changed.
