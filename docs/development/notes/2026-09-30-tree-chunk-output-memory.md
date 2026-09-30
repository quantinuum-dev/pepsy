# 2026-09-30 — TreeSampler output memory and backend preservation

The [review](../../../history/2026-09-30-tree-sampling-chunk-review.md)
identified that chunk collection retained every result before concatenating
another full output. The user authorized fixing that allocation and keeping
sampling on the selected tree backend/device.

## Implemented

- Dense chunk collection allocates final arrays once, using the first chunk's
  dtypes and the existing backend allocation helper. Each chunk fills its
  output slice and is released before the next batch. Requests fitting in one
  chunk return the kernel outputs directly.
- The recursive sampling visitor clears its self-reference in `finally`.
  Otherwise its closure retains completed buffers until cyclic garbage
  collection, even after the collector deletes its own references.
- Sampling, amplitude evaluation, and probability evaluation enter the captured
  CuPy device context. Torch allocations continue to use the recorded device;
  explicit `backend="numpy"` still selects host results. Native Symmray's
  sequential sampling algorithm is unchanged.
- Full output and uniform draws still scale with requested sample count.
  Uniform draw order, Born probabilities, dtype, and source state are preserved.
  The 1 GiB environment tile target remains an intermediate-memory target,
  not a total-memory guarantee. No tensor contraction or truncation changed.

Implementation: [tree sampler](../../../src/pepsy/sampling/tree.py).
Public contract: [tree sampling guide](../../api/sampling/tree.md).

## Validation

Shared py312 environment, local sources first, one BLAS/OpenMP thread. GPU
checks used one available NVIDIA RTX A5000 and small states.

- Sampler plus public API/layout suites: **120 passed, 1 skipped, 1 failed**.
  The skip requires two CUDA devices. The failure is the existing installed
  Pepsy 0.4.0 versus project 0.5.0 metadata mismatch.
- New lifetime/backend tests cover NumPy, Torch CPU, Torch CUDA, and CuPy,
  complex64/complex128, exact 4+4+3 shot batches, seeded parity, independent
  probability scoring, and dtype/device preservation. Weak references verify
  immediate release with cyclic garbage collection disabled.
- Single-chunk tests verify output reuse. Native Symmray code/occupation tests
  cover both default sampling and an explicit chunk size.
- The lifetime and single-chunk reuse regressions both fail against the old
  implementation loaded in an isolated temporary process.
- A final explicit NumPy-output check from a Torch tree passed with both
  unchunked and chunked sampling. Ruff and whitespace checks passed.
- Two-device placement and full-package validation remain unverified.

## Bounded memory measurement

Compared commit `dacc200` with this working tree: 16-site plus product state,
complex128, 32,768 samples, chunk size 512, seed 7, and 4,456,448 output bytes.
Both implementations were warmed. Returned configurations and probabilities
were exactly equal within each backend.

| Backend / metric | Baseline peak bytes | Fixed peak bytes |
| --- | ---: | ---: |
| NumPy / tracemalloc | 13,292,293 | 9,099,267 |
| Torch CUDA / additional peak allocated | 13,139,968 | 8,950,784 |
| CuPy / peak active allocator-pool bytes | 13,222,400 | 8,949,760 |

These single-run allocation measurements show about a 32% reduction for this
output-dominated probe. They exclude total process/device overhead and are
not a prediction for large-bond production trees. Temporary script and data:
`/tmp/pepsy_tree_chunk_fix_memory.py` and
`/tmp/pepsy_tree_chunk_fix_memory.json`.

## Dependency audit

Installed Quimb 1.15.1.dev66+ge927f06e1, Autoray
0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f,
Symmray 0.4.1.dev7+g83fb22865, NumPy 2.5.2, Torch 2.6.0+cu124,
and cupy-cuda12x 14.1.1. Inspected existing sampler allocation helpers and
installed NumPy/Torch/CuPy allocation and Autoray dispatch capabilities.

Consulted the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The [Symmray array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
was unavailable; installed code and repository information were available.
**Adopt:** existing native allocation, slicing, and CuPy device-context APIs.
**Defer:** dependency upgrades. No compatibility shim or installed-library
change is needed.

## Follow-up verification after the fix

The user requested another correctness and efficiency review of the final
implementation. No additional implementation change was needed.

- **96 additional sampling calls passed** across NumPy, Torch CPU, Torch CUDA,
  and CuPy; complex64/complex128; ternary trees; physical roots present/absent;
  physical dimension 3; chunk sizes 3 and 50 for 17 requested shots. A 256-byte
  environment target forced tiled contractions through the complete public
  sampling path. Explicit seeds and two successive persistent-RNG calls
  matched unchunked configurations and probabilities. Probabilities also
  matched an independently normalized dense statevector, with maximum absolute
  error 2.47e-8 across both dtypes.
- Warmed 12-site, D=8, complex128 throughput probe: 1,024 samples, chunk 128,
  three interleaved timings per implementation, one CPU numerical thread,
  GPU synchronization around each timed call. Median seconds:

  | Backend | Baseline dacc200 | Fixed working tree |
  | --- | ---: | ---: |
  | NumPy | 0.03949 | 0.03951 |
  | Torch CPU | 0.06505 | 0.06484 |
  | Torch CUDA | 0.07516 | 0.07522 |
  | CuPy | 0.11269 | 0.11309 |

  Differences below 0.4% in this small probe do not establish a speedup or
  a regression. The fix reduces memory without a measured material throughput
  penalty in this case. Chunk-size tuning and production performance remain
  workload dependent.
- Eight successive fixed-implementation calls on each GPU backend, with
  cyclic garbage collection disabled and returned results discarded, returned
  to the same live allocation count after every call: **0 bytes retained
  growth** in Torch CUDA allocated memory and CuPy active-pool memory.
- Temporary script/data: `/tmp/pepsy_tree_chunk_final_review.py` and
  `/tmp/pepsy_tree_chunk_final_review.json`. The earlier suite results and
  two-device/full-package limitations still apply; no production run was used.
