# 2026-10-05 — MPS prefix continuation and GPU gate batches

## Scope and implementation

User-requested follow-up to the [precision/memory work](2026-10-05-mps-precision-memory.md):
preserve completed work at branch caps, batch compatible gate/SVD work, and
exercise larger/slow circuits plus real GPU MPI ranks. Changes are uncommitted
on `develop`, based on `2039fbc`, alongside the earlier uncommitted changes.

**Implemented:** local `MpsOptimizer.run(strategy="auto")` owns continuation
for both stream-local channels and `PauliErrorModel`, including threaded
coalescing. Before a possible next split exceeds the branch budget, it retains
the current counted frontier, copies each prefix as needed, and completes the
remaining count-one trajectories serially. Control registers, leakage state,
weights, original event indices and histories survive. Retention is checked
against the existing memory planner before expansion. Explicit coalesced caps
and low-level automatic restart behavior remain unchanged. MPI auto continues
to select independent shots.

The split decision uses a conservative bound **before drawing outcomes**.
Rejecting an already sampled overflowing draw and resampling would bias the
ensemble. Continuation can therefore begin before an actual overflow; seeded
draw ordering differs from independent restart. Count-aware results expose
`continued_from_cap` and `continued_shots`. Retained count-one leaves can
exceed the coalescing cap; the separate retention budget still applies.

**Implemented:** `_trajectory_batch.py` batches gate contractions and SVDs for
one-gate deterministic segments with ascending adjacent two-site gates in
`mode="swap"`. Dense Torch CUDA/CuPy parents are grouped after canonicalizing
their centers. Autoray performs stacking, contractions and SVD. A small rank
vector is downloaded; each parent is sliced to its own rank under the same
`chi` and abs/rel/sum1/sum2/rsum1/rsum2 cutoff policy. There is no rank padding.
The prepared tensor update is consumed by ordinary optimizer replay, retaining
its validation, norm ledger, stabilization and metadata handling. Batch size
is bounded to 32 parents and a conservative 32 MiB workspace estimate.

**Implemented:** real MPI testing exposed process-dependent Torch tensor
pickle hashes inside compiled plans. A configuration-only Pickler hashes
backend arrays by dtype, shape and values, ignoring storage IDs and device
ordinal. This downloads gate payloads once at runner construction. The prior
NumPy-only pickle representation is unchanged and regression-tested. Backend
array checkpoints from the old fingerprint scheme may need regeneration.

## Upstream audit and classification

The unchanged environment and broad upstream audit from the linked note were
reused: NumPy 2.5.2, Torch 2.6.0+cu124, CuPy 14.1.1, JAX 0.10.2,
Quimb 1.15.1.dev79, Autoray 0.11.1.dev9, Cotengra 0.8.3.dev7,
Symmray 0.4.1.dev11, Python 3.12. No shared dependency was changed.
`mpi4py` 4.1.2 was installed only under `/tmp/pepsy-mpi-probe` for validation.

- **Adopt:** existing Autoray `linalg.svd` dispatch, including the registered
  Torch stabilized/non-stabilized drivers. The caller's `TorchLinalgConfig`
  stays in force. Installed dispatch was tested with batched complex inputs
  and both backward policies.
- **Defer:** upstream batched splitting as a complete truncation solution.
  The installed Quimb `array_split(x, method='auto', absorb='auto',
  max_bond=None, cutoff='auto', cutoff_mode='rel', renorm=None, info=None,
  **kwargs)` shares a batch rank and does not supply all per-parent cumulative
  cutoff behavior needed here. See the [official decomposition API](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/decomp/index.html).
- **Adopt:** public CuPy batched SVD. Its small-matrix Jacobi route and larger
  matrix behavior explain why benefits vary; see [CuPy SVD](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html).
  No approximate driver, global dispatch replacement, copied upstream
  decomposition implementation or installed-library edit was introduced.
- **Compatibility shim:** content-based MPI backend-array fingerprints,
  scoped to configuration serialization and covered by real multi-rank runs.

## Measured performance

RTX A5000, complex128, 16 compatible parents, 14-site random MPS, one random
adjacent unitary on sites (6, 7), `mode="swap"`, cutoff 1e-12. Six measured
repetitions after warmup, alternating order, synchronization around the whole
gate replay; optimizer copies excluded. CPU BLAS/OpenMP threads set to one.
State seed 618, gate QR seed 193. This is a gate-stage microbenchmark, not an
end-to-end noisy-circuit throughput claim.

| Backend | Bond dimension | Serial ms | Batch ms | Speed ratio |
| --- | ---: | ---: | ---: | ---: |
| Torch CUDA | 8 | 103.64 | 21.31 | 4.86x |
| Torch CUDA | 32 | 742.65 | 745.10 | 1.00x |
| Torch CUDA | 64 | 2409.76 | 2401.54 | 1.00x |
| CuPy | 8 | 41.25 | 24.33 | 1.70x |
| CuPy | 32 | 108.78 | 101.51 | 1.07x |
| CuPy | 64 | 248.70 | 241.49 | 1.03x |

Temporary reproducer/output: `/tmp/pepsy-trajectory-batch-benchmark.py` and
`/tmp/pepsy-trajectory-batch-benchmark-final.log`. These final timings include
the retained-slice copy. A focused test subprocess overlapped part of the
large Torch timing; no speedup is claimed for that size. Numerical validation includes
different retained ranks, both complex dtypes, all cutoff modes, nontrivial
Torch gradients with/without truncation, tags, canonical centers, norm
accounting and unsupported-route fallback.

## Validation

- Larger GPU replay: 18 sites, chi 12, 24 noisy/gate pairs, 32 shots; both
  Torch and CuPy match unbatched MPS overlaps, ranks and norm ledgers.
- Real `mpiexec -n 2` and `-n 3`: **32 tests passed per rank** at each rank
  count, covering the new eight GPU cases plus existing MPI integration tests.
  GPU cases cover both dtypes/backends, independent/coalesced replay, CPU
  references and streamed observable reductions. Logs:
  `/tmp/pepsy-mpi-gpu-2.log`, `/tmp/pepsy-mpi-gpu-3.log`.
- Full-suite attempt with `--maxfail=3`: **83 passed, 2 skipped, 3 failed**.
  The failures remain the baseline BP tests documented in the prior note:
  `test_sequential_loop_series_compression_reuses_projected_messages`,
  `test_sequential_loop_series_refreshes_topology_cache_after_reduction`,
  `test_simultaneous_loop_series_compression_uses_one_boundary_snapshot`.
  Log: `/tmp/pepsy-frontier-batch-full.log`. This is not a passing full suite.
- Expanded domain run, including slow cases: **962 passed, 2 failed,
  2 skipped** (426.91 s). All **32 slow cases passed**, including native
  3x4 Hubbard replay across modes/symmetries. The two failures were legacy
  diagnostics test doubles omitting new optional counters; the accumulator
  now defaults missing counters to one. Both unchanged tests pass on recheck.
  Log: `/tmp/pepsy-frontier-batch-domain.log`.
- Final gate-batch suite after copying retained U slices: **44 passed**,
  including the two larger GPU circuits. Torch's registered Autoray `copy`
  deliberately detaches; the batch path reuses the optimizer's existing
  differentiable copy helper instead. Log: `/tmp/pepsy-batch-copy-final.log`.
- Final consolidated domain run: **933 passed, 2 skipped, 32 deselected**
  (94.18 s). This reruns the trajectory, GPU backend, norm/control,
  compression/fermion, MPI unit, public API and package-layout selections
  after the fixes above; the deselected slow cases were checked separately.
  Skips are the unavailable second JAX device and Torch Metal device.
  Log: `/tmp/pepsy-frontier-batch-verified-domain.log`.
- Final branch-bound/memory checks: **89 passed, 1 skipped**. Known zero-weight
  mixture outcomes now do not force expansion of an otherwise deterministic
  retained ensemble. The memory check for serial continuation uses one active
  worker, even if the preceding coalescing used threads. Both refinements are
  covered with serial/threaded regressions. Log:
  `/tmp/pepsy-frontier-bound-final.log`.
- Ruff (`src tests`), `git diff --check`, and local documentation links pass.

## Limits and deferred work

Only one physical GPU is available: the MPI runs are genuinely multi-process,
but all ranks share that GPU. Distinct-device/multi-node scaling is unverified.
Gate batches exclude other compression modes (including default `direct`),
multi-gate segments, reversed/nonlocal gates, active layouts, explicit
normalization, timing/finite/quality diagnostics, native symmetry, CPU/JAX and
multi-worker execution. Those cases retain the ordinary path. No requested
compression algorithm is changed to gain batching. Larger matrices showed
little benefit on this device. Kernel fusion, wider segment batching and
multi-GPU performance remain deferred rather than claimed as completed.
