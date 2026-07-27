# Plan: SymDMRG2 compiled matvec fanout GEMM

## Status

Implemented and retained after a warmed 6x6 PBC chi=64 hot-loop A/B. The
compiled path now builds bounded source-fanout groups for eligible bosonic,
unfused NumPy plans and reports the census, static storage, predicted savings,
and actual GEMM-call diagnostics. Native-fermionic, fused, outer-product, and
metadata-mismatch paths retain their existing fallbacks.

## Evidence and decision

The Fermi-Hubbard input MPS is natively fermionic, but SymDMRG2 intentionally
bosonizes the local state, two-site theta, active MPO operands, and projected
contractions. The benchmark-critical local path therefore uses the existing
NumPy compiled block plan, rather than Symmray's per-block `tensordot` path.

The current plan batches equal `(M, K, N, dtype)` output products. Its
identical-right-source case still uses batched `numpy.matmul`, which can retain
one small-GEMM dispatch per output internally and excludes compatible outputs
whose row counts differ. The bounded sector effective-Hamiltonian cache is
correctness-first and validates only on small right-first windows; it is not
the next 6x6 performance delivery.

The next experiment is a *source-fanout GEMM*: group outputs that use the same
dynamic right block schedule and evaluate all their static left maps with one
ordinary two-dimensional GEMM.

## Scope and non-goals

In scope:

- NumPy-backed, unfused compiled `_BlockPairContraction` plans.
- Output blocks with an identical `right_specs` schedule, common reduced
  dimensions `(K, N)`, and compatible input/output dtypes.
- Static left-matrix stacking during plan construction and row-slice scattering
  during each subsequent matvec.
- Counters and timing needed to make a warm 6x6 performance claim auditable.

Out of scope:

- Native-fermionic, mixed, outer-product, fused, non-NumPy, or metadata-mismatch
  paths: retain the existing direct/compiled fallbacks exactly.
- Changing the public `SymDMRG2` API or the theta sector basis.
- Replacing the two-contraction projected matvec with an unbounded dense local
  Hamiltonian.
- Retuning SVD or the local eigensolver in this iteration.

## Target algorithm

For a group of output plans with identical dynamic schedule, the existing
compiled representation has

```text
A_i : (M_i, K)       static left map for output i
B   : (K, N)         dynamic right matrix, built once from right_specs
C_i : (M_i, N) = A_i @ B
```

The fanout representation precomputes once:

```text
A_stack = concatenate((A_0, A_1, ..., A_g), axis=0)  # (sum_i M_i, K)
row_slices = ((0, M_0), (M_0, M_0 + M_1), ...)
```

Each matvec then performs:

```text
B       = right_matrix(right_specs)   # one transpose/reshape/concatenation
C_stack = A_stack @ B                 # one standard 2D GEMM
C_i     = C_stack[row_slices[i]].reshape(output_shape_i)
```

This differs from a batch dimension: `M_i` may vary, while the dynamic right
matrix is shared. It should replace an eligible group of small products with
one BLAS GEMM and avoid dynamic right-side packing. A group of one retains the
current single-output path. Existing packed batches for *different*
`right_specs` remain unchanged in the first iteration.

Fanout groups must own their output plans exclusively: they may not also appear
in the existing batched or single lists. Bound any additional static stacked
matrix storage using the initial plan census before enabling a group globally;
do not introduce an unmeasured duplicate-matrix footprint.

## Work phases

1. **Census and baseline**
   - Instrument compiled plans to report eligible fanout groups, output count,
     maximum group size, static stack bytes, and predicted matmul-call savings.
   - Capture a warmed 6x6 PBC chi=64 baseline with the established fixed model,
     solver options, seed, and sweep schedule.
   - Record compiled-left/right elapsed time, dynamic pack time, total compiled
     matmul calls, total wall time, energy, and Lanczos matvec count.

2. **Plan construction and application**
   - Build fanout groups keyed by the exact `right_specs` tuple plus `(K, N)` and
     dtype compatibility, deliberately excluding `M` from the key.
   - Pre-stack left matrices and store row offsets plus output-plan indices.
   - In `_apply_compiled_block_plan`, build the right matrix once, issue
     `A_stack @ B`, and scatter rows to the existing output templates.
   - Retain current broadcast batches, packed batches, and singles for all
     outputs that are not fanout-owned.

3. **Correctness and diagnostics**
   - Add fanout-specific counters: group count, output count, maximum fanout,
     static bytes, and GEMM calls. Keep existing batch counters meaningful.
   - Use the actual 3x2 periodic Fermi-Hubbard construction at bond 4 and a
     seeded random theta layout that has a multi-output fanout candidate with
     non-uniform `M_i` when possible.
   - Compare every output block against newly constructed direct
     `tensordot(mode="blockwise")` contractions at `atol=rtol=1e-12`.
   - Verify repeated matvecs reuse the plan, and retain native-fermion
     dummy-mode/lazy-phase regression coverage.

4. **Warm benchmark and decision**
   - Repeat exactly the baseline 6x6 PBC chi=64 run after the implementation.
   - Require solver-tolerance energy agreement, unchanged solver convergence
     behavior, fewer compiled matmul calls, and lower compiled-matvec elapsed
     time. Report wall time separately.
   - Keep and commit the change only if the warm A/B demonstrates a real gain.
     Otherwise retain the current batching and move the next investigation to
     the dominant SVD or local-solver phase.

### 2026-07-27 result

On the actual 6x6 PBC U/t=8 construction, a fixed `chi=64` block-fill state
at the central projected window was warmed once and then evaluated 100 times
with single-threaded BLAS. The paired compiled right/left contractions fell
from 2.2492 s (pre-change) to 2.0618 s (fanout), an 8.3% reduction; total
measured application time fell from 2.3156 s to 2.1279 s. The active right
contraction formed 1,481 fanout groups covering 3,841 output blocks, with
1,271,840 bytes of added static maps and 2,360 predicted output-product
savings per application.

A matching one-sweep `chi=64` solver control retained its energy to
`4.9e-13` and used the same 280 Lanczos matvecs. Its end-to-end wall time was
not used as a throughput claim because it includes cold projected-plan setup
and shared-host noise.

The required full control used the same 6x6 PBC product ramp, density-matrix
mixer, `variational_sector_basis="off"`, seed, and single-threaded BLAS for 30
sweeps. It produced indistinguishable energy (`-16.547808951298467` before,
`-16.547808951298432` after) and the same 7,533 Lanczos matvecs. Aggregated
compiled left/right contraction time fell from 43.3675 s to 41.4009 s (4.5%).
End-to-end wall time was effectively tied (198.6 s before, 199.3 s after), so
this is a retained hot-loop optimization rather than a claim of full-solver
speedup. Additional scale runs should report the fanout timing fields
separately from plan-build and total wall time.

## Test commands

```bash
source ~/envs/py312/bin/activate
NUMBA_CACHE_DIR=/tmp/pepsy-numba-cache \
  pytest -q -o addopts='' tests/test_symmetric_tensors.py tests/test_optimize_mps.py
NUMBA_CACHE_DIR=/tmp/pepsy-numba-cache \
  pytest -q tests/test_public_api.py tests/test_package_layout.py
python -m ruff check src tests
```

Use the existing warm external 6x6 Fermi-Hubbard benchmark configuration for
the A/B rather than adding a performance harness to this package repository.

## Risks and stop conditions

- A fanout stack can increase static memory. Do not enable it until the census
  identifies a conservative storage bound and useful group coverage.
- NumPy/BLAS behavior for small matrices is backend-dependent. A reduced Python
  call count is necessary but not sufficient; the warmed elapsed-time A/B is
  the decision criterion.
- The fanout result has a different dispatch shape from individual products.
  Tight direct-blockwise comparison is mandatory, and any metadata-incompatible
  path must stay on the existing fallback.
- If fanout candidates account for too little compiled time or fail to lower the
  warm compiled-matvec total, stop rather than expanding the optimization into
  unsupported fermionic or dense-operator cases.
