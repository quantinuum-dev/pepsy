# 2026-10-01 — Exact factor/group/cache TreeSampler path

Scope: the user requested further work on the factor, grouping and cache
prototype after reviewing the available implementations and their defaults.
Branch/base: `develop` / `5eadd9f`. This implementation is in the working tree,
not committed or published. Earlier solver fixes and vector-sampling changes
were preserved.

## Implemented, opt-in

`TreeSampler(strategy="factor")` is an experimental exact path in the existing
sampler, with one shared traversal rather than a copied recursive algorithm.
The default is `strategy="standard"`; native Symmray keeps its block-sparse
algorithm for either strategy. The earlier literal batched-QR center prototype
was not integrated.

- Carry compact known amplitude factors when their structural size is suitable,
  without eigenvalue/rank tolerances or numerical truncation. Densities still
  handle mixed sibling environments; both factors and later-sibling densities
  can stay indexed by unique full prefixes instead of expanding to every shot.
- Group conditional environments by the entire measured prefix. Group
  collapsed tensor messages/remainders by subtree configurations only. Those
  messages contain selected source tensors and do not depend on outside
  measurements; conditional probabilities do depend on them.
- Keep large remainders compact through the traversal. Tile gathers during
  child collapse, vector projection and mixed-factor projection, so computing
  a small result does not first expand a large tensor into every shot.
- Reuse cached child-density hits plus already-computed misses when cache
  admission is refused. A full-cache regression reduces observed transfer
  batches from the old `[2, 1, 2]` behavior to `[2, 1]` with identical values.
- Use a call-local context across chunks, cleared in `finally`. No sampler-owned
  scratch flags, persistent numerical cache or installed-library edits.
  Reentrant explicitly seeded calls and refresh after source replacement pass.
- Allocate/group within the captured device context; add Torch unique-index
  recovery using stable inverse sorting. Torch CUDA keys use integer multiply
  and reduction rather than assuming an integer GEMM implementation.
- Use exact row grouping when mixed-radix integer keys exceed signed 64-bit
  range; disable cross-chunk caching for that fallback. Unit physical
  dimensions contribute zero coefficients, allowing the valid `2**63 - 1`
  maximum key without trying to store a `2**63` radix coefficient.
- Root subtree and full-prefix numeric keys have identical sorted order, but
  overflow row keys can sort differently because their column orders differ.
  The root's no-copy reuse is therefore restricted to numeric keys. A known
  two-Bell-pair state on 70 binary sites validates the fallback independently:
  removing that safeguard changes 494 configuration entries and violates
  38 pair correlations in the targeted NumPy probe.

The implementation lives in [tree.py](../../../src/pepsy/sampling/tree.py)
and [_tree_factor.py](../../../src/pepsy/sampling/_tree_factor.py); the public
[sampling guide](../../api/sampling/tree.md) documents controls and limits.

## Resource policy and limitations

`cache_bytes=128 * 1024**2` limits retained density key/value payload. Zero
disables admission but preserves within-chunk grouping. Cache growth,
sorting/gathers, dictionaries and autograd graphs are additional memory.

`workspace_bytes=512 * 1024**2` targets individual remainder/density batches
and tiled projection/gather work. A metadata estimate bounds ordinary
shot-dependent densities and small ungrouped remainders. Actual grouped
remainder/result dimensions are checked before allocation; exceeding the
target retries a smaller chunk against the same precomputed uniform slice.
Existing cached prefixes remain valid during retry. At least one shot or
child-index slice is processed even when larger than the target.

These controls do not establish a global memory cap. Source/captured tensors,
simultaneously live buffers, source reshapes, sorting/concatenation copies,
the complete output/draw arrays and Torch backward graphs remain relevant.
Sharing thresholds (`parent >= 64`, remainder size `>= 65536`) are profiling
heuristics. A high-entropy or small-bond workload can gain less or pay more
grouping overhead. The user's production checkpoint was not provided.

## Fresh validation

- Activated the existing Python 3.12 environment. Versions and official
  upstream audit were unchanged from the
  [vector-sampling investigation](2026-10-01-tree-sampling-vectors.md#upstream-audit-and-classification).
  Re-inspected installed NumPy/Torch/CuPy unique/searchsorted signatures and
  dispatch, and verified available Torch CUDA/CuPy devices. CuPy allocator
  instrumentation uses public `using_allocator` and memory-pool APIs.
- `python -m pytest -q -o addopts='' tests/test_tree_factor_sampler.py tests/test_tree_sampler.py tests/test_tree_canonical_regions.py tests/test_public_api.py tests/test_package_layout.py`:
  **323 passed, one skipped**, two compatibility deprecation warnings. The
  skip requires two GPUs; single-GPU CUDA is available and was exercised.
- After adding explicit native output/device assertions, the complete factor
  suite ran again: **134 passed**. It includes 288 dense-reference sampling
  calls across NumPy, Torch CPU/CUDA and CuPy, three dtypes, several arities,
  physical/virtual roots, sliced noncontiguous arrays, heterogeneous local
  dimensions, explicit/persistent RNG and small workspace/cache budgets.
- Eight Torch CPU/CUDA checks compare all free source-tensor derivatives to
  independently scored, explicitly normalized probabilities. Further tests
  cover exact zero GHZ branches, cache exhaustion, compact remainders, chunk
  reduction/retry, overflow order and cardinality boundaries, known Bell-pair
  correlations, nested calls, refresh, source array/center ownership and
  scratch release on success/failure with cyclic GC disabled.
- Existing native Abelian and fermionic sampling tests exercise both strategies;
  dense fermionic samples retain charge conservation and occupation decoding.
- Ruff (`src tests`) and whitespace checks pass. No full package suite or
  production-state validation was performed. Factor execution on two distinct
  GPUs and Torch graph-memory bounds remain unverified.

Classification: **adopt as an experimental exact opt-in**, no compatibility
shim or dependency change; **defer** changing the default or introducing
approximate rank/precision/layout policies.

## Final performance and allocation measurements

Synthetic balanced 30-site trees use root arity two or three, state seed 19,
sample seed 2, complex128, `threads=1`. Bond dimensions are capped by both chi
and the smaller side's Hilbert space. State construction and sampler capture
are excluded; uniforms, call-local metadata/caches and the complete sampling
call are included. Variants warm up and run serially; GPU timings synchronize
before and after sampling. Allocator instrumentation runs separately after
all timing runs for that state. No competing benchmark was run.

CPU uses chi=32, 2,048 shots/chunk 256. CuPy on RTX A5000 uses actual chi=256,
8,192 shots/chunk 2,048. Both final GPU cases process four 2,048-shot chunks;
the final compact representations avoid the early implementation's smaller
512-shot batches. The prior prototype is frozen against its old shared-density
base for comparison. It remains NumPy/CuPy-only and uses its original controls.

| State/backend | Standard times (s) | Prior prototype times (s) | New factor times (s) | Standard/factor median ratio |
| --- | --- | --- | --- | --- |
| NumPy, three root children | 6.684, 9.263 | 4.283, 4.836 | 4.226, 4.771 | 1.77x |
| CuPy, three root children | 17.708, 17.829 | 2.917, 2.940 | 2.945, 2.956 | 6.02x |
| CuPy, two root children | 14.160, 14.151 | 3.559, 3.553 | 3.543, 3.559 | 3.99x |

New CuPy medians are **2.951 s** and **3.551 s**. The prior prototype medians
are 2.928 s and 3.556 s: this work preserves its speed rather than establishing
a further speedup over it. CPU timings have substantial variability; these
two-repeat comparisons do not establish universal or production speedups.

All measured/profiled configurations match exactly between variants and all
probabilities agree at `rtol=1e-10`, `atol=1e-22`. Each timed result also passes
independent bottom-up scoring for 16 configurations.

The memory metric is the maximum increase in **live default CuPy-pool
allocations**, sampled immediately after each allocation through the public
allocator hook, relative to the warmed sampler's initial live payload. It
includes sampling outputs, uniforms, scratch and caches; it excludes initial
state/captured arrays, unused reserved blocks, allocations outside that pool
and Torch backward graphs. It is not total GPU memory or reserved pool size.

| Root children | Standard increment (GiB) | Prior prototype increment (GiB) | New factor increment (GiB) | Reduction versus prototype |
| --- | --- | --- | --- | --- |
| Three | 10.013 | 6.494 | **2.150** | 66.9% |
| Two | 12.009 | 6.029 | **1.670** | 72.3% |

The final retained factor caches hold 12,080,904 bytes (three children) and
4,583,208 bytes (two children), and are cleared on return. Keeping grouped
conditional factors and densities compact and tiling subsequent gathers is
the main improvement over the old prototype's expand-then-contract path.

Temporary reproducibility artifacts are
`/tmp/pepsy_tree_factor_verified_benchmark.py`, `.log` and `.json`. The script
loads the frozen old prototype and synthetic builder through
`/tmp/pepsy_tree_factor_final_benchmark.py`; earlier prototype source/builder
paths are listed in the September 30 notes. These files can disappear; the
construction, conditions and raw results above are durable evidence.
Focused logs are `/tmp/pepsy_tree_factor_final_checks.log` and
`/tmp/pepsy_tree_factor_native_output_checks.log`. Earlier conservative and
pre-compaction measurements were intermediate designs, not the final result.

## Finalization validation — 2026-10-01

The user requested finishing the current factor sampler. The review found no
need for another production algorithm change. Added 16 regressions for actual
default budgets and grouping thresholds, covering single-site leaf/root plans
and branching trees on NumPy, Torch CPU/CUDA and CuPy. Explicit and persistent
RNG calls, with chunks unset/seven, match standard configurations and independent
dense Born weights at `rtol=2e-11`, `atol=1e-14` (96 factor calls, each paired
with a standard call).
The factor test module now belongs to the `tree`, `integration` and `optional`
profiles; collection with all three markers selects its complete 150 cases.
The public guide and constructor help now distinguish an unset shot limit from
workspace-driven batching; the guide also describes density reuse across chunks
within one sampling call.

Fresh focused validation:

```bash
python -m pytest -q -ra -o addopts='' \
  tests/test_tree_factor_sampler.py tests/test_tree_sampler.py \
  tests/test_tree_canonical_regions.py tests/test_tree_entropy.py \
  tests/test_tree_unitary_stability.py tests/test_public_api.py \
  tests/test_package_layout.py
```

Result: **408 passed, one skipped**, four compatibility warnings, 69.77 s.
Single-GPU Torch/CuPy paths were available and exercised; the skip requires
two CUDA devices. Ruff (`src tests`), whitespace checks and relevant local
documentation links pass. Installed dependency versions match the earlier
audit; the installed canonization, NumPy unique and sampler signatures were
rechecked. No dependency changes or installed-library edits were made.

The complete package run used the same activated Python 3.12 environment:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLBACKEND=Agg NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig \
  python -m pytest -q -ra -o addopts=''
```

Result: **6,493 passed, 287 skipped, seven failed**, 716 warnings, 2041.65 s
(34:01). This establishes a completed CPU run, not a passing full suite or a
full accelerator matrix. CUDA skips in that run reflect the process-local
visibility setting; the focused run above supplies the sampler's available
single-GPU coverage. Multi-device and multi-rank checks remain unvalidated.

All seven failures reproduce in isolated selections. Their implementation and
test files are unchanged by this task:

- `tests/test_mpi.py::test_mps_optimizer_run_mpi_keyword_covers_all_modes[dmrg1]`
- `tests/test_trajectory_noise.py::test_unitary_shot_replay_has_a_valid_path_for_each_mps_mode[dmrg1]`
- `tests/test_trajectory_noise.py::test_canonical_mps_modes_replay_kraus_shots[dmrg1]`
- `tests/test_peps_sampler_4x4.py::test_peps_4x4_end_to_end[truncated-torch]`
- `tests/test_peps_sampler_4x4.py::test_peps_4x4_end_to_end[small-caps-absolute]`
- `tests/test_peps_sampler_4x4.py::test_peps_4x4_actual_row_cache_and_refresh`
- `tests/test_peps_sampler_4x4.py::test_peps_4x4_backend_parity[jax-complex64]`

The first three request `dmrg1`, which the committed MPS constructor rejects
before replay or sampling. The PEPS failures concern amplitude/reference or
backend-parity differences; their numerical cause remains outside this review.
Neither the MPS mode policy nor the PEPS algorithms/assertions were changed to
make the broad gate pass.

Logs: `/tmp/pepsy_tree_finalize_focused_final.log`,
`/tmp/pepsy_tree_finalize_domain_collection.log`,
`/tmp/pepsy_tree_finalize_full.log`,
`/tmp/pepsy_tree_finalize_unrelated_mpi.log`,
`/tmp/pepsy_tree_finalize_unrelated_peps.log`, and
`/tmp/pepsy_tree_finalize_unrelated_remaining.log`.

Classification remains **adopt as an experimental exact opt-in**. The review,
regression/profile integration and documentation are finalized in the working
tree on `develop` at `5eadd9f`; nothing was staged, committed or published.
Performance tables above are earlier synthetic measurements, not fresh results
from this finalization. Production-state profiling, default promotion and Torch
graph-memory bounds remain deferred/unverified; no checkpoint was supplied.
