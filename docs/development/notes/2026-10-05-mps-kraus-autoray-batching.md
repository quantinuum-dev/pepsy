# 2026-10-05 — Backend-resident MPS Kraus batching

Implemented in the working tree on `develop` / `bec773a`, extending the prior
[trajectory corrections](2026-10-05-mps-trajectory-corrections.md) and
[CUDA performance review](2026-10-05-mps-trajectory-cuda-review.md). The user
authorized GPU optimization using Autoray consistently for Torch and CuPy.
Nothing was staged, committed or published; pre-existing edits were preserved.

## Implemented behavior

- Dense MPS/exact Kraus evaluation prepares one normalized amplitude block per
  channel. Nonadjacent supports are routed once on an untruncated private MPS,
  preserving the declared physical-leg order and tracked-center contract.
- Reuse the existing backend `_array_namespace` helper, backed by
  `autoray.get_namespace` with the existing late-dispatch fallback. Stacking,
  matmul, reshape, max, abs, where, norm and concatenate use that namespace.
  There are no new Torch/CuPy-specific numerical kernels or eager imports.
- Cache converted operator batches in the existing locked stream cache, keyed
  by channel identity, batch interval, backend, dtype and device. Probability
  evaluation never downloads state tensors or projected amplitude arrays.
- Bound each batch's operator-plus-projected element count to 2**20, allowing
  at least one outcome when a single block exceeds that target. This limits
  additional outcome batching, not total state/cache/autograd memory.
- Scale each outcome before its norm reduction, transfer its norm in one small
  vector, then square in host float64. This preserves tested probabilities of
  1e-60 for complex64 states. Host RNG/importance calculations remain unchanged.
  The existing base-norm scalar validation remains a separate readback.
- Preserve native Symmray, tree and stabilizer probability routes. Share their
  existing probability validation/diagnostic normalization with the dense path.
  No public signature, execution strategy or worker default changed.

Source: [noise.py](../../../src/pepsy/optimizers/noise.py).
Regressions: [test_mps_kraus_batch.py](../../../tests/test_mps_kraus_batch.py),
38 new cases covering five backends, reversed/nonadjacent/exact supports,
bounded batches, readback boundaries, real GPU dtype changes, importance weights,
rare/zero outcomes and preservation of a live Torch state gradient.

## Dependency audit and decisions

**Adopt:** public Autoray namespace dispatch and Quimb canonicalize/swap/tensor
APIs. Installed dispatch probes confirmed stack, matmul, norm(axis=1), and
max(axis=1, keepdims=True) on NumPy, Torch CUDA and CuPy. The new regression
matrix also exercised JAX. No upstream internals were copied or patched.

Reviewed the user's [Autoray documentation](https://autoray.readthedocs.io/en/latest/index.html),
its [API](https://autoray.readthedocs.io/en/latest/autoapi/autoray/index.html)
and [repository](https://github.com/jcmgray/autoray), the
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and the
[Symmray repository](https://github.com/jcmgray/symmray). The Symmray abelian-array
documentation returned an internal error; retained the existing native path
and used installed signatures/source plus its official repository. The served
Autoray documentation identifies older versions, so installed dispatch checks
are the capability evidence for this environment.

Installed versions were unchanged: Autoray 0.11.1.dev9+g1291702f9,
Quimb 1.15.1.dev79+gb5e316200, Cotengra 0.8.3.dev7+g1d7fd333f,
Symmray 0.4.1.dev11+g1a3481803, NumPy 2.5.2, Torch 2.6.0+cu124,
CuPy 14.1.1. Inspected Quimb `canonicalize(..., info=...)` and
`swap_site_to(..., info=None, inplace=False, **compress_opts)` signatures.

**Defer:** canonical shot-template reuse, changed automatic workers, and
whole-trajectory compilation. This change isolates the measured probability
bottleneck and maintains the existing replay/normalization policy.

## Measured implementation performance

RTX A5000; NumPy/Torch/CuPy complex64; one host math thread; synchronized CUDA
wall timing. Warm-up excluded; seven kernel repetitions and three full-replay
repetitions, reporting medians. GPU was not reserved. The pre-change noise
module was saved before editing, then loaded separately; full-run comparison
temporarily replaced only `_kraus_probabilities` in the benchmark process.
No installed or repository library was modified by that baseline comparison.

Kernel: seed=932 random 16-site MPS, chi=32, four computational projectors.

| Backend / support | Before | Implemented | Speedup |
| --- | ---: | ---: | ---: |
| NumPy / adjacent `(6,7)` | 1.096 ms | 0.558 ms | 1.96x |
| NumPy / nonadjacent `(4,11)` | 17.492 ms | 4.625 ms | 3.78x |
| Torch CUDA / adjacent | 1.692 ms | 0.690 ms | 2.45x |
| Torch CUDA / nonadjacent | 23.332 ms | 6.189 ms | 3.77x |
| CuPy / adjacent | 2.310 ms | 0.923 ms | 2.50x |
| CuPy / nonadjacent | 26.174 ms | 6.876 ms | 3.81x |

Full replay: same state, H on even sites, disjoint adjacent CNOTs, then one
four-outcome channel at `(4,11)`; direct mode, cutoff=1e-8, eight shots,
seed=173, one worker, retain=none, progress disabled. Initial setup/upload
excluded, per-shot preparation included.

| Backend / strategy | Before | Implemented | Speedup |
| --- | ---: | ---: | ---: |
| NumPy / independent | 0.385 s | 0.294 s | 1.31x |
| NumPy / coalesced | 0.0734 s | 0.0626 s | 1.17x |
| Torch CUDA / independent | 0.676 s | 0.547 s | 1.24x |
| Torch CUDA / coalesced | 0.133 s | 0.115 s | 1.16x |
| CuPy / independent | 0.855 s | 0.703 s | 1.22x |
| CuPy / coalesced | 0.218 s | 0.205 s | 1.07x |

Replacing the final four-outcome channel with four one-site amplitude-damping
events (p=.15 at sites 0,5,10,15) gave Torch CUDA independent 0.677 → 0.557 s
(1.22x); Torch coalesced 0.0899 → 0.0903 s (essentially unchanged). CuPy
independent 0.700 → 0.693 s and coalesced 0.174 → 0.172 s were also essentially
unchanged. Benefits depend on channel/support and the share of replay time
spent evaluating probabilities; do not apply the 3.8x kernel factor to full runs.

A separate warmed adjacent-event Torch profile changed kernel launches 43 → 18,
scalar extractions 9 → 1, and copies/stream synchronizations 9 → 2. The two
remaining host boundaries are the base norm check and outcome norm vector.
Unlike the earlier prototype, these timings include production validation,
scaled rare-weight handling and the batch/cache policy.

Temporary reproduction files: `/tmp/pepsy-noise-pre-gpu-opt.py`,
`/tmp/pepsy-gpu-opt-benchmark.py`, `/tmp/pepsy-gpu-opt-amplitude-benchmark.py`,
and their `.log` outputs.

## Validation

- `pytest -q -ra -o addopts=''` on `test_mps_kraus_batch.py`,
  `test_trajectory_noise.py`, `test_trajectory_importance_regressions.py`,
  `test_trajectory_review_regressions.py`, `test_mps_trajectory_density_reference.py`,
  `test_mps_dynamic_controls.py`, and `test_mps_fermions.py`, with `-m 'not slow'`:
  **411 passed, 1 skipped, 30 deselected**. Skip requires a second JAX device.
- Public API and package layout: **54 passed**.
- Existing CUDA/CuPy selection from `test_mps_gpu_backend.py` and
  `test_trajectory_importance_regressions.py`: **16 passed, 2 failed,
  48 deselected**. Failures remain the previously recorded
  `test_backend_ledger_matches_cpu_and_preserves_state_dtype[cuda-False]`
  and `[cuda-True]` infidelity comparisons. Their dense-state checks pass;
  no tolerance was weakened. This is not a clean full GPU suite.
- All-source/test Ruff, relative documentation links and `git diff --check`
  pass. Full repository suite was not rerun; prior BP/tree/GPU findings remain
  documented in the earlier correction audit.
