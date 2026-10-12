# 2026-09-30 — Boundary amplitudes by default and proposal-only draws

- Scope: user explicitly requested removing the standalone exact-amplitude
  default, adding an option to skip amplitudes, and explaining cap convergence.
- Branch: `develop`; prior task baseline `f01f596`, current HEAD `5316cb0`
  after concurrent work. This session did not stage, commit, or push. Concurrent
  work staged shared sampler files during validation; the index was preserved.
- Supersedes the compatibility-default decision in
  [the preceding amplitude handoff](2026-09-30-peps-boundary-amplitudes.md).

## Behavior

`PepsSampler` now defaults to `amplitude_mode="boundary"`. `"exact"` is explicit.
`"none"` skips every amplitude evaluator in serial, grouped, chunked, and
streamed sampling. It returns `ps=None`, accumulated proposal probabilities,
zero log averaging weights and uniform normalized weights. Diagnostics identify
`weight_kind="proposal"`; `log_mean_weight=None` avoids inventing a norm
estimate. Reading unavailable log amplitudes raises clearly.

With no explicit amplitude cap, boundary amplitudes inherit χ′. If neither is
set, the boundary sweep is uncapped but still uses cutoff. Proposal-engine
auto selection is unchanged: no χ/χ′ still selects exact conditionals; supplied
caps select a boundary proposal. The amplitude method never falls back to a
full-network exact contraction. Large caps and vanishing cutoff/solver errors
recover q = normalized |Ψ|²; q does not recover complex phase or global scale.

Changing the default exposed boundary overflow/underflow for extreme physical
tensor scales. Both amplitude methods now share cached normalized physical
slices. Boundary contraction keeps physical exponent metadata separate from
backend float32 arithmetic and contracts only its final one-dimensional
boundary through the public Cotengra zero-aware scaled contraction. Quimb's
generic final-contract options do not accept `check_zero`; use the final 1D
tree's public `contract` method instead. No installed dependency was changed.
The preceding active-task upstream audit remains applicable.

The roughening CLI exposes `--peps-sample-amplitude-mode none`. Its selected-time
diagonal observables then average proposal draws uniformly. Amplitude fields
are omitted from snapshots and `amplitude_evaluated=False` is saved. Prior
3×3 experiment data/notebook are preserved; no experiment rerun was requested.

## Validation evidence

New tests forbid amplitude evaluators in none mode, compare identical seeded
draws/q across modes, check all draw interfaces on NumPy/Torch, and enumerate
a small system to show larger χ/χ′ recovering Born probabilities at cutoff zero.
Existing exact-plan tests now request exact mode explicitly. Extreme physical
scale tests cover both exact and boundary modes, including zero amplitudes and
large exponent metadata.

The amplitude-focused selection passed 68 checks. The downstream PEPS,
observable, and sweep selection passed 91 checks; both saved-schema regressions
passed again after the stability fix. Changed-file and full Pepsy Ruff passed.
The broader CPU sampler/amplitude/API/layout selection completed with
**367 passed, 2 skipped, 2 failures** in 449 seconds. One failure was the
already-corrected legacy exact-amplitude test (the process had collected its
old definition); its focused rerun passed. The other was the existing installed
version metadata mismatch. Smoke gave 92 passes and that same version failure;
an earlier selected public-API/layout smoke run gave 54 passes. No full-suite
success is claimed.

A JAX GPU probe failed in existing future-environment/conditional construction,
before amplitude evaluation (cuBLAS autotuning failure in Quimb-MPS; a DMRG
proposal comparison also disagreed). CPU JAX validation is run separately;
this task makes no JAX GPU compatibility claim. An obsolete pre-fix test run
was interrupted after 202 passes; its two failures were tests assuming exact
amplitudes by default and have been made explicit reference tests. Both passed
in their focused rerun. The installed/project version metadata mismatch remains
an independent environment failure, not addressed by changing installations.

## Follow-up cache verification

The user asked to verify fixed future boundaries and DMRG-style site environment
reuse. Both are already implemented: `_prepare_future_environments` constructs
the future MPSs once per refresh; `_build_factored_row_cache` selects `X{x}` and
site/BRA tags, builds suffixes once per conditioned slice, `_row_local_rho`
combines left/local/right, and `_advance_row_prefix` projects both ket/bra and
updates the left side. Prefix groups share suffixes until they move to a new
slice; different histories must not share their conditioned numerical values.

An instrumented 3×3 D=2 probe with default DMRG futures, factored cache, two
12-shot batches (chunk size 4), and a likelihood query observed one future
preparation and one initial-row cache build. Future tensors were unchanged.
Later conditioned rows rebuilt as required by distinct histories; the final
query reused the initial-row cache and advanced six site prefixes. None mode
performed zero amplitude contractions. Memory budget fallback remains explicit
in `row_cache_stats`; caches are not unconditionally enabled for oversized rows.

## Autoray audit requested by the user

Read the linked [Autoray index](https://autoray.readthedocs.io/en/latest/index.html)
and [dispatch/namespace guide](https://autoray.readthedocs.io/en/latest/automatic_dispatch.html).
Checked the installed 0.11.1.dev3+g1b476b305 `get_namespace(like,device,dtype,submodule)`
and `do` signatures. Tensor arithmetic and creation use inferred Autoray
namespaces; Quimb/Cotengra dispatch contractions to the native arrays. Python/
NumPy usage in the sampler is dtype metadata, integer grouping, and explicit
public scalar/diagnostic output. Result weight/log postprocessing is documented
as host NumPy; the implementation is not an entirely device-resident Python loop.

Added `tests/test_peps_sampler_backend_audit.py` covering boundary/none modes
and complex64/complex128 on NumPy, Torch CPU/CUDA, JAX CPU, and CuPy CUDA. It
forbids floating tensor host transfers and checks rho, private PEPS, future
boundaries, conditioned MPS, suffix caches, and amplitude slices retain their
backend/dtype/device. The local versions were NumPy 2.5.2, Torch 2.6.0+cu124,
JAX 0.10.2, and CuPy 14.1.1. All 20 backend/dtype/mode audit cases passed
(74.84 seconds). The two JAX GPU factored/reference proposal comparisons also
passed with `JAX_DEFAULT_MATMUL_PRECISION=highest` (115.65 seconds). Default GPU
matmul precision had produced approximately 1e-4 differences against tight
reference tolerances; this precision setting is required for that comparison.
