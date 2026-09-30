# 2026-09-26 — PEPS sampler cache audit

- Scope: verify future boundary MPS and within-row cache reuse, conditioning,
  numerical accuracy, memory, and efficiency.
- Branch / baseline: `develop` / `a13031b`, preserving the uncommitted
  [scaling](peps_sampler_boundary_scaling.md) and
  [rho-repair](peps_sampler_rho_repair.md) changes.
- Status: fixes implemented in the working tree; no commit or publication.

## Cache ownership and lifetime

The sweep samples increasing `y`, then increasing `x` within a row. The names
are coordinate conventions; the roles are unsampled future and sampled prefix.

| Object | Reuse | Invalidation |
| --- | --- | --- |
| Future double-layer MPS, capped by χ | One preparation sweep; shared by every sample/query | `refresh()` |
| Initial-row transfers and right suffixes, when enabled | Shared across calls because no previous row was sampled | `refresh()` |
| Later-row transfers and right suffixes | Shared by descendants of the same incoming sampled prefix within a batch | End of that row |
| Left within-row prefix | Updated immediately after fixing both physical layers; separate after branching | End of that row |
| Conditioned single-layer ket, capped by χ′ | Shared only while preceding sampled configurations agree | Project and compress after completing each row |

Future MPS reuse was already implemented correctly. This task extends reuse to
initial-row transfers. It does not reuse an unconditioned lower norm after
sampling or apply a second approximation to the within-row suffixes.
`refresh()` rebuilds all private networks and invalidates both persistent cache
levels. Source mutations require that explicit call.

Later-row caches do not accumulate across calls. The cache no longer retains
its unused center network. The final scalar prefix of a row is not contracted,
because no later site consumes it. The cache memory estimate includes the
retained initial row in addition to bounded active groups and workspace.
`row_cache_stats` counts actual builds, nonterminal prefix updates, and
`initial_row_cache_hits` in transfer mode; zero-probability queries report the
work actually completed before their early return.

## Confirmed numerical bug and fix

The old transfer route multiplied rare branch weights into an unscaled prefix.
It could underflow before the sampler accumulated the next log conditional.
For a single row of product vectors `[1, a]`, querying all ones gives:

| dtype | Width | a | Previous cached result | Fixed log q |
| --- | ---: | ---: | --- | ---: |
| complex64 | 5 | 1e-6 | `-inf` | -138.1551143633 |
| complex128 | 12 | 1e-20 | non-finite error at site (8, 0) | -1105.2408446371 |

Both fixed results agree with `-width * log(1 + a**-2)` at their respective
precisions and with the reference contraction. The original complex64 expected
value in double precision is -138.1551055796; the difference is ordinary
single-precision logarithm accumulation.

The fix removes positive scalar factors from private cached columns, traces,
prefixes, and suffixes. These factors cancel from normalized local probabilities,
including the homogeneous spectral-repair policies. Complex arrays divide their
real and imaginary components by a real scale to avoid reciprocal overflow in
complex division of subnormal values. No tensor phase is removed, no probability
floor is introduced, and zero tensors remain zero. Rho normalization is left to
the existing conditional kernel rather than duplicated in the contraction helper.

This does not rescale the source PEPS, physical amplitude contraction, or input
to absolute-cutoff boundary compression. It does not guarantee recovery when
individual entries are already unrepresentable or severe cancellation destroys
relative information. NaN/Inf input remains invalid. Finite χ/χ′ proposal errors
are distinct from cache errors.

Before/after probes on shapes 1×1, 1×3, 3×1, 2×3, 3×2, and 3×3, D=2,
complex128, χ=8, χ′=4, both future engines, and absolute rho repair agreed with
the reference to about 1.3e-14 in log q after scaling. Seeded batch configurations
matched. Independent dense Born probabilities and original amplitudes are
checked in the untruncated cache lifecycle regression.

## Measured efficiency

One CPU thread, complex128, D=2 random PEPS seed 313, greedy contraction,
Quimb future boundaries, χ=8, χ′=4 for 2×3/3×3; χ=0, χ′=1 for 8×1.
Four shots per call, one warmup, median of seven repetitions with rotating
method order and seeds 800–806. Torch uses CPU `inference_mode`. Construction
is excluded. The earlier class is the saved source immediately before this
cache task, including the preceding rho repair and boundary scaling.

| Backend / shape | Method | Previous dense cache (ms) | Fixed dense cache (ms) | Reference (ms) |
| --- | --- | ---: | ---: | ---: |
| NumPy 2×3 | grouped | 14.61 | 15.50 | 13.51 |
| NumPy 3×3 | grouped | 27.63 | 30.21 | 27.51 |
| NumPy 8×1 | grouped | 7.79 | 6.57 | 11.39 |
| NumPy 8×1 | serial | 17.42 | 11.55 | 17.37 |
| Torch 2×3 | grouped | 18.79 | 21.31 | 18.24 |
| Torch 3×3 | grouped | 45.89 | 56.13 | 43.14 |
| Torch 8×1 | grouped | 12.84 | 11.58 | 16.65 |
| Torch 8×1 | serial | 30.48 | 23.79 | 30.27 |

All configurations, log proposals, and log amplitude magnitudes matched the
reference within 3e-12 in this timing probe. The retained first row helps most
when it is a substantial fraction of the calculation. Stable rescaling has a
cost on multiple-row cases: there is no general dense-cache speedup.

The default `row_cache_max_bytes=0` remains appropriate for these small grids;
positive budgets explicitly enable the existing conservative routing heuristics.
A positive budget is not a total process/device memory cap. The benchmark ran
on a workstation with other production calculations active; no GPU performance
claim is made. Temporary reproduction files: `/tmp/pepsy_cache_timing.py` and
`/tmp/pepsy_cache_timing_final.log`.

### Factored-cache prototype — deferred

A temporary subclass kept columns factored and formed prefix/suffix tensors
without materializing dense local super-transfer tensors. NumPy complex128,
D=3, four shots, greedy contraction, absolute rho repair:

| Shape / caps | Reference median (ms) | Prototype median (ms) | Held prototype row arrays (bytes) | Old conservative dense estimate (bytes) |
| --- | ---: | ---: | ---: | ---: |
| 3×3, χ=8, χ′=4 | 36.00 | 39.68 | 36,288 | 170,714,432 |
| 4×4, χ=16, χ′=8 | 82.78 | 90.46 | 129,888 | 2,419,963,904 |

Held arrays and the conservative workspace estimate measure different things;
these are not measured peak-memory reduction factors. The prototype agrees
with reference probabilities but was 9–10% slower in these bounded cases.
It is not integrated or enabled. Native shape-grouped contractions remain
future work. Temporary files: `/tmp/pepsy_factored_cache_probe.py` and its log.

## Upstream evidence and decisions

The same-session official dependency audit in the
[rho-repair note](peps_sampler_rho_repair.md#upstream-evidence) was reused; the
environment is unchanged. Installed versions: NumPy 2.5.2, Quimb
1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305, Cotengra
0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1, Symmray 0.4.1.dev7+g83fb22865,
Torch 2.6.0+cu124, JAX 0.10.2. The unavailable Symmray abelian-array page was
already recorded there. This sampler remains dense; native Symmray support
was not added or claimed.

Installed public `tensor_contract` supports `preserve_tensor`, `get="tree"`,
and `strip_exponent`. Cotengra's exponent-stripped contractions scale their
intermediates, explaining why the reference likelihood remained stable. The
cache fix uses public Quimb contraction/copy/modify APIs and native Autoray
reductions/division; it introduces no extra explicit scalar transfer to the host.

- **Adopt:** positive scale removal, initial-row reuse, unused-object release,
  explicit memory accounting, and focused cache lifecycle regressions.
- **Prototype / defer:** factored row layout; lower held storage but no measured
  speed advantage in the tested cases. Larger native batches/GPU timing deferred.
- No compatibility shim, dependency update, or installed-library edit.

## Validation

Focused cache/device checks: 21 passed, 215 deselected, 3 existing Quimb
`mode`/`method` warnings. Coverage includes NumPy/Torch/JAX CPU, both complex
precisions, JAX nondefault CPU placement, Torch conditional gradients,
branching/exact probabilities, immutable suffixes, refresh, and memory fallback.

The final complete sampler suite passes: **236 tests**, 37 existing Quimb
`mode`/`method` warnings, 300.73 seconds. API/layout has 58 passes and smoke
has 162 passes plus one skip; both retain only the known version-metadata
failure (installed/runtime 0.4.0 versus checkout 0.5.0). Ruff, documentation
link/catalog checks, and diff checks pass. Details are recorded in the
[session handoff](https://github.com/quantinuum-dev/pepsy/blob/develop/history/2026-09-26-peps-sampler-cache-audit.md).
