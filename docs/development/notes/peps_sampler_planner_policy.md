# 2026-09-26 — Reusable PEPS planning and directional row caches

## Decision

Adopt the existing public `pepsy.tensors.build_optimizer` (compatibility name
`build_contraction`) for full-network contractions. Each default PepsSampler
owns one reusable Cotengra optimizer with `parallel=False`. Small cached-row
contractions use `auto-hq`; callers can configure both separately. Explicit
legacy optimizer arguments still propagate to rows unless overridden.

The roughening runner and benchmark use the same policy. The runner keeps one
structural optimizer across snapshots while creating fresh numerical samplers. No installed library
was modified. This adopts an existing Pepsy abstraction, not a compatibility
shim or a new contraction algorithm. The unchanged dependency/API audit from
the [efficiency work](peps_sampler_efficiency.md) was reused; installed public
builder, optimizer, contraction-tree cost/size, and slicing signatures were
inspected for this change.

## Two different caches

The reusable optimizer caches contraction plans by graph/index dimensions.
It does not cache numerical contraction outputs. It survives sampler refresh.

The row environment cache uses X-column tags to identify local blocks. Right
suffixes are built once per conditioned row; left prefixes update immediately
after sampling. The bottom ket boundary advances after each row. These values
are tied to the sampled prefix and the physical snapshot, like directional
FIT/DMRG environments. Refresh clears them. Sharing tags or matching dimensions
is insufficient to reuse values for a different prefix. Existing numerical
cache behavior was preserved rather than replaced with tag-only memoization.

## Exact amplitude checks

Optional `amplitude_max_intermediate_bytes` and `amplitude_max_cost` inspect the
selected tree before normalized-leaf allocation or exact execution. A cached
tree is checked again before use, so adjusting a limit cannot bypass it.
`amplitude_plan_info` exposes the last estimates, including a rejected plan.
Rejection leaves cache initialization retryable. No approximate amplitude or
partial public batch is silently returned.

These checks limit one estimated intermediate and Cotengra's estimated cost.
They are not a total process-memory, planner-memory, GPU-memory, or time limit.
Both default to None. Exact slicing can be configured through a supplied public
optimizer; this change does not impose slicing or any extra truncation.

Runner controls: `--peps-sample-opt pepsy` (default),
`--peps-sample-row-opt auto-hq` (default), optional
`--peps-sample-max-intermediate-mib` and `--peps-sample-max-cost`.
The existing finite-cap proposal/importance-weighting rules are unchanged.

## Large-case measurements

Same evolved-state methodology as the [large corner probes](peps_sampler_large_corner_cases.md):
five dt=.2 Ising simple-update steps, D=4, complex128 NumPy, chi=16,
chi_prime=8, explicit absolute rho repair, 64 MiB factored budget, one BLAS/OpenMP
thread, four proposal draws in two chunks, then two public draws in chunks
of one. Existing production jobs remained active; some probes overlapped, so
these are resource checks and indicative timings, not isolated speed claims.
Each process had 16 GiB address-space and 150 CPU/180 wall-second limits.

| Full planner / row path | Shape | First / second two-shot proposal chunk | Public two-shot batch | Peak process RSS | Largest exact-plan tensor |
| --- | --- | ---: | ---: | ---: | ---: |
| auto-hq / factored | 9×9 | 1.30 / 1.26 s | 1.56 s | 645 MiB | 16 MiB |
| auto-hq / reference | 9×9 | 18.38 / 2.49 s | 2.75 s | 647 MiB | 16 MiB |
| auto-hq / factored | 10×10 | 1.66 / 1.73 s | 2.91 s | 675 MiB | 16 MiB |
| Pepsy builder / factored auto-hq | 9×9 | 1.38 / 1.27 s | 1.50 s | 659 MiB | 16 MiB |
| Pepsy builder / factored auto-hq | 10×10 | 1.65 / 1.64 s | 3.63 s | 880 MiB | 64 MiB |

All completed with finite proposal/amplitude logs and normalized weights.
The builder cases enabled a 2 GiB largest-intermediate and 5e10 cost limit.
The inspected tree was used for the initial exact amplitude and then reused
by the public batch; proposal calculations used the stated full/row policy.
Hyper-optimized paths can differ between runs. No 81/100-site dense oracle was
constructed; small-system tests provide independent numerical checks.

The 10×10 auto-hq result corrects any inference that the earlier greedy 4 GiB
plan was unavoidable. The reusable builder is selected for explicit ownership
and caching control, not because it beat auto-hq on every measured graph.
Warm reference performance also shows why greedy's large slowdown should not
be used as a general cache speedup claim.

## Automatic selection and GPU status

Evaluated factored versus reference behavior at 9×9 with auto-hq, together with
the earlier 4×4 and 8×4 measurements. Factored reuse helps these wide cases;
small earlier cases regressed. A follow-up 4×4 D=4 check under the new builder/auto-hq policy (16 shots,
chunks of four, two warm repeats) measured 0.621 s reference versus 0.542 s
factored. Both matched dense amplitudes within 2.6e-17; finite-cap log q still
differed from Born by about 1.04e-4. This differs from the earlier greedy-based
small-case ranking and reinforces the dependence on planner and workload.
A universal automatic selector is **deferred** pending broader width, chi,
prefix fragmentation, and cold/warm measurements. Library cache activation remains opt-in, with the existing
memory estimate/fallback; the roughening sampling integration explicitly
chooses the factored path.

An isolated GPU run was not available: the only device is an RTX A5000 (24 GiB),
occupied by the authorized DMRG job at about 99% utilization when inspected.
It was not stopped or reconfigured. No new shared-device GPU timing is
presented as isolated validation. This requested validation remains outstanding.

Raw bounded probes: `/tmp/peps-large-*-auto-hq-full.json` and
`/tmp/peps-large-*-pepsy-builder.json`. Reproduction scripts are temporary;
principal settings and results are retained here.

## Validation

- Existing sampler/shared-result/API/layout: 396 passed, 6 CUDA/CuPy skips,
  one known installed-version failure (0.4.1 runtime versus 0.5.0 checkout).
- New-feature suite: 46 passed, including default builder ownership, explicit
  overrides, exact numerical references, limits, rejected-plan retry, and
  rechecking cached plans.
- Runner integration: 60 passed, plus one new default/limit regression; final
  PEPS runner suite after cross-snapshot plan reuse: all 26 passed.
- Repository smoke: 162 passed, 1 skipped, same known metadata failure.
- Changed Python lint, whitespace, benchmark dense oracle, and local report
  links passed. No full numerical suite or isolated GPU success claimed.
