# Larger-square MPO DMT: stability and contraction costs

**Implemented:** cheaper exact region contractions and protected routing for
later nonlocal MPO gates. All output caps, physical-region preservation checks,
native dtypes/devices, transactional commits and explicit precision failures
remain active. This extends the
[earlier replay measurements](2026-11-16-mpo-dmt-replay.md).

## Exact contraction improvements

- Build each region's panel from both ends. For physical dimension $p$ and
  four-site plaquettes, either side carries at most $p^2$ open coefficients
  per bond, instead of a one-sided $p^3$ intermediate.
- Reduce strided ordinary-MPO physical legs directly with native einsum;
  flattening virtual legs previously copied entire cores repeatedly.
- Reuse identical one-site protection projections within each active pair.
  The cache expires before any core write or gauge move. Logarithmic scale
  tracking remains active for both panel halves.

The pre-change 8×8/χ=128 profile spent 10.17 of 14.43 s in region panels and
2.84 s constructing protection vectors; cumulative frames overlap. Thus the
large-square bottleneck differed from the small replay benchmark's target QR.

## Matched compression measurements

CPU, one BLAS/OMP thread, complex128, `cutoff="auto"`, mapped open plaquettes
in snake-row-major order. Input is the same seed-321 rank-136 positive product
mixture across χ values. No global dense operator is formed. Three warmed,
alternating repetitions in one process compared saved pre-change contraction
helpers with the new ones, using the same protected SVD.

| Square / final χ | Before | After | Speed ratio | Maximum local defect, both |
| --- | ---: | ---: | ---: | ---: |
| 6×6 / 96 | 2.825 s | 1.366 s | 2.07× | 7.12e-15 |
| 8×8 / 128 | 13.078 s | 4.845 s | 2.70× | 7.89e-15 |

χ=64 was rejected on this rank-136 input at a conservative protected budget
of 65. This is an algorithmic construction bound, not a proof that no MPO of
that rank can satisfy the marginal constraints. It is not a universal minimum
χ for these lattice sizes. χ=96 and χ=128 passed on both sizes.

A separate final χ sweep (same rank-136 input, three warmed repetitions)
gave compression medians:

| Square | χ=64 | χ=96 | χ=128 |
| --- | --- | ---: | ---: |
| 6×6 | rejected, bound 65 | 1.398 s | 1.699 s |
| 8×8 | rejected, bound 65 | 4.359 s | 5.036 s |

Maximum complex128 local/trace defects were `7.89e-15`/`2.40e-14`.
Complex64 also passed both feasible χ values on both sizes, with maximum
local/trace defects `1.07e-6`/`3.22e-6`. Absolute CPU times vary across runs;
only the alternating table above supports the reported speed ratios.

## Protected routing for distant gates

The old center-edge benchmark was interrupted during full-target
canonicalization/compression; no completed large-replay baseline time is
claimed. A full ket/bra target can enlarge many consecutive bonds before
output compression starts.

The new route moves one endpoint toward the other with **untruncated QR
SWAPs**, applies the physical gate, then compresses during the return SWAPs.
Requested regions are remapped at every return step. Remaining operations
after the gate are permutations, so preserving those remapped reduced
operators preserves the final physical regions:

$$
\operatorname{Tr}_{\bar R}{\color{purple}\widetilde\rho}
=\operatorname{Tr}_{\bar R}{\color{blue}\mathcal G(\rho)}.
$$

Reference panels are computed immediately after the exact gate and their
physical indices reordered back for a final all-region check. A temporary
ordering can need a larger protected construction rank: that bound is retained
temporarily and enlarged bonds are compressed after restoring the original
order. Final `max_bond <= chi` remains mandatory. Reports distinguish this
through temporary split ranks and final bond dimensions. Geometry plans are
built only for the active temporary cut.

The first gate still establishes the global cap with a full protected sweep.
One-site gates, dtype changes and routed precision/final-budget failures use
the full protected reference path. The route changes the approximation
schedule; it is not a globally optimal constrained MPO fit, and ordinary
complex DMT still does not guarantee positivity/global Hermiticity.

### Routed square observations

The initial routed implementation completed the four-gate rank-24 center
schedule with these single-pass, oracle-subtracted observations:

| Square / χ | Time | Largest region defect | Largest trace defect |
| --- | ---: | ---: | ---: |
| 6×6 / 96 | 7.285 s | 1.54e-15 | 4.77e-15 |
| 6×6 / 128 | 9.761 s | 1.99e-15 | 4.22e-15 |
| 8×8 / 96 | 10.901 s | 3.92e-15 | 1.18e-14 |
| 8×8 / 128 | 13.322 s | 4.08e-15 | 1.47e-14 |

Each used one initial target/sweep, one adjacent update and two routed updates;
final bonds obeyed χ. χ=64 was rejected at bounds 65 (6×6) and 74 (8×8).
There is no completed large-replay before/after timing comparison. These
observations precede the trace-only roundoff stabilization below; no precise
replay speed ratio is inferred from them. No full 6×6/8×8 lattice layer was run.

### Trace roundoff and final stability fix

The original complex64 8×8/χ=128 routed run **failed** the independent `1e-5`
benchmark trace check: trace defect `2.78e-5`, local defect `7.39e-6`.
The tolerance was retained. Density routing now contracts the two-site input
density before any forward QR, applies the actual ket/bra payload to obtain
the target trace, and corrects only bounded accumulated roundoff at the final
canonical center. The allowed relative correction is at most `128 * N * eps`;
larger drift fails the routed check. All physical panels are rechecked afterward.

This preserves the gate's computed trace, **including nonunitary changes**,
rather than normalizing to unity. Native trace numerator/denominator graphs
remain connected; logarithmic panel scales cancel their detached normalizers.
Traceless `variant="operator"` routing does not use this correction.

The final worst-case complex64 replay check passed with trace defect `6.29e-7`
and local defect `2.47e-6`, without a full-target fallback. This final run
overlapped regression checks, so its time is not used as performance evidence.

## Reproducible sweep and independent oracle

```bash
python examples/mpo_dmt_benchmark.py --square 6 8 --chis 64 96 128 \
    --initial-rank 136 --operation compress --methods dmt --repeats 3
python examples/mpo_dmt_benchmark.py --square 6 8 --chis 64 96 128 \
    --initial-rank 24 --operation replay --schedule center --methods dmt \
    --repeats 1 --single-pass
```

The four-gate center schedule includes adjacent edges, interior edges and a
physical edge spanning `2 * size - 1` MPO bonds. It is **not a complete lattice
layer**; `--schedule layer` requests every edge. The starting rank is fixed
across a χ sweep and reported explicitly. Compression and replay use different
starting ranks here, so their times must not be treated as a convergence study
on the same evolved state.

Replay validation contracts the input density on the requested region plus
gate support, applies the small dense unitary and traces extra sites. Regions
disjoint from the support and the trace are invariant to gate working
precision. Small-system checks compare this independent oracle against a
full dense target, including reversed supports. It does not use DMT's bases,
workspace, routing or preservation panels.

`--single-pass` performs one validated replay per case, subtracting measured
oracle time from wall time. It includes construction and production checks,
has no explicit warmup, and is labeled `one_run_wall_minus_oracle` in JSON.
It avoids repeating costly large evolutions solely for instrumentation.
The ordinary benchmark retains its warmed-repeat protocol and separate
untimed validation. Process peak RSS includes validation and is cumulative;
`state_storage_mib` measures the output arrays separately.
Compression rows also report an independent global relative Hilbert–Schmidt
error against the product-mixture input, using small product overlaps and the
canonical-center norm. A dense small-system test validates that diagnostic;
it is separate from protected-region accuracy and is not a fidelity estimate.

## Validation and audit

The existing upstream audit and installed environment were reused unchanged;
no dependency upgrade or global driver registration was made. **Adopt:** native
einsum/GEMM and Quimb exact QR/protected SVD. **Defer:** native PEPO, GPU scaling,
fully compiled adaptive replay and global constrained-fit optimality.

Focused tests include independent strided NumPy/Torch/JAX panels, zero
environments, qutrit matrix units, and 64-site extreme scales whose full norm
would underflow/overflow. Large public-compressor checks compare every 6×6 and
8×8 plaquette with a product-mixture oracle. Discarded-norm assertions use a
size-dependent physical scale because trace-one density Hilbert–Schmidt norms
shrink exponentially with chain length. Replay checks include mapped physical
regions, reversed ket/bra gates, temporary budgets, rollback/fallback and Torch
derivatives. Final test counts and measurement results are recorded in the
[handoff](../../../history/2026-11-16-mpo-dmt-replay.md).
