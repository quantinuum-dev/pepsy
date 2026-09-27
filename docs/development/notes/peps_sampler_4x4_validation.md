# 2026-09-26 — Real 4×4 D=4 OBC sampling validation

## State and independent reference

The user requested a full end-to-end test after the
[cache audit](peps_sampler_cache_audit.md). The test uses an entangled random
qubit PEPS, seed 271, with 16 sites, 24 open-boundary virtual bonds, and dimension
4 on every virtual bond. It is not a product-state or reduced-D fixture.
Both real and imaginary tensor components are present.

The original single-layer ket is contracted to all **65,536 complex amplitudes**
in the sampler's increasing-y/increasing-x order. This oracle does not use any
sampler boundary environments. Its exact squared norm is
`8.314067465028782e23`. The greedy dense plan has 1,048,576 elements in its largest
intermediate (16 MiB at complex128) and took about 0.07 seconds on this machine.
Complex64 tests cast the same tensors; their oracle contracts those rounded
inputs in complex128, separating input precision from contraction error.

The durable reproduction is
[`tests/test_peps_sampler_4x4.py`](https://github.com/quantinuum-dev/pepsy/blob/develop/tests/test_peps_sampler_4x4.py).
It is marked `integration`, `slow`, and `sampling`, outside the default smoke
profile. No sampler algorithm or public default changed in this task.

## Main sampling results

**8,192 main draws**, in 128-shot prefix batches with fixed seeds 290 onward.
Additional draws check replay, serial sampling, backend parity, and cache
lifecycle. All runs use CPU with one BLAS/OpenMP thread. Torch uses
`inference_mode`; no GPU benchmark or production-job change was made.

| Backend / future engine | χ / χ′ | dtype | Draws | ESS / N | Estimated norm / exact norm |
| --- | --- | --- | ---: | ---: | ---: |
| NumPy / Quimb, cutoff=0 | 1024 / 16 | complex128 | 4096 | 100.000% | 1.0000000000000013 |
| Torch / Quimb, auto cutoff | 32 / 16 | complex64 | 2048 | 99.834% | 1.000446 ± 0.000901 |
| NumPy / DMRG, auto cutoff | 32 / 16 | complex128 | 1024 | 99.857% | 0.998306 ± 0.001181 |
| NumPy / Quimb, absolute rho repair | 8 / 4 | complex128 | 1024 | 69.582% | 0.993387 ± 0.020535 |

The ± values are one estimated standard error. The first three cases use the
default Hermitian/diagonal policy; the last explicitly selects
`rho_positivity="absolute"`. Automatic cutoff resolves to 1e-6 for complex64
and 1e-12 for complex128; `cutoff_mode="auto"` resolves to relative squared
singular-value weight. Ket compression is Quimb in all four cases; the DMRG
row changes the future-environment provider, not the ket compressor.

Weights are computed against the independently known target norm:
`w_normalized = p_Born(S) / q(S)`. Norm and observable comparisons use ordinary
means of these weighted quantities, not a fitted self-normalization that could
hide a norm error. ESS is `(sum(w))**2 / sum(w**2)`.

For every main draw, the returned **complex amplitude**, including its phase,
is checked against the full-state oracle. Maximum relative amplitude errors:
6.04e-15 (exact-limit NumPy), 1.75e-6 (Torch complex64), 4.92e-15 (DMRG),
and 7.38e-15 (small caps). Returned log weights agree with the amplitude/proposal
formula. Replaying public likelihood queries agrees with sampled log proposals
to 5.33e-15 in complex128 and 2.20e-6 in complex64.

### Observables and distribution checks

The tests check all 16 local Z values, all 24 nearest-neighbor ZZ values,
checkerboard Z2 imbalance, and all 16 outcomes of the first-row joint marginal.
The statistical assertions allow six estimated standard errors plus a small
roundoff floor. Observed errors were smaller: the largest observable discrepancy
was 2.79 standard errors and the largest first-row-bin discrepancy was 2.94.

The exact checkerboard imbalance is **0.0211482593**. Sample estimates:

| Case | Weighted imbalance ± one standard error |
| --- | ---: |
| Exact-limit NumPy | 0.0208435 ± 0.0037799 |
| Torch χ=32 | 0.0197638 ± 0.0054758 |
| DMRG χ=32 | 0.0283377 ± 0.0076101 |
| χ=8, χ′=4 | 0.0367499 ± 0.0102339 |

The exact-limit case also passes unweighted observable checks. At finite caps,
q is generally different from the Born distribution: maximum observed log-q
versus log-Born deviations were 0.134 (Torch), 0.116 (DMRG), and 4.34 (small
caps). Valid importance estimates do not imply that raw finite-cap draws are
exact Born samples.

### Exact-limit and rare-configuration checks

With χ=1024, χ′=16, and cutoff=0, **all 4096 drawn log probabilities** agree with
the dense Born oracle to 3.55e-14; the weight variation is at roundoff scale.
A separate set of 68 configurations includes the 16 least likely, 16 most
likely, uniformly chosen configurations, and all-zero/all-one/alternating
patterns. Minimum Born probability is 1.38693e-12; maximum log-probability
error over this set is 1.40e-11.

A useful qualification emerged from the real-size test: χ=256 is sufficient
for the final four-site double-layer boundary rank, but is **not sufficient
for every temporary boundary** used by the installed Quimb sweep. With cutoff
zero it still gave a maximum log-probability deviation of 0.00392 over eight
fixed configurations. Raising χ to 1024 reduced that to 2.84e-14; the final
stored boundaries still have maximum bond 256. Installed Quimb
`TensorNetwork2D._contract_boundary_core` compresses inside the layer loop,
including between ket and bra contractions. Final stored rank alone therefore
does not certify exact sampling. This is a truncation-policy finding, not a
cache invalidation defect. The API guide now states that distinction.

## Caches, boundaries, and precision

- Future MPS objects and all their arrays/indices/tags remain unchanged across
  draws and likelihood queries; a guard forbids any repeated preparation.
- The conditioned single-layer boundary always respects χ′. Native backend,
  dtype, and device signatures are checked on retained boundary arrays.
- Source PEPS tensors, indices, and tags remain unchanged; seeded serial and
  grouped calls are reproducible.
- Positive-χ dense row-transfer estimates exceed the 64 MiB budget for this
  problem. The public route correctly falls back before allocation. These main
  draws therefore reuse future MPSs and use reference within-row contractions.
- A separate **same 4×4, D=4 state** case, χ=0 and χ′=2, safely enables actual
  dense row transfers. Four 4-shot batches agree with the reference route to
  1e-11, use the initial-row cache once, and return exact physical amplitudes.
  A physical filter plus `refresh()` invalidates that cache and matches a fresh
  sampler. This identity-future case checks cache mechanics, not Born accuracy.
- Another physical-filter test verifies positive-χ future-cache replacement
  after `refresh()`, equality with a freshly constructed sampler, preserved
  old-cache snapshots, and correct filtered-state amplitudes.
- Raw finite-cap rho can remain non-Hermitian: the DMRG run reached a defect
  of 0.0855. Conditionals use the implemented Hermitian part. The absolute
  repair case encountered only roundoff-scale spectral correction (2.9e-16),
  so it does not add a new large-negative-eigenvalue stress test; that coverage
  belongs to the earlier deterministic rho tests.

### Backend parity and completed validation

Native proposals at χ=32/χ′=16 were also compared with a NumPy sampler using
the same input precision; complex amplitudes were checked against the dense
oracle. Maximum log-proposal differences:

- torch complex128: 5.33e-15.
- jax complex64: 3.29e-06.
- jax complex128: 3.55e-15.

All **10 new integration checks passed**: the eight initially collected cases
in 347.64 seconds (six existing Quimb mode/method warnings), the subsequently
added rare-configuration check in 9.99 seconds, and the positive-χ refresh check
in 4.50 seconds. Ruff, local documentation links/catalog entries, and diff checks
pass. The earlier 236-test suite was not rerun because runtime implementation
was unchanged by this validation task. See the
[handoff](https://github.com/quantinuum-dev/pepsy/blob/develop/history/2026-09-26-peps-sampler-4x4-validation.md).

## Timing and limits

Main sampling/checking loops took 82.27 s (4096 exact-limit NumPy draws), 39.98 s
(2048 Torch draws), 18.27 s (1024 DMRG draws), and 16.24 s (1024 small-cap draws).
Each includes one additional 128-shot reproducibility replay, diagnostics,
comparison overhead, and normal amplitude evaluation; these are integration
check timings, not isolated sampler benchmarks. Construction took 3.61, 0.19,
0.20, and 0.05 seconds respectively. Other production work was active.

This validates one full random PEPS plus filtered variants, the stated caps,
and CPU backends. It does not prove convergence for every PEPS, enumerate the
finite-cap proposal over all 65,536 configurations, certify target support for
arbitrary truncations, test GPU throughput, or add native Symmray support.
No off-diagonal observable Monte Carlo estimator is tested here, though complex
amplitude phase is checked directly.

Installed versions rechecked: NumPy 2.5.2, Quimb 1.15.1.dev66+ge927f06e1,
Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1,
Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124, JAX 0.10.2. Public dense
contraction signatures and the installed layer-compression loop were inspected.
No upstream behavior was patched. **Adopt:** independent full-state integration
validation and the documented temporary-rank caveat. **Defer:** GPU performance
and broader state ensembles.
