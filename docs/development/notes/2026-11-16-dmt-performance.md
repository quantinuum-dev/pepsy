# 1D and lattice DMT: API and CPU bottlenecks

**Measured:** faster gauge movement and regional scaling reduced compression
time by about 1.5–1.6× in the workloads below. Small persistent evolution cases
improved little or modestly; their protected splits remain expensive.

## Workloads and measurement

Use the [benchmark script](../../../examples/pauli_dmt_benchmark.py) and
[workflow guide](../../api/optimizers/pauli_dmt.md). Inputs are seeded positive
product mixtures, Bloch-vector length 0.5, seed 321. Lattice regions are open
plaquettes with `snake-row-major` ordering. Both versions use `cutoff="auto"`,
normal preservation checks and no optional constraint diagnostics.

Baseline kernel: `b2df2dba:src/pepsy/operators/_pauli_dmt.py`; optimized kernel:
the current working tree. Loaded both modules in one process, warmed each,
then alternated order across seven timed runs of each version. Medians below
use one CPU BLAS thread. Input construction and validation are untimed.
Evolution includes workspace initialization and two timesteps of size 0.03;
the 2D case evolves adjacent **chain** bonds with lattice-region protection.

| Case | Dtype | Sites / rank / cap | Before | After | Ratio |
| --- | --- | --- | ---: | ---: | ---: |
| 1D compression | float64 | 64 / 48 / 16 | 101.80 ms | 63.47 ms | 1.60× |
| 2D compression | float64 | 5×5 / 96 / 72 | 366.24 ms | 229.72 ms | 1.59× |
| 1D evolution | float64 | 16 / 16 / 12 | 36.48 ms | 36.43 ms | 1.00× |
| 2D-region evolution | float64 | 3×3 / 32 / 40 | 81.21 ms | 74.97 ms | 1.08× |
| 1D compression | complex128 | 64 / 48 / 16 | 227.87 ms | 149.29 ms | 1.53× |
| 2D compression | complex128 | 5×5 / 96 / 72 | 753.28 ms | 462.53 ms | 1.63× |
| 1D evolution | complex128 | 16 / 16 / 12 | 74.67 ms | 74.00 ms | 1.01× |
| 2D-region evolution | complex128 | 3×3 / 32 / 40 | 190.04 ms | 175.03 ms | 1.09× |

These are limited CPU measurements, not GPU or general scaling claims.
Discarded residual norms are nonzero and final bonds reach their caps.
The script reports them in Pauli-coefficient units, which can be exponentially
smaller than physical Hilbert–Schmidt units on long normalized chains.

## What changed and what remains costly

- **Adopt:** reshape gauge/tensor contractions into native matrix products.
  Unoptimized general einsums dominated preparation for the larger input
  ranks: centre movement accounted for roughly 36% of the profiled 1D
  compression and 22% of the 2D compression before this change.
- **Adopt:** maximum-entry scaling for intermediate regional panels and
  identity environments. Only the final output panel needs Frobenius
  normalization. This retains log scaling, final physical reference norms,
  all regions and before/after checks. The 5×5 profile's Euclidean norm
  calls fell from 11,714 to 4,314 across the complete operation.
- Regional verification remains the main 5×5 compression cost: about 57%
  of the optimized instrumented profile. The full `4**k` panel and MPO span
  still matter; no verification panel is cached across tensor writes.
- Protected support/basis factorizations and residual SVDs dominate the
  sampled persistent 1D evolution. Changing their rank policy or removing
  checks is **deferred**; these timings do not justify numerical changes.

## Accuracy evidence and limits

Independent product-mixture oracles check every protected compression region.
Maximum Frobenius errors were `2.31e-14` (1D float64), `9.21e-15` (1D
complex128) and `6.02e-15` (2D). Before/after compression marginals agreed
exactly in three cases and to `2.72e-15` in the 2D complex case. All eight
comparisons retained the same final bond dimensions; trace errors were below
`6.1e-14`.

**Evolution is not bitwise invariant.** Changing contraction summation order
can alter subsequent finite-bond truncations. Maximum before/after final
three-site/plaquette differences in the two-step evolution cases ranged from
`1.48e-7` to `2.38e-6`. Per-gate protected-target checks still pass. Separate
dense Trotter-circuit probes with complex128 inputs gave relative global
operator errors:

| Probe | Baseline | Optimized |
| --- | ---: | ---: |
| 8-site chain, rank 16, cap 12 | 0.17557 | 0.17549 |
| 3×3 regions, rank 32, cap 40 | 0.059390 | 0.059345 |

These probes showed comparable compression accuracy, not an improvement
guarantee. They illustrate why marginal preservation and a small trace error
do not certify complete evolution. Check bond, timestep and precision
convergence for an application.

New deterministic regressions compare arbitrary one- to four-site region
panels with independent mixture coefficients after alternating `1e10`/`1e-10`
virtual gauges, complex phases, and global scales `1e-25`, 1 and `1e25`.
All four floating/complex dtypes and NumPy, Torch and JAX CPU are covered.

## Environment and audit

Python 3.12.14, NumPy 2.5.2, Quimb 1.15.0, Autoray 0.8.4, Cotengra
0.8.3.dev8+g8954240f2, Cotengrust 0.2.1, Symmray 0.4.1.dev15+g0374aaa3c;
macOS arm64. Reconfirmed versions and the installed public `array_split`
signature. Reused the unchanged environment's official upstream audit from
the [earlier DMT review](2026-11-16-dmt-gradient-review.md). Native reshape,
transpose and matrix products require no new dispatch registrations or
dependencies. Native Symmray remains unsupported by DMT.

The [handoff](../../../history/2026-11-16-dmt-api-performance.md) owns final
validation results and commit status.
