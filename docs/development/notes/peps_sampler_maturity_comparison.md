# 2026-09-26 — PEPS sampler maturity and Verstraete comparison

- Scope: review the current sampler's maturity and efficiency and compare it
  with Frank Verstraete's direct/perfect-sampling work.
- Branch / baseline: `develop` / `a13031b`, initially clean and synchronized.
- Status: audit only. No implementation, dependency, source PEPS, production
  job, or shared environment was changed. This report and its session handoff are uncommitted;
  nothing from this assessment was staged or published.

A subsequent authorized implementation addresses the overflow described below;
see the [boundary-scaling follow-up](peps_sampler_boundary_scaling.md). The
measurements here record the pre-fix audit.

## Assessment

The conditioned-boundary algorithm and backend API are well tested on small
finite dense OBC systems. The implementation remains an eager research sampler.
It has neither a native batch-axis contraction engine nor demonstrated large-D
GPU throughput. A newly reproduced complex64 truncation overflow prevents an
unqualified maturity claim for the Quimb future-environment route.

### Newly confirmed numerical issue

The installed Quimb `rsum2` truncation squares singular values without scaling
before the sum. On NumPy and Torch complex64, this public operation keeps two
singular values at scale 1 but only one at scale 1e20:

```python
import numpy as np
import quimb.tensor as qtn

for scale in (1.0, 1e20):
    tensor = qtn.Tensor(
        np.diag([scale, 0.5 * scale]).astype('complex64'), inds=('a', 'b')
    )
    u, s, vh = tensor.split(
        ['a'], get='arrays', absorb=None, max_bond=2,
        cutoff=1e-6, cutoff_mode='rsum2',
    )
    print(scale, len(s))
```

Expected relative discarded weight for removing the smaller singular value is
0.2, so rank two is required at both scales. Complex128 retains rank two at
both scales. The installed generic and NumPy/Numba cutoff paths both directly
form squared singular values; this is independent of Pepsy.

It affects an actual 5x6 D=2 random PEPS (seed 101), generated once in complex128
and cast to Torch complex64, with chi=32, chi_prime=16, greedy contractions,
32 samples, seed 800:

| Variant | ESS / N | Future bonds for rows 0,1 |
| --- | ---: | --- |
| Quimb future, complex64, auto cutoff | 85.4099% | 1, 1 |
| Same state, complex128, auto cutoff | 100.0000% | ample ranks |
| Same state, complex64, cutoff=0 | 100.0000% | ample ranks |
| Complex64, auto, privately rescaled tensor copies | 99.99993% | 14, 15 |
| DMRG future, complex64, auto | 100.0000% | 16, 16 |

The rescaled copies differ only by an overall scalar, so their normalized
Born distribution is unchanged. Queried proposal log probabilities nevertheless
change by as much as approximately 0.713 in the three inspected configurations.
NumPy complex64 also exhibits the scale sensitivity. The unscaled Torch
Quimb case has a maximum reported rho Hermiticity defect of 0.4293.
This is a concrete numerical problem, distinct from legitimate finite-chi
proposal error. No workaround was installed and no original tensors were
normalized. Fixing private compression scale handling while preserving cutoff
semantics and original amplitudes is the first recommended follow-up.
The successful DMRG probe is a bounded case, not a universal immunity claim.

### Existing limits reconfirmed

- Default `row_cache_max_bytes=0` uses a current-row contraction at each site.
  The future environments are cached; within-row suffix environments are not
  reused by this default path. The optional dense transfer cache is guarded
  and can be expensive; enabling it is not a general optimization.
- Prefix groups share early calculations, then split. Distinct groups still
  call contractions and compression in Python loops. Probability validation
  and draws are grouped per site; this is not batched boundary linear algebra.
- `_projected_amplitude` fully contracts the original projected ket for each
  distinct final configuration. This is valuable for importance correction,
  but its cost is not capped by chi or chi_prime.
- Amplitude output is scaled only after its raw contraction. Log-probability
  bookkeeping does not protect every intermediate or amplitude from overflow.
- Small chi_prime can remove nonzero target support: the prior 2x2 Bell-row
  probe still gives q=(1,0) at chi_prime=1 and q=(1/2,1/2) at chi_prime=2.
  Importance weighting cannot recover the missing branch.
- Result logs are available, but an ESS/weight-quality summary is not exposed
  directly. Finite pilot ESS does not certify target support or convergence.

## Comparison with the literature and authors' code

Relevant primary sources:

- [Vieijra, Haegeman, Verstraete, Vanderstraeten, direct PEPS sampling](https://arxiv.org/abs/2109.07356), sections III.B, V.D and appendix B.
- [Authors' Julia reference](https://github.com/tvieijra/DirectSamplingPEPS.jl),
  inspected at commit `797ec041807ab13835747cf344e66b6ef7d9dde8`.
- [Ferris and Vidal, perfect sampling with unitary tensor networks](https://arxiv.org/abs/1201.3974).

The Verstraete paper samples an auxiliary proposal with importance correction.
Exact perfect sampling of unitary networks is the related Ferris/Vidal result.
Pepsy's chi corresponds to the paper's chi_m and chi_prime to chi_s.

| Detail | Authors' inspected implementation | Current Pepsy |
| --- | --- | --- |
| Future marginals | Cached opposite to sampling | Same distinction |
| Conditioned boundary | Single layer | Same distinction |
| Compression timing | Absorb and compress unmeasured row, then sample it | Sample row, then absorb/compress its fixed ket |
| Row environment | Array edge environments reused across the row | Full-center default; optional dense transfer cache |
| Multiple shots | Julia threaded sample loops | Python prefix groups, grouped draws |
| Backend | Concrete Julia Array storage | NumPy/Torch/JAX inference and converter |
| Weights/amplitudes | Truncated boundary and normalization factors | Separate original-ket amplitude and explicit log proposal |

The truncation order means finite-cutoff proposals need not coincide. Changing
Pepsy to pre-sampling compression would change the user's selected algorithm.
The authors' reference was read, not installed or benchmarked: no speed ratio
between implementations is claimed.

## Fresh measurements

Environment unchanged: NumPy 2.5.2; Quimb 1.15.1.dev66+ge927f06e1;
Autoray 0.11.1.dev3+g1b476b305; Cotengra 0.8.3.dev7+g1d7fd333f;
Cotengrust 0.2.1; Symmray 0.4.1.dev7+g83fb22865;
Torch 2.6.0+cu124; JAX 0.10.2. Inspected public contraction, compression,
MPO application, namespace, and contraction-tree signatures.

Random seed 101, chi=32, chi_prime=16, both cutoffs auto, default cache disabled,
greedy planning, fit_n_iter=2. CPU, one BLAS/OpenMP/Torch thread, Torch inference
mode. One eight-shot warmup and two measured calls per batch size (32 and 128).
Other production CPU/GPU jobs continued; GPU was at 99% utilization and was not
benchmarked. These are shared-machine timings, not controlled backend rankings.
Initial dtype timing cases use their own dtype-specific random construction;
the separate precision diagnosis above uses exactly the same source state.

| Shape, D | Backend / dtype | Future | 128-shot median | Shots/s | ESS/N |
| --- | --- | --- | ---: | ---: | ---: |
| 4x4, 4 | NumPy complex128 | Quimb | 1.787 s | 71.64 | 99.88% |
| 4x4, 4 | Torch complex64 | Quimb | 2.312 s | 55.37 | 99.84% |
| 4x4, 4 | Torch complex128 | Quimb | 2.397 s | 53.41 | 99.86% |
| 4x4, 4 | NumPy complex128 | DMRG | 1.970 s | 64.96 | 99.88% |
| 5x6, 2 | NumPy complex128 | Quimb | 4.450 s | 28.76 | 100.00% |
| 5x6, 2 | Torch complex64 | Quimb | 5.968 s | 21.45 | 87.69% |
| 5x6, 2 | NumPy complex128 | DMRG | 4.458 s | 28.72 | 100.00% |

All measured sampled/query log probabilities agreed within 1e-3 for complex64
and 1e-8 for complex128; original arrays were unchanged. This agreement checks
proposal bookkeeping, not agreement of the truncated proposal with Born weights.

On 4x4 D=4 NumPy, a separate 32-shot comparison measured 0.841 s serial versus
0.582 s grouped (about 1.44x). The 128-shot stage profile attributed 32.8% to
local rho contractions, 34.9% to conditioned-boundary updates, 10.9% to center
construction, 5.2% to original amplitudes, and 0.4% to grouped draws. A separate
32-shot cProfile recorded 1,200 tensor-network copies. Profile instrumentation
is separate from throughput measurements; overlapping cumulative times are
not summed.

Shape-only greedy planning for full projected-amplitude contraction estimated
largest complex128 intermediates of 256 MiB for 5x6 D=16 and 4 GiB for 10x10 D=4.
These are particular path estimates, not lower bounds or total peak-memory
predictions. No large tensors were allocated for those estimates.

## Validation and priorities

- Fresh complete PEPS sampler suite: 132 passed in 203.32 seconds, with 25
  existing Quimb mode/method deprecation warnings. NumPy/Torch/JAX CPU paths
  were exercised; JAX used two CPU devices in this isolated domain process.
- New 3x3 D=2 enumeration: all 512 configurations for each of four settings.
  Quimb and DMRG with chi=32, chi_prime=16 agree with exact Born probabilities
  within 2.43e-17 maximum absolute error. Both proposal sums agree with one
  within 4.45e-16.
- At chi=2, chi_prime=1, both normalized proposals remained evaluable for all
  configurations; total variation from Born was 0.1555 (Quimb), 0.1674 (DMRG),
  with population ESS fractions 76.30% and 76.85%. No target support was lost
  in this random example; the separate Bell-row example does lose support.
- The eight independent 2x2 SVD probes confirm the scale bug above.
- No entire repository suite was rerun for this audit.

Recommended order, all proposals rather than implemented changes:

1. **Compatibility shim:** narrowly stabilize affected complex64 compression
   scales using public APIs, preserving requested cutoff semantics and source
   norms, with scale-invariance regressions. Do not edit installed Quimb.
2. **Adopt:** explicit stable ESS/max-weight diagnostics and convergence checks.
3. **Prototype:** factored within-row edge environments that preserve the current
   proposal and avoid both full-row recontractions and huge dense transfers.
4. **Prototype:** an explicit, separately controlled approximate amplitude mode
   for large systems, keeping original-ket contraction as the reference.
5. **Defer:** native shape-bucketed batch contractions and QR/SVD until the
   previous paths have accuracy and effective-samples-per-second evidence.

Official Quimb changelog, Autoray repository, Cotengra docs/changelog, and
Symmray repository were checked. Symmray's abelian-array documentation failed
in the browser; no Symmray behavior is changed. Latest upstream documentation
is not evidence of installed capabilities. Autoray namespaces/device dispatch
are already adopted; they do not remove Python prefix/control loops.

Temporary evidence: `/tmp/pepsy_sampler_maturity_audit.py/.jsonl/.log`,
`/tmp/pepsy_sampler_maturity.prof`, `/tmp/pepsy_sampler_oracle_audit.py/.log`,
`/tmp/pepsy_sampler_precision_probe.py/.log`, `/tmp/pepsy_sampler_scale_probe.py/.log`,
and `/tmp/pepsy_sampler_maturity_tests.log`.
