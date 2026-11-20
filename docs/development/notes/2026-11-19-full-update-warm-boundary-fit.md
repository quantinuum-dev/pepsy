# Full-update boundary FIT planning and warm reuse

`PepsOptimizer` forwards `boundary_kwargs['fit_contraction_opt']` to the
boundary FIT. `CompBdy` already accepts this option and includes its resolved
planner in the cut cache key. This adopts the existing Quimb/Cotengra planner
interface. The Eli runner selects `auto-hq` for one-site DMRG FIT by default
and offers `--full-update-fit-contraction-opt greedy` for a matched control.
The outer norm contraction uses `greedy` in Eli. The existing-MPS initializer
and three-sweep setting were used for the GPU comparison.

## RTX A5000 audit

Torch CUDA complex128, PEPO D=6 on all 176 active bonds, boundary χ=120,
three one-site FIT passes per rebuilt cut, and a fresh cut cache. The starting
operator was a real 10×10 Eli PEPO prepared at Δt=0.05. The four diagnostic
gates were lower-leg RZZ at Δt=0.025. For the first gate the saved MPS guesses
were expanded from cap 56 to 120 with zero padding and canonicalization.
Later gates kept the live boundary cache and fitted MPSs.

| Diagnostic gate | Cache hits | Fits | Wall time |
| --- | ---: | ---: | ---: |
| First gate in strip | 0 | 9 | 90.38 s |
| Next gate in same strip | 9 | 0 | 10.90 s |
| Neighboring strip | 8 | 1 | 25.12 s |
| Return to original strip | 8 | 1 | 24.72 s |

The returned strip refit used the MPS fitted for that cut on the first gate,
starting directly at stored rank 120. Its 11.63 s FIT used two sweep
environment reuses and ran all three passes. Synchronized phase times were
4.85 s for 30 effective-tensor contractions, 4.82 s for 27 moving environment
contractions, 1.70 s for fixed-side preparation, and 0.26 s for canonicalization.
The four PEPO pair ALS solves took 0.14 s total. Even with nine boundary
cache hits, the same-strip update still assembled a new norm metric.

The earlier single-gate planner comparison took 191.63 s with `greedy` and
91.16 s with `auto-hq` inside FIT. Their normalized norm matrices differed by
6.24e-16 relative norm. Both studies used finite-bond diagnostic updates.
The updated Eli runner also completed the 640-action RZZ-guided Δt=0.025
prefix in 8.62 s. It recorded `auto-hq`, 130 cut cache hits, and 14 rebuilds.

## Selected D=6 timing pilot

For the Eli GPU pilot, select boundary χ=86 and two one-site DMRG sweeps with
the existing-MPS initializer and `auto-hq`. Replaying the same four diagnostic
gates took 57.17 s at χ=86 with two sweeps versus 151.12 s at χ=120 with three
sweeps. Both runs had the same cut hit and rebuild pattern. The warm refit
took 3.92 s versus 11.63 s. Peak allocated CUDA memory fell from 5.23 GiB to
2.72 GiB. The 640-action Δt=0.025 RZZ prefix took 5.67 s at χ=86.
The data compare finite-bond runtimes. Full-evolution accuracy is pending.

## Validation

Installed Torch 2.14.1+cu126, Quimb 1.15.0, Autoray 0.8.4, Cotengra
0.8.4.dev8+g646239731, Cotengrust 0.2.1, and Symmray 0.5.1.dev2+g20035beb7.
Checked the installed `CompBdy` and `FIT.run_eff` interfaces and the
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/).
Consulted the Symmray repository when its array documentation was unavailable.

The focused package policy/cache/API check passed 69 tests. The Eli allocation
and full-update check passed 51 tests with one skip, and the D=6, χ=120 CLI
preflight selected `auto-hq`. Ruff and whitespace checks passed. The broader
PEPS/PEPO and FIT-hotpath checks reproduced their 11 and 16 failures,
respectively, on an isolated HEAD checkout with the same environment.
