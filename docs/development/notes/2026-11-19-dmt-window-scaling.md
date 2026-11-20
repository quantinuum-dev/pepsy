# Larger DMT windows, bonds and chains

**Measured:** four-/five-site constraints survive genuine compression to
roundoff. Wider protection alone does not ensure better dynamics at fixed
chi. An existing feasible-seed refinement improves the best tested pure-state
case, and indexing crossing regions speeds up long-chain initialization.

[Summary figure](2026-11-19-dmt-window-scaling.png),
[SVG](2026-11-19-dmt-window-scaling.svg),
[timing table](2026-11-19-dmt-window-scaling-timings.csv),
[accuracy samples](2026-11-19-dmt-window-scaling-accuracy.csv), and
[complete results/profiles](2026-11-19-dmt-window-scaling-summary.json)
retain successful runs and the native-SVD failure evidence.

## Protocols

- **Compression:** Torch CPU complex128; L=64,128,256; chi=32,64,128;
  contiguous protection widths 3,4,5. Seed-321 positive product mixtures
  have Bloch length 0.2 and input bond `chi + max(8, chi//4)`. Each result
  checks every protected reduced density against the physical product-mixture
  oracle, outside timing. One warmup and three timed fresh compressions;
  one CPU thread, zero cutoff, strict constraints, warning policy.
- **Cross-checks:** ordinary MPO tensors at L=64/128, chi=64 and all widths;
  NumPy complex128 ordinary/Pauli MPOs at L=64, chi=64 and widths 4/5.
- **Exact dynamics:** L=12 near-Y figure state, tilted Ising Hamiltonian,
  20 steps of dt=0.15 in Pauli-ZZ units. The independent statevector uses
  the identical gate schedule, isolating compression from Trotter error.
  All widths use the same three-/five-site RDM accuracy measures. Every gate
  in the first and final steps is audited against its current exact gated
  approximate target. Readout and audits are outside evolution timing.
- **Long-chain evolution:** positive mixtures are first compressed to the
  requested cap. Two tilted-Ising sweep steps of dt=0.1 then operate on
  occupied bonds, with eight independently audited gates per configuration.
  These audits verify each current target, not an exact full trajectory.

Full-region verification is disabled for timing, with the protected
construction and bond checks retained. Independent oracle checks are untimed;
the new focused tests also exercise enabled full-region checks. All reported
successful compressions reduce ranks and all evolution cases include actual
loss. These are CPU measurements, not GPU claims.

## Larger windows: preservation versus accuracy

All **27 Pauli compression cases** complete, including L=256, chi=128,
five-site windows. Maximum protected-window Frobenius error is 3.28e-14.
Six ordinary MPO and four NumPy cross-check cases also complete with errors
near roundoff. Larger-window compression therefore preserves the requested
data; the nontrivial issue is subsequent trajectory accuracy.

Including paired timing/oracle runs at L=512, the largest measured window
error is 1.27e-12 (chi=64 with the supported fallback policy). The 3.28e-14
maximum above refers specifically to the 27-point L<=256 Pauli grid.

All **13 long-chain evolution cases** complete, through L=128, chi=128,
five-site protection. Across their eight audited gates per case, maximum
current-target window defect is 2.79e-15 and maximum final trace defect is
6.89e-14. Each case has two full sweep steps, not a long-time accuracy claim.
Representative five-site evolution timings, excluding preparation/readout:

| L | chi | CPU linalg policy | Median seconds per sweep step |
| --- | ---: | --- | ---: |
| 64 | 64 | Native Torch | 7.46 |
| 128 | 64 | Native Torch | 12.41 |
| 64 | 128 | Regularized policy with GESVD fallback | 28.53 |
| 128 | 128 | Regularized policy with GESVD fallback | 90.74 |

These gate timings operate on occupied bonds and different states/policies;
they are separate from the isolated old/new planning comparison below.

For generic full protected ranks under the default smaller-half cover:

| Width | Interior protected rank sum p+q | Residual budget at chi=32 | At chi=64 |
| --- | ---: | ---: | ---: |
| 3 | 8 | 24 | 56 |
| 4 | 20 | 12 | 44 |
| 5 | 32 | 0 | 32 |

These are sufficient construction budgets; actual state-dependent ranks
can be smaller, and failure of the conservative budget is not proof of
mathematical infeasibility. Protecting more can remove space for longer
correlations needed by future evolution.

Final five-site RDM Frobenius RMSE at t=3 against the exact gated statevector:

Here “paper” names Pepsy's conservative residual-SVD rule, extended to the
requested regions; these four-/five-site numbers are new measurements, not
values reported in the original paper.

| chi | Paper / 3 sites | Paper / 4 sites | Paper / 5 sites | Refined seeds / 5 sites |
| --- | ---: | ---: | ---: | ---: |
| 32 | 0.137672 | 0.230430 | 0.396739 | 0.087111 |
| 64 | 0.0336755 | 0.0418272 | 0.0341444 | 0.00901848 |
| 128 | 0.00708738 | 0.00667292 | 0.332031 | 0.00161952 |

“Refined seeds” means the existing `truncation="optimized"` with
`max_refinement_sweeps=0`: guarded feasible paper/Schur SVD seeds, with no
alternating updates. It remains an explicit choice; the default is unchanged.
At chi=128 its five-site RDM error is **4.38 times smaller** than standard
three-site DMT at the same cap, and its three-site RMSE is 6.76e-5 versus
9.97e-4. Audited current-target preservation remains near roundoff.

The plain five-site chi=128 failure is retained in the results: trajectory
accuracy is not monotonic with chi or protection width. Refined four-site
cases also do not improve uniformly. None of these methods enforces
positivity; the best final five-site minimum eigenvalue is still about
-2.92e-4. No default truncation or positivity repair was silently installed.

This short small-chain experiment favors MPS: chi=64 already represents
every 12-site pure state, and its error is at roundoff. It is not a claim
that DMT beats MPS for this regime, nor an exact large-L accuracy result.
The separate [L=64 paper comparison](2026-11-19-dmt-l64-paper-comparison.md)
remains an energy benchmark.

## Profiling and implemented improvement

**Adopt:** `_region_patterns` now computes the maximal region inventory once
and indexes each region onto the bonds it crosses. Each cut sees only those
crossing regions. Protected strings, their ordering, cover tie-breaking,
truncation mathematics and verification policy are unchanged.

The old implementation recomputed full-inventory subset tests at every cut.
For fixed-width windows this led to approximately cubic planning work in L.
The new index removes that repeated full-inventory scan. Gapped regions
still cost their span, and the initial maximal-region pass remains.

Five-site planning speedup, alternating old/new order over three repeats:

| L | Planner speedup |
| --- | ---: |
| 128 | 14.9x |
| 256 | 42.8x |
| 512 | 139x |

These are planning-only ratios. Complete Torch complex128 compression at
chi=32 gives the smaller, relevant end-to-end gain:

| L | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| 64 | 0.1983 s | 0.1916 s | 1.04x |
| 256 | 1.0337 s | 0.7382 s | 1.40x |
| 512 | 3.5839 s | 1.5161 s | 2.36x |

At L=512, chi=64, the supported fallback policy gives 6.879 s before versus
4.894 s after (1.41x). Both sides use the same linalg policy. The native
policy fails on this input and is recorded separately rather than omitted.

Each timed operation uses a fresh input, preceded by an untimed warmup.
Old/new order alternates; both use the same current kernel except for the
region-plan builder. Every result receives an independent marginal oracle.
Exact plan equality and bitwise compressed-array equality are regression
tested. Background jobs and hardware affect absolute CPU times.

At larger chi, linear algebra dominates. L=256, chi=128, five-site compression
takes a median 8.69 s; its instrumented profile attributes about 50% to native
QR and 23% to native SVD. The planning change is not a solution to those
matrix-factorization costs. The full timing/profile samples are retained.

## CPU SVD robustness

Native Torch SVD fails on an attempted L=64, chi=128 mixture initialization,
and on the L=512, chi=64 timing input. The saved failing 136-by-136 complex128
matrix is finite, with norm 0.379856. SciPy GESVD reconstructs it with relative
error 2.95e-15. The failure is not a detected NaN input.

The benchmark exposes the existing `TorchLinalgConfig` policy. The optional
`--stabilized-linalg` uses native forward SVD with GESVD fallback and native
QR rank policy; it retains Torch storage and complex128. This addresses SVD
convergence failures, not DMT positivity or universal trajectory accuracy.
An always-GESVD full trajectory was stopped for cost; it is not presented as
a completed comparison. No installed dependencies or backend defaults changed.

## Reproduce

```bash
python examples/dmt_window_scaling.py --case compression \
  --lengths 64 128 256 --chis 32 64 128 --widths 3 4 5 --profile
python examples/dmt_window_scaling.py --case dynamics --lengths 12 \
  --chis 32 64 128 --widths 3 4 5 --methods pauli mps
python examples/dmt_window_scaling.py --case dynamics --lengths 12 \
  --chis 32 64 --widths 3 4 5 --truncation optimized
python examples/dmt_window_scaling.py --case large-dynamics --lengths 128 \
  --chis 128 --widths 5 --steps 2 --dt .1 --stabilized-linalg
```

Each run writes its configuration, raw timing samples, errors and profiles
to JSON. `--cpu-svd scipy_gesvd` selects that driver for every forward SVD;
the fallback option avoids paying that cost on every successful native call.

## Validation and compatibility

The DMT domain selection has **399 passed, 1 skipped, 1 expected failure**.
The final new focused file has **43 passed**. The domain selection predates
the last example-only storage assertions and extra planning parameterizations;
the subsequent focused selection includes them. Ruff and whitespace checks
pass. No full-suite or GPU validation is claimed.

New checks cover all four-/five-site densities after lossy compression in
both MPO bases and NumPy/Torch, independently embedded gate targets, exact
small-system references, real rank reduction, and identical legacy/indexed
plans for contiguous/gapped/nested inventories, several physical dimensions,
identity labels and all cover selectors.

Upstream audit: Quimb changelog, Autoray repository, Cotengra documentation
and changelog, and Symmray repository were checked; Symmray's array-doc URL
was unavailable. Installed versions remain NumPy 2.5.2, SciPy 1.18.0,
Quimb 1.15.0, Autoray 0.8.4, Cotengra 0.8.3.dev8+g8954240f2,
Cotengrust 0.2.1, Symmray 0.4.1.dev15+g0374aaa3c and Torch 2.9.1.
The existing public `array_split` signature and native dtype route are
unchanged. **Defer:** approximate decompositions and mathematical changes
to address positivity. No shims or dependency upgrades were introduced.
