# CuPy full update, strip scheduling, and proposed refinement

Working-tree follow-up on `develop` (`fe11e24`), 2026-10-08. The preceding
[native ALS review](2026-10-08-full-update-native-als.md) validated Torch CPU.
This follow-up implements and exercises CuPy on an NVIDIA RTX A5000 with
CuPy 14.1.1. No dependency installations or upstream source changes.

## Implemented: native CuPy and safe numerical reuse

The full-update wrapper previously rejected CuPy and directly used Torch
tensor operations. It now reuses Autoray and Pepsy's existing backend helpers
for contraction, transpose, QR/SVD, eigensystem projection and reconstruction.
Gate conversion uses `infer_backend_converter_from_sample`; the state must
have one backend, complex dtype and device. Quimb's public ALS and Pepsy's
weighted-QR solver both stay on the input backend. Full-update remains
non-differentiable and does not accept native Symmray arrays.

CuPy's [current-device semantics](https://docs.cupy.dev/en/stable/user_guide/basic.html#current-device)
require native allocation/linear algebra to run on the state's device. The
public run wrapper enters that device context and restores it afterwards.
The installed Quimb/Autoray solver capabilities were inspected and reused
(**adopt**, no compatibility shim). The Quimb tensor-fitting documentation URL
was inaccessible, so the public signature and installed implementation were
used. The upstream audit from the preceding active task is otherwise unchanged.

Torch mutation counters still validate cached source and output arrays.
CuPy instead retains one exact device snapshot per live tracked array and
compares values on the GPU, transferring one boolean. A changed value,
including a write through an alias, advances its revision. Weak references
release snapshots with their source arrays. This has explicit memory and
comparison overhead; it avoids probabilistic checksums or unsafe pointer-only
reuse. Shared boundary-MPS and strip-prefix/suffix caches now recognize CuPy.
Scalar norms are not cached; Cotengra continues to reuse topology plans.

## Implemented: layer-aware strip traversal

`gate_order="column"` / `"row"` sorts contiguous commuting blocks. Arbitrary
fixed two-qubit gates on disjoint supports can now be sorted. Overlapping
gates can share a block only when both are exactly diagonal. Single-site,
trainable or unsupported gates remain barriers; a conflicting two-site gate
starts another block. This is a conservative ordering heuristic, not a global
search for the cheapest contraction schedule.

Within a block, all bonds of the preferred orientation are visited strip by
strip before turning to the other orientation. Previously orientation was
the final sort key, which could interleave rows and columns and repeatedly
clear the active-strip cache. A 4×4 all-bond diagonal layer now has seven strip
transitions: four columns and four rows (or the reverse), with snake order
inside strips. The default remains `gate_order="input"`. Submit one depth per
run when depth boundaries must be enforced independently of commutation.

The exact circuit is preserved. Truncating after each gate can still produce
different approximations when commuting gates are reordered.

## Measured GPU checks

Optional GPU tests compare CuPy with Torch for complex64/complex128,
Quimb/weighted-QR ALS, and fixed/adaptive boundary caps. They guard bulk host
transfers through Autoray and CuPy conversion APIs, assert retained backend,
dtype/device, and check normalized final-state agreement. Additional tests
verify both cache levels against fresh environments along rows and columns,
invalidate mutations inside/outside the active strip, and release snapshots.
Two additional GPU tests exercise `eff` and `dmrg2` boundary compression,
guard host transfers and compare the reported fidelity with a dense reference.
CPU circuit tests verify disjoint non-diagonal gate reordering, overlapping
gate dependencies, exact state reconstruction and strip transition counts.

These tests use zero boundary cutoff for reference comparisons. During the
audit, default complex64 cutoff legitimately retained rank three instead of
four on one boundary, causing the directional convergence check to report
`limit_unconverged`. Increasing chi alone need not undo a cutoff error. The
production convergence tolerance was not loosened to conceal that result.

Three CuPy 4×4 TFIM experiments repeat the preceding independent reference
setup with `gate_order="column"`, D=2, boundary chi=32, complex128:

| Experiment | End time | Infidelity vs exact Trotter | Final energy |
| --- | ---: | ---: | ---: |
| Real, cached, 4 steps of 0.1 | 0.4 | 1.65302777e-3 | -15.79010091 |
| Real, fresh, same schedule | 0.4 | 1.65302777e-3 | -15.79010091 |
| Imaginary, cached, 3 steps of 0.05 | 0.15 | 4.31221072e-8 | -20.60673592 |

Cached/fresh real-time states agree to 9.38e-9 in phase-aligned norm, with
maximum component difference 4.36e-10. Norm error stays below 1.6e-15 across
these runs. The real-time cached run reports 398 outer-boundary hits / 175
rebuilds and 64 strip hits / 128 rebuilds. Imaginary time reports 313 / 116
and 48 / 96 respectively. No timing speedup is claimed: the small GPU problem
includes launch/synchronization overhead, snapshot comparisons and first-run
kernel initialization. Multiple GPUs and large-D performance remain untested.

Temporary reproducer: `/tmp/pepsy_itf_4x4_cupy_review.py`; JSON/vector outputs:
`/tmp/pepsy_itf_4x4_cupy_results/`. Test results and logs are recorded in the
[handoff](../../../history/2026-10-08-full-update-cupy.md).

Final validation: 431 combined affected checks passed (66 warnings,
114.74 seconds), followed by two newly added CuPy DMRG-boundary checks
(33 upstream warnings, 3.77 seconds). Default smoke: 94 passed, two warnings,
32.14 seconds. Ruff and whitespace checks pass. These are focused results;
the preceding review's 24 reproduced baseline BP failures remain outside this
change, and no new clean full-suite claim is made.

## Proposed only: optional ALS refinement of a strip

**Status update:** this proposal was subsequently implemented and tested in
the [strip-refinement follow-up](2026-10-08-full-update-strip-refinement.md).
The text below preserves the original design discussion.

The user's suggested extension is mathematically distinct from applying a
two-site gate: the gate changes two local tensors exactly, but the best
fixed-D approximation can benefit from changes elsewhere. A full update's
environment represents the whole lattice; it does not make every tensor a
variational parameter.

The smallest useful extension would retain standard two-site full updates and
optionally run one one-site ALS pass along the current row/column before
leaving it. Keep bond dimensions fixed initially. Reuse Quimb's public fitter
with prebuilt norm and candidate–target overlap networks, including cached
left/right partial contractions and Cotengra paths. Rebuild only dependencies
of changed tensors. Add neighboring strips only after measuring whether the
extra improvement is useful.

The target must be the untruncated gated state for the refined block. For a
single gate, retain its existing exact target. For one cleanup after a strip's
gates, snapshot the state before that block and apply those gates without
intermediate truncation to construct the target. Refitting the compressed
candidate to itself cannot recover projection error. A separate overlap
boundary is necessary once the candidate differs outside the original pair;
the current pair-only norm shortcut cannot simply be reused as that overlap.

Existing `SweepOptimizer` supplies norm/overlap boundary orchestration but
currently optimizes PEPS slices with gradient solvers. Its DMRG/FIT options
compress boundary MPS; selecting them does not turn PEPS refinement into ALS.
A native strip-ALS adapter with an explicit active strip would be needed.
This is not implemented in this follow-up. Start with a single bounded pass,
retain the best candidate against the same target/metric, and compare its
accuracy and wall time against ordinary two-site full update before expanding
the region or scheduling additional passes.
