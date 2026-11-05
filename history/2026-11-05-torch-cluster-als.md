# 2026-11-05 — Torch cluster ALS, fidelity tracking and stream scaling

- Scope: implement the reviewed speed optimization, optional local fidelity
  tracking, and investigate Torch compilation at the Pepsy level.
- Branch / baseline: `develop`, `4d50e53`.
- Commit status: included in this publication commit on `develop`, following
  the user's request to commit and push both repositories. The implementation
  and second review originally ended uncommitted; publication is a later step.
  Saved checkpoints and running simulations were not changed.

## Changes and measured behavior

Added `solver="torch"` with batched physical RHS contractions, certified
Cholesky solves and reference spectral fallback. Default automatic dispatch
remains Quimb. Added optional local fidelity/infidelity/status fields to
reduced ALS results, `LoopClusterUpdateTracker` for scalar records across
stream chunks, optional AOT contraction capture, and stream-owned per-pair
RMS extraction via `strip_exponent=True`. Avoid retaining full update results
across a stream unless requested. Public exports remain lazy and owning docs
and changelog are updated.

The [numerical and timing record](../docs/development/notes/2026-11-05-torch-cluster-als.md)
contains the conditioning certificate, upstream audit and bounded χ6 replay.
Three GPU pair updates improved by 1.64–1.87×, including fidelity measurement,
with matched 4/4/5 ALS sweeps. Compilation did not consistently beat eager;
keep it optional. No new full-trajectory accuracy claim is made.

## Validation

- Final integrated selection: **480 passed** in 65.09 s, including the BP
  compression/reduced/patch/local-gauge suites, new Torch ALS suite, public
  API/package layout, PEPS/PEPO full update, and paired PEPO full update.
  All ran with one numerical thread; CUDA checks were enabled.
- New solver tests: **36 passed**, including CPU/CUDA spectral-floor fallback,
  independent full-state fidelity, eager/AOT derivatives with native and
  stabilized QR/SVD, changing capture shapes, scaling and warm starts,
  optional measurement, scalar tracker edge cases, and 100-gate stream scale
  preservation, and autograd/inference mode chart coverage.
- Initial extended integration run: **479 passed, 1 failed**. The failure
  was a new tiny-product assertion that accidentally requested exact equality
  after logarithmic accumulation; specifying relative tolerance fixed it.
  The corrected 36-test new suite and final 480-test integration run passed.
- Full repository fail-fast: **264 passed, 2 skipped, 1 failed**, at
  `test_bp_symmray.py::test_fermionic_long_range_hopping_sign_survives_su_and_bp_gauges`.
  This same exact-readout failure was independently reproduced on the earlier
  clean baseline in the [publication review](2026-11-05-publication-review.md).
  The full repository suite is not green.
- Full `ruff check src tests`, local documentation-link targets, and
  `git diff --check` passed.

Temporary evidence: `/tmp/pepsy-cluster-speed-20261105/`.
The two pre-existing untracked audit journals were preserved. The optional
tree-reduction/message-floor/native-U1 issues documented there remain open;
only the multigate scale hazard is addressed by the new explicit stream option.

## Second review

The user requested another review of the implementation. Checked contraction
index order, complete-sweep QR/LQ, Cholesky certification and spectral fallback,
fidelity phase/normalization, tracker retention, stream scale preservation,
and the optional AOT cache. Reused this task's unchanged upstream audit.

Found and fixed one additional defect: the AOT key omitted autocast state and
dtype. A graph first captured without mixed precision produced a backward
dtype error when reused under CPU bfloat16 autocast. The key now includes
each input device's autocast state/dtype. CPU bfloat16 and CUDA float16
regressions both pass while toggling off/on/off and comparing eager forward
and gradients. No mixed-precision trajectory accuracy claim is made.

Added independent rectangular-layout/full-state checks with physical dimensions
2 and 3 and exterior dimensions 2 and 3, covering real64/complex64/complex128,
CPU/CUDA and eager/AOT at two ranks. Extended native/stabilized QR/SVD gate
gradient checks to CUDA. The new shape/dtype tests passed, and an interim
focused integration selection passed **238 tests** before the autocast fix.
The new autocast regressions passed **2 tests**. The final combined BP cluster,
reduced/compression/local-gauge and public API/layout rerun passed **240 tests**
in 33.11 s, including all **48** Torch ALS tests. Full Ruff, whitespace and
local documentation-link target checks passed. The larger 480-test selection
and full-suite fail-fast above belong to the earlier implementation check;
they were not rerun for this isolated cache correction. All changes remain
uncommitted.

## Publication preparation

The user subsequently authorized committing and pushing Pepsy and Pepsy
examples. The existing downstream 3×2 PEPO benchmark now selects this guarded
Torch solver and tracker; its small CPU/CUDA integration checks pass against
the independent complete operator. Its boundary-MPS scalar policy stays at
3χ. The separate diagonal-wall PEPS frontend was distinguished explicitly;
no new wall-state evolution mode is included in this publication.

All implementation, tests, owning docs, numerical notes and the two earlier
audit journals belong to this task's commit. The 240-test second-review
selection above is the latest package numerical check; prior full-suite
failure evidence remains scoped and unresolved. Push success is verified
against the remote before the user handoff.
