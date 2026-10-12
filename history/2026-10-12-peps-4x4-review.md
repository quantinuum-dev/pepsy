# 2026-10-12 — Independent 4×4 PEPS optimizer review

- Scope: follow-up review and actual 4×4 sweep/global checks.
- Branch / baseline: `develop` / `131d7eb`.
- Changes: numerical regression tests and this record; no optimizer changes.
- Commit status: this entry accompanies the follow-up validation commit,
  continuing the session's authorized publication to `origin/develop`.

## Reference and controls

The 16-qubit reference starts from a complex128 random PEPS, D=2, seed
`20261012`, normalized using its dense vector. Apply gates directly to this
65,536-element vector, independently of PEPS gating and boundary contraction.
The full circuit has 16 `exp(-0.13i X)` rotations, followed by all 12 vertical
and 12 horizontal `exp(-0.2i ZZ)` gates. Compare a complete warm-start-only
run with the optimized circuit, and check every fitted batch against its
own exact target. This is a correctness probe, not a solver ranking.

Sweep runs retain the published default strip/SU/ALS policy and both nested
two-round-trip budgets. Global runs use five L-BFGS-B steps with Torch or
JAX autodiff. Default norm/overlap caps are `(8, 16)`; the tighter runs use
`boundary_chi=(16, 64), normalize_chi=16, evaluation_chi=(16, 64)`.
All probes use CPU arrays and one BLAS/OpenMP thread per library.

## Measured exact circuit infidelity

| Run | Warm start only | Optimized | Exact output squared norm |
| --- | ---: | ---: | ---: |
| NumPy sweep, default caps | 0.273759309 | 0.220569513 | 1.000107681 |
| Torch sweep, default caps | 0.273759309 | 0.220569513 | 1.000107681 |
| NumPy sweep, tighter caps | 0.273759309 | 0.219900638 | 1.000000000 |
| Torch global, default caps | 0.298224053 | 0.242487317 | 0.999477572 |
| Torch global, tighter caps | 0.298224053 | 0.241598083 | 1.000000000 |
| NumPy sweep, general gates, default caps | 0.705968696 | 0.625725091 | 0.999795172 |
| Torch global, two strips, general gates | 0.361438359 | 0.306933568 | 0.999908238 |
| JAX global, two strips, diagonal gates | 0.102165138 | 0.080072326 | 0.999782339 |

General gates replace ZZ by `exp(-0.2i H)`, with
`H=(A+A.conj().T)/2` and complex Gaussian 4×4 `A` from NumPy seed 84.
Two-strip cases contain six vertical gates on columns 1 and 2, use SU guesses,
and retain layered targets with direct BRA/KET boundary compression.

Every fitted batch improved exact fidelity. NumPy/Torch sweep circuit
infidelities agreed within `6e-16`. Fitted boundary guesses were reused
between compatible batches; the first change of orientation correctly
discarded incompatible overlap guesses. Inputs and fixed targets retained
their arrays and tensor metadata, and outputs remained finite with D≤2.

Default caps give small, measurable norm/objective errors on this entangled
state. At `(16, 64)`, the output squared norm differs from one by less than
`4e-15`; local reported losses agree with dense fidelity within `1.2e-14`.
These bounds describe this fixture, not arbitrary 4×4 circuits.

Torch directional finite differences used step `1e-5`. The general-gate
two-strip derivative agreed within `1.7e-11`. The full diagonal circuit's
gradient differed by `1.13e-6` (relative `6e-5`), within the requested
`atol=2e-7, rtol=1e-4`; it emitted the existing rank-deficient QR warning.
This does not establish unrestricted accuracy on singular derivative charts.
The JAX two-strip directional derivative agreed within `8.8e-11`.

An initial JAX probe incorrectly demanded bitwise equality between NumPy and
JAX dense contractions; they differed by at most `3.2e-17`. Comparing the
input before/after on the same backend fixed the probe, and the rerun passed
the exact input-preservation and gradient checks. No numerical implementation
was changed to accommodate this reference issue.

## Reproduction and validation

The retained [4×4 regressions](../tests/test_peps_optimizer_4x4.py) exercise
both full-circuit fitting modes, exact per-batch loss, exact final fidelity,
input/target preservation, normalization, and bond caps. They are marked slow
and integration; the global case additionally requires Torch.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q -o addopts='' tests/test_peps_optimizer_4x4.py
```

Fresh regression result: **2 passed** in 26 seconds. Ruff on `src tests` and
`git diff --check` passed. The suite reported upstream NumPy deprecations and
the existing Torch QR warning described above. No new full-suite run was
needed for these test-only additions; prior publication results remain
separate evidence.

The compatible isolated environment and installed versions from the
[publication review](2026-10-12-peps-strip-publication.md) are unchanged;
the same upstream audit applies. Expanded diagnostic scripts/results live
under `/tmp/pepsy_4x4_review.py` and `/tmp/pepsy-4x4-*` and may be temporary.
No production trajectory or shared dependency was changed.

## Matched Torch/JAX follow-up

The earlier table compares each path with its own unoptimized run. Its
40-gate full-circuit and six-gate two-strip rows must not be used to compare
backend accuracy. The values are **infidelity**, so smaller is better.

A controlled rerun built one NumPy target and one SU guess, then copied their
arrays exactly into Torch and JAX. Both used the same six diagonal gates on
columns 1 and 2, exact initial normalization, complex128 CPU data, layered
targets, direct BRA/KET compression, cutoff zero, canonicalization enabled,
no exponent stripping, and five L-BFGS-B steps. Initial parameter vectors
were bitwise identical. Both backends made seven optimization evaluations;
Torch used eager execution and JAX used JIT.

| Norm/overlap caps | Backend | Initial exact infidelity | Final exact infidelity |
| --- | --- | ---: | ---: |
| `(8, 16)` | Torch | 0.102158635429621 | 0.080258389691159 |
| `(8, 16)` | JAX | 0.102158635429621 | 0.080258389691160 |
| `(16, 64)` | Torch | 0.102158635429621 | 0.080175937759808 |
| `(16, 64)` | JAX | 0.102158635429621 | 0.080175937759808 |

At default caps, final fidelities differ by `8.9e-16`; at tighter caps they
agree at the displayed precision and in the returned Python floats.
Initial full-gradient component differences are at most `6.8e-15`; each
backend also passes its directional finite-difference check, with absolute
error below `3.2e-10`. The tighter-cap objectives agree with exact dense
infidelity to roundoff. All outputs retain D≤2 and improve exact fidelity.

The shared warm start is normalized and constructed once before backend
conversion; this removes the backend-specific driver preparation from the
comparison. Its initial infidelity therefore differs slightly from the
previous independently constructed two-strip warm starts. No optimizer code
was changed. Temporary reproduction and machine-readable results are
`/tmp/pepsy_4x4_matched.py` and `/tmp/pepsy-4x4-matched.json`.
