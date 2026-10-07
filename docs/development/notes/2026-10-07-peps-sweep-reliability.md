# PEPS sweep reliability investigation

User scope: finish investigating the unresolved D=4 sweep behavior, including
failure to obtain reliable results by increasing boundary chi. Baseline:
develop at e8a3f96. This note distinguishes control-flow corrections from
finite-chi contraction error and finite-D approximation error.

## Confirmed control-flow problems

The preserved 5x6 chi=(32,56) full-layer trajectory recorded all 47 local
updates invalid at several time steps (including 28 and 60), but marked the
steps `optimized=true`. The completed sweep lacked a failure status; the
objective-only driver consequently accepted a run that applied no updates.
The new status counts actual writebacks and rejected updates. Fully rejected
sweeps return `success=False`, `no_valid_local_updates`, and raw invalid
slice records. The driver retains its compressed warm start and continues
the gate stream without claiming a successful refinement. Explicit zero
cycles remains an intentional successful no-op.

Clipping negative losses to zero also let a contraction artifact become the
best state or satisfy early convergence. Diagnostic clipping remains, but
selection uses admissible raw values (with the existing 1e-10 roundoff
allowance). A small negative initial value continues the requested sweeps.
The outer sweep driver likewise refines after an unreliable clipped
precheck, forwards its raw value when reused, and does not compare a
candidate against the artificial zero. This is not proof that a candidate
improves on an unreliable precheck. Independent accurate checks remain
necessary for such a claim. If a postcheck retry raises the cap and the
remeasured warm-start score becomes admissible, normal improvement comparison
resumes using the two scores at the same cap.

## Exact-reference evidence

An entangled 3x3 complex128 case tests local values and autograd derivatives
after actual slice perturbations through forward, backward, and return passes
on both axes. Pepsy/direct and Quimb-MPS boundaries at chi=64 match exact
statevector contraction to about 1e-14. This is now a regression test.
It rules out a basic indexing/gradient/update defect on this reference;
it does not prove finite-chi environments are accurate on a wider lattice.

A separate A100 4x4 D=4 case starts from random seed 317, applies all
nearest-neighbor RZZ gates with phase 0.1, builds the exact D=8 target, and
compresses to D=4. The exact target vector is independently obtained by
applying diagonal phases to the input vector. Target and guess are normalized
using their exact vectors for this diagnostic. The exact initial infidelity
is 0.1467183450792866. The same interior-row local objective gives:

| Boundary chi | Approximate infidelity | Absolute discrepancy |
| --- | --- | --- |
| 32 | 0.1526723938 | 0.00595405 |
| 64 | 0.1500636533 | 0.00334531 |
| 128 | 0.1471054729 | 0.000387128 |
| 256 | 0.1467949812 | 0.0000766361 |
| 512 | 0.1467187847 | 0.000000439639 |

Bounded real SciPy fits (20 iterations per slice, one forward pass per axis,
8 applied updates, no invalid updates) improve the exact infidelity:

| chi | Exact infidelity after fitting | Best local estimate |
| --- | --- | --- |
| 128 | 0.1060865195 | 0.1073138279 |
| 512 | 0.1052112988 | 0.1051993327 |

Thus chi=128 is not automatically sufficient even on 4x4. Increasing chi
improves contraction accuracy; it does not eliminate finite-D fit error.
No t=6 5x6 accuracy claim follows from these bounded references.

Scripts/logs: `/tmp/pepsy_sweep_local_exact_audit.py`,
`/tmp/pepsy_sweep_chi_dense_audit.py`,
`/tmp/pepsy_sweep_dense_fit_audit.py` and corresponding `.log` files.
The initial diagnostic target used uncapped generic splits and exhausted
memory by retaining zero singular values; its replacement uses the known
rank-two ZZ bound (D to 2D). No production job was affected.
An initial fit-script option spelling was ignored; the numbers above come
from the rerun using verified `optimizer` / `optimizer_options` controls.

## Dependencies and validation

Installed versions: Quimb 1.15.1.dev75+g4112e304a,
Autoray 0.11.1.dev9+g1291702f9, Cotengra 0.8.3.dev7+g1d7fd333f,
Symmray 0.4.1.dev11+g1a3481803, Torch 2.11.0.
Inspected installed contract and balance_bonds signatures. Reviewed official
[Quimb release notes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/)
and [changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray](https://github.com/jcmgray/symmray).
Symmray's abelian-array documentation returned an error. Classification:
defer unrelated upstream changes; retain existing supported public APIs and
Pepsy's Cotengra builder. No installed-library or dependency changes.

Final focused validation: 284 passed (safeguards, performance, batching,
optimizer, timing, and exact boundary numerics). This includes the final
outer-precheck and higher-cap remeasurement regressions. Old tests that
expected clipping to bypass refinement now explicitly check the refinement
attempt and rollback/continuation instead. CPU test workers require execution
outside the PID sandbox: inside it, loky/psutil fails to find worker PIDs.
Ruff and Pyflakes are unavailable. Broader validation is recorded in the
session handoff.

The old-policy 5x6 Torch complex128, D=4, chi=(64,64), direct-boundary replay
uses GPU0 and preserves existing jobs on GPUs1/3. Its bounded diagnostic log
is `/tmp/pepsy_sweep_invalid_capture.log`; all twelve requested steps (t=1.2)
completed without a large invalid local loss. Small negative estimates were
observed. The last step took 257 seconds. This
replay was started before the new control-flow code was loaded and is not
validation of the patched trajectory or late-time accuracy.

## Subsequent patched production comparison

The user subsequently launched direct chi=(64,64), direct chi=(128,128),
and DMRG (`eff`) chi=(128,128) trajectories on GPUs 0, 1, and 2, respectively.
The launch archive is
`/tmp/pepsy_examples_runs/peps_sweep_comparison_D4_20261007_192044/`.
Each initial full-layer fit used 49 two-site gates, an exact D=8 target,
and 47 applied local updates. At a later check, all three were still on
angle 1, reaching t=1.3, 1.2, and 1.7 without rejected updates.

The direct runs nevertheless show early local loss excursions: a small
negative initial estimate is followed by a positive local final loss of
about 0.41--0.67. The sweep restores its best recorded state, but the
independent post-fit overlap is disabled in this experiment. These remain
unresolved numerical concerns; the control-flow fixes do not certify that
every local update improves the state or that the selected best approximate
score is accurate. Accumulated products of fit scores are diagnostics, not
measured global fidelity against an exact trajectory.
