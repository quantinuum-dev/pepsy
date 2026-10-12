# 2026-10-06 — PEPS boundary chi, exponent, and rollback investigation

- Scope: investigate the failed 5x6 D=4 Torch complex128 gate-by-gate sweep;
  verify chi=64 norm/overlap settings, target normalization, exponent handling,
  and convergence over boundary caps. No production restart requested here.
- Branch / baseline: `develop`, `4f8935e47a41361f1e1c691898f9af8956cb33f1`.
- Commit status: this journal is uncommitted. Existing working-tree changes
  from earlier tasks are preserved; no algorithm changes in this investigation.

## Reproduction and convergence

CUDA:0 diagnostic replay used the latest saved runner arguments: 5x6, D=4,
dt=0.1, delta theta=18*pi/88, Torch complex128, SciPy L-BFGS-B, two round
trips per axis, one two-site gate per batch, direct boundary compression,
boundary/normalization/evaluation caps all (64,64). Other GPU jobs untouched.
Original job root:
`/tmp/pepsy_examples_runs/peps_sweep_5x6_D4_gpu0_scipy_lbfgs_2rounds_chi64_64_batch1_20261006_225941`.

The original run failed with infidelity -0.01422. The same-settings replay
failed in the warm-start precheck of time step 3 with -0.01407130159884.
This is a reproduced failure, not a recovered original checkpoint; numerical
roundoff changes which near-tolerance batches invoke fitting.

On the identical saved failing state/target, chi=16,32,64,96,128,192,256 all
gave the same metrics (current direct-compression cutoff retained):

| Metric | Value |
| --- | --- |
| Candidate norm squared | 1.0000000005560448 |
| Target norm squared | 1.0140713065988338 |
| Overlap | 1.0070110735055056 - 4.072751909e-9 i |
| Infidelity assuming target norm=1 | -0.014071301598841846 |
| Infidelity using measured target norm | 4.93061180772969e-9 |

Changing `strip_exponent=True` to False at chi=64 reproduced these results.
Moving both input exponents into tensor data and setting exponent metadata
to zero preserved the normalized infidelity within 2.4e-14. Compensated
exponent shifts (+3 for candidate, -2 for target) agreed within 3.7e-14.
Explicit target normalization (including the default bond balancing) yielded
target norm squared 0.9999999998323108 and infidelity 8.6736606786e-10.
The small difference after balancing is at the finite-contraction error scale.
There is no evidence that missing exponents or insufficient chi cause this
percent-level discrepancy; this is not a claim of convergence at later times.

## Confirmed rollback ownership bug

A separate bounded replay captured the first rejected fit. The saved state
before fitting had norm squared about 1.00000000035. After rejection, the
retained state differed in tensor entries by up to 0.0015592755 and its norm
squared was `10**3.5820016841e-5`, about 1.00008248. Both snapshots had the
same stored exponent. Its remeasured normalized infidelity was 1.2953531e-7,
although the outer rollback kept the original pre-fit diagnostic 2.6553537e-9.
The rejected candidate's reported postcheck was 1.3158638e-7.

`PepsOptimizer.run` saves the warm start with shallow `warmstart.copy()`.
`solvers.gradient._as_trainable_tensor` uses `detach().contiguous()` without
an ownership copy, and SciPy `_assign_flat_params` writes through `copy_`.
A one-parameter complex128 SciPy probe confirmed that caller data is mutated
and the returned solution shares storage with it. The live replay confirms
that this reaches the supposedly preserved PEPS rollback state. The exact
target tensors and exponent did not change during the five observed fits.

Suggested correction: give trainable solver parameters independent storage,
and validate rejected-fit state restoration and its diagnostic consistency.
Measuring or normalizing the target removes the invalid unit-norm assumption,
but alone would mask the rollback ownership bug. No correction has yet been
implemented or committed in this investigation.

## Evidence and checks

- `/tmp/pepsy_chi_exponent_probe.py` and `.log`: replay, failure capture, cap scan.
- `/tmp/pepsy_chi_exponent_probe/failure.pkl`, `metrics.json`: saved inputs/results.
- `/tmp/pepsy_exponent_representation_probe.py` and `.log`: exponent equivalence.
- `/tmp/pepsy_chi_exponent_probe/exponent_checks.json`: exponent results.
- `/tmp/pepsy_rollback_probe.py` and `.log`,
  `/tmp/pepsy_chi_exponent_probe/rollback.pkl`: rejected-fit snapshots.
- Diagnostic runs only; no full-suite claim, production restart, commit, or push.

## Follow-up

The user subsequently requested a fix and commit/push. The implementation and
validation are recorded in [the solver storage fix](2026-10-06-peps-solver-storage-fix.md).
The statements above describe the investigation before that correction.
