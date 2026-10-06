# 2026-10-06 — Isolate solver storage to preserve PEPS rollback

- Scope: user requested a careful fix for the confirmed rollback bug, then
  explicitly requested commit and push.
- Branch / baseline: `develop`, `4f8935e47a41361f1e1c691898f9af8956cb33f1`.
- Commit scope: solver ownership fix, its tests/docs/changelog, and the
  associated investigation records. Earlier pending optimizer-policy and
  global-cutoff changes remain separate in the working tree.

## Change

`solvers.gradient._as_trainable_tensor` now clones once with contiguous
memory layout, on the input device, before making parameters trainable.
Torch `detach()` and `contiguous()` previously allowed caller storage to
remain aliased; host solvers and Torch optimizers then wrote trial iterates
through the PEPS warm-start snapshot. NumPy-to-Torch conversion had the same
ownership problem. JAX dispatch and immutable arrays are unchanged.

This fixes the cause instead of adding another whole-PEPS rollback copy or
changing boundary chi, normalization policy, fidelity clipping, solver
budgets, or physical target construction. Background workers already running
have their original loaded code; no production worker was restarted.

## Regression evidence

See [the investigation](2026-10-06-peps-chi-exponent-investigation.md) for the
same-input chi convergence and exponent-equivalence tests. The 1.4% negative
infidelity was explained by a target norm squared around 1.014, following
corrupted rollback, rather than inadequate chi.

Before this fix, the new SciPy/NLopt input-ownership checks failed and the
real 2x3 PEPS rejected-fit test failed with tensor changes up to 0.0158532.
After the fix:

- 40 new solver cases cover SciPy, NLopt, Torch L-BFGS/Adam and FD-SciPy,
  Torch/NumPy, complex64/complex128, contiguous/strided inputs. They assert
  input values and gradient metadata are preserved during objective calls,
  dtype/device/shape are retained, loss improves, and modifying the returned
  result cannot mutate the caller.
- A real SciPy PEPS fit exercises forced outer rejection and checks exact
  tensor/exponent restoration, normalized dense state, consistency of the
  retained fidelity diagnostic, and unit norm of the next unitary target.
- Current working tree: gradient solver, JAX solver, PEPS safeguards,
  batching, performance, and optimization suites: **350 passed**, 5 warnings,
  54.23 s. This includes earlier pending optimizer changes.
- Exact staged code copied with `git checkout-index` and tested with its own
  `PYTHONPATH`: gradient solver and PEPS safeguards suites: **108 passed**,
  1 sandbox CUDA-initialization warning, 10.60 s. The tests ran on CPU.
- Compileall and `git diff --check` passed. Ruff is not installed in the
  selected cloudspace environment; no full-suite claim.

## CUDA replay

`/tmp/pepsy_fixed_rollback_replay.py` replays the original 5x6 D=4 settings
on CUDA:0: Torch complex128, dt=0.1, theta=18*pi/88, chi=(64,64) for all
boundary/normalization/evaluation policies, direct compression, SciPy,
two round trips and one two-site gate per batch. It asserts unchanged
target arrays and exact rollback restoration during real fits.

Log: `/tmp/pepsy_fixed_rollback_replay.log`.
Step results: `/tmp/pepsy_fixed_rollback_replay/summary.json`.
The replay completed through t=0.3, past the previous failing batch, with
49 gate batches at each of t=0.1,0.2,0.3. Step 3 took 524.65 s and completed
13 fits. All 9 rejected fits restored the saved tensors and exponent exactly;
the remaining 4 fits were accepted. All checked target tensors/exponents
remained unchanged during fitting. The final norm squared was
0.9999999999834298. No normalization/fidelity exception occurred. Expected
small finite-contraction warnings remained; one loky worker-stop warning
occurred, after which the replay continued and finished successfully.
This is a bounded regression replay, not a t=6 production trajectory.
