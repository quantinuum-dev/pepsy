# 2026-10-06 — Global PEPS JAX/JIT review

Scope: investigate whether global optimization supports JAX/JIT and whether
boundary MPS contraction requires `cutoff=0`. This review does not change
implementation or defaults. Branch `develop`, baseline `8f7c896`, including
the uncommitted [global fixes](2026-10-06-peps-global-mode-fixes.md).

## Findings

- **Measured:** positive boundary SVD cutoff fails under JIT with
  `ConcretizationTypeError` in Quimb's `_trim_and_renorm_svd_result` at
  `int(n_chi)`. Retained rank depends on traced singular values. Setting
  `cutoff=0` removes this rank selection; the configured boundary bond cap
  still applies.
- **Measured:** `cutoff=0` alone is insufficient with the PEPS driver's
  default `strip_exponent=True`. `GlobalOptimizer._as_scaled_scalar` calls
  `.item()` on the JAX exponent, causing another `ConcretizationTypeError`.
  Its special preservation branch currently handles Torch only.
- **Measured:** disabling JIT does not fix exponent differentiation.
  With stripping enabled, an actual directional gradient was approximately
  zero, while the finite difference was `-0.0007239261856639699`.
- **Measured workaround:** `cutoff=0, strip_exponent=False` in global
  `loss_kwargs`, with `autodiff_backend="jax", jit_fn=True`, works on the
  tested dense complex128 CPU paths. Both JIT and non-JIT directional
  derivatives were `-0.0007239261878815074`, absolute finite-difference
  discrepancy `2.22e-12`.

JIT compiles the objective and gradient, not the Python gate driver or NLopt
loop. Disabling stripping gives up its overflow/underflow protection; these
small normalized examples do not establish safety for arbitrary large
networks. Native symmetry, GPU execution, and other contraction modes are
not established by this review.

## Reproduction and evidence

Installed JAX 0.10.2, x64 explicitly enabled. Reused the unchanged upstream
audit from the earlier global review; inspected installed Quimb `JaxHandler`,
`TNOptimizer` vectorized loss/gradient methods, and SVD truncation dispatch.
The optimizer's default device is CPU; successful checks explicitly set
`JAX_PLATFORMS=cpu`. CUDA discovery in an initial probe is not a GPU test.

Temporary scripts `/tmp/pepsy-global-jax-check.py`,
`/tmp/pepsy-global-jax-unstripped.py`, and
`/tmp/pepsy-global-jax-gradient.py` used single-threaded BLAS/OMP.

- Standalone: normalized random 3x3 D=2 complex128 PEPSs, seeds 291/293,
  target inner indices mangled, MPS cap 2, greedy final contraction,
  known target norm 1. Compare JIT on/off and cutoff 0/1e-12. With stripping
  disabled, eight LD_LBFGS evaluations lower loss from
  `0.9987868226299679` to `0.86374857840988` for both JIT and non-JIT.
- Gradient: same tensors, cutoff 0, random real parameter direction seed
  337, normalized direction, central difference step `1e-5`. Results above
  compare stripped non-JIT with unstripped JIT/non-JIT.
- Full PEPS driver: normalized random 3x3 D=2 complex128 seed 311,
  diagonal RZZ with phases `exp(-0.37j * [1,-1,-1,1])` on `(0,0),(0,1)`,
  chi 2, boundary caps `(8,10)`, greedy contraction, eight default global
  LD_VAR2 evaluations. NumPy input with JAX/JIT and both workaround options
  improves measured infidelity `0.08020151686720578` to
  `0.050734785364784374`; returned dense squared norm is 1 and backend NumPy.
  Repeating with JAX input arrays and gate gives
  `0.08020151686720389` to `0.050734785364782486`, dense squared norm 1,
  and JAX output arrays.

These are real diagnostic contractions and gradient checks, not a full
pytest suite. No production changes or dependency modifications were made.
Local links and `git diff --check` passed.

## Proposed follow-up

Preserve differentiable JAX exponents as backend scalars, add a JAX/JIT
boundary-cutoff policy, and add gradient regressions before claiming that
the driver's default loss options support JAX. Classification: **defer**
implementation pending task scope; the explicit workaround is measured.

The [JAX error documentation](https://docs.jax.dev/en/latest/errors.html)
explains concretization failures, and the
[JAX sharp bits guide](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)
documents static shape requirements and x64 configuration.
