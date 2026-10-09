# 2026-10-09 — Reduced/default and full-tensor pair L-BFGS

Historical implementation stage. The dense full-pair path below was replaced
by [cached scalar TN objectives](2026-10-09-cached-pair-tn.md); its measurements
describe the earlier implementation only.

## Implemented

`PepsOptimizer` two-site full update now exposes `tensor_mode="reduced"`
(default) and `tensor_mode="full"`. The existing working tree already had
`gauge=False`; this remains the default. Full mode selects joint L-BFGS and
requires gauges off. Reduced mode retains automatic Quimb/QR ALS and accepts
explicit L-BFGS, with or without optional environment conditioning.

The full fit reuses validated strip boundaries and positive norm projection.
Its exterior maps are identities; the two full tensors are reconstructed
directly, without projecting back onto frozen QR/LQ bases. Spectator arrays,
indices, tags, dtype, device, and exponent bookkeeping are preserved. The
original warm start is retained if the fit does not lower the positive-metric
cost. Analytic complex Euclidean gradients cover both factors jointly.

SciPy L-BFGS-B owns real host parameter vectors; objective/gradient contractions
stay on native Torch/CuPy arrays. This is not differentiation through a PEPS
update. Symmray, JAX and NumPy PEPS remain outside this two-site path's contract.

## Papers and scope

- [Lubasch et al., Sec. III B](https://arxiv.org/pdf/1405.3259) treats reduction
  and gauge conditioning separately. The latter improves numerical conditioning
  but is not needed to define the reduced optimization problem.
- [Haghshenas and Sheng, Sec. III C](https://arxiv.org/pdf/1711.07584) alternates
  reduced optimized regions and uses a positive approximant. Its comparison
  with fully optimized tensors uses conjugate gradients. The new L-BFGS option
  is not an implementation of its second-neighbor plaquette algorithm.
- Preserve the current clipped PSD spectrum. The second paper's spectral
  absolute-value prescription differs; that behavior was not changed silently.

## Upstream audit

Activated the existing `envs/py312` environment and inspected public signatures
of SciPy `minimize`, Quimb `tensor_network_fit_als` and `Tensor.split`.
Installed: Quimb 1.15.1.dev90+g6a3906cbe, Autoray 0.11.1.dev14+g014a3f69a,
Cotengra 0.8.3.dev8+g8954240f2, Cotengrust 0.2.1,
Symmray 0.4.1.dev15+g0374aaa3c, Torch 2.6.0+cu124, SciPy 1.17.1.
CuPy execution on CUDA was available and exercised.

Reviewed official Quimb/Cotengra changelogs, Cotengra docs and Autoray/Symmray
repositories. The Symmray array documentation returned an error; the official
repository and installed source remain available. **Adopt:** public SciPy
analytic-Jacobian L-BFGS-B and current Quimb contractions, with existing
backend converters. **Defer:** native symmetry expansion and upstream upgrades.
No compatibility shim or dependency/environment modification was needed.

## Validation

Fresh combined selection: **160 passed**, three warnings, 30.61 seconds:

```text
tests/test_peps_pair_lbfgs.py
tests/test_peps_full_update.py
tests/test_peps_strip_refinement.py
tests/test_peps_layer_refinement.py
tests/test_public_api.py
tests/test_package_layout.py
```

The new regressions check analytic gradients against Torch autograd, exact
asymmetric-gate targets and fidelities in both site orders/orientations,
complex64/complex128, leaving the original reduced subspace, untouched input
and spectator tensors, normalization scale/phase, constructor validation,
driver integration, and CuPy/Torch agreement. Ruff `src tests` and
`git diff --check` passed. No full-repository suite was run.

## Limits

This is a bounded dense full-pair implementation. Its norm dimension is
`D**6` for two bulk square-lattice sites, storage scales as `D**12`, and
PSD projection is expensive. `full_max_matrix_size=1024` rejects oversized
pairs before allocating the full target/environment; bulk D=4 needs an
explicit override. Larger-D performance and long trajectories are unmeasured.
Reduced mode remains the practical default. A matrix-free full-pair solver
is deferred, not implied by using L-BFGS.
