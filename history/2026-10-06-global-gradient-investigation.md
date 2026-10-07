# 2026-10-06 — Global NLopt early-stop investigation

User asked how to fix repeated NLopt early stops in the live 5x6 D=4 global
roughening run. Baseline develop 4f8935e plus the uncommitted diagnostic policy
change. Production process/settings were not changed. Temporary isolated
probes shared CUDA0 with production, so timings are not clean benchmarks.

## Reproducer and evidence

Loaded the saved first actual refinement input from
`/tmp/pepsy_sweep_perf_case_20261006.pkl` (5x6 D=4 Torch complex128, boundary
caps 32/46). Called GlobalOptimizer's configured TNOptimizer loss and gradient
on the same vector, comparing centered finite differences along a seeded
random unit direction and the normalized autodiff gradient. This tests 15,660
real parameters without changing the state in the live process.

Probe: `/tmp/pepsy_global_gradient_probe.py`. Logs under
`/tmp/pepsy_global_gradient_*.log` retain scalar values, steps, warnings, and
directional derivative comparisons. No installed library was edited; candidate
contraction options were injected only into the isolated probe process.

- Current stabilized QR/SVD: objective 2.1837e-11, gradient norm 2.52e22.
  Random-direction AD 1.43e20 versus FD around 1e-8; gradient-direction FD is
  approximately zero. The gradient is unreliable at this input.
- QR adaptive policy without exponent stripping: gradient norm 3.01e23;
  disagreement remains. Exponent stripping is not the explanation here.
- Existing composed projector factorization: gradient norm .00562, with or
  without exponent stripping, but still fails finite-difference agreement.
  Removing QR reductions/canonicalization also does not fix the mismatch.
- Explicit projector canonicalization/reduction truncation at 1e-12 remains
  unstable. Stronger 1e-8 truncation produces locally consistent derivatives
  on two selected checks: random AD 1.52445e-5 vs FD 1.52262e-5 at h=1e-5;
  gradient-direction AD .00124817 vs FD .00124885 at h=1e-6. Larger steps cross
  changing retained-rank regions, and smaller differences show numerical noise.
  The forward approximate loss changes to 3.18507e-7; this is a materially
  different contraction policy, not a drop-in validated numerical fix.

Official references checked:
[Torch QR](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html)
documents incorrect autodiff for dependent columns;
[NLopt introduction](https://nlopt.readthedocs.io/en/latest/NLopt_Introduction/)
recommends finite differences to validate analytic gradients. These support
the diagnostic approach, not a proof of a complete fix for this trajectory.

Classification: prototype. No gradient/backend production changes adopted.
The issue requires a reliable derivative through rank-deficient boundary
contractions; clipping scalar fidelity cannot repair that derivative.

## Optimization check

`/tmp/pepsy_global_fit_probe.py` tries 50 LD_VAR2 evaluations with the pruned
projector prototype and then applies the original direct-boundary (32,46)
normalization/acceptance check. Results are stored in
`/tmp/pepsy_global_pruned_fit.json`. The prototype completed without an NLopt
early-stop error after 29 evaluations (32.30 s), reducing its own loss from
3.18507e-7 to 1.66533e-15. However, the original acceptance infidelity worsened
from 2.70579e-9 to 3.82759e-8. This candidate must be rejected. Eliminating the
solver error alone is insufficient: stronger pruning can optimize a biased
surrogate. The prototype was not applied to production. A complete global
derivative repair remains unresolved. All temporary diagnostic processes
finished; the existing global process was left running. git diff --check passed.
