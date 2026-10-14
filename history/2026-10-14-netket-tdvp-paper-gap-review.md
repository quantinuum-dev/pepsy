# 2026-10-14 — Remaining NetKet PEPS tVMC wiring

The basic fixed-step dense-spin TDVP bridge is wired; paper-scale solver and
storage mechanisms, plus several operational controls, remain separate work.

- Scope: read-only comparison beyond compression-gradient validation, gauge
  removal, optional symmetries and large-scale convergence studies.
- Branch / baseline: `develop` / `a820f3c` (published implementation).
- Publication scope: this review accompanies the user-authorized practical
  TDVP fixes commit. No numerical code changed during the review itself;
  concurrent compression, fitting and Torch changes were preserved.

## Findings

| Addition | Current evidence | Role |
| --- | --- | --- |
| Sample-space/minSR dynamics | NetKet TDVP uses parameter-space QGT solves; `VMC_SR` belongs to optimization | Scalable alternative, with a correctly derived real-coordinate form for compressed amplitudes |
| Physical-slice small-o derivatives | The bridge stores generic sample Jacobians | Paper-specific memory optimization; verify sparsity for the actual contraction |
| Actual-stage solve accuracy policy | Native TDVP discards solve info; adapter rejects nonfinite accepted parameters; endpoint diagnostics perform a fresh solve | Reliability improvement: detect finite inaccurate solves used during integration |
| Resumable checkpoints | TFIM example writes parameter/reference snapshots without sampler/RNG or integrator state and has no resume option | Long-run operations; distinguish fresh-sampler continuation from reproducible continuation |
| Adaptive integration | Builder exposes fixed-step Euler/midpoint/RK4; installed NetKet has RK45 tolerances and step limits | Optional integration extension, not required by the paper's basic RK workflow |
| PEPS sequential sampling | Ising builder uses native MetropolisLocal | Optional performance/mixing alternative to reproduce the paper's sampling implementation |

The first two additions are substantial algorithm work rather than missing
arguments. The paper's uncentered holomorphic minSR equation cannot be copied
unchanged into explicit real/imaginary coordinates. Duplicate samples and
restricted support also require rank handling. CG already avoids a dense
parameter metric, but does not implement either of these additions.

Product-state startup is already present in the example. Born-support collapse
remains a known limitation, especially after projective local quenches; neither
small solve residuals nor gauge removal resolves it.

## Sources and validation

- Re-read [Wu and Nys, PRX Quantum 7, 033035](https://journals.aps.org/prxquantum/pdf/10.1103/tggc-8fjx),
  Secs. II, III B–E, IV G and VI. Paper uses sample-space solves, small-o
  storage and sequential sampling, and identifies support/product-state issues.
- Checked [native TDVP documentation](https://netket.readthedocs.io/en/stable/api/_generated/experimental/dynamics/netket.experimental.TDVP.html)
  and [RK45 controls](https://netket.readthedocs.io/en/stable/api/_generated/experimental/dynamics/netket.experimental.dynamics.RK45.html).
- Inspected installed NetKet 3.22.4 signatures and TDVP source on CPU: native
  RHS computes `qgt.solve(...)` and discards its second return value. No
  installed-library modifications. An initial automatic-device import logged
  the known CUDA library mismatch; this review makes no GPU validation claim.
- Read [bridge](../src/pepsy/vmc/netket.py) and
  [TFIM example](../examples/tfim_netket_tdvp.py).
  No numerical tests rerun: this is an API/source/paper review. Earlier
  implementation tests remain documented in the
  [basics audit](2026-10-14-netket-tdvp-basics-audit.md).

Suggested next work, subject to user scope: actual-stage solver residual
monitoring and resumable checkpoints for operations; sample-space TDVP and
small-o storage for the paper's scalable algorithm.
