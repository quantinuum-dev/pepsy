# 2026-10-13 — PEPS/NetKet/Symmray review before scaling

- Scope: user requested code and upstream review, fixes and validation before
  increasing lattice size, D and χ.
- Branch / baseline: `develop`, `82ee387`.
- Commit status: uncommitted. Earlier benchmark changes and concurrent MPS
  progress work were preserved; no files staged and no push in this audit.

## Changes and evidence

- [VMC adapter](../src/pepsy/vmc/netket.py) honors build-time sampler thinning,
  applies/restores call-time burn-in across released/development NetKet,
  detects incomplete optional imports, retains complex SR derivatives by
  default, and chooses reverse Jacobians for compressed PEPS. Explicit
  incompatible forward-mode requests fail early.
- [Shared configuration](../src/pepsy/vmc/api.py) delegates default SR mode
  to each backend; Torch preserves its general real/imaginary update.
- TDVP accepts named native dense/JAX CG solvers and kwargs. A configurable
  1 GiB limit checks single dense metric allocations. CG avoids metric
  materialization while retaining the sample Jacobian.
- [Tests](../tests/test_netket_tdvp.py) compare CG and dense dynamics with ED,
  forbid dense metric allocation during CG, check actual sampler proposal
  counts and truncated SR updates. The mesh fixture now restores an explicit
  empty mesh if JAX first imports during a test, supporting older JAX.
- [API guide](../docs/api/vmc.md), example solver choices and changelog
  describe the behavior changes; the
  [dated audit](../docs/development/notes/2026-10-13-netket-symmray-audit.md)
  records exact versions, upstream adopt/shim/prototype/defer decisions,
  numerical evidence and remaining scaling constraints.

## Validation and remaining work

- Isolated main/JAX 0.11.1: 120 CPU VMC/TDVP/public API checks passed.
  Released NetKet 3.22.4/JAX 0.8.3: 78 CPU VMC/native symmetry checks passed.
  Each also passed three subsequent compressed reverse-SR checks.
- GPU: small exact-evolution CG checks and a 4×4 D=4 χ=4 step passed.
  User subsequently supplied locally patched NetKet 3.22.4/JAX 0.11.2;
  installed-stack GPU dense/CG comparisons passed after clearing the inherited
  system CUDA library path for the process. The same D=4 probe completed in
  40.0 s, residual `8.7605e-11`.
- Final user's installed stack: 135 focused CPU checks, four GPU
  exact-evolution checks and four GPU dense/flat-Z2 derivative checks passed.
- Ruff over src/tests/example and diff whitespace checks passed. No full-suite
  success claimed. Installed environment and shell profile were not edited.
- Symmray is already current; packed NumPy/Numba storage is not a replacement
  for flat-Z2 JAX arrays. Native `balance_bonds_()` still fails for both flat
  and sparse Z2; no densification workaround was introduced.
- Increase system size, D, χ and samples separately. Paper-level scalable
  tVMC still needs validated gauge/tangent conditioning, dynamics sample-space
  solves, convergence evidence and complete restart checkpoints.
