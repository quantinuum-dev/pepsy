# 2026-10-05 — MpsOptimizer trajectory re-review after synchronization

- Scope: review MpsOptimizer, especially trajectories, and report findings.
- Branch / baseline: `develop` / `bec773a`, after the latest upstream sync.
- Commit status: no implementation/test edits, staging, commit, or push.
  This review journal is new and uncommitted. Existing working-tree changes
  were preserved; the reviewed MPS/noise implementation matches HEAD.
- Earlier evidence: [trajectory review and subsequent fixes](2026-10-05-mps-trajectory-review.md),
  [density references](2026-10-05-mps-trajectory-density-reference.md).

## Confirmed findings

1. **P1: default auto/coalesced exact-mode controls crash.**
   `noise.py:4385` calls `_measurement_probabilities` without first rebuilding
   an MPS from an exact-mode TensorNetwork. Ordinary control replay does this
   through `_ensure_mps_state` in `mps/_controls.py:66`, but the coalesced
   probability preflight runs before that boundary. Public reproduction:

   ```python
   sim = pepsy.MpsOptimizer(
       qtn.MPS_computational_state("00", dtype="complex128"),
       [("h", 0), ("measure", "Z", 0)], chi=4, mode="exact",
   )
   sim.run(shots=4, seed=1, workers=1, progress=False, progbar=False)
   ```

   Raises `AttributeError: 'TensorNetwork' object has no attribute 'canonize'`.
   Reproduced for `exact` and `exact-batch`, with `auto` and `coalesced`.
   `independent` succeeds in both modes. Also reproduced for entangled reset
   after H/CNOT. The API mode table explicitly promises exact control replay.
   Proposed: perform the exact-to-MPS preparation before coalesced probability
   evaluation, preserving backend, scale, and canonical metadata.

2. **P2: multi-site Kraus Gram contractions corrupt rare-event weights.**
   The one-site amplitude path is stable, but `noise.py:3036-3051` computes
   a Gram expectation for larger supports, losing small positive weights
   through cancellation. Let `u=|++>`, `v=|-->`, and
   `psi=(u+1e-10*v)/sqrt(1+1e-20)`. Build the two-site MPS with
   `MatrixProductState.from_dense(..., dims=[2,2], cutoff=0)`, and channel
   `K_rare=outer(v,v.conj())`, `K_common=I-K_rare`.
   Directly applying K to the actual optimizer dense state gives probability
   `1.0000012757041736e-20`. The public trajectory record instead reports
   `5.551115123125783e-17`. With proposal `[.5,.5]`, the rare branch's recorded
   likelihood ratio is `1.1102230246251565e-16`, versus the reference
   `2.0000025514083473e-20`: approximately 5,551 times too large.
   Reproduced in direct, dmrg2, and svd, independently and coalesced, with
   `chi=4`, `cutoff=0`, `n_iter=4`, 16 shots, seed 6. Thus this affects
   importance estimates even without truncation. Proposed: evaluate projected
   amplitudes for multi-site dense channels, or use an accuracy-preserving
   amplitude fallback for near-null Gram expectations.

3. **P2: generated noise gates fail on real-valued Torch states.**
   `noise.py:3205-3209` delegates to a converter that preserves the generated
   complex dtype, although X and amplitude-damping matrices here have no
   imaginary content. A Torch float64 `|10>` state, `chi=4`, and either
   `[("x_error", .3, 0)]` or `[("amplitude_damping", .3, 0)]` fail through
   `MpsOptimizer.run(shots=4, seed=1, workers=1, progress=False, progbar=False)`.
   X-error raises the backend/dtype validation TypeError; amplitude damping
   raises `RuntimeError: both inputs should have same dtype` during contraction.
   Both pass after converting the initial state to Torch complex128.
   Proposed: safely retain the real state dtype for real-valued generated
   operators and define explicit behavior for genuinely complex outcomes.

4. **P2: default JAX GPU precision violates probability accuracy checks.**
   The one-site local gate contraction (`noise.py:3033`) and exact Gram path
   (`noise.py:3036`) inherit JAX's default matrix-product precision. On this
   CUDA device, complex64 amplitude damping of `|1>` at gamma=.3 returns
   `[.6997834378226164, .3002165621773835]`, not `[.7,.3]`.
   A four-site canonical test also misses its reference by `4.88e-5`.
   Both existing tests fail in isolation at default precision and pass with
   `JAX_DEFAULT_MATMUL_PRECISION=highest`. Both pass on CPU as well.
   This is reproducible backend precision sensitivity, not GPU allocation
   failure. Proposed: adopt a scoped public JAX precision policy for these
   probability contractions and test it on GPU; no global policy was changed.
   Consulted the official [JAX precision context documentation](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html).

## Fresh validation

Activated the existing `~/envs/py312` environment. Commands used one
OpenBLAS/OpenMP thread and disabled JAX GPU preallocation where relevant.

- `test_trajectory_noise.py`, `test_trajectory_importance_regressions.py`,
  `test_mps_trajectory_density_reference.py`, `test_mps_controls.py`,
  `test_mps_dynamic_controls.py`, `test_mpi.py`, with `-o addopts=''`:
  **292 passed, 2 failed, 1 skipped** (43.79 s).
  Failures: `test_one_site_exact_kraus_gram_stays_on_backend[jax]` and
  `test_kraus_probabilities_use_tracked_center_without_global_norms[jax]`.
  Skip: a second JAX device was not configured. CUDA/Torch and CuPy cases ran.
- Those two failures repeat in isolation; **2 passed** with highest JAX
  matmul precision. CPU rerun including the nondefault logical CPU device:
  **3 passed** with `JAX_PLATFORMS=cpu` and
  `XLA_FLAGS=--xla_force_host_platform_device_count=2`.
- `test_mps_gpu_backend.py` and `test_mps_normalization.py`:
  **83 passed, 9 failed, 1 skipped** (20.66 s). Seven failures concern JAX
  (`exact_reconstruction` direct/perm, `random_fit` random/random_expand,
  `warm_controls`, and `backend_ledger` restore False/True); two concern
  CUDA `backend_ledger` restore False/True. Metal is unavailable.
- Highest-precision rerun of all eight JAX backend cases: **7 passed,
  1 failed**. Remaining `backend_ledger[jax-True]` infidelity differs from
  NumPy by about `4.05e-6`, exceeding the `3e-6` tolerance, after the state
  comparison passes. The two isolated CUDA ledger cases still fail by about
  `3.84e-6` and `3.70e-6`. These small diagnostic discrepancies remain
  unclassified; no tolerance was weakened or additional runtime bug inferred.
- Combined original selections: **375 passed, 11 failed, 2 skipped**.
  Precision/CPU reruns are separate evidence, not a clean default suite.
- Public standalone probes confirm findings 1-3. Temporary script/log:
  `/tmp/pepsy-trajectory-public-repros.py` and matching `.log`.
  Suite logs have `/tmp/pepsy-trajectory-*` and `/tmp/pepsy-mps-*` names.
- `git diff --check` passes. No full-package suite, multi-rank MPI,
  performance benchmark, or new native Symmray trajectory audit.

Installed: Quimb `1.15.1.dev79+gb5e316200`, Autoray
`0.11.1.dev9+g1291702f9`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev11+g1a3481803`, NumPy `2.5.2`, Torch `2.6.0+cu124`, JAX `0.10.2`,
CuPy `14.1.1`. JAX uses `CudaDevice(0)`, x64 disabled, default matmul
precision unset. Inspected installed Quimb signatures for
`local_expectation_canonical`, `compute_local_expectation` (route keyword),
and MPS `normalize`.

These findings are reported, not fixed. The earlier support/callback
regressions and density-matrix ensemble references pass in this selection;
they do not cover the new exact-control, real-Torch, and rare two-site cases.
