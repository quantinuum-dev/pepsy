# 2026-11-06 — Noisy optimizer capability review

MPS `direct` and `exact-batch` support noisy trajectories; `PepsOptimizer`
does not expose the integrated shot/Kraus workflow.

- Scope: inspect existing noisy simulation support and explain its limits;
  no implementation changes requested.
- Branch / baseline: `develop`, `ad523b6`.
- Commit status: this journal is uncommitted; existing JAX sampler changes
  were preserved. Nothing staged, committed, or published.

## Findings

- The [MPS mode matrix](../docs/api/optimizers/mps.md#shot-aware-replay) and
  [noise guide](../docs/api/optimizers/noise.md) agree with current execution:
  fixed mixtures, state-dependent Kraus channels, measurement/reset and
  feed-forward use trajectory replay. Dense Kraus probabilities are evaluated
  from projected amplitudes before selected-branch compression.
- `exact-batch` fuses gates on a dense trajectory state, with exponential
  memory. Finite-shot estimates still have sampling error. `direct` also has
  finite-bond/cutoff bias, including changes to later state-dependent weights.
- PEPS public probes: `run(shots=2)` raises an unexpected-keyword TypeError;
  native amplitude-damping entries fail gate-queue parsing; the generic
  trajectory runner fails because PEPS lacks the expected state-norm protocol.
  Non-unitary gate replay alone does not implement a complete noisy channel.
  Manually sampled unitary-fault PEPS circuits remain possible in principle;
  no PEPS noise integration was implemented or validated here.

## Fresh validation

Activated `/Users/rezah/envs/genpy` and used its Python 3.12 against local `src`.
Quimb `1.15.1.dev90+g6a3906cbe`, Autoray `0.11.1.dev14+g014a3f69a`,
NumPy `2.5.2`, pytest `9.0.2`.

- `python -m pytest -q -o addopts='' tests/test_mps_trajectory_density_reference.py
  tests/test_trajectory_review_regressions.py tests/test_mps_exact_batch.py`
  passed: **72 passed, 6 skipped**, one Quimb default-decomposition warning.
- Four additional direct calls to the existing independent density-matrix
  reference test with `mode="exact-batch"` passed: independent/coalesced replay
  crossed with ordinary/importance sampling. These were standalone probes,
  not four additional pytest-collected tests. The oracle checks all sixteen
  two-qubit Pauli products after damping, phase noise, measurement/feed-forward
  and further damping.
- A Bell circuit followed by two-qubit depolarization (`p=.001`) and damping
  (`gamma=.25`, qubit zero), 10,000 shots, seed 7, auto strategy, zero cutoff:
  both modes gave Z0 = 0.250857142857143 versus analytic 0.25. Both retained
  twelve leaves and agreed to floating-point precision at `chi=64`.

No full-suite, new GPU/MPI, or production performance benchmark was run.

## Follow-up: DMRG modes and automatic GPU scheduling

- Confirmed noisy-shot support for `dmrg`/`fit`, `dmrg2`, and `dmrg3`.
  The latter two retain their named block schedules through `_dmrg_mode_alias`
  and the shot factory, although `optimizer.mode` normalizes them to `dmrg`.
- Four standalone density-reference calls passed for `dmrg`/`dmrg3` crossed
  with independent/coalesced replay, without importance sampling. All four
  public spellings also passed a seeded Bell/damping `strategy="auto"` probe.
- Automatic scheduling, continuation and memory suites passed:
  **64 passed, 11 skipped** (`test_mps_trajectory_auto.py`,
  `test_mps_trajectory_continuation.py`, `test_mps_trajectory_memory.py`,
  with `-q -o addopts=''`). These are separate from the 72 tests above.
- Local Torch reports neither CUDA nor MPS availability. GPU policy was
  inspected and its scheduler/query tests ran; no new hardware-GPU claim.
- `strategy="auto"` selects prefix sharing versus independent replay;
  `workers="auto"` separately chooses one accelerator worker. Device-memory
  planning can tighten branch limits. Local automatic cap continuation keeps
  completed prefixes and finishes remaining shots individually without pruning.
- Torch CUDA/CuPy coalesced one-site Kraus probabilities batch compatible
  parents. Full gate/SVD parent batching is restricted to eligible adjacent
  `swap` gates; `direct` and DMRG gate updates remain their ordinary paths.
  No mode, device, dtype, bond cap or cutoff is automatically retuned.
