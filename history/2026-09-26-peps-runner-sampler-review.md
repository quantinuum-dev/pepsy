# 2026-09-26 — Review PEPS roughening and standalone sampler

- Scope: user requested a review and report, not implementation or launches.
- Pepsy branch / baseline: `develop` / `a13031b`; examples `main` with
  pre-existing working-tree edits. Reviewed current working trees.
- Commit status: this review handoff is uncommitted; no algorithm changes,
  staging, publication, dependency changes, or production-job changes.
- Applied the repository `peps-sampler` review skill.

## Findings

1. The roughening PEPS frontend explicitly rejects shots and entropy and
   dispatches to `PepsSimpleUpdate`. It does not call `PepsSampler` or the
   MPS paper-sample collector. Existing functionality is boundary local Z,
   imbalance, and selected-time boundary/BP/loop norms. Shot-based interface
   observables require a separate integration task.
2. The PEPS runner saves observation arrays and metadata, not serialized
   core tensors plus gauges or absorbed PEPS snapshots. These outputs alone
   cannot feed later PEPS sampling or restart evolution at a saved time.
3. Simple update keeps core tensors and external gauges, uses gate renorm=False,
   and absorbs gauges into a private measurement snapshot. Evolution D,
   measurement boundary chi, and sampler future/conditioned chis are distinct
   convergence controls. No 5x6 t=10 accuracy certification follows from the
   small-system checks or from a norm close to one.
4. Boundary sampler configurations follow an approximate proposal. Returned
   original-ket amplitudes and proposal likelihoods support importance
   weighting, but finite truncation can remove target support. Spectral rho
   repair is opt-in and changes the proposal; it cannot guarantee support.
5. Each distinct final sampled configuration contracts the full original ket
   for its amplitude. Sampling boundary caps do not bound this contraction's
   complexity. Prefix groups retain separate conditioned states after
   branching; this is not native GPU batch-axis sampling. Dense row caches
   are opt-in, with default budget zero, and do not cap total batch memory.
6. The earlier complex64 relative-cutoff overflow finding is already addressed
   in current uncommitted code by private proposal rescaling and future
   equalization. It should not be reported as still unfixed. Arbitrary-scale
   raw amplitudes remain outside that fix.

## New checks

- CPU-only checks use the existing py312 environment, one BLAS/OpenMP thread,
  and no visible GPU to avoid using the production GPU.
- Independent bridge probe: evolved 2x3 PEPS, D=4, complex128, three dt=0.05
  steps, zero cutoff; enumerated all 64 configurations. Exact, Quimb future,
  and DMRG future samplers gave normalized distributions; maximum Born error
  2.22e-16. Sampled complex amplitude error <=3.53e-16 and log proposal replay
  error <=8.89e-16. Evolution core arrays remained exactly unchanged.
- New combined regression run: `tests/test_peps_sampler.py`,
  `tests/test_peps_sampler_4x4.py`, and the examples runner `tests/test_peps.py`:
  **259 passed, 2 skipped**, 45 Quimb mode/method deprecation warnings,
  679.20 seconds. Skips are JAX nondefault-device cases with only one exposed
  CPU device. This includes the full 4x4 dense-reference integration cases.
- Installed versions confirmed unchanged: Quimb 1.15.1.dev66+ge927f06e1,
  Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f,
  Cotengrust 0.2.1, NumPy 2.5.2, Torch 2.6.0+cu124, JAX 0.10.2.
- `git diff --check` passed. No new numerical regression was identified in
  these tested paths; integration and scaling limits above remain.
- Test log: `/tmp/peps-review-20260926-tests.log`.

## Evidence and remaining limits

- Runner: sibling `pepsy_examples/experiments/mps_magnetization/benchmark/`
  `magnetization/engines/peps.py` and `magnetization/runners/roughening.py`.
- Sampler: [implementation](../src/pepsy/sampling/peps.py) and
  [API guide](../docs/api/sampling/samplers.md).
- Earlier, separate [4x4 validation](../docs/development/notes/peps_sampler_4x4_validation.md)
  checks a genuine D=4 entangled state and full dense amplitude oracle; its
  finite-cap results do not certify arbitrary PEPS or production GPU throughput.
- No numerical upstream behavior or dependency was changed. This is not a
  full repository test run or a GPU performance benchmark.
- Suggested next work, not authorized implementation: connect sampled
  readout to absorbed PEPS snapshots, preserve site-order mapping, report
  proposal-weight/ESS diagnostics, and check evolution/measurement/sampling
  cap convergence separately on evolved-state references.
