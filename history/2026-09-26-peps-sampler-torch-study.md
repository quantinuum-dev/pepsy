# 2026-09-26 — PepsSampler Torch/Autoray study

- Scope: study backend/device preservation through PepsSampler and whether
  recent Autoray changes help. No sampler implementation changes requested.
- Branch / baseline: develop / 80f451a; existing working-tree changes retained.
- Commit status: new audit note and this handoff are uncommitted; nothing
  staged, committed, installed, or published.

## Findings and evidence

- Exact and positive-marginal Quimb/DMRG boundary sampling preserve Torch
  arrays in contractions/compression on CPU and CUDA, but local rho,
  diagnostics, RNG, grouping, bookkeeping, and returned results cross to host.
- Reproduced a Torch CPU/CUDA failure with marginal_chi=0: NumPy identity
  caps cause mixed-backend tensordot. A temporary namespace-based identity
  subclass works on CUDA; the package remains unchanged.
- Bounded 2x3 probes: nine successful backend/path cases and three reproduced
  identity-cap failures. Captured CUDA copies and scalar reads; no timing or
  large-system performance claims. Prior production jobs were not changed.
- Installed Autoray 0.11.1.dev3 already provides useful namespace and RNG
  facilities. Main is three commits ahead, changing compiler/lazy thread
  safety rather than eager sampling. No dependency update needed for the
  demonstrated namespace/RNG capabilities; package-minimum compatibility
  remains a consideration for later use of newer features.
- Full details, source locations, versions, reproduction artifacts, and
  proposed work are in the [audit note](../docs/development/notes/peps_sampler_torch_audit.md).

## Validation and limits

Study scripts under /tmp exercised Torch CPU complex128 and CUDA
complex128/complex64, kept source tensors unchanged on successful paths,
and checked sampled amplitudes against dense contraction. A separate probe
verified device-local Autoray identities and seeded categorical draws.
Documentation link/whitespace checks passed. No package suite rerun or new
sampler regressions are claimed; package behavior was not changed.
