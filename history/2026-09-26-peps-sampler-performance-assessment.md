# 2026-09-26 — PEPS sampler performance assessment

- Scope: assess performance, practical sampling settings, and implementation
  priorities after the backend fix.
- Branch / baseline: develop / 80f451a, with prior working-tree changes.
- Commit status: new note and this handoff only; no implementation edits,
  staging, commits, installs, publication, or production-job changes.

## Measured findings

- 23 timing cases on random and SU-evolved D=4 PEPS, plus equivalence,
  allocation, and CUDA readout probes.
- A 3x3 eight-shot dense-cache call took 15.934 s versus 0.07663 s through
  the existing reference path. Same configurations; probabilities and
  amplitudes agree within rtol=1e-11, atol=1e-14. Routing prototype is confined
  to /tmp; the package still uses the original heuristic.
- Dense local transfer measured at 1 GiB on 3x3; metadata predicts 16 GiB
  individual transfers on the 4x4 case. Larger allocations were not made.
- 4x4 NumPy 32-shot batching: 0.877 s versus 1.006 s serial. Local-rho work
  and boundary updates dominate; many Quimb object copies remain.
- SU snapshot t=2, D=4, 128 shots: marginal_chi=0 gives roughly 31% ESS,
  versus 99.9% with marginal_chi=8, about 4.7x effective-sample throughput.
- Shared-GPU CUDA timing is limited by contention and small operations;
  sampler still performs hundreds of scalar/choice readouts per batch.
- Found float32 Hermiticity-diagnostic overflow on large unnormalized rhos;
  temporary scaling makes the diagnostic finite. Not fixed in this task.

## Recommended next implementation

Memory/work-aware cache routing first; then group native probability checks
and draws, reduce temporary network construction, and make norm diagnostics
numerically stable. Full batched boundary kernels are a larger follow-up.

Detailed methodology, measured tables, proposed work, existing best practices,
versions, official references, and artifact paths:
[performance assessment](../docs/development/notes/peps_sampler_performance_assessment.md).

## Validation and limitations

New validation is the bounded profiling and numerical route comparison.
No production-scale speedup, idle-GPU result, JAX timing, or new full-suite
pass is claimed. Prior backend tests remain recorded in the preceding
[handoff](2026-09-26-peps-sampler-backend-fix.md).
Documentation links and git diff --check were checked before handoff.
