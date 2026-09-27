# 2026-09-26 — PEPS sampler maturity and paper comparison

- Scope: user requested a fresh correctness/performance assessment and a
  comparison with Frank Verstraete's PEPS direct/perfect-sampling work.
- Branch / baseline: `develop` / `a13031b`, clean at start.
- Status: assessment only; documentation edits are uncommitted. No sampler
  implementation, dependency, production job, or shared environment changed.

## Findings

- Current algorithm is validated for small finite dense OBC PEPS. Prefix
  grouping and backend-native probabilities do not provide batched boundary
  contractions or demonstrated high-throughput CUDA performance.
- Newly reproduced installed Quimb complex64 `rsum2` overflow changes retained
  rank under a pure scale change. It affects the Quimb-future sampler on an
  unnormalized 5x6 D=2 state. The DMRG future route passes that probe. This issue
  remains unfixed and is the first recommended follow-up.
- The authors compress an enlarged row before sampling; Pepsy compresses after
  fixing the row, as requested by the user. Finite-cutoff proposals can differ.
- Original-ket amplitude contraction remains uncapped by the boundary chis;
  factored row environments and explicit amplitude-cost control are the main
  scalability proposals. No head-to-head Julia runtime comparison was made.

## New validation

- PEPS sampler domain: 132 passed, 25 existing warnings, 203.32 seconds.
- Full 3x3 D=2 enumeration for four settings: ample-cap maximum Born error
  2.43e-17; aggressively truncated proposals normalized but differ from Born.
- Fourteen timing cases, same-state precision/scale probes, and eight minimal
  public Quimb SVD cases. GPU simulations continued untouched.
- Detailed settings, timings, sources, reproduction snippet, and open limits:
  [assessment](../docs/development/notes/peps_sampler_maturity_comparison.md).

The sandbox patch helper could not access files due to the existing bubblewrap
mountinfo failure. Documentation edits used checked replacements via authorized
execution. No commit or publication was requested for this assessment.
