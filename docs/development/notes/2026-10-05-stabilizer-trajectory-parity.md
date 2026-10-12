# Stabilizer MPS trajectory execution parity — 2026-10-05

The user requested that ordinary MPS and stabilizer MPS trajectories mature
together. Baseline: develop `7a01b05`. The reviewed changes are included in the
local priority-1 commit; see the [complete validation record](../../../history/2026-10-05-clean-baseline.md).

## Implementation

Both frontends now call the same local execution dispatcher in `noise.py`.
It owns shared-prefix continuation, explicit coalesced caps, retention checks
before expanding prefixes, and fallback after releasing failed-frontier
tracebacks. Prepared trajectory plans are reused for threaded ordinary MPS
replay as well as stabilizer replay.

StabilizerMpsSimulator now reuses ordinary MPS CPU/accelerator worker planning,
allocator-aware memory helpers and diagnostic annotation. Its public
`memory_budget="auto"` accepts explicit local byte budgets or `None`, with
estimated tableau/classical storage added to possible coefficient bond growth.
Explicit byte budgets remain unsupported for MPI; automatic accelerator MPI
workers default to one. Estimates are snapshots, not allocator reservations or
universal OOM guarantees. Unsupported allocator queries leave auto planning
inactive, including the current CPU path.

Frame Kraus probabilities reuse normalized Pauli expectations across outcomes
of each channel. The cache is state-local and discarded immediately. Channel
Gram decomposition retains rare terms instead of applying the physical gate
operator tolerance. Parent norm multiplication/division is avoided. Sampled
positive branches use zero normalization floor, preserving the ordinary
forced-projector floor. This fixes a reproduced failure at probability 1e-30;
importance weights, normalized states and the recorded physical probability
are checked together.

## Upstream audit

Installed: Pepsy 0.5.0, Quimb 1.15.1.dev66+ge927f06e1, Autoray
0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f, Symmray
0.4.1.dev8+gc45f91457, Stim 1.16.0 and Torch 2.9.1.
Installed Autoray `get_namespace(like=None, device=None, dtype=None,
submodule=None)` and Quimb `gate_with_submpo(..., method='direct', ...,
**compress_opts)` signatures were inspected. No dependency or compression
policy was changed.

- **Adopt:** existing public allocator queries, shared MPS scheduling and
  Autoray backend reductions; retain Quimb coefficient compression ownership.
- **Defer:** ordinary MPS GPU gate/SVD and cross-parent Kraus batches for STN.
  Their dense local-gate assumptions do not describe a physical STN gate after
  tableau mapping. STN falls back explicitly to its frame path; batch sizes
  remain one. No equivalent GPU throughput or unavailable device validation
  is claimed.
- Sources checked: [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
  [Autoray source](https://github.com/jcmgray/autoray),
  [Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
  [changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
  [Symmray source](https://github.com/jcmgray/symmray). The Symmray abelian-array
  documentation returned an internal error; installed source and official
  repository served as fallback.

## Validation

- Final shared-dispatch domain selection: **464 passed, 38 skipped**, covering
  trajectory/importance/MPS memory and continuation, stabilizer compression
  and available backend paths, public API and package layout.
- Dedicated parity suite after MPI-policy and reversed-support additions:
  **39 passed, 2 skipped**. NumPy and actual Torch CPU complex64/128 pass;
  two CUDA cases skip because CUDA is unavailable. The MPI worker test is
  a dispatch unit test, not a real multi-rank accelerator run.
- Stabilizer suite plus the earlier parity selection: **394 passed, 3 skipped**.
  These selections overlap; counts must not be added.
- Final backend-signature selection: **9 passed**, verifying shared accelerator
  classification and explicit dispatcher/worker behavior.
- Full-suite attempt with Agg/`--maxfail=2`: **1017 passed, 12 skipped, 2 failed**.
  A clean exported `7a01b05` reproduces both Hamiltonian backend failures and
  the same counts (`SVD_real requires a real Torch tensor`). The initial
  non-headless run aborted in the macOS plotting backend. See the
  [session journal](../../../history/2026-10-05-stabilizer-trajectory-parity.md).
  This is not a clean full-suite result.
- Ruff and `git diff --check` passed.

This improves shared execution robustness and API parity. Frame-specific GPU
batching still needs a separate implementation and hardware validation.
