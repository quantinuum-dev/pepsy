# 2026-10-05 — MPS trajectory publication preparation

- Scope: user requested committing and pushing the accumulated MPS trajectory
  corrections, automatic execution policy and GPU probability batching.
- Branch: `develop`; initial baseline `bec773a`, fast-forwarded to upstream
  `dec6960` before staging. Combined upstream and pending changelog additions.
- Status at writing: prepared for the user-authorized commit and push; Git
  history and the final session response record the resulting commit and push.

The publication includes MPS implementation, focused tests, API documentation,
research notes and preceding MPS handoffs. Unrelated MPO, solver, tree and
sampling edits remain outside this commit. Earlier handoffs retain their
historical uncommitted status.

Prior validation and remaining limitations are recorded in the
[GPU batching handoff](2026-10-05-mps-gpu-frontier.md) and its linked research.
Three original complex64 ledger comparisons remain failing: Torch CUDA with
both normalization settings and JAX with restored normalization. Public
configured-SVD checks pass; tolerances were not changed. No full-suite success
is claimed. Publication checks cover staged whitespace, scope and local links.

Fresh validation on an exported staged snapshot after the upstream fast-forward
(excluding all unrelated working-tree edits): Kraus batching, GPU frontier,
automatic execution, trajectory review regressions and public API/layout suites
completed with **202 passed**. All-source/test Ruff passed on that snapshot.
Staged whitespace and relative Markdown links also passed. This focused run
does not include the three known ledger failures described above.
