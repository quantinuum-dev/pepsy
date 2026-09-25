# 2026-09-24 — Mixed diagonal gate-stream compaction

- Scope: improve MpsOptimizer exact-batch replay speed by compacting gate streams while keeping full-state memory bounded.
- Branch / baseline commit: develop, 44c931a (local branch was two commits ahead of origin/develop).
- Commit status: committed locally in this change; not pushed.

## What changed

- Consecutive diagonal RZ/Z-like one-qubit and RZZ/ZZ-like two-qubit gates now compact repeated supports before one grouped phase pass. Reversed ZZ endpoints merge, mutable gates are re-inspected on every replay, and non-diagonal or control boundaries remain barriers.
- Fixed the two-class mask ordering for interleaved diagonal gates. Added a GPU cost gate so short pure-RZ runs keep the faster bounded broadcast route.
- Updated the [MPS API guide](../docs/api/optimizers/mps.md), [implementation map](../docs/development/modules/optimizers.md), [MPS skill](../.github/skills/mps-optimizer/SKILL.md), and changelog. Detailed algorithm, upstream audit, and measurements are in the [exact-batch note](../docs/development/notes/mps_exact_batch.md#mixed-zzz-stream-compaction-2026-09-24).

## Validation

- Focused exact-batch suite: 27 passed on the active NumPy/CuPy/Torch/Symmray environment.
- Adjacent MPS controls, Quimb compatibility, public API, and package layout: 106 passed, one inherited JAX complex64 direct-mode Kraus-probability precision failure. The failure was reproduced on the remote baseline in the [prior handoff](2026-09-24-mps-exact-batch-rebase.md).
- Ruff, MPS skill quick validator, catalog validator, and git whitespace check passed. Full suite was not run.
- The 102-gate mixed RZ/RZZ 4x5-grid spot check used one state pass rather than five bounded blocks. Warm CuPy application at 22 qubits measured 0.307 ms versus 0.523 ms; CPU at 20 qubits measured 5.0 ms versus 15.2 ms in one short run. Planning plus application also favored compaction. Pure-RZ GPU runs at 17-22 qubits favored bounded broadcast and now use it.

## Decisions and remaining work

- Stream compaction is opt-in through exact-batch and adds no state-sized diagonal or persistent plan cache. The grouped kernel still needs one output state array.
- General diagonal two-qubit gates, many phase value classes, commuting RXX/RYY basis transforms, immutable plan reuse, in-place state mutation, and cuStateVec integration remain deferred pending workload-specific speed, memory, and ownership evidence. No claim of global optimality is made.
- Preserved the separate uncommitted tree-layout files and their changelog entry outside the MPS commit.
