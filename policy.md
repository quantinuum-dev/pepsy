# Agent Code-Change Policy

This policy keeps the reasoning behind Pepsy changes available to future
developers and agents. Code should explain not only what it does, but also
why the chosen behavior is necessary when that reason is not obvious from the
implementation.

## Explain non-obvious decisions in the code

When an agent changes existing logic or adds new logic, it must check whether
a future maintainer could reasonably misunderstand the decision. If so, add
a concise comment next to the affected code that explains:

- the problem, invariant, or external constraint that motivates the behavior;
- why this implementation was chosen; and
- any important trade-off, limitation, or failure mode.

Comments should explain *why*, not merely repeat the syntax or restate the
function name. Do not add comments for code whose purpose is already obvious.

For numerical, tensor-network, backend, or performance-sensitive code, call
out relevant assumptions such as mutation and ownership, canonical or gauge
invariants, dtype or backend requirements, cutoff and tolerance behavior,
fallback paths, or stability safeguards. For a bug fix, describe the failure
mode and the reason the fix prevents it.

## Keep rationale close and maintainable

- Put the comment immediately above or beside the logic it explains.
- Use stable technical language; do not refer to a temporary conversation,
  prompt, or unnamed agent.
- Keep comments concise and precise, with enough context for someone who has
  not seen the original issue.
- Update or remove comments when the implementation changes so that comments
  never describe stale behavior.
- Prefer a design note in `docs/development/` when the rationale spans
  multiple modules or needs historical detail; link to it from the code when
  useful.

## Preserve evidence for behavior changes

When a code change alters observable behavior:

- add or update a focused regression test for the relevant invariant;
- document public API changes in the appropriate handwritten Markdown file;
- include important backend, dtype, seed, and tolerance assumptions in tests
  or documentation; and
- make the commit message describe the user-visible or correctness-related
  change.

Tests and documentation complement code comments: comments preserve local
reasoning, tests preserve executable behavior, and documentation preserves
the public contract.

## Review checklist for agents

Before handing off a change, an agent should verify:

1. Non-obvious logic has a nearby rationale comment.
2. The comment states the constraint or failure mode, not just the action.
3. The comment matches the final implementation.
4. A focused regression test exists when behavior or correctness changed.
5. Unrelated code and generated files were not modified.
