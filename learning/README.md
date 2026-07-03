# learning/ — concepts & design notes

Conceptual documentation for the in-progress workstreams in `PLAN.md`. These
notes explain *the ideas and the intended implementation* — the "why" behind the
code — so that contributors and agents share a mental model. They are not
auto-generated API docs (those live in `docs/api/`).

Keep these in sync as the design firms up; when something is finalized and
user-facing, promote it into `docs/` (tutorials / how-to).

## Contents

- [`bp.md`](bp.md) — Belief propagation for tensor networks, BP gauging, and the
  loop series / loop cluster expansion corrections.
- [`quimb.md`](quimb.md) — How pepsy leans on `quimb` (and `cotengra`,
  `autoray`); the "pepsy concept → quimb API" map.
- [`symmray.md`](symmray.md) — Block-sparse abelian-symmetric and fermionic
  tensors via `symmray`, and how they bridge into pepsy's tagging conventions.
- [`fermionic_mpo.md`](fermionic_mpo.md) — Fermionic Fermi-Hubbard MPO
  conventions, the Symmray/Jordan-Wigner bridge, and current validation status.

## Relationship to other docs

- `PLAN.md` — the roadmap (what/when).
- `learning/` — the rationale (why/how it should work).
- `history/` — the journal (what happened each session).
- `docs/` — the finished, published documentation.
