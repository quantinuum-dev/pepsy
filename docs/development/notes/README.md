# Design and research notes

Conceptual documentation for Pepsy workstreams. These notes explain the ideas
and intended implementation — the "why" behind the code — so contributors and
agents share a mental model. They are not auto-generated API docs (those live
in `docs/api/`).

Keep these in sync as the design firms up; when something is finalized and
user-facing, promote it into `docs/` (tutorials / how-to).

## Contents

- [`belief_propagation.md`](belief_propagation.md) — Belief propagation for tensor networks, BP gauging, and the
  loop series / loop cluster expansion corrections.
- [`quimb.md`](quimb.md) — How Pepsy leans on `quimb` (and `cotengra`,
  `autoray`); the "pepsy concept → quimb API" map.
- [`symmray.md`](symmray.md) — Block-sparse abelian-symmetric and fermionic
  tensors via `symmray`, and how they bridge into pepsy's tagging conventions.
- [`fermionic_mpo.md`](fermionic_mpo.md) — Fermionic Fermi-Hubbard MPO
  conventions, the Symmray/Jordan-Wigner bridge, and current validation status.

## Relationship to other docs

- [`../plans/project.md`](../plans/project.md) — the project roadmap.
- [`../modules/`](../modules/README.md) — concise implementation maps.
- `history/` — the journal (what happened each session).
- `docs/` — the finished, published documentation.

```{toctree}
:hidden:

belief_propagation
quimb
symmray
fermionic_mpo
higher_order_mpo_benchmarks
```
