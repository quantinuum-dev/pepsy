---
name: pepsy-maintainer
description: 'Plan, implement, review, and validate focused changes in the Pepsy tensor-network package. Use for cross-cutting Pepsy maintenance, package layout, public API, tests, documentation, CI, or branch-workflow tasks; route specialized numerical work to the matching domain skill.'
---

# Pepsy Maintainer

Use this as the entry point for work in `quantinuum-dev/pepsy`. It is a
router, not a replacement for the domain skills below.

## Startup and scope

1. Read the repository [`AGENTS.md`](../../../AGENTS.md) and inspect
   `git status --short --branch` before editing.
2. For any skill add/update/deprecation/removal, read the catalog
   [`SKILL_POLICY.md`](../SKILL_POLICY.md) before editing the skill tree.
3. Keep edits inside Pepsy. Do not change Tensy, Gaugy, examples, or sibling
   repositories as part of a Pepsy task.
4. Use canonical `pepsy.<domain>` namespaces and preserve the `develop` →
   `main` workflow. Never push, merge, release, delete data, or stage
   unrelated changes without explicit approval.
5. Read only the domain skill(s) needed for the task. Keep focused tests,
   Ruff, and the full suite proportional to the change; report anything not
   run and any remaining risk.

## Domain routing

- MPS replay, layouts, or canonicalization → [`mps-optimizer`](../mps-optimizer/SKILL.md)
- FIT sweeps, variational compression, active rank growth, or FIT profiling → [`tensor-fitting`](../tensor-fitting/SKILL.md)
- Tree replay, TTN layout, trajectories, or measurement → [`tree-optimizer`](../tree-optimizer/SKILL.md)
- Stabilizer tensor networks → [`stabilizer-tensor-networks`](../stabilizer-tensor-networks/SKILL.md)
- Tree stabilizer simulation → [`tree-stabilizer-optimizer`](../tree-stabilizer-optimizer/SKILL.md)
- Symmetry-conserving DMRG2 → [`symdmrg2`](../symdmrg2/SKILL.md)
- Belief propagation or loop/PNE methods → [`belief-propagation`](../belief-propagation/SKILL.md)
- Direct PEPS sampling, conditioned boundary MPS, or PEPS proposal batching → [`peps-sampler`](../peps-sampler/SKILL.md)
- Torch/NetKet/JAX variational Monte Carlo → [`pepsy-vmc`](../pepsy-vmc/SKILL.md)
- Fermion operators, Symmray charges, or fermionic gates → [`pepsy-fermion-operators`](../pepsy-fermion-operators/SKILL.md)
- qMERA energy optimization → [`qmera-energy-optimizer`](../qmera-energy-optimizer/SKILL.md)

When a change crosses domains, read the maintainer skill first and then the
smallest set of domain skills that own the affected invariants. Do not copy
their rules into this router.

For skill-catalog work, run
`python .github/skills/pepsy-maintainer/scripts/validate_catalog.py` after
the individual skill validator and before reporting completion.

## Handoff

Use [`.github/skills/agent-bundle.yaml`](../agent-bundle.yaml) as the
canonical list of files for a Workspace Agent. In a local checkout, the
relative links above are authoritative. In Agent Studio, preserve the
`references/skills/<name>/SKILL.md` target paths when uploading reference
copies so the same routing remains readable.

Finish with changed files, branch, focused tests, Ruff/full-suite status,
commit or push status, and remaining risks.
