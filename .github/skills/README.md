# Pepsy agent skills

Each skill is self-contained in one directory with a required `SKILL.md` and
optional `references/` and `agents/` subdirectories.

Start with [`pepsy-maintainer/SKILL.md`](pepsy-maintainer/SKILL.md) for a
cross-cutting task. The repository-level
[`agent-bundle.yaml`](agent-bundle.yaml) is the upload map for the Pepsy
Maintainer Workspace Agent; it separates stable references, core skills, and
optional domain skills. Keep the skill directories flat and upload
`SKILL.md` plus any `references/**`; `agents/openai.yaml` is local UI metadata,
not ordinary agent context.

Read [`SKILL_POLICY.md`](SKILL_POLICY.md) before adding, renaming, merging,
deprecating, or removing a skill. It defines the selection workflow, package
contract, lifecycle, and quality gate. Run the catalog validator after any
catalog or skill-package change.

## Core workflows

- [MPS optimizer](mps-optimizer/SKILL.md)
- [Tree optimizer](tree-optimizer/SKILL.md)
- [qMERA energy optimizer](qmera-energy-optimizer/SKILL.md)

## Domain workflows

- [Belief propagation](belief-propagation/SKILL.md)
- [Direct PEPS sampling](peps-sampler/SKILL.md)
- [Fermion operators](pepsy-fermion-operators/SKILL.md)
- [Variational Monte Carlo](pepsy-vmc/SKILL.md)
- [Stabilizer tensor networks](stabilizer-tensor-networks/SKILL.md)
- [Tree stabilizer optimizer](tree-stabilizer-optimizer/SKILL.md)
- [SymDMRG2](symdmrg2/SKILL.md)

When adding a skill, follow the same layout and add it to this catalog. Keep
shared repository rules in `AGENTS.md`; keep skill-specific procedures here.

All user-facing skills also carry `agents/openai.yaml` metadata so the catalog
and invocation chips stay consistent. The maintainer router and Tree
Stabilizer skill compose the domain skills rather than duplicating their
shared invariants.
