# Pepsy agent skills

Each skill is self-contained in one directory with a required `SKILL.md` and
optional `references/` and `agents/` subdirectories.

## Core workflows

- [MPS optimizer](mps-optimizer/SKILL.md)
- [Tree optimizer](tree-optimizer/SKILL.md)
- [qMERA energy optimizer](qmera-energy-optimizer/SKILL.md)

## Domain workflows

- [Belief propagation](belief-propagation/SKILL.md)
- [Fermion operators](pepsy-fermion-operators/SKILL.md)
- [Variational Monte Carlo](pepsy-vmc/SKILL.md)
- [Stabilizer tensor networks](stabilizer-tensor-networks/SKILL.md)
- [Tree stabilizer optimizer](tree-stabilizer-optimizer/SKILL.md)
- [SymDMRG2](symdmrg2/SKILL.md)

When adding a skill, follow the same layout and add it to this catalog. Keep
shared repository rules in `AGENTS.md`; keep skill-specific procedures here.
