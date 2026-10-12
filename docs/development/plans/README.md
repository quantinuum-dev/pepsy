# Implementation plans

This is Pepsy's single planning index. Domain plans live in this directory
under descriptive lowercase filenames; root `plan.md` and `PLAN.md` are retired.
For supported behavior, use the [API guides](../../api/index.md) and current code.

## Choose a route

| Need | Owner |
| --- | --- |
| Public behavior and compatibility | [API guides](../../api/index.md) and [stability policy](../../stability.md) |
| Cluster/downstream capability and validation | [Current status](../cluster_optimization_status.md) |
| Reusable task instructions | [Maintainer router](../../../.github/skills/pepsy-maintainer/SKILL.md) and its domain skills |
| Package workstream history | [Historical project roadmap](project.md), snapshot 2026-09-14 |
| Deferred native fermionic PEPO design | [Native fermionic PEPO proposal](native_fermionic_pepo.md) |
| Plan format | [Markdown policy](../markdown_policy.md) |

## Domain plans

Each document records its own scope and baseline. Implementation statements
inside an older plan describe that record, not a fresh package-wide validation.

| Area | Design or progress record |
| --- | --- |
| Operators and exponentials | [Operator API](operator_api.md) |
| Native fermionic PEPO construction | [Deferred proposal](native_fermionic_pepo.md) |
| Package and import costs | [Lightweight package](lightweight-package.md) |
| Boundary providers and sweeps | [Optimizer boundaries](optimizers.md), [PEPS split](peps_optimizer.md), [sweep providers](sweep_optimizer.md) |
| PEPS direct sampling | [Direct sampler](peps_sampler.md) |
| Tree networks | [Tree PEPS](tree_peps.md), [FIT traversal](tree_fit_path_traversal.md), [optimizer consolidation](tree_optimizer_consolidation.md) |
| Symmetric DMRG | [Matvec fanout](symdmrg_matvec_fanout.md) |
| Stabilizer simulation | [Stabilizer tensor networks](stabilizer_tn.md) |
| Regional CTMRG/RG | [Deferred regional design](regional_ctmrg_rg.md) |
| Historical package workstreams | [Project roadmap](project.md) |

## Maintain a plan

State status, baseline, scope, intended result, next steps and validation.
Link API contracts, measurements and long derivations in their owning guides
or notes. Update an existing domain plan instead of adding a competing overview.
Preserve superseded designs and completed journal evidence with their dates.
Plans describe work; the user's scope authorizes its implementation.
