# Implementation plans

These documents record proposed work and implementation progress. Some are
historical or deferred; check each plan's status against the current code and
tests. For supported behavior, use the [API guides](../../api/index.md).

## Choose a route

| Need | Owner |
| --- | --- |
| Current public behavior | [API guides](../../api/index.md), implementation and focused tests |
| Cluster/downstream capability and validation | [Current status ledger](../cluster_optimization_status.md) |
| Reusable task instructions | [Maintainer router](../../../.github/skills/pepsy-maintainer/SKILL.md) and the selected domain skill |
| Package workstream history | [Project roadmap](project.md), with its stated historical baseline |
| Deferred native fermionic PEPO design | Root [plan.md](../../../plan.md); this remains a scoped proposal |
| Plan writing | [Markdown policy](../markdown_policy.md) |

Keep each new or revised plan compact: status/baseline, scope, intended result,
next steps, validation and owning references. Move long derivations to an owning
note and record completed measurements in the session journal. Plan status is
separate from implementation and validation evidence.

## Domain plans

- [Project roadmap](project.md)
- [Operator and exponential API plan](operator_api.md)
- [Lightweight package plan](lightweight-package.md)
- [Optimizer plan](optimizers.md)
- [SymDMRG2 matvec fanout-GEMM plan](symdmrg_matvec_fanout.md)
- [PEPS optimizer plan](peps_optimizer.md)
- [TreePeps and tree-embedded PEPS plan](tree_peps.md)
- [Tree FIT path traversal plan](tree_fit_path_traversal.md)
- [Tree optimizer consolidation plan](tree_optimizer_consolidation.md)
- [PEPS direct sampler plan](peps_sampler.md)
- [BP-guided regional CTMRG/RG plan](regional_ctmrg_rg.md)
- [Sweep optimizer plan](sweep_optimizer.md)
- [Stabilizer tensor-network plan](stabilizer_tn.md)
