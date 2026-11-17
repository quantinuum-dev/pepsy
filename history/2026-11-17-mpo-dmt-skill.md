# 2026-11-17 — MPO DMT mathematical skill

- Scope: add a repository DMT skill recording dimensions, local-density
  constraints, the full one-bond optimization, and the paper's actual SVD rule.
- Branch / baseline commit: `develop`, `328979f4`.
- Commit status: uncommitted; nothing staged or pushed. Existing sampling,
  test-harness, changelog and historical working-tree changes were preserved.

## What changed

- Added [`mpo-dmt`](../.github/skills/mpo-dmt/SKILL.md), UI metadata,
  a [mathematical reference](../.github/skills/mpo-dmt/references/mathematics.md),
  and a [deterministic probe](../.github/skills/mpo-dmt/scripts/check_math.py).
- Registered the domain skill in the catalog, maintainer router and upload
  manifest, including its executable mathematical checks.
- Recorded $A=U_cM'V_c^\dagger$, the full constrained Frobenius objective,
  the three physical/virtual legs of each computational trace map, transpose
  conventions, protected L-shape, connected residual and conservative rank bound.
- Distinguished feasibility, the restricted SVD optimum, and the full
  fixed-basis completion optimum. Included an invertible-corner mathematical
  solution and a counterexample to unrestricted optimality; neither adds a
  new solver or changes production DMT behavior.

## Validation

- Skill creator `quick_validate.py`: passed.
- Pepsy skill catalog validator: passed, 13 skills.
- Ruff on the added mathematical probe: passed.
- `check_math.py --check-kernel`: passed complex six-site trace identities,
  both operator bases, embedded/compressed dimensions, rank and tail norms,
  invertible-corner completion, and the current kernel counterexample.
  Maximum three-site defect in that dense-reference probe: $2.52\times10^{-15}$.
- Reviewed representative dimension, optimality and mapped-region/replay
  requests against the skill's routing and equations.
- Local links: 64 resolved; UI metadata and executable upload entry validated.
  `git diff --check` passed. No package behavior changed; no package-wide
  numerical suite was run.

## Mathematical scope

The paper's connected free-block SVD minimizes a restricted residual-rank
family. The current kernel reserves `max_bond - p - q` free directions.
For $M=\operatorname{diag}(1,0.3,0.1)$, $p=q=1$, and cap two, its output
has squared error $0.10$, while a different constraint-satisfying rank-two
completion has error $0.01$. This demonstrates the optimum's scope and does
not invalidate local-density preservation. The reference documents the
special-case exact completion separately from Pepsy's implemented algorithm.
