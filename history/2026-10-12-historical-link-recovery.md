# 2026-10-12 — Recover historical example links

- Scope: user-authorized commit/push and remaining Markdown cleanup.
- Branch / baseline: `develop` / `793c088`.
- Publication: the [documentation reorganization](2026-10-12-markdown-organization.md)
  was committed as `793c088` and pushed successfully to `origin/develop`.
  This record accompanies the authorized recovery commit; push results are
  checked after committing.

## Recovered navigation

Replaced 29 missing local link destinations in dated notes and journals with
immutable GitHub file links. All 12 target artifacts exist in published Git
history; their containing commits are ancestors of the corresponding remote
branch. Only link destinations changed in those records: prose, formulas,
measurements, validation limits and original publication statements are preserved.

| Historical assets | Verified snapshot |
| --- | --- |
| Five Pepsy cluster examples/benchmarks | `5316cb03c5778cb116432539724231b864567e90`, before removal in `50d319e` |
| Pepsy PEPS-sampling benchmark | `6486a616c1fc14290cd11fac6f077b1a029becc5`, before removal in `b97297b` |
| Gaugy Examples explicit 1D demos, guide, diagrams and joint-notebook test | `af9ec7317bbc3d1d4f6d5242a1c88dd283bc00c5` |
| Retired joint-gauge optimization notebook | `96e274d4cbe7c2f5407b3629ed81f515641a3ef0`, before removal in `d3e1a26` |

These snapshots preserve access to historical source; they do not certify
compatibility with today's package APIs or reproduce every earlier workspace.
The benchmark commands in those records describe their original checkout.
Current workflows start from the [API guides](../docs/api/start_here.md).

## Validation and remaining scope

The complete audit covers **723 Markdown pages and 1,443 local links**, with
zero missing files, anchors or filename case errors. Git object checks verify
every recovered file and its published ancestry. Body comparisons verify that
all 19 affected older records changed only their link destinations. Whitespace
checks pass.

No numerical source, tests, dependencies, notebook output or device-local policy
changed. Numerical tests and remote HTTP checks were not run for link repairs.
Deferred numerical proposals remain outside this documentation cleanup.
