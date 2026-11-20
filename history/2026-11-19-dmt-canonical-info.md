# 2026-11-19 — Consistent canonical metadata ownership

- Scope: clarify existing MPO `info_c`, give each Pauli object its own info,
  and apply canonical reuse consistently without merging storage classes.
- Branch/baseline: `develop`, `04a712435391712b47ad42777bdb4829d9dd7519`.
- Status: uncommitted; prior working-tree research/cache changes preserved.

`MpoOptimizer` already owned `info_c`; earlier work extended its reuse.
`PauliMPO` and `PauliDMT` now own separate stable dictionaries using the
same `cur_orthog` convention and `sync_canonicalization(site)` interface.
Native QR/SVD/DMT establish actual center records; copies/snapshots have
independent metadata, managed changes invalidate it, and symbolic metadata
access compiles no cores. Shared cheap array tokens bind native reuse to
actual cores and available version counters. Known-center transport does
only the required QR steps; DMT initialization retains a validated gauge.

Driver metadata refreshes after partial errors. Failed explicit recovery
after raw state edits leaves unknown metadata, including the ordinary MPO
owner. Exact driver sync retains DMT reports and physical time. Existing
symbolic `canonicalize()` meaning is unchanged. Automatic native Pauli input
to `MpoOptimizer` is not implemented; explicit `to_mpo()` remains available.

[Contract and checks](../docs/development/notes/2026-11-19-dmt-canonical-info.md):
31 metadata checks pass, including canonical Grams, independent copies,
mutation/failure behavior, Torch versions and JAX complex128 JIT/gradients.
Broad selection: 454 passed, one skipped, nine existing warnings. Final
ordinary-MPO canonicalization/public-API validation is recorded in the note.
The former has six passes; the latter has 54 passes and the existing
installed-version mismatch (0.5.0 versus checkout 0.6.0). Ruff, catalog,
local links and whitespace checks pass.
No full-suite/GPU/native-fault-fix or package-upgrade claim is made.
