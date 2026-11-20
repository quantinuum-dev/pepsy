# 2026-11-19 — Default cached canonical MPO DMT updates

- Scope: default canonical reuse and maintain `info_c` carefully and
  efficiently across updates/calls.
- Branch / baseline: `develop`, `04a712435391712b47ad42777bdb4829d9dd7519`.
- Status: uncommitted. Earlier 4×4/6×6 work is preserved. No commit, push,
  package upgrade, installed-library edit or numerical-driver change.

## Changes and evidence

`MpoOptimizer.run(dmt_update="canonical")` retains owned canonical cores
and trace environments across matching calls. Existing adjacent/routed
algorithms stay in place. One-site payloads act exactly at their tracked
center, retaining density nonzero-trace and finite-data requirements without
a full sweep. `dmt_update="full"` is the explicit reference.

`info_c` keeps dictionary identity across commits, reinitialization and
rollback; scratch targets use separate info. Strong references and
state/tensor/array, layout/isometry, shape, dtype/device, exponent, policy and
available version tokens guard reuse. Raw NumPy writes require
`sync_canonicalization()`; no content hashing or implicit numerical isometry
scan is used. Manual DMT compression can seed matching replay. Changing only
the gate queue preserves the cache.

[Record and benchmark](../docs/development/notes/2026-11-19-dmt-canonical-cache.md):
L16/chi32 complex128 onsite passes fall from median 0.20784 to 0.01870 s
after untimed preparation, with 80 versus zero timed full sweeps. Independent
product-mixture triple-RDM errors are 8.07e-15 and 1.14e-15. This workload's
11.1× ratio is not a general entangling-dynamics claim.

## Validation and limits

Broad MPO selection: 319 passed, eight skipped, four existing warnings.
Focused checks additionally cover lower chi and non-DMT option validation.
The final focused cache selection has 31 passes. Public API/layout checks
have 54 passes and the known installed-version mismatch (0.5.0 versus
checkout 0.6.0); it was not repaired through an unrelated environment change.
Coverage includes actual lossy local preservation, canonical Gram identities,
invalidation/rollback, stable info ownership, Torch complex128 gradients and
JAX complex128 replay. Final details are in the record. No full suite, GPU,
global variational solver or fix of chi192 failures is claimed.

Upstream audit refreshed against unchanged installed dependencies; the
Symmray array-doc URL was unavailable. Existing unrelated installed-version
and backend-gradient baseline failures remain separate from this work.
