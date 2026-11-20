# Canonical metadata ownership across MPO representations

**Adopt:** `PauliMPO` and `PauliDMT` now expose independent stable `info_c`
dictionaries, following the existing `MpoOptimizer` convention. Earlier
[cache work](2026-11-19-dmt-canonical-cache.md) extended that optimizer's
existing metadata lifetime and reuse; it did not invent the field.

## Ownership and behavior

| Owner | Metadata meaning |
| --- | --- |
| `MpoOptimizer` | Its live ordinary MPO gauge; scratch targets have separate info |
| `PauliMPO` | Its native Pauli-coefficient gauge, unknown on raw construction |
| `PauliDMT` | Its owned persistent-workspace gauge |

All use `info_c["cur_orthog"]`: a Quimb-style `(lo, hi)` span, normally
`(center, center)`, or `None` for unknown. Each offers
`sync_canonicalization(site)` returning the resulting center pair. A raw
Quimb MPO itself uses caller-supplied `info`; no custom field is installed
on upstream classes.

Native QR, non-flat SVD compression and DMT record their established center.
Copies and driver snapshots retain the gauge with independent dictionaries
and storage. Managed gates, channels and damping invalidate native metadata
when isometries may change. Raw buffer writes still require synchronization;
failed explicit rebuilding of an untrusted state leaves an unknown claim.
Editing the dictionary alone cannot assert a gauge.

`PauliMPO.canonicalize()` keeps its symbolic-word meaning for sparse inputs.
Native gauge preparation remains explicit through `canonicalize_native()`.
Unknown symbolic metadata access compiles no tensor train. When the native
gauge is known and still matches its core objects/version tokens, QR transport
visits only the required center distance; a same-center in-place call does
no QR or array-copy work. Native DMT initialization can keep this validated
center instead of returning through a full preparation sweep. Density trace
resolution and norm checks now use the actual center.

The array metadata token helper is shared with ordinary MPO replay. DMT still
uses the same protected split; storage/basis differences are separate from
that numerical algorithm. Persistent Pauli-driver metadata refreshes on both
success and partial numerical failure. Exact synchronization retains the
latest DMT warning/report evidence and leaves physical time unchanged.

## Optimizer boundary

The existing strong ordinary-MPO optimizer can already use
`MpoOptimizer(op.to_mpo(), ..., chi=..., mode="dmt")`. That is an explicit
matrix-unit conversion. This change does not add automatic native-Pauli
dispatch into that optimizer or merge the two storage classes. A native
storage adapter would be separate work; metadata and the protected DMT
kernel are already consistent without duplicating the solver mathematics.

## Evidence

- 31 new metadata checks pass: owner/copy isolation, canonical Gram identities,
  native compression forms, mutation invalidation, raw-buffer recovery,
  partial failures, and Torch version tracking.
- A seven-site center move 1→4 uses exactly three QR calls, 4→2 uses two,
  and an in-place same-center request uses zero. Known-gauge DMT construction
  and an already-centered gate need no initial canonical transport.
- A 256-site sparse operator exposes unknown metadata without compiling cores.
- JAX complex128 JIT and gradients through cached native QR moves agree with
  an independent trace derivative. Existing backend/DMT gradients remain in
  the broad selection.
- Broad Pauli/MPO selection: 454 passed, one skipped, nine existing warnings.
  The final two failure-sync cases are included in the 31 focused checks.
- Ordinary-MPO canonicalization/rollback selection: six passed. Public
  API/package-layout checks: 54 passed; the known installed-version mismatch
  remains (installed 0.5.0 versus checkout 0.6.0). Ruff, skill catalog, local
  links and whitespace checks pass.

Installed dependencies and upstream audit remain those of the preceding
cache record; no driver registration, dependency upgrade or compatibility
shim changed. Full-suite/GPU coverage and prior chi192 fault resolution are
not claimed. All work remains uncommitted.
