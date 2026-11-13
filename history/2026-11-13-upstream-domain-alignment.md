# 2026-11-13 — Symmray/Quimb cross-domain development audit

- Scope: user-requested latest-development compatibility review of VMC,
  fermionic PEPS/PEPO/MPS, trees, TreeMPO, and Pauli/operator paths.
- Branch / baseline: `develop` / `1d0a894`.
- Commit status: uncommitted working-tree changes; nothing staged or pushed.
- Preserved the existing gate/FIT repairs and unrelated full-update changes.

## Changes and findings

Installed Symmray `20035beb7` and Quimb `3371ff40c` match GitHub `main` at
review time. Autoray remains `7a474cab3`, Cotengra `646239731`. No dependency
was installed or modified. Read the Docs Symmray pages were unavailable;
official GitHub documentation and installed implementations supplied the audit.

The [detailed record](../docs/development/notes/2026-11-13-upstream-domain-alignment.md)
contains exact revisions, upstream links, API/capability findings, validation
selections, and prototype limits.

- Fixed [`bp/series.py`](../src/pepsy/bp/series.py): native rho corrections
  with no blocks contribute zero trace without attempting dense conversion.
  This resolves both BP failures reported in the
  [preceding repair handoff](2026-11-13-native-fermion-gates-fit-fixes.md).
- Expanded the existing public open-rho sweep regression to `Z2Z2`; U1,
  U1U1, Z2, and Z2Z2 all pass. Updated the BP API and changelog.
- Existing capability adapters use current positive fermionic messages,
  native inverse/eig, modern boundary signatures, and seeded SRC correctly.
- Eight small packed CPU MPS prototypes retain packed arrays and match the
  sparse reference in direct and one-/two-/three-site DMRG modes. This does
  not certify packed VMC, tree/PEPO workflows, gradients, or performance.

## Validation

- Six domain selections: **854 distinct passing cases**, after removing the
  shared BP/compatibility cases. Detailed per-command counts are in the audit.
- 25 tree hardware/configuration skips; 31 MPS/PEPO slow/large cases deselected.
- Torch compiled/vmap/gradient checks: 14 passed, included above.
- JAX/NetKet JIT, flat arrays, boundary methods, and complex derivatives pass.
- Expanded open-rho focused rerun: 4 passed.
- Default smoke profile: 112 passed.
- Ruff, touched Markdown local links, and `git diff --check`: passed.

Tests used local source in the shared Python 3.12 environment, one CPU thread
per numerical library, `CUDA_VISIBLE_DEVICES=''`, and `JAX_PLATFORMS=cpu`.
Logs: `/tmp/pepsy-upstream-audit-*.log`. Probe source/results:
`/tmp/pepsy_upstream_alignment_probe.py` and corresponding `.json`.

## Limits

No full-suite, CUDA, multi-rank, or production convergence claim. The separate
TreePeps/TreePepo constructors remain dense in the checked workflows; native
TTN/TreeMPO support must not be confused with that separate representation.
Native Symmray full-PEPO fitting remains explicitly unvalidated. Existing
TreeFIT odd-parity restrictions remain. NumPy 2.5 shape-assignment warnings
come from upstream Quimb optimization packing and remain visible.

## Publication preparation

The user subsequently authorized committing and pushing the pending Pepsy
work, including the earlier full-update changes. Before publication, the
focused full-update, PEPO, gate-order, smart-order, and sweep-default selection
passed 255 tests with 63 skips in the same CPU environment. Ruff and staged
diff checks passed. The fetched `origin/develop` matched baseline `1d0a894`.
The native repairs, full-update changes, and their existing audit/handoff
records are included together in the maintenance commit.
