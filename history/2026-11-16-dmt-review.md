# 2026-11-16 — DMT correctness review

- Scope: resume DMT work; user prioritized reviewing and fixing potential
  problems in the existing implementation.
- Branch / baseline commit: `develop`, `b2df2dba`.
- Commit status: working-tree changes only; nothing staged, committed or pushed.
  Existing sampling, test-harness, changelog and historical edits were preserved.

## Findings and changes

- **Fixed:** trainable Torch `ZZ` Hamiltonians produced NaN observable
  gradients through gate preparation's repeated-eigenvalue EIGH backward.
  Trainable Torch inputs now use the existing native matrix-exponential
  dispatch, retaining per-call shared-gate reuse. Other inputs retain EIGH.
- Eight independent dense-reference gradient cases cover real/complex double
  precision, zero/nonzero coupling, conjugate/Heisenberg evolution, and shared
  generators. The first case failed with NaN gradients before the fix; all
  eight pass afterward.
- Added per-update dense plaquette/trace checks for genuinely truncated
  three-timestep lattice evolution, with continuation across calls, for both
  density/conjugate and traceless-operator/Heisenberg policies.
- Verified that rejected persistent gates preserve the preceding physical
  state and leave canonical/trace caches usable for subsequent gates, for
  default chain and explicit lattice protection. These checks found no defect.
- Updated the public API guide and changelog. The
  [review note](../docs/development/notes/2026-11-16-dmt-gradient-review.md)
  records the environment, upstream audit, cause and adopted route.

## Validation

- Before changes: four focused DMT modules excluding slow tests, **133 passed**,
  four deselected, seven rank-deficient Torch QR warnings.
- After the fix: `tests/test_pauli_dmt_optimizations.py`, **17 passed**, one
  rank-deficient Torch QR warning. The four added lattice/recovery cases also
  passed separately.
- Five DMT modules (`test_pauli_dmt.py`, `test_pauli_dmt_cutoff.py`,
  `test_pauli_dmt_lattice.py`, `test_pauli_dmt_optimizations.py`,
  `test_pauli_dmt_large.py`), with `-o addopts='' -m 'not slow'`:
  **167 passed**, 23 slow tests deselected, seven existing rank-deficient
  Torch QR warnings.
- `python -m ruff check src tests` and `git diff --check` passed.

## Limits

- CPU checks only. No full-suite, GPU, or higher-order-gradient claim.
- Existing finite-precision limitations and conservative protected-rank
  rejection remain; this fix concerns Hamiltonian gate derivatives, not
  differentiability of arbitrary rank-changing tensor factorizations.
- No new lattice gate scheduling or changes to truncation constraints.
