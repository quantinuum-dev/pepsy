# 2026-11-05 — Review for publication of FU and local-cluster updates

User authorized committing and pushing Pepsy and the sibling Pepsy examples.
Pepsy branch: `develop`; baseline: `ee22198`. This entry accompanies the
package publication commit. Earlier journals retain their original commit
status and validation scope.

Reviewed the existing FU factor reuse, guarded boundary starts, local SU
gauge extraction, centered PEPO patches, accelerator preservation and
operand-order cache changes. Corrected the BP guide's remaining half-weight
description to match the full exterior-weighted split. No numerical behavior
was changed during publication review. Existing frozen simulation sources
and running jobs were left untouched.

Fresh validation used Python 3.12 and one numerical thread, with CUDA hidden:

- Nine affected suites (`test_bp_reduced_update`, `test_bp_compression`,
  `test_bp_cluster_patches`, `test_bp_local_gauges`, `test_bp_symmray`,
  `test_optimize_peps`, `test_pepo_full_update`, `test_peps_full_update`,
  `test_prepare_boundary_inputs`): **721 passed, 54 skipped, 5 failed**.
- A clean `git archive ee22198` checkout reproduced all five failures:
  spinless and spinful U1/U1U1 long-range hopping exact readout, and U1/U1U1
  open-rho sweeps. Its native reduced-compression test also failed; that
  additional baseline failure passes with the reviewed changes.
- Full collection with `-o addopts='' --maxfail=1`: **251 passed, 14 skipped,
  1 failed**, stopping at the same baseline spinless hopping readout failure.
  The complete repository suite is therefore not green.
- Full `ruff check src tests` and whitespace checks passed. Downstream Eli
  integration: **88 passed, 10 skipped**; its saved-data plotting notebook
  executed into `/tmp`, and its report regenerated consistently.
- Default public-API/package smoke profile: **112 passed**.

Installed dependency versions match the prior upstream audit: Quimb
`1.15.1.dev101+g1b64353d9`, Autoray `0.11.1.dev14+g014a3f69a`, Cotengra
`0.8.3`, Symmray `0.4.1.dev25+gdc377f033`, Torch `2.14.1+cu126`.
Earlier CUDA checks remain historical evidence; this review does not claim
fresh accelerator validation or full-suite success. Temporary test logs and
the clean baseline copy live under `/tmp/pepsy-commit-review-20261105/`.
