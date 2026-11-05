# 2026-11-05 — Current cluster-update usage routes

User requested clear usage documentation and commits/pushes in Pepsy and
Pepsy examples. Pepsy branch: `develop`, baseline `9e57d5e`; the companion
example documentation starts from `0a46133` on `main`.

- Root and operator guides now point to the centered six-site gate API.
- The BP guide links the maintained downstream command and distinguishes
  local gate updates from its global trace readout policy: boundary MPS,
  scalar cap 3χ, per-pair RMS-to-exponent bookkeeping.
- The status ledger names the published fix and uses the recorded MPS-cap18
  χ6 estimate, 0.0516255525, for routine comparison. Historical journals and
  exact-readout diagnostics retain their original scope.
- Corrected two moved symmetric-tensor example paths. The example runbook
  separates current corrected sources from historical snapshot/patch replay.

Documentation-only validation: current example command passed allocation-free
preflight with 23 spins, χ6 and MPS cap18; Markdown source links/anchors,
skill catalog and whitespace checks passed. No new numerical test suite or
simulation was run. The prior publication review records numerical validation
and the five reproduced baseline fermionic readout failures.
