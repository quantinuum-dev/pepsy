# 2026-11-11 — direct full-update boundary default

On `develop` at baseline `1d0a894`, the user selected direct boundary
compression and CUDA `gesvd` after the bounded GPU comparison in
[the preceding handoff](2026-11-11-smart-full-update-gpu.md). Changes are
uncommitted. Pepsy full update now defaults to `fit_mode="direct"`; explicit
DMRG remains available. Switching run style also resolves automatic boundary
layer tags for the selected direct mode. Pepsy retains its existing public
Torch linalg policy; the Eli evaluator and profiler explicitly scope `gesvd`
for full update and restore the caller's policy afterward.

The Eli resolution queue records direct for new plans and respects the
recorded boundary mode when executing, including explicit `--boundary-mode
dmrg` for historical plans. Its direct/gesvd defaults, smart ordering and
source barriers are documented in the examples repository. Direct and DMRG
can produce different finite-cap approximations; the prior prefix and
synthetic benchmarks do not establish complete-evolution accuracy.

Validation: 290 Pepsy full-update, ordering and sweep-default tests passed;
36 Eli full-update/profiler tests, including a CUDA check that observed
complex128 `gesvd` and restoration of an outer `gesvdj` policy, passed; and
15 resolution tests passed. Ruff and `git diff --check` passed. An initial
broader Eli selection also reported 14 known SU failures from the installed
Quimb's missing `transpose` gate option; four DMRG-initializer tests were
subsequently corrected to request DMRG explicitly and passed. The earlier
full-suite fermionic failure and Quimb gate-option limitations remain.
