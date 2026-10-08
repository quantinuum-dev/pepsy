# PEPS gate-by-gate cache and two-site update handoff

Branch develop, baseline 0ba2a05; pre-existing edits preserved. Changes remain
uncommitted. Examples were not edited. See the
[implementation and validation note](../docs/development/notes/2026-10-08-peps-boundary-reuse-full-update.md)
and [public API](../docs/api/optimizers/peps.md).

Implemented explicit update scope and safe commuting-gate traversal, checked
boundary reuse in both directions, local cut invalidation, and optional
QR/LQ/PSD two-site full update through existing Pepsy and Quimb ALS. User
confirmed row/column and two-site choices, without a separate one-site mode.
The overlap handoff previously rebuilt because bra/ket orientation differed;
this is corrected. Local norm/overlap checking remains mandatory by default.

Final focused tests: 91 passed. Broader PEPS/API checks: 338 passed and the
existing installed-version metadata mismatch. Full suite interrupted after
403 passed and 23 skipped; Ruff absent. CUDA 3×3 reference comparison and bounded 5×6 D4
execution completed. The 5×6 cap64 probes did not converge: no accuracy claim.
Calibration remains the measured bottleneck (25.15 of 26.64 seconds).

User then requested stopping the current PEPS run and returning later. Sent
SIGTERM to verified launcher 935204 and worker 935589, confirmed both exited
and GPU allocation cleared. Results preserved under
`/tmp/pepsy_examples_runs/peps_D4_sweep_dmrg_normtol_gpu1_20261007_221424`, with
`stopped_by_user.json`. Earlier CUDA:0/2 PEPS runs were already stopped.
CUDA:3 MPS worker 89592 remains running. Do not restart PEPS without a new
request. No commit or push was requested for this change.
