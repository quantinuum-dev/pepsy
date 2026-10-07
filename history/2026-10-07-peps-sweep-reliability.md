# 2026-10-07 — PEPS sweep reliability

- Scope: investigate and fix unresolved sweep behavior, including why larger
  boundary chi did not establish reliable results.
- Branch/baseline: develop / e8a3f96. This handoff accompanies the local
  commit requested by the user; no push was requested for this commit.
- Fixed: false convergence/best-state selection from clipped negative losses;
  outer sweep precheck bypass; false optimized status when all local updates
  are rejected; loss of raw rejected-slice diagnostics.
- Preserved: finite-D target construction, direct boundary compression,
  solver objectives, diagnostic clipping, and gate-stream continuation.
- Numerical evidence: exact moving-slice losses/gradients pass for both
  boundary implementations; a real D=4 4x4 fit improves exact fidelity, but
  chi=128 retains measurable contraction error. Chi=512 is substantially
  better on that case, not a universal recommended cap for 5x6.
- Details and evidence: [investigation](../docs/development/notes/2026-10-07-peps-sweep-reliability.md).
- Tests: final focused selection 284 passed, including exact boundary
  numerics and outer-precheck/higher-cap comparison regressions.
  Log: `/tmp/pepsy_sweep_reliability_final_tests.log`.
  Ruff and Pyflakes unavailable; `git diff --check` passed. The additional
  full suite was stopped after roughly 15 minutes at about 30%, with no
  reported failures up to that point. This is incomplete validation, not a
  full-suite pass. Log: `/tmp/pepsy_sweep_reliability_full_tests.log`.
- Runtime: the bounded old-policy diagnostic replay on GPU0 completed its
  twelve steps (t=1.2), without a large invalid local loss; it does not validate
  the patched implementation or late-time accuracy. This initial diagnostic
  preceded the user-requested production comparison described below.
- Production status checked during this task: all three corrected 5x6 exact
  sweeps (dt=0.1, 0.25, 0.05) finished with 12 completion records and 12 sample
  files each. The GPU3 MPS chi=4096 job remained active on angle 7/12.

## Subsequent live comparison and remaining limits

The user then authorized three 5x6 D=4 sweeps using the patched code: GPU0
direct chi=(64,64), GPU1 direct chi=(128,128), and GPU2 one-site DMRG (`eff`)
chi=(128,128). All use Torch complex128, auto batching, dt=0.1, depth=60,
12 angles, two round trips per axis, and SciPy L-BFGS-B maxiter=50. Separate
overlap checks and readout remain disabled; retained states are normalized.
Launch commands, source patches, PIDs, and checked effective settings are in
`/tmp/pepsy_examples_runs/peps_sweep_comparison_D4_20261007_192044/`.

At the subsequent status check, angle 1 had reached t=1.3, 1.2, and 1.7
respectively, with no rejected updates or failed fits. The direct runs each
contained two early local updates with clipped negative initial losses and
large positive final local losses (roughly 0.41--0.67). Best-state restoration
is in place, but these excursions remain a numerical concern, not something
this commit claims to solve. Finite-chi local objectives and products of
per-step fit scores do not establish exact global fidelity. Larger boundary
chi does not remove finite-D representation error. Full t=6 accuracy remains
unverified. Existing jobs continue unchanged by this commit.
