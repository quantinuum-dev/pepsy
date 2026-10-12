# 2026-10-07 — Batch full nearest-neighbor layers by actual bond growth

- User scope: implement the intended per-dt algorithm in Pepsy itself: apply
  the complete second-order gate stream, build a target within 2D for the
  nearest-neighbor ZZ layer, then fit once. Single-site gates must preserve
  their order and do not count toward the two-site gate budget.
- Branch/baseline: develop at `01c0d5e`. Existing objective-only sweep edits
  from the previous turn are retained. All current changes remain uncommitted.
- Root cause: auto batching stopped on shared endpoints before checking actual
  bond growth. On the 5×6 circuit, 49 ZZ gates became 25 batches, each running
  47 local solves. One dt cost 1,175 local solves and about 21 minutes.
- Fix: remove endpoint collision stopping and its private helper. Keep actual
  target bond budget checking, exact candidate construction/rollback, circuit
  ordering, diagonal 2D/general 4D budgets, and oversized-single-gate handling.
  Repeated/routed gates may still split a batch if actual bond growth requires
  it. This is not a universal claim that arbitrary layers fit in 2D.
- The frontend queues one dt at a time; auto can now absorb its full ZZ layer
  and trailing single-site gates before refinement. Leading single-site gates
  remain exact direct applications. Targets already inside D bypass fitting.
- Regression: dense NumPy/Torch second-order reference comparisons; mixed
  one-site/noncommuting gate order and physical-site aliases; real 5×6 D=4
  target reaching D=8 with all 49 edges and unchanged input; existing routed
  look-ahead/budget tests. The callback failure test now explicitly requests
  k_2q_batch=1, since two same-site gates no longer imply separate auto batches.
- Companion examples bump execution policy to 4, update CLI/docs, and verify
  the real SciPy/no-readout run performs one refinement for the complete layer.

## Live verification and restart

Stopped the old shared-site-batched cuda:0 PEPS parent/child 1410678/1410904 at
01:40:47 UTC. Saved results are preserved at the old root ending
`auto_scipy50_2roundtrips_chi32_56_no_checks_20261007_010307`; stop_record.json
records the completed t=0.3. Other GPU jobs were not stopped.

Bounded Torch complex128 CUDA0 probe: `/tmp/pepsy_full_layer_gpu_probe.py`,
log `/tmp/pepsy_full_layer_gpu_probe.log`, per-step records and SciPy counts
in `/tmp/pepsy_full_layer_gpu_probe/`. Production numerical settings and
thread policy, D=4, norm/overlap caps 32/56, direct boundaries, SciPy maxiter=50,
two round trips, no independent overlap checks. At t=0.1 and 0.2, one target
per step stayed within D and needed no refinement. At t=0.3, one D=8 target
contained all 49 ZZ gates; one sweep performed 47 real SciPy calls and returned
finite D=4 CUDA complex128 tensors. Step wall time 72.36 s, sweep 70.55 s.
This verifies the changed batching, not independently measured accuracy.

The t=0.4 probe also used one D=8 target, one sweep, and 47 SciPy calls,
returning finite D=4 tensors. Wall time 114.74 s (sweep 112.75 s). Across both
refinements, all 94 minimize calls received maxiter=50; the maximum observed
iteration count was 50. No invalid inner losses were recorded; small negative
approximate losses were retained/clipped under the existing continuation policy.

Validation: 268 PEPS batching/performance/safeguard/timing/optimizer tests
passed, 5 warnings, 54.13 s; 140 example PEPS/roughening-sweep tests passed,
11 warnings, 186.55 s. API/layout checks: 54 passed, one known installed
metadata mismatch (0.4.0 vs project 0.5.0), unchanged from earlier evidence.
Ruff is unavailable; diff checks pass. The same active-task upstream audit
was reused; dependencies are unchanged. Full unrelated package suite not run.
All 12 production cases passed dry-run validation. New production root:
`/tmp/pepsy_examples_runs/peps_sweep_5x6_D4_gpu0_full_layer_auto_scipy50_chi32_56_20261007_014148`.
It stores launch.sh, launch_record.json, source patches, and logs. Runtime
pointers now select this root. No commit or push performed.

Production verified: parent 1477661, first child 1477842, CUDA_VISIBLE_DEVICES=0
with no thread overrides. Child checkpoint reports Torch complex128 cuda:0,
requested caps/solver controls, execution policy 4, no samples/norm readouts.
First two steps each saved one batch containing all 49 two-site gates; t=0.2
completed and the first full refinement was running at verification.

## Publication request

The user subsequently requested committing and pushing both repositories.
This changeset includes the full-layer batching correction and preceding
objective-only sweep controls, with their tests, API docs and handoffs. The
validation above was reused because no numerical code changed afterward.
Final review found only these task changes; generated results remain outside
Git. The requested destination is the existing develop branch.
