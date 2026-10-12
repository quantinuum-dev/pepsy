# 2026-10-06 — Automatic D=4 to D=8 target batching after rollback fix

- Scope: user requested testing `k_2q_batch="auto"` after the solver storage
  fix, and clarified fitted D=4 with automatic target up to D=8.
- Baseline: develop `dfdd9c6920d22699eaf000322d2755ca8493ec90` (pushed).
  Earlier pending optimizer diagnostic/global-cutoff edits remain in the
  working tree; this test loaded that same working tree as the prior replay.
- No package algorithm changes or production restart in this task. This
  validation note is uncommitted.

## Test and result

Bounded CUDA:0 replay: 5x6, Torch complex128, dt=0.1, theta=18*pi/88,
fitted D=4, all norm/overlap/normalization boundary caps (64,64), direct
boundary compression, SciPy L-BFGS-B, two round trips, automatic batching.
No sampling or physical readout; optimizer norm diagnostics were measured.

Completed t=0.1,0.2,0.3, with 25 batches per step rather than the 49 used by
gate-by-gate evolution. Automatic batches contained up to three two-site
gates. Every retained state had max bond <=4; every target had max bond <=8,
with D=8 reached during the third step. Assertions checked every batch.

| Time | Evolution seconds | Measured norm squared |
| --- | --- | --- |
| 0.1 | 5.5463 | 1.000000000000001 |
| 0.2 | 4.4981 | 1.000000000581562 |
| 0.3 | 425.1598 | 1.000000000255573 |

Third step: 10 actual fits, 3 accepted and 7 rejected; 15 other batches
were below the fit threshold. All seven rejected fits restored the saved
tensor arrays and exponent exactly. Target arrays and exponents were
unchanged during all ten fits. The earlier automatic-batch run had failed
after 12 completed batches toward t=0.3 with infidelity -0.004267; this run
completed all 25. No normalization/fidelity exception recurred. Small
finite-contraction warnings and transient loky worker-stop warnings remained;
the process finished successfully. This does not establish t=6 accuracy or
completion. Other GPU jobs were untouched.

## Artifacts

- Probe: `/tmp/pepsy_auto_rollback_probe.py`.
- Log: `/tmp/pepsy_auto_rollback_D4_target8_20261006.log`.
- Root: `/tmp/pepsy_auto_rollback_D4_target8_20261006` contains settings,
  per-step summary, complete status and per-batch records.
- `batch_metrics.json` uses the correct `batch_fidelity` label. The original
  log/`batches.jsonl` called that value `fidelity_product`; it is the local
  batch fidelity, not a cumulative product. The probe label was corrected
  for future runs and the derived JSON retains the original numerical values.
- Exit code zero; completed at 2026-10-06 23:53:58 UTC. No full-length job was
  launched by this diagnostic test.
