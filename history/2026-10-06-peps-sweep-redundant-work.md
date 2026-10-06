# 2026-10-06 — Remove repeated PEPS sweep work

- Scope: user asked to fix the identified duplicate work, compare one/four
  round trips, then explicitly requested a Pepsy commit. Examples changes
  remain outside that commit request. No push requested.
- Baseline: develop / 7cb1044 plus the uncommitted timing and negative-metric
  changes documented in the other two journals from this session. A complete
  source snapshot was preserved under `/tmp/pepsy_sweep_perf_baseline_20261006`.

## Changes

- The driver skips the sweep's duplicate final diagnostic when it performs
  the authoritative post-normalization acceptance check. Explicit debug
  diagnostics remain available. Matching default direct-boundary prechecks
  are reused; custom policies, different caps, and iterative FIT keep their
  own sweep initial checks. Reuse/skipping is recorded in sweep summaries.
- Direct sweep boundaries initialize lazily, including restored best states.
  This avoids eagerly constructing unused axes/directions. No contraction
  values survive a local state change through the new cache.
- Local objectives cache paths only within a fixed slice environment, keyed
  by ordered indices/shapes. String optimizer presets use this cache;
  explicit tree/optimizer objects preserve their prior handling. Optional
  FLOP/peak diagnostics are disabled by default.
- Optional boundary-MPS average-norm diagnostics were performed after every
  slice and at completion. They now require collect_boundary_norms=True.
  This does not disable physical state normalization or objective norms.
- Four round trips remain the default. One-trip mode was benchmarked but a
  single easy case does not establish sufficient accuracy for all later times.

## CUDA benchmark

Used the first actual refinement input encountered by the prior 5x6 D=4,
Torch complex128, delta_theta=18*pi/88, dt=.1 trajectory. Input/target were
captured before fitting and saved as CPU tensors; each benchmark loaded the
same tensors onto idle physical GPU0. Caps (32,46), direct boundary compression,
NLopt defaults, negative allowance 1e-6. No concurrent GPU0 production job.
Scripts/artifacts: `/tmp/pepsy_sweep_perf_probe.py`,
`/tmp/pepsy_sweep_perf_case_20261006.pkl`, and
`/tmp/pepsy_sweep_perf_{before4,after4,after1}.json`.

| Variant | Fit seconds | With normalization/postcheck | Slices | Peak allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| Baseline, 4 trips | 14.2309 | 14.5289 | 46 | 92,770,304 |
| Updated, 4 trips | 8.0834 | 8.5603 | 46 | 47,035,392 |
| Updated, 1 trip | 4.7356 | 5.2218 | 16 | 46,173,696 |

Initial local infidelity was 2.705790e-9. Postchecks were respectively
2.726742e-9, 2.727094e-9, and 2.727076e-9; all three rejected the candidate
and retained the same initial approximation. The four-trip fit was about
43% faster with about 49% less peak tensor memory on this one case. This is
not a full-trajectory performance or global-fidelity guarantee.

## Validation

- Domain selection after implementation: 227 passed (46.81 s), covering
  optimizer, batching, sweep safeguards, timing, and performance parity.
- Added opt-in boundary-report tests subsequently: all 7 performance tests
  passed (5.92 s). NumPy/Torch fit parity and no duplicate metric contractions
  are tested. Benchmark inputs use independent tensors to avoid in-place
  solver mutation contaminating the reference comparison.
- Final performance/safeguard selection after early-exit metadata adjustment:
  35 passed (8.49 s); sandbox CUDA initialization warning during CPU Torch test.
- Downstream direct/no-readout, default delegation, and CLI serialization:
  3 passed (15.50 s).
- Public API/layout plus domain selection: 281 passed; one environment
  mismatch failure: installed Pepsy distribution metadata is 0.4.0 while the
  checkout pyproject version is 0.5.0. No shared-environment reinstall made.
- Default smoke suite: 93 passed, one failure (45.44 s), the same package
  version mismatch. Verified that mismatch also occurs with the untouched
  baseline source and HEAD pyproject version.
- compileall and git diff --check passed. Ruff is unavailable in cloudspace.
- Existing upstream audit reused; installed Quimb/Autoray/Cotengra/Symmray
  versions are unchanged. Inspected Quimb's public contraction_path signature
  and implementation. Adopted public path reuse and existing lazy BdyMPS;
  no upstream monkeypatches or new dependencies.

## Production status

The previous direct-boundary run independently failed after t=.4, while
advancing toward t=.5, with raw infidelity -3.267e-6 exceeding its 1e-6
allowance. It had loaded the pre-performance code. User was informed; this
performance task did not silently widen the tolerance again or restart it.
Outputs remain under the earlier direct/tol1e6/no-readout directory.
