# D-dependent boundary convergence and reusable guesses

This supersedes the default geometric policy in the
[initial adaptive implementation](2026-10-07-peps-adaptive-boundary.md).
The user requested D-squared probe sizes, lower cost, better reliability,
fixed selected norm/overlap caps throughout each fit, and output normalization
with the selected norm cap on the same Torch backend/device.

## Implemented

- Default probes: D², 2D², 3D²,... up to 8D², using the fitted PEPS cap D.
  Explicit `start_chi` and `max_chi` accept scalar/pair overrides. This adaptive
  path owns norm/overlap/normalization caps instead of legacy fixed-cap flags.
  Geometric search remains an explicit option, as does disabling calibration.
- Two consecutive stable cap increases, separate state/target norm checks,
  complex normalized-overlap checks, and cross-direction agreement are required.
- DMRG keeps three boundary guesses per axis only during calibration of the
  unchanged state/target pair. It grows them through existing boundary APIs
  and refits them using direct initialization. Block FIT grows locally through
  splits. Fresh contractions in both directions must confirm a candidate before
  declaring convergence; they also supply continuation values at the ceiling.
  No environments survive a PEPS update. Direct compressors do not reuse guesses.
- Constructor caps lock the ensuing sweep. Stale optimize-time chi overrides
  are cleared to avoid changing caps or redundant eager expansion. Measured
  target norms are forwarded. Final normalization uses selected norm chi.
- Diagnostics save named comparisons, individual thresholds, stable counts,
  warm/fresh initialization, actual boundary ranks, selected caps, and stop reason.
- Example execution policy 6 prevents reuse of earlier results; new start-chi
  CLI control complements max/threshold controls. The initial manifest and log
  now show the resolved D-dependent policy.

## Validation and practical limits

- 26 focused adaptive tests passed, including exact NumPy references, Torch
  complex64/128 fits, normalization/cap forwarding, false warm-guess plateaus,
  and fresh-at-ceiling behavior. The final cap-lock revision was included.
- 284 fixed-cap optimizer/numerical regressions passed.
- 39 example optimizer/control tests passed before the final logging addition;
  the final affected selection then passed 12 tests (28 deselected), including
  the added explicit start-cap case and resolved-policy manifest checks.
- Bounded 3x3 CUDA complex128 direct and DMRG fits selected chi 12 and
  preserved output device/dtype and exact norm within 1e-8. DMRG included a
  fresh confirmation. This check was before the final ceiling fallback and
  stale optimize-chi suppression; those have separate regression coverage.
- A harder seeded 4x4 D=4 complex128 reference with a single RZZ target tested
  calibration against dense vectors. Reused guesses before ceiling confirmation
  took 26.85s; fresh probes throughout took 79.20s. The final reused policy,
  including fresh contractions at the ceiling, took 47.81s. Final selected chi
  was 128 and **convergence was not established** in all three cases. The final
  norm, target-norm and overlap absolute errors were approximately 9.26e-5,
  4.02e-4, and 3.71e-4, matching fresh contractions. The new lower maximum is a
  cost limit, not sufficient accuracy for every PEPS.
- The old geometric 128→256→512 comparison exhausted GPU memory trying to
  allocate another 16 GiB. Only the diagnostic process failed; all production
  children were verified alive afterward. Do not report a completed old-policy
  timing. The benchmark ran alongside production work, so timings are indicative.
- Ruff remains unavailable. Diff checks pass. No full-suite claim or dependency
  upgrade. The installed/source package version mismatch from the initial note
  is unchanged. Native Symmray warm-start calibration is not newly validated.

Temporary probes: `/tmp/pepsy_d2_cuda_probe.py`,
`/tmp/pepsy_d2_boundary_benchmark.py`, and
`/tmp/pepsy_d2_boundary_benchmark_final.py`, with matching logs/JSON.
The read-only summary utility `/tmp/pepsy_report_boundary_chi.py` writes CSVs
from completed batch records, including named comparisons and actual ranks.

The dependency environment and audit are unchanged from the initial adaptive
task. Adopted existing Pepsy boundary retuning and public contraction APIs;
no installed libraries were patched. Pre-fit convergence does not certify
every later moving local environment or accumulated global fidelity.
