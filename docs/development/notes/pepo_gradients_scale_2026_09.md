# Local PEPO gradients and gauge scaling — 2026-09-26

Classification: **adopt** a tested local differentiation policy and correct
scale bookkeeping. No installed library edits, dependency upgrades, or
replacements of Quimb's contraction algorithms are involved.

## Defects and changes

Gauge renormalization previously divided by `rms + smudge` but restored
`rms` through the network exponent. This attenuated both reconstructed values
and derivatives, and direct squaring overflowed/underflowed at finite extreme
weights. The shared helper now computes RMS relative to the maximum magnitude,
uses one detached positive scale for both operations, validates the scale
floor, and gives zero gauges scale one. If a nonzero subnormal RMS rounds to
zero, its positive peak is used instead. Gaugy calls this shared implementation.

Native SVD backward can return NaNs at repeated/zero singular values even when
the composed trace loss is finite. The existing stabilized SVD handles the
tested SU cases. However, automatically regularizing every small QR pivot
produced large directional-gradient errors in Gaugy's composed SU losses.
The new opt-in adaptive QR policy uses native VJPs when finite, skips undefined
native differentiation at exactly zero pivots, and warns when a batch member
requires the regularized fallback. It raises if the fallback is still nonfinite.
Zero cotangents propagate zero without evaluating an undefined native VJP.

QR nodes capture their forward policy, so restoring an enclosing dispatch
context before backward no longer changes the selected rule. Existing global
defaults and `warn`, `native`, and `error` choices remain unchanged.

Gaugy's public local PEPO losses select stabilized SVD/adaptive QR during
trainable Torch evaluation. Explicit per-loss `torch_linalg_config` overrides
the choice; forward driver preferences are inherited by default. Registrations
are restored afterward. Structurally zero fixed targets bypass SU and return
connected zero first derivatives. Raw matrix penalties reduce all entries.

## Evidence and limits

- `tests/test_gauge_scale.py`: NumPy/Torch reconstruction from zero through
  `1e-200 .. 1e200`, three scale floors, unit reconstruction gradients,
  invalid floor rejection, and a smallest-positive-float regression.
- `tests/test_qr_adaptive.py`: real/complex native-gradient agreement after
  scope exit and isolated fallback for mixed batched singular inputs.
- Gaugy `tests/test_pepog_gradients.py`: all three public costs at zero and
  near-zero overlap against independent dense values/gradients; zero targets;
  matrix penalty gradients; directional derivatives through actual SU caps
  at bond two/four, warm/perturbed parameters, and exact/MPS trace routes.
- The perturbed bond-two direction changed from stabilized `0.32922256`
  versus finite difference `0.33186222`, to approximately `0.33186221865`
  versus `0.33186221879`. The remaining stabilized-SVD discrepancy in the
  perturbed bond-four fixture is about `7.2e-8`; regression tolerances are
  `atol=1e-7, rtol=1e-6` across three finite-difference steps.
- Additional 3-by-3 directional probes at two random seeds, with a bond-two
  SU cap and exact/MPS contractions, agreed within `6e-10` at step `1e-5`.

These are first-order tests. A finite extension at a singular chart is not a
unique mathematical derivative, and truncation rank changes/ties can be
nonsmooth. Higher derivatives and CUDA execution are not certified here.
This is numerical scale tracking, not Hilbert--Schmidt normalization.

## Upstream/environment check

The unchanged environment and official upstream audit from
[the operator convention correction](operator_conventions_2026_09.md) were
reused: Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Torch `2.9.1`.
Installed Quimb `gate_simple`, gating/split signatures, and Pepsy's Torch VJPs
were inspected again. SU still uses the ordinary reduced split, requested
bond/cutoff controls, and Quimb gauge handling; no full tensor-product split
was substituted for numerical convenience. Full-suite and documentation
results are recorded in `history/2026-09-26-pepo-gradient-scale-fixes.md`.
