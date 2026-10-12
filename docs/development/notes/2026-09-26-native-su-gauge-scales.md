# Native SU gauge scales — 2026-09-26

Gaugy's requested spin-Z2 SU workflow exposed an unsupported Autoray `mean`
dispatch on Symmray `BlockVector` gauges. `renorm_gauge` now reduces magnitudes
over backend blocks, dividing the squared sum by the total number of stored
singular values, not the number of sectors. Dense arrays follow the same rule.
Each block is detached before the scale calculation. The identical positive
scale divides the gauge and contributes `log10(scale)` to the exponent.

## Upstream evidence and classification

**Adopt:** Symmray's public `blocks` dictionary and Autoray operations on its
backend arrays. No installed package modifications or dependency upgrades.
Environment audit reused from the same active task: Symmray
`0.4.1.dev8+gc45f91457`, Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Torch `2.9.1`.
Installed `BlockVector.size` also totals individual block sizes; `to_dense`
concatenates them but is unnecessary for scale extraction.

The requested [Symmray documentation](https://symmray.readthedocs.io/en/latest/index.html)
and array page were unavailable to the browser during this audit. Used the
[official source](https://github.com/jcmgray/symmray) and installed
`BlockVector`/`Z2Array.from_dense` implementation instead. Earlier upstream
Quimb/Autoray/Cotengra checks from this active task remain applicable to the
unchanged environment.

## Validation

`pytest -q -o addopts='' tests/test_gauge_scale.py tests/test_gate.py`:
**165 passed, 1 skipped**, one preexisting dtype warning. Native regressions
cover NumPy/Torch, unequal sector sizes, weights 0, 1e-200, 1e-12, 1, 1e200,
floors 0 and 1e-12, reconstruction, detached scales, and unit reconstruction
derivatives. Dense subnormal and extreme-scale tests remain in the suite.
Ruff over all `src` and `tests` and `git diff --check` pass.

The complete Pepsy suite was not rerun for this single-helper extension.
Downstream Gaugy's full suite passes with native SU loss-gradient tests.
