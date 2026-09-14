# Flat contraction backend scalars — 2026-09-14

`contract_flat` historically formatted every final scalar as a Python number.
That behavior is convenient for reports and remains the default, but it
detaches Torch/JAX results and prevents downstream packages from optimizing a
flat tensor-network objective through this public Pepsy boundary.

The opt-in `preserve_backend=True` path returns `BoundaryContractResult.cost`
without scalar formatting. With `strip_exponent=True`, both mantissa and
exponent remain backend values exactly as returned by Quimb or Pepsy's FIT
boundary engine. This is an **adopt** of existing backend-native Quimb return
semantics, not a contraction-algorithm change. Exact and boundary-MPS Torch
regressions should check values and gradients separately; CTMRG gradients
through complex eigenspace projectors can remain phase-ambiguous.

The 2026-09-14 compatibility audit used Quimb
`1.15.1.dev55+gd0591eb70`, Autoray `0.11.1.dev3+g1b476b305`, Cotengra
`0.8.3.dev7+g1d7fd333f`, and Symmray `0.3.2.dev8+g6c6dd34b5`. Installed
signatures confirm that Quimb's exact, boundary-MPS, CTMRG, and HOTRG routes
all support stripped-exponent final contraction output. No compatibility shim
is needed; only Pepsy's final Python-scalar formatter is bypassed on request.
