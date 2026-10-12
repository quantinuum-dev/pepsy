# 2026-09-30 — Public gate automatic truncation

Implemented `cutoff="auto"` and `cutoff_mode="auto"` defaults at the public
`gate` / `gate_simple` boundary. Both reuse `dtype_auto_cutoff`, the existing
MPS policy: complex128/float64 1e-12, complex64/float32 1e-6, 16-bit 1e-3.
Automatic mode resolves to rsum2 before upstream dispatch or SU scale checks.
Numeric overrides remain available. Dtype inspection uses network array
metadata without conversion or block densification. Internal routed gates
receive resolved values; path compression also accepts auto. `gate`'s final
`chi_cutoff` defaults to the same automatic numeric threshold.

The roughening example no longer imports the private cutoff helper or
resolves the policy itself. It forwards the request to `gate_simple`; saved
parameters retain that request. Its PEPS defaults remain 60 steps at dt=0.1,
loop size 6, and optional 4096-shot weighted sampling.

## Dependency audit

Installed in the shared Python 3.12 environment: Quimb
1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305, Cotengra
0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1, Symmray 0.4.1.dev7+g83fb22865,
Torch 2.6.0+cu124. Inspected signatures for `tensor_network_gate_inds`,
`TensorNetworkGenVector.gate_simple_`, and `tensor_split`. Installed Quimb
simple update defaults to numeric 1e-10, while tensor_split accepts auto and
defaults to relative singular-value mode. Pepsy explicitly supplies its own
shared policy rather than depending on those different upstream defaults.

- **Adopt:** existing public Quimb gate/split dispatch and native Symmray
  block-preserving array operations; no installed dependency edits.
- **Defer:** dependency upgrades, other builder defaults, and optimizer
  redesign; no compatibility shim was required for this policy resolution.

Reviewed the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The requested Symmray `abelian_arrays.html` documentation page was unavailable;
used the installed native array API and official source instead.

## Validation

- Gate cutoff, gate routing, SU scale, and SimpleUpdateGen tests: 179 passed.
  New numerical checks exercise a Schmidt weight between the two precision
  thresholds, explicit overrides, copy isolation, native Symmray tensors,
  and routed PEPS on NumPy/Torch CPU.
- Roughening PEPS tests: 40 passed, including automatic versus explicit
  cutoff evolution and output metadata.
- Public API/layout: 53 passed; the existing installed-distribution version
  mismatch (0.4.0 versus project 0.5.0) still fails.
- Package Ruff (`src tests`) and whitespace checks passed.
- No full package suite or GPU-specific validation was performed.
