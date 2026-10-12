# 2026-10-07 — Large Torch vector sampling precision

The 5×6 `rough_exact_mps` investigation found a sampling defect in
`pepsy.sampling.vector.VecSampler._draw_samples`. For more than 2**24
categories, the Torch fallback accumulated a CDF and generated uniforms in
the probability dtype. For complex64 states this was float32, causing narrow
probability intervals to vanish and biasing low-order sampled bits.

## Measured and implemented

- Before the fix, an exactly uniform 25-qubit complex64 state produced 2071
  final-bit ones in 8192 shots (seed 62, chunk size 1200), versus 4096
  expected. The measured fraction was 0.2528076171875.
- A deterministic four-outcome forced-fallback test selected the wrong
  configuration when the uniform draw lay inside a 2**-26-probability interval.
  This test was observed failing before implementation.
- The fallback now adopts Torch's public float64 `cumsum` and float64 `rand`.
  It normalizes the whole CDF by its own total, preserving monotonicity and
  avoiding an artificial final-outcome mass from setting only the last entry.
  The discrete draw CDF is detached; returned Born weights still support
  the existing `track_grad=True` contract.
- State, probability dtype/device, site ordering, and the small-category
  multinomial route are preserved. The large-route seeded sequences change.
  The CDF alone uses eight bytes per outcome (8 GiB for 30 qubits); chunking
  shots does not reduce its size. No persistent CDF cache was introduced.

## Environment and upstream audit

Activated the examples' selected `~/envs/py312` environment with local Pepsy
first on PYTHONPATH. Installed versions: Torch 2.6.0+cu124, NumPy 2.5.2,
Quimb 1.15.1.dev79+gb5e316200, Autoray 0.11.1.dev9+g1291702f9,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1, and
Symmray 0.4.1.dev11+g1a3481803.

Classification: **adopt** public Torch precision controls, no compatibility
shim or installed-library changes. Verified installed `torch.cumsum` accepts
`dim` and `dtype`, `torch.rand` accepts `dtype`, `device`, and `generator`,
and `torch.searchsorted` supports the existing `right=True` convention.
See [Torch 2.6 cumsum](https://docs.pytorch.org/docs/2.6/generated/torch.cumsum.html).
The Torch 2.6 rand documentation page was unavailable through the web tool;
installed docstrings and execution confirmed the used keyword capabilities.

Reviewed the required [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray source](https://github.com/jcmgray/symmray).
The Symmray array documentation page was unavailable; its repository was
accessible. Classification: **defer** unrelated contraction/compression and
native-array changes; this defect is confined to dense Torch draw arithmetic.

## Validation and limits

`tests/test_sampler.py tests/test_public_api.py tests/test_package_layout.py`
with default addopts disabled: **165 passed**, two compatibility-alias
deprecation warnings. This includes the actual 25-qubit route, with all bit
frequencies within 0.03 of 0.5, and narrow-interval checks with/without
returned gradients. `ruff check src tests` and `git diff --check` passed.
No full numerical suite or CUDA-device validation was performed.

Existing exact 5×6 complex64 shots used the affected route, but the size of
their observable bias is unmeasured. Independently recomputed raw-shot wall
statistics agree with saved figures, so plotting does not repair the issue.
No state vectors were found in the selected archives; no production run was
launched or data changed. Resampling or rerunning remains necessary to
establish corrected exact-reference curves. Earlier notebook rendering tests
did not validate sampling accuracy.
