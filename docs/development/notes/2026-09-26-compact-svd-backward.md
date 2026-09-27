# 2026-09-26 — Compact stabilization for Torch SVD backward

## Scope and diagnosis

Gaugy's two strict expected failures still reproduced on the initial checkout.
The dense/native full-V gradient errors were respectively `8.67e-7` and
`8.58e-7`, although the cost values matched an independent dense calculation.
The shared Lorentzian reciprocal damped every spectral denominator.

Using exact reciprocals outside the existing gap window reduced the error to
about `1e-7` but did not resolve it. Isolating inverse singular values from
pairwise gap terms identified the remaining cause: rectangular/complex-phase
terms treated small, numerically resolved singular values as degenerate gaps.

## Implemented policy

- Both real and complex SVD VJPs use exact reciprocals outside compact
  stabilization regions. The singular gap/sum threshold remains `1e-6*smax`.
- Inverse singular values use `max(m,n)*finfo(dtype).eps*smax`, the standard
  numerical-rank criterion, separately for each matrix or native block.
- For threshold `t`, the extension inside is `(2*y-y**3)/t`, `y=x/t`.
  It matches the reciprocal's value and slope at both ends and is zero at
  zero. Clamp `t` to the dtype's smallest normal positive value. Neither the
  spectrum nor the denominator is squared; inactive branches remain finite.
- Keep the native simulation default, forward decompositions, dtype/device,
  configuration API, raw Quimb split routing, and adaptive QR unchanged.
  `safe_inverse` remains unchanged for QR and other callers.
- The extension is a surrogate at singular charts. This does not establish
  exact gradients inside its stabilization regions, at truncation-rank
  crossings, or at retained/discarded ties. No second-order claim is made.

## Upstream audit

Installed: Torch 2.9.1; Quimb 1.15.1.dev66+ge927f06e1;
Symmray 0.4.1.dev8+gc45f91457; Autoray 0.11.1.dev3+g1b476b305;
Cotengra 0.8.3.dev7+g1d7fd333f. No installed dependency was edited/upgraded.

- **Adopt:** numerical-rank criterion documented by
  [Torch pinv](https://docs.pytorch.org/docs/2.9/generated/torch.linalg.pinv.html).
  Only its threshold is adopted; this implementation does not call `pinv`.
- **Compatibility shim (retain):** existing scoped Autoray registration and
  raw Quimb split drivers. Installed `svd_truncated(x, cutoff=-1.0,
  cutoff_mode=2, max_bond=-1, absorb=0, renorm=0, info=None, **kwargs)` and
  `qr_stabilized(x, absorb=1, stabilized=True, **kwargs)` remain compatible.
  [Autoray](https://github.com/jcmgray/autoray)'s installed registration accepts
  `(backend, name, fn=None, *, wrap=False, module=None, alias=None,
  wrapper=None, inject_dtype=None, inject_device=None)`.
- **Defer:** [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html)
  changes to default cutoff modes; this patch does not change forward
  truncation policy. [Cotengra docs](https://cotengra.readthedocs.io/en/latest/)
  and [changelog](https://cotengra.readthedocs.io/en/latest/changelog.html)
  require no contraction-planner changes for this VJP correction.
- **Compatibility shim (retain):** native Symmray blocks use the same Torch
  driver. The requested old
  [array-doc URL](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
  was unavailable; checked the [official repository](https://github.com/jcmgray/symmray)
  and installed `symmray.linalg.svd(x, *args, **kwargs)`, which delegates to
  `x.svd`. Dense/native end-to-end checks validate the existing block routing.

## Validation

Focused backend suites: 112 passed, 1 skipped. New tests cover real/complex,
rectangular/batched rank-reduced gradients, native VJP comparison with scales
`1e-100` to `1e100`, finite differences, small resolved singular values,
zero/repeated/rank-deficient spectra in both precisions, and reciprocal scale
handling from `1e-200` to `1e200`.

Gaugy's former failures now pass at their original tolerances; maximum
gradient errors across renormalization on/off and target scales 1/1.3 are
`2.56e-15` dense and `2.49e-14` native Z2. All three public losses pass new
directional finite differences with actual bond-four to bond-two truncation,
both representations, and renormalization on/off. Its 25 existing
zero-overlap/penalty/truncation gradient tests pass as well.

Full Pepsy suite: **4752 passed, 121 skipped**, 735 warnings, 336.53 seconds.
Full Gaugy suite: **414 passed**, 8 warnings, no xfails. Ruff, CI mypy, diff
checks, and strict Sphinx build pass. The documentation build required network
access to fetch official intersphinx inventories. CUDA was not tested.
