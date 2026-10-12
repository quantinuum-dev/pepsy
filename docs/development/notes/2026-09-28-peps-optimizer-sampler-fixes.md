# 2026-09-28 — PEPS target, cutoff, and sampler result fixes

## Scope and environment

This task fixes the four implementation defects reproduced in the
[review handoff](../../../history/2026-09-28-peps-optimizer-sampler-review.md):
truncated two-site targets, dropped warm-start cutoff mode, invalid infidelity
coercion, and mutable batch result aliasing. The documented extreme-scale
exact-draw limitation is separate and remains unchanged.

The selected environment is `envs/py312`. Installed versions at the start:
Pepsy distribution metadata 0.4.1 (checkout source 0.5.0), Quimb
1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305, Cotengra
0.8.3.dev7+g1d7fd333f, Symmray 0.4.1.dev7+g83fb22865, and Torch
2.6.0+cu124. `PEPS.compress_all` accepts `**compress_opts` and forwards them
through bond compression; installed `tensor_split` accepts
`cutoff_mode='rel'`. The implicit `PEPS.compress_all` cutoff behavior is not
inferred from that lower-level default: on the reproducible 2x2 D=4 seed-2
state, implicit compression retained four rank-2 bonds, while explicit
`cutoff_mode='abs'` retained four rank-3 bonds at cutoff 0.1 and cap 3.

## Upstream audit and decision

- **Adopt:** explicitly pass `cutoff_mode` to Quimb `compress_all` and use
  `cutoff=0`, `max_bond=None`, `path_compress=False` for the exact target. The
  [Quimb split reference](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/decomp/index.html)
  defines the distinct cutoff policies. Its
  [changelog](https://quimb.readthedocs.io/en/latest/changelog.html) reports
  changing lower-level defaults, reinforcing explicit selection at this call.
- **Defer:** no Autoray, Cotengra, or Symmray behavior needs a compatibility
  shim for these changes. Checked the official
  [Autoray repository](https://github.com/jcmgray/autoray),
  [Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
  [Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
  and [Symmray repository](https://github.com/jcmgray/symmray). The
  [Symmray Abelian array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
  returned an internal error; the repository and installed implementation
  were used for the capability check. No installed dependency was changed.

## Behavior and evidence

The optimizer keeps the generated post-gate target exact under run-level gate
truncation, with explicit target truncation requests rejected. Requested
`cutoff_mode` reaches both normal and routed warm-start compression. Invalid
non-finite or materially negative infidelity values raise; tiny negative
roundoff still clips to zero. Prefix-grouped sampling retains shared
contractions and amplitudes but returns one configuration list per shot.

A 2x2 |0000> PEPS evolved by exp(-0.1i X⊗X) with run cutoff 0.1 must retain
both cos(0.1) and sin(0.1) amplitudes at chi=2 and report `within_chi` only
when that exact target fits. Before the fix, the normalized output omitted
the small component and had 0.009966711079379076 infidelity against the
cutoff-zero dense reference while reporting zero. The regression checks
retained amplitudes and bond dimension. Further regressions check the
cutoff-mode bond ranks, rejection of invalid boundary infidelity, and
independent configuration lists from duplicate shots.

## Validation on the edited tree

- `python -m pytest -q -o addopts='' tests/test_optimize_peps.py
  tests/test_peps_sampler.py`: **299 passed, 2 skipped**, 43 existing
  upstream/compatibility warnings, 223.42 seconds. Includes the target
  reconstruction, requested cutoff mode, invalid metric, and three-backend
  independent-shot regressions.
- `python -m ruff check src tests`: passed.
- `git diff --check` and local Markdown link checks: passed.
- Whole-repository run before correcting a stale integration assertion:
  **5177 passed, 105 skipped, 2 failed** in 1529.07 seconds. One failure is
  the pre-existing installed Pepsy metadata 0.4.1 versus checkout 0.5.0.
  The other was `test_peps_4x4_actual_row_cache_and_refresh` expecting
  `mode="transfer"` while leaving the sampler at its current default
  `row_cache_mode="factored"`. The test now selects `row_cache_mode="dense"`
  explicitly, preserving its intended dense transfer-cache check; its
  isolated rerun passed in 2.63 seconds. The complete suite was not rerun
  after this test-only correction.
