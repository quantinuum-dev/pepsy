# 2026-10-02 — MPO delinearization review fixes

Implemented in the working tree on `develop`, baseline `d7d509b`, following
the [ten-commit review](../../../history/2026-10-02-ten-latest-commits-review.md).
The scope is the three confirmed dense-MPO regressions introduced by `c796530`.
No dependency upgrade, installed-library edit, or backend registration changed.

## Reduction safeguards

Both pre-materialization array sweeps and tensor-network edge reductions now
check reconstruction independently for every original bond channel. They also
weight the channel residuals by the opposing tensor's row maxima. Separate
endpoint normalization avoids multiplying extreme scales. Nonfinite or
subnormal endpoint scales conservatively retain the original bond.

A purely local matrix tolerance could discard a tiny channel whose distant
partner makes its global contribution order one. Checking individual channels
protects that direction even on longer chains before the sweep has propagated
the compensating scale. The weighted check additionally bounds the immediate
two-tensor residual. These are local roundoff safeguards, not a global MPO
error certificate; ill-conditioned channels can retain larger bonds.

Numerical rank selection now skips Torch tensors requiring gradients and JAX
tracers. A dependency that holds at one parameter value need not hold for its
derivative, so the original channels remain until a parameter-independent
reduction is available. Tests check `X⊗X + θ Z⊗Z` at `θ=0` and `θ=.1`
against an independent derivative of one, including a trainable opposing
endpoint. The earlier gradient test now checks this conservative contract
while retaining its value and gradient assertions.

Dtypes outside float32/float64/complex64/complex128 skip numerical structural
reduction. This preserves Torch float16 CPU construction without an unsupported
SVD or silent promotion. Entire skipped delinearization passes report
`skipped_reason`, zero sweeps and no reductions. Explicit numerical compression
remains the caller's separate policy. Structured Symmray paths are unaffected.

## Upstream compatibility audit

Installed versions inspected: Quimb `1.15.1.dev79+gb5e316200`, Autoray
`0.11.1.dev9+g1291702f9`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev11+g1a3481803`, Torch `2.6.0+cu124`, JAX `0.10.2`.

Public signatures inspected: `qtn.MatrixProductOperator(arrays, *, sites=None,
L=None, shape='lrud', tags=None, upper_ind_id='k{}', lower_ind_id='b{}',
site_tag_id='I{}', **tn_opts)` and `Tensor.modify(self, **kwargs)`.
Autoray dispatches Torch `linalg.svd` to its SVD wrapper and identifies Torch
half arrays as `float16`; the original CPU SVD reproducer confirms that dtype
cannot use this decomposition.

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The [Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
was unavailable with an internal error; the official repository and installed
implementation were available. Classification: **compatibility shim** for the
Pepsy dense-MPO reduction eligibility and acceptance checks; **defer** changes
to upstream decomposition policies. The shim preserves backend, dtype and
parameter derivatives and stays within the owning operator helper.

## Fresh validation

Activated `~/envs/py312`; set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`
and `XLA_PYTHON_CLIENT_PREALLOCATE=false` for the operator selection:

```text
python -m pytest -q -o addopts='' \
  tests/test_mpo_delinearization_regressions.py \
  tests/test_structural_compression.py tests/test_mpo_automaton.py \
  tests/test_ham.py tests/test_tree_mpo.py \
  tests/test_cluster_channel_automaton.py tests/test_mpo_cluster_compression.py \
  tests/test_public_api.py tests/test_package_layout.py
```

**244 passed, 5 warnings in 52.83 seconds.** The new module includes 28 cases:
NumPy/Torch/JAX/CuPy float32 and float64, both scale orientations on two- and
four-site MPOs, direct arrays and converted tensors, independent Torch
derivatives, float16, and both Hamiltonian builder routes. Existing tree and
cluster tests validate other callers of the shared structural helper.

Running the new module's non-JAX/non-CuPy selection against the archived
original `d7d509b` produced **20 failed, 8 deselected**, confirming the old
behavior fails these regressions. Some cases additionally assert the newly
documented skip report. The original three standalone reproducers now pass:
unbalanced-term error zero, zero-initialized derivative one, and float16
construction succeeds without changing dtype.

Ruff and `git diff --check` passed. No full-package suite or performance
benchmark is claimed. The focused run uses the current working tree, preserving
the user's unrelated solver and test edits.
