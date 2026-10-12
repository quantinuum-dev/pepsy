# 2026-09-29 — Numerical MPO delinearisation and frontier comparisons

## Scope and implementation

Implemented `pepsy.operators.delinearize_mpo` in the owning
[operator module](../../../src/pepsy/operators/mpo_delinearize.py) and added
frontier/delinearisation comparisons to both 1D Gaugy example notebooks.
The implementation and this evidence are uncommitted working-tree changes.
Existing channel implementations were already present; this work does not
claim their earlier validation as new tests.

**Adopt:** the column-basis/neighbor-transfer principle of
[Hubig et al.](https://arxiv.org/abs/1611.02498), with scaled pivoted QR
(`scipy.linalg.lstsq`, `lapack_driver="gelsy"`), original-column retention,
relative column-fit checks, and paired sweeps. Among equally sparse columns,
larger scales are considered first to avoid large transfer coefficients.
The [API guide](../../api/operators/mpo_delinearize.md) specifies the
conservative differences from the paper's full Appendix C.

**Prototype:** this is numerical dependency reduction at the current values,
not exact symbolic parameter-family minimization. It has no target bond cap,
minimum-rank guarantee or optimal SVD approximation guarantee. Local residuals
are not global error bounds. NumPy dense floating tensors are supported;
Torch/JAX, Symmray and cyclic storage are rejected without conversion.
Frontier planning remains the exact differentiable construction alternative.

**Defer:** native-sector/autodiff delinearisation and stronger symbolic or
variational reduction. No installed library, shared environment, default
cluster assembly, SVD or channel-plan policy was changed.

## Measurements

Six sites, J1=1, J2=0.5, h=1, t=0.06, the notebook's XX/YY/ZZ ordered product
for the joint target. Frontier uses exact formal sharing and no QR projection;
numerical delinearisation uses rtol=1e-12, preserve_zeros=True, max_sweeps=4.
Errors use independent `ExplicitClusterSum` full matrices. Stored dimensions
are not measurements of peak allocation, and PBC here is the interaction
graph, not cyclic MPO storage.

| Model | Boundary | p | Frontier bonds | Delinearised bonds | Relative error vs explicit C_p |
| --- | --- | ---: | --- | --- | ---: |
| single | OBC | 2 | (5, 25, 25, 25, 9) | (4, 16, 24, 16, 4) | 1.743e-15 |
| single | OBC | 3 | (5, 33, 65, 81, 13) | (4, 16, 54, 16, 4) | 9.054e-15 |
| single | OBC | 4 | (5, 33, 161, 129, 13) | (4, 16, 51, 16, 4) | 2.021e-14 |
| single | OBC | 5 | (5, 33, 209, 129, 13) | (4, 16, 50, 16, 4) | 2.417e-14 |
| single | PBC | 2 | (5, 25, 125, 113, 17) | (4, 16, 64, 18, 4) | 4.656e-15 |
| single | PBC | 3 | (5, 41, 281, 233, 21) | (4, 16, 64, 17, 4) | 2.994e-14 |
| single | PBC | 4 | (5, 41, 281, 233, 21) | (4, 16, 64, 18, 4) | 5.920e-14 |
| single | PBC | 5 | (5, 41, 329, 233, 21) | (4, 16, 64, 20, 4) | 6.496e-14 |
| joint | OBC | 2 | (5, 25, 25, 25, 9) | (4, 16, 24, 16, 4) | 5.024e-15 |
| joint | OBC | 3 | (5, 33, 65, 81, 13) | (4, 16, 61, 16, 4) | 1.215e-14 |
| joint | OBC | 4 | (5, 33, 161, 129, 13) | (4, 16, 64, 16, 4) | 2.410e-14 |
| joint | OBC | 5 | (5, 33, 209, 129, 13) | (4, 16, 64, 16, 4) | 2.861e-14 |
| joint | PBC | 2 | (5, 25, 125, 113, 17) | (4, 16, 64, 17, 4) | 1.745e-14 |
| joint | PBC | 3 | (5, 41, 281, 233, 21) | (4, 16, 64, 16, 4) | 3.099e-14 |
| joint | PBC | 4 | (5, 41, 281, 233, 21) | (4, 16, 64, 16, 4) | 6.158e-14 |
| joint | PBC | 5 | (5, 41, 329, 233, 21) | (4, 16, 64, 16, 4) | 4.818e-14 |

At single OBC p=5, fixed/frontier/delinearised/SVD maximum dimensions are
649/209/50/35. Joint OBC p=5 gives 649/209/64/64. The frontier MPO is still
materialized before delinearisation; the original fixed collection MPO is
avoided on this path. Preparation maps and local residuals can remain large.

## Failure found and corrected

The first PBC notebook execution rejected single p=2: a local fit below
1e-12 produced relative full-matrix error 2.73e-9. Keeping tiny columns before
larger columns of the same sparsity generated transfer weights up to 5.35e5.
Preferring the larger column within each sparsity class removed this failure;
the corrected case agrees within 4.66e-15. No notebook acceptance tolerance
was loosened. Three regression cases cover t=0.03/0.06/0.12, and all 16 model /
boundary / order combinations above pass independent reference checks.

Initial zero/one-site test fixtures also exposed Quimb restrictions on
`MPO_identity(1)` and scalar multiplication by zero. Fixtures now construct
the one-site matrix directly and scale a core; upstream code was unchanged.

## Validation and environment

- New delinearisation and public API/layout selection: **81 passed**, two
  existing deprecation warnings. Includes nonparallel complex dependencies,
  all four supported dtypes, weak independent channels with large neighbors,
  zero policy, independent complete single/joint targets under OBC/PBC, and
  rejection of native/autodiff data. SVD and NumPy's SVD-based least-squares
  entry points are patched to raise in numerical reconstruction tests.
- Existing frontier/structure/channel/assembly and MPO compression selection:
  **81 passed, 6 skipped** (CUDA hidden for this CPU run). No full suite claim.
- Gaugy downstream channel/binding/materialization selection: **27 passed**.
- Notebook execution and final presentation checks are recorded in the
  [handoff](../../../history/2026-09-29-mpo-delinearisation.md).

Used the existing Python 3.12 environment: NumPy 2.5.2, SciPy 1.17.1,
Quimb 1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305,
Cotengra 0.8.3.dev7+g1d7fd333f, Cotengrust 0.2.1, Symmray
0.4.1.dev7+g83fb22865. Inspected installed Quimb `copy`, `bond`,
`permute_arrays`, `Tensor.modify`, MPO attributes and Pepsy/SciPy signatures.

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The Symmray array-documentation page returned an internal error; installed
source and explicit native-array rejection were used for this scoped path.
No compatibility shim or dependency update was needed.
