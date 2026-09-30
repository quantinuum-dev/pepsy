# 2026-09-28 — Explicit fixed-cluster MPO bond compression

## Scope

Uncommitted working-tree change on `develop`, baseline `4e398e4`. The user
requested a concrete error/time benchmark for numerical bond compression after
SVD-free cluster construction. No cluster residual, PEPO, symmetry, cache or
fixed factorization policy changed. Existing unrelated working-tree edits are
preserved.

The [MPO guide](../../api/operators/mpo_cluster.md#explicit-bond-compression-after-fixed-construction)
now shows the explicit API. A runnable [benchmark](../../../examples/cluster_mpo_bond_compression.py)
reports bond sizes, construction and compression times, and Frobenius error.
It uses `factorization="fixed"`, `assembly="recursive"`, `cutoff=0`, no
intermediate bond cap, then `compress_numerical(max_bond=chi, cutoff=0,
estimate_error=True)` separately. The error is relative to the **uncompressed
p-cluster operator**, not to the global exponential.

## Numerical finding and correction

For a small relative compression error, Quimb's direct-sum difference MPO
could lose digits when its doubled network was contracted directly. On the
2x3 p=2 model at chi=32, the old norm contraction reported about 3.05e-8,
while direct dense subtraction gave 3.68e-14. QR-canonicalizing the dense MPO
difference before the optional norm contraction gives about 3.68e-14. At
chi=16, the corrected estimate and dense reference both give about 3.46e-9.
The operation is inside the opt-in `estimate_error=True` path; ordinary
compression and the fixed constructor are unchanged. Native sector estimates
retain their existing route because the correction is scoped to dense MPOs.
The estimator still has ordinary floating-point limits at sufficiently small
errors and its contraction may be expensive at large bond dimensions.

**Adopt:** installed Quimb `MatrixProductOperator.left_canonize` (QR), followed
by its existing `norm` contraction. Quimb 1.15.1.dev66+ge927f06e1,
Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f,
Symmray 0.4.1.dev7+g83fb22865 and Torch 2.6.0+cu124 were unchanged from
the earlier [upstream audit](2026-09-28-mpo-recursive-assembly.md#upstream-audit--classification).
Installed `left_canonize`, `norm` and `compress` signatures were checked.
No upstream library was edited. **Defer:** native sector error stabilization
until a corresponding sector-aware numerical check is available.

## Scoped CPU benchmark

Run from the repository root with the activated development environment:

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python examples/cluster_mpo_bond_compression.py
```

2x3 open square, uniform 0.2 X + 0.4 ZZ, step=-0.03i, p=2, complex128,
one BLAS/OpenMP thread. Structure was compiled and backend dispatch warmed
before timing. Three repeated evaluations give median fixed-construction time
**12.36 ms**; the initial MPO bond dimensions are (58, 106, 157, 106, 58).
Each chi has three compression-only timings. The optional error run performs
compression again plus a tensor-network Frobenius contraction, with no dense
operator. This is a small, scoped CPU measurement, not a 5x6 performance claim.

| chi | Final bonds | Median compression only | Compression plus error measurement | Relative error vs p=2 MPO |
| ---: | --- | ---: | ---: | ---: |
| 2 | (2, 2, 2, 2, 2) | 6.02 ms | 23.78 ms | 1.697e-2 |
| 4 | (4, 4, 4, 4, 4) | 5.82 ms | 24.08 ms | 2.493e-4 |
| 8 | (4, 8, 8, 8, 4) | 5.87 ms | 23.22 ms | 2.494e-7 |
| 16 | (4, 16, 16, 16, 4) | 6.33 ms | 25.49 ms | 3.456e-9 |
| 32 | (4, 16, 32, 16, 4) | 6.85 ms | 24.35 ms | 3.682e-14 |

The small dense regression independently compares the report with direct
Frobenius norms for chi=4,16,32. Torch is checked on a nontrivial chi=1 case.
These checks establish the diagnostic on those backends and sizes; they do
not establish compression convergence for larger or different models.

The affected MPO/fixed-cluster/API/layout gate passed **274 tests**, with two
existing deprecation warnings (60.40 s). After narrowing the QR guard to
dense MPO data, **22 focused compression/sector checks passed**, with 117
deselected (11.42 s). Full Ruff, whitespace and 21 affected relative-link
checks passed. The [handoff](../../../history/2026-09-28-cluster-bond-compression.md)
records the validation scope. The previous full-suite result predates this
change; it was not rerun.

Post-compression can reduce stored bonds and downstream contraction cost, but
it **cannot** reduce the peak cost of constructing the uncompressed fixed MPO.
For a 5x6 p=4 graph the existing plan has 33,514 states, but its exact
fixed-channel numerical assembly has not been measured. When the exact MPO
cannot be built, the existing `assembly_chi` numerical policy is the separate
working-bond option and requires a convergence study. Its SVD is outside the
SVD-free fixed policy. No gradient-through-compression guarantee is claimed.
