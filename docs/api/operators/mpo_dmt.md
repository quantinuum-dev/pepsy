# Protected SVD for ordinary MPO tensors

Use `compress_mpo_dmt` to reduce ordinary ket/bra MPO bonds while preserving
local reduced operators. **Protected SVD remains the direct compression
method**; it does not invoke a gradient optimizer.

```python
import quimb.tensor as qtn
from pepsy.operators import compress_mpo_dmt

rho = qtn.MPO_identity(6) / 64
rho_small, report = compress_mpo_dmt(
    rho, max_bond=16, cutoff="auto", return_report=True,
)
```

The default protects every contiguous window of three sites (the whole chain
for lengths below three). Alternatively, pass `regions=[(0, 3), (1, 4)]` to
preserve those reduced operators. Region entries are **MPO positions**;
[`square_lattice_dmt_regions`](../optimizers/pauli_dmt.md#2d-regions-and-physically-local-gates)
builds physical neighborhoods for a 2D lattice mapped to an MPO.
Larger regions can require much larger protected bond budgets.

## Trace protection without a Pauli basis

For matrix units $E_{st}=|s\rangle\langle t|$, the local trace is

$$
{\color{blue}t_{st}}=\operatorname{Tr}E_{st}=\delta_{st},\qquad
{\color{purple}\widetilde\rho_R}
=\operatorname{Tr}_{\bar R}\widetilde\rho
=\operatorname{Tr}_{\bar R}\rho.
$$

The implementation fuses the ket/bra pair, contracts this trace covector
outside protected regions, and uses the same guarded splitting kernel as
[`PauliMPO.compress_dmt`](pauli_mpo.md#protected-density-matrix-truncation-dmt).
It does not convert to Pauli coefficients or form a global dense matrix.
Only the protection vectors use an independent trace-aligned orthonormal
basis; the stored tensors stay in the matrix-unit basis.
Both bases span the same operator space. Degenerate residual singular values
can select different equally good bond approximations, so complete sweeps
need not produce identical tensors across bases.

## Contract

- Accepts dense NumPy, Torch or JAX `MatrixProductOperator` tensors on an
  **open chain**, with one tensor per site and uniform square local dimension
  $d$. Float32/64 and complex64/128 are supported. Native Symmray arrays and
  cyclic MPOs are explicitly rejected.
- `variant="density"` uses the connected-correlation construction and requires
  a numerically resolved nonzero trace. `variant="operator"` also supports
  traceless and non-Hermitian operators. Neither variant guarantees positivity
  or global Hermiticity: a complex SVD can select a non-Hermitian combination
  at a degenerate residual cutoff. Protected reduced operators retain their
  Hermiticity through the marginal constraints. Real Pauli coefficients remain
  useful when global Hermiticity must be maintained through such cuts.
- `max_bond` must accommodate the protected subspaces. An insufficient
  conservative budget raises `ValueError`; insufficient working precision
  raises `FloatingPointError`. Marginal checks remain active without reports.
- `cutoff=0.0` is the default. `"auto"` uses Pepsy's dtype policy. Quimb's
  `rel`, `abs`, `sum1`, `rsum1`, `sum2`, and `rsum2` policies act only on the
  free residual spectrum. Absolute cutoffs use physical Hilbert–Schmidt units,
  including Quimb's external exponent; squared-sum cutoffs use squared units.
- Returns a new MPO by default. `inplace=True` writes only after preservation
  checks succeed. Tensor/index order, tags, external exponent, array backend,
  dtype and device are retained. Exact Pepsy construction-history attachments
  are invalidated after numerical compression.
- `return_report=True` returns `(mpo, MPODMTReport)`. `MPODMTReport` and
  `MPODMTBondReport` share the existing Pauli diagnostic schema. Here,
  discarded weights and absolute constraint errors are in physical
  Hilbert–Schmidt units; Pauli reports use coefficient units.

The SVD optimizes the **free residual at one canonical bond** for the chosen
protected construction and rank budget. It is not a globally optimal MPO fit
under only the requested marginal constraints. The adaptive decomposition
and precision guards use host decisions; this compressor is not a fully
JIT-compiled or generally differentiable solver. Native Torch graphs are
retained. For autodiff, use `TorchLinalgConfig(stabilized=True).activated()`
from `pepsy.backends`: the projected residual has zero singular values and
native SVD backward formulas can produce NaNs. Even with the stabilized
policy, rank changes and degenerate truncation boundaries remain limitations.

For repeated Pauli-basis gate evolution, the persistent
[`PauliDMT`](../optimizers/pauli_dmt.md) driver remains available. This ordinary
MPO entry point performs one compression sweep; it does not add a new replay
mode or a constrained variational fitter.
