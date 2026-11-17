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
- `cutoff="auto"` and `cutoff_mode="auto"` are the defaults, matching MPO
  DMT replay and `MpoOptimizer.compress_dmt`. Auto cutoff is `1e-12` for
  float64/complex128 and `1e-6` for float32/complex64. Auto mode chooses
  `"rel"`: a dtype-scale threshold relative to the **free residual's**
  largest singular value. It does not use the much larger protected part
  to judge small correlations. Explicit `cutoff=0.` retains zero-cutoff
  behavior. Quimb's
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
MPO entry point performs one compression sweep. For gate-stream evolution,
use [`MpoOptimizer(mode="dmt")`](../optimizers/mpo.md#dmt-gate-stream-replay),
or call its `compress_dmt` method for the current state. Neither entry point
implements a constrained variational fitter.

## Larger mapped squares

Use plaquette regions with a snake ordering and check the protected bond
budget at each chosen `chi`. The budget depends on both the state and region
inventory; a value that works on 4×4 need not work on 6×6 or 8×8. DMT rejects
an insufficient conservative budget and keeps the requested cap unchanged.

Region checks contract the kept sites from both ends with logarithmic scale
tracking. Overlapping protection patterns reuse local projections within
each active pair. These exact contractions reduce time and temporary storage
while retaining every marginal check and the native dtype/device.

Run a reproducible CPU sweep with a fixed rank-136 input across χ values:

```bash
python examples/mpo_dmt_benchmark.py --square 6 8 --chis 64 96 128 \
    --initial-rank 136 --operation compress --methods dmt --repeats 3
```

For interior gate replay use `--operation replay --schedule center`; set
`--initial-rank` to select the starting state complexity. The four-gate center
schedule includes long MPO spans; `--schedule layer` covers every lattice edge.
JSON rows include explicit numerical/budget rejections, actual bond cap,
density/trace defects, state storage and cumulative process peak RSS.
See the [large-square measurements](../../development/notes/2026-11-16-mpo-dmt-large-squares.md)
for tested inputs, precision limits and the remaining distant-gate cost.
