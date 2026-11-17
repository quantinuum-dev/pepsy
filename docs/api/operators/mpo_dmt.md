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

## Optional improved bond truncation

`region_selection="rank_aware"` compares several valid covers using their
state-dependent protected ranks. Contained region requests are derived from
their parent regions. This is a bounded search, not a global rank optimum.

For small bond caps, `constraint_mode="relaxed"` first tries exact DMT and
the compatible-border solution. If these cannot fit, a weighted SVD fallback
keeps the trace while allowing local-density errors. `constraint_weight`
controls the weight on protected directions (default 100); it is not a direct
plaquette-error penalty. Inspect `report.relaxed_bonds`,
`report.relative_density_error` for the whole compression, and
`report.max_relative_density_error` for individual splits. Relaxed replay
retains reports even without diagnostic collection. `check_regions=False`
skips repeated full-region checks while retaining bond and trace checks;
in that mode density-error report fields are `None`. Use
`dmt_check_regions=False` for `MpoOptimizer.run`. Routed gates still contract
regions for their trace and routing bookkeeping.

```python
rho_small, report = compress_mpo_dmt(
    rho, max_bond=8, regions=regions, region_selection="rank_aware",
    constraint_mode="relaxed", check_regions=False, return_report=True,
)
print(report.relaxed_bonds)  # density errors are not measured in fast mode
```

The same options work with `PauliMPO.compress_dmt`, `PauliMPO.apply_gate_dmt`,
`PauliDMT`, and `MpoOptimizer.compress_dmt`. For gate replay use
`dmt_region_selection`, `dmt_constraint_mode`, `dmt_constraint_weight`, and
`dmt_check_regions`. Fast checks do not certify a full physical-region
contraction at every step. Enable full checks for accuracy investigations.

At fixed χ, exact preservation refers to the current post-gate approximate
state. Discarded longer-range correlations can affect future local densities;
compare trajectories at increasing χ to assess long-time accuracy. The
compressor does not guarantee global positivity. CuPy execution has not been
validated in the current environment.

The existing fast method remains `truncation="paper"`. To spend more work
reducing the error at each cut, use:

```python
rho_small = compress_mpo_dmt(
    rho, max_bond=16, truncation="optimized", max_refinement_sweeps=8,
    regions=[(0, 1, 3, 4), (1, 2, 4, 5)], region_selection="min_cover",
)
```

In orthonormal protected/free coordinates, the candidate has the form

$$
{\color{purple}A(X)}=
\begin{pmatrix}{\color{blue}P}&{\color{blue}B}\\
{\color{blue}C}&X\end{pmatrix},\qquad
\min_{\operatorname{rank}A(X)\le\chi}\|X-D_0\|_F^2.
$$

The blue border is fixed. The optimized route keeps the paper candidate,
tries a generalized Schur SVD only when numerical range/conditioning checks
pass, then performs equality-constrained alternating factor updates. Each
accepted update preserves the original trace maps and cannot increase the
one-cut error, up to floating-point roundoff. `max_refinement_sweeps=0`
tries the SVD seeds only. A well-conditioned compatible corner gives the
full one-cut optimum at zero cutoff; incompatible corners use local
refinement without a global optimum certificate. A sweep is not a globally
optimal MPO fit, and comparing complete sweeps does not inherit the one-cut
nonincrease guarantee because their intermediate targets differ.

Cutoffs govern the seed residual SVDs. Refinement can reuse the available
rank up to `max_bond`; it does not impose a final spectral cutoff or discard
protected data afterward. This is an optional accuracy tradeoff and can be
slower than paper DMT. Reports add `truncation_method` (`paper`, `schur`, or
either followed by `+als`) and `refinement_sweeps`; `residual_rank=None`
denotes optimized splits, where there is no unique paper residual rank.
`discarded_weight` is then the measured one-cut Frobenius error.

For explicit regions, `region_selection="min_cover"` minimizes the total
number of protected physical strings using a bipartite vertex cover. Every
requested cross-cut observable has at least one protected endpoint. Trace
directions remain on both sides. This shares protection across overlapping
plaquettes rather than choosing each smaller half independently. It does
**not** minimize state-dependent virtual ranks or guarantee a faster sweep;
the default `"smallest"` retains the original rule. Large requested regions
still require exponentially many physical strings.

These options also work on `PauliMPO.compress_dmt`, `apply_gate_dmt`, the
persistent `PauliDMT` constructor, and `MpoOptimizer.compress_dmt`. Replay
uses the names `dmt_truncation`, `dmt_max_refinement_sweeps` and
`dmt_region_selection` in `MpoOptimizer.run`.
See the [derivation and validation](../../development/notes/2026-11-17-dmt-refinement.md).

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
