# MPO delinearisation without SVD

`pepsy.operators.delinearize_mpo` reduces numerical channel dependencies in
an existing open-chain NumPy MPO. It returns a copy, preserving dtype, tensor
tags, physical indices and normalization. It never constructs the full dense
operator and never calls SVD. The source MPO remains unchanged.

```python
from pepsy.operators import delinearize_mpo, prepare_cluster_channels

# builder describes fixed, uncapped cluster factors and exact graph assembly.
channels = prepare_cluster_channels(
    builder, step=-1j*t, preparation="frontier", max_bond=None,
)
frontier_mpo = channels.exp(-1j*t)
mpo, report = delinearize_mpo(
    frontier_mpo, rtol=1e-12, preserve_zeros=True,
    max_sweeps=4, return_report=True,
)
print(report.initial_bond_dimensions, report.final_bond_dimensions)
```

Frontier construction avoids the original collection-expanded MPO.
Delinearisation then reduces the constructed frontier MPO; it does not
prevent allocation of that frontier MPO or its preparation maps. No claim of
bounded peak memory or small exact ranks follows from the reduced output.
The cluster cutoff and Hamiltonian are unchanged.

The same call accepts an MPO built with `preparation="automaton"`:

```python
automaton = prepare_cluster_channels(
    builder, step=-1j*t, preparation="automaton", max_bond=None,
).exp(-1j*t)
reduced = delinearize_mpo(automaton, rtol=1e-12, preserve_zeros=True)
```

This combines exact Pauli/state reduction with numerical dependency removal
at the current parameters. It need not give smaller bonds than delinearising
the frontier MPO; check both stored dimensions and final operator error.

Pepsy also integrates this operation into `ClusterChannelPlan.exp` and
`trace_exp` with `delinearize=True, delinearize_opts={...}`. The one-shot
`exp_mpo_cluster` and `exp_mpo_cluster_product` facades accept those controls
with `preparation="automaton"` or `"frontier"` and fixed uncapped construction.
See the [integrated API example](mpo_cluster.md#integrated-automaton-plus-qr-construction).

## Algorithm and controls

The routine reshapes one core into columns, selects original independent
columns, and solves for each other column as their linear combination. It
absorbs the resulting transfer matrix into the neighboring core. Both sweep
directions repeat until a full pair of sweeps removes no bonds, or
`max_sweeps` is reached. Dense physical operator matrices are never needed.

- `rtol=1e-12`: each accepted column fit must have maximum entrywise residual
  at most `rtol` times that column's largest entry. This is a **local** test,
  not a global operator-error bound. Nonzero columns are never dropped merely
  for having small norm; exactly zero columns may be removed.
- `preserve_zeros=True`: only retained columns that vanish at a target
  column's exact zeros are eligible. False permits cancellation there and can
  find more dependencies. Original retained columns preserve their entries.
- `max_sweeps=4`: maximum number of paired left/right sweeps.
- `return_report=True`: returns `(mpo, MPODelinearizationReport)` with initial
  and final bonds, tolerance, zero policy, sweep count, convergence flag and
  largest accepted local residual. The report is also attached as
  `mpo.pepsy_delinearization_report`. Convergence means stable bond counts,
  not proof of a minimum representation.

Within each sparsity class, larger columns are considered first to avoid
large transfer weights. Column scaling and SciPy `lstsq(lapack_driver="gelsy")` use pivoted QR rather
than SVD. A final residual test accepts or rejects each fit, and unstable
large transfer coefficients are rejected. No arbitrary bond cap is enforced.
This is a conservative implementation of the delinearisation principle in
[Hubig, McCulloch and Schollwoeck](https://arxiv.org/abs/1611.02498), not all
Appendix C heuristics: it does not perform the row-deparallelisation prepass,
coefficient snapping, alternate row-factorization choice, or cancellation
cleanup. Its optional exact-entry zero restriction is stricter than the
paper's operator-block zero restriction.

## Accuracy and scope

Delinearisation is **not** an SVD replacement with the same optimality
property. Truncated SVD gives a best rank-limited approximation to a matrix
unfolding; this routine greedily finds channel dependencies and may retain
more bonds. Neither statement guarantees a globally optimal fixed-bond MPO.
Validate final matrix or observable error separately from cluster error.

This numerical routine acts at the **current parameter values**. Unlike
frontier's symbolic sharing, its choices are not identities for a whole
parameter family and cannot be reused as differentiable maps. It supports
NumPy float32/float64/complex64/complex128 with a common dtype. Other
backends, native Symmray tensors and cyclic MPO storage are explicitly
rejected, without detaching or flattening. Periodic interaction graphs stored
as open MPO chains are supported. Use exact channel plans for autodiff.

Semantic history and block-plan attachments are invalidated on the returned
MPO. The default cluster builders, channel plans and SVD paths are unchanged.
The single and joint 1D example notebooks compare fixed, frontier,
frontier plus delinearisation, and SVD at the same time and cluster sizes.
The single-exponential `cluster_1d` notebook additionally compares automaton
plus delinearisation, including dense reconstruction and cut-rank checks.
