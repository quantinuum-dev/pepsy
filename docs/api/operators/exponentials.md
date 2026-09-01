# Exponential API: MPO and PEPO

This page is the short usage guide for the two higher-order exponential
builders. The rule is simple:

For the longer-term ownership map, canonical vocabulary, and staged
refactoring roadmap, see the [operator and exponential API plan](../../development/plans/operator_api.md).

> Use `exp(step, ...)` for the operator exponential. Use `compile_exp(...)`
> when the operator topology is reused. The `step` is the scalar in
> `exp(step * H)`; it is not automatically a physical time.

## Choose the analytical construction mode

The higher-order history mode is independent of the final numerical `chi`
compression:

| Mode | Passes | Meaning |
| --- | --- | --- |
| `"base"` | Algorithms 1--2 | Conservative exact history path |
| `"exact"` | Algorithms 1--3 | Include the selected next-order replay |
| `"folded"` | Algorithms 1, 2, and 4 | Fast Algorithm-4 correction folding |
| `"hybrid"` | Algorithms 1--4 | Exact extension followed by folding |
| `"auto"` | Budget-dependent | Symbolic Algorithm-3 estimate: `exact` when it is within `extension_budget`, otherwise `folded` |

The older names `"algorithm4"`, `"optimal"`, and `"approximate"` remain
accepted as compatibility aliases. `chi` is a separate numerical bond cap;
it does not select or disable Algorithm 4.

For `mode="auto"`, the estimate is made from reachable symbolic histories
before the numerical left/right Cartesian pair plan is built. The default
`extension_budget` is 1,024 selected extension terms and can be overridden:

```python
U = exp_mpo(
    terms,
    -1j * tau,
    order=4,
    mode="auto",
    extension_budget=10_000,
    return_semantic=True,
)
print(U.metadata)
```

The semantic result records `requested_mode`, `mode`,
`estimated_extension_terms`, `extension_budget`, and `mode_reason`.

## Choose the representation

| Goal | Canonical entry point | Result |
| --- | --- | --- |
| One-off term-centric MPO | `exp_mpo(terms, step, ...)` | Compiled Quimb `MatrixProductOperator` |
| One parameterized 1D Hamiltonian | `MPOBasis.from_pauli_terms(...)` or `MPOBasis.from_local_terms(...)` | Reusable `MPOBasis` |
| One MPO exponential | `basis.exp(step, parameters=...)` | Semantic `FirstDegreeMPO` |
| Repeated MPO exponentials | `basis.compile_exp(...).exp(step, ...)` | Cached `CompiledMPOExp` call plus semantic MPO |
| Connected/joint MPO clusters | `MPOClusterProductExpansion` / `MPOGraphClusterProductExpansion` | One MPO assembled from local connected residuals |
| Product of two existing MPOs | `compress_mpo_product(A, B, ...)` | Lazy `A @ B`, then one ordinary compressed MPO |
| Raw MPO tensors for a compiled kernel | `basis.compile_exp(...).exp_arrays(step, ...)` | Backend-native tensor tuple |
| Quimb MPO interoperability | `semantic_mpo.to_mpo()` | Quimb `MatrixProductOperator` |
| One fixed-channel square-lattice PEPO | `PauliPEPOBasis.compile(...)` | Reusable `PauliPEPOBasis` |
| One PEPO exponential | `basis.exp(step, ...)` | `ActivePEPOBlocks` by default |
| Repeated PEPO exponentials | `basis.compile_exp().exp(step, ...)` | Cached `CompiledPEPOExp` call |
| Ordered PEPO product | `PEPOClusterProductExpansion.from_bases(...)` | One composed PEPO |
| Dense PEPO materialization | `active_blocks.to_pepo()` or `materialize=True` | Quimb `PEPO` |
| Coefficient-dependent real-time PEPO | `build_real_time_cluster_expansion_pepo(...)` | Quimb `PEPO` or active blocks |
| Fractional-step PEPO composition | `compose_cluster_expansion_pepo(...)` | Quimb `PEPO` |

The MPO and PEPO APIs deliberately have the same top-level vocabulary. They
do not have the same output layout: an MPO is a 1D semantic operator, while a
PEPO is first kept as sparse active virtual-sector blocks.

There are three distinct construction axes: SciPost higher-order MPO history,
connected MPO cluster size, and PEPO spatial cluster order. In both cluster
families, `exp(A) @ exp(B) @ exp(C)` is a joint local-residual expansion: the
ordered target is formed on each small connected support and inserted into one
MPO or PEPO topology. It is not sequential multiplication of three separately
truncated full-lattice layers. See the [MPO cluster guide](mpo_cluster.md)
and [PEPO cluster guide](cluster_expansion.md).

## Term-centric MPO construction

For the common case, callers only need to provide an operator, its support,
and a coefficient. `exp_mpo` infers a chain length or regular 2D/3D lattice,
maps lattice coordinates through the default snake ordering, canonicalizes
the support of commuting local factors, and shares common MPO channels:

```python
from pepsy.operators import exp_mpo

terms = [
    {"operator": "ZZ", "location": ((0, 0), (1, 0)), "coefficient": J},
    {"operator": "X", "location": (0, 0), "coefficient": h},
]

# Returns a Quimb MatrixProductOperator by default.
U = exp_mpo(terms, -1j * tau, shape=(4, 4), order=4, mode="exact")
```

The compact tuple spellings used by `ham_tn.build_mpo` are accepted too:

```python
terms = [
    ((0, 0), "X", h),
    (("ZZ", J), ((0, 0), (1, 0))),
]
U = exp_mpo(terms, -1j * tau, shape=(4, 4), mode="exact")
```

When `chi` is supplied, Pepsy first compiles the complete higher-order term
MPO and then applies one final numerical Quimb compression sweep. Use
`cutoff="auto"` and `cutoff_mode="auto"` for the dtype-aware cutoff policy
and `form` / `create_bond` for the main Quimb controls. Other Quimb keywords
can be passed through `compress_opts`:

```python
U = exp_mpo(
    terms,
    -1j * tau,
    shape=(4, 4),
    order=3,
    chi=64,
    cutoff="auto",
    cutoff_mode="auto",
    form="left",
    compress_opts={"renorm": False},
)
```

To multiply two already-built MPOs while keeping the product intermediate
lazy, use the separate numerical product facade:

```python
from pepsy.operators import compress_mpo_product

AB = compress_mpo_product(
    A,
    B,
    chi=64,
    method="auto",          # direct/SDC for mild products, FIT/DMRG2 otherwise
    cutoff="auto",
    cutoff_mode="auto",
)

AB_exact = compress_mpo_product(A, B, chi=None)  # no numerical compression
```

`method="dmrg"`, `"dmrg2"`, and `"dmrg3"` use Pepsy's native `FIT` solver;
`"direct"`, `"dm"`, `"sdc"`, and `"src"` dispatch to Quimb's 1D
compression methods. This `chi` compression is numerical and separate from
the analytical history Algorithms 1--4.

The DMRG methods first create a disposable rank-`chi` guess, then refine the
exact lazy product target with `FIT.run_eff`. The latter reuses its left/right
environments across full-chain sweeps. The default `guess_method="auto"`
selects deterministic SDC; dense products can opt into an SRC warm start:

```python
AB = compress_mpo_product(
    A,
    B,
    chi=64,
    method="dmrg2",
    guess_method="src",
    guess_seed=0,
)
```

SRC warm starts are currently dense-only. Native Symmray products retain
charge-sector structure and should use the default SDC or an explicit
`guess_method="direct"` until a sector-aware randomized SRC path is
available. The result metadata records `guess_method`, `guess_seed`, and
`fit_solver="FIT.run_eff"`.

For a live diagnostic of a large-order build, pass `progress=True`. The bar is
headed `exp(order=N)` and uses separate colors for history construction,
analytical algorithm passes, the boundary contraction, and final numerical
`chi` compression. Algorithm 4 is selected with `mode="folded"` and shown as
`A4 analytical-compress`; it is an order-controlled analytical history
reduction, not an SVD or `chi` cutoff.
The final stage is shown compactly as `chi-compress (chi=...)`; its backend and
method remain in `numerical_compression`. A call with `order=3` builds order 3
directly; it does not silently rebuild orders 1 and 2. To compare those
timings, call the builder once for each order:

```python
for order in (1, 2, 3):
    U = exp_mpo(
        terms,
        -1j * tau,
        shape=(4, 4),
        order=order,
        mode="base",
        chi=64,
        progress=True,
    )
```

When enabled, the completed result also stores stage timings in
`U.pepsy_exp_metadata["timings"]` (or
`U.pepsy_first_degree.metadata["timings"]` for an uncompressed Quimb MPO, or
`result.metadata["timings"]` for a semantic MPO). The displayed `maxchi` is
the actual current/final MPO bond size; `chi` is the requested upper bound.
The metadata fields `analytical_compression` and `numerical_compression` make
the two compression layers explicit. `chi=None` means
`numerical_compression="none"`; it does not disable Algorithms 1, 2, or 4.
`timing_history` stores the completed stage timings under the requested order,
and `order_seconds` stores the total elapsed construction time. The bar keeps
its main description stable while transient site/bond information is placed in
the postfix, and the completed line includes `order_s`.

`location` can also be a 1D integer site or a sequence of chain sites. In a
2D/3D term, one coordinate is used for a one-site operator and a sequence of
 coordinates is used for a product operator. Coefficients may be Python
numbers, Torch/JAX scalars, `MPOParameter` references, or callables supported
by `MPOBasis`; their slots remain independent even when their structural MPO
path is shared. Pass a configured `OneDMap` with `mapper=` when a custom
ordering is needed. Pass `symmetry=` and `physical_charges=` to enable the
native bosonic block-sparse compilation. Set `return_semantic=True` to keep
the history-aware `FirstDegreeMPO` instead of materializing the Quimb MPO.
Pass `to_backend=pepsy.backend_torch(...)` or another array converter to move
the compiled operator blocks and coefficient assembly onto a backend before
higher-order contractions; the final ordinary Quimb MPO is checked with
`apply_to_arrays` as well. `to_backend` is currently for dense MPO execution
and cannot be combined with native `symmetry=` compilation.
When `chi` is requested, use `compression="fixed_rank"` (or
`differentiable=True`) with `return_semantic=True`; ordinary Quimb compression
cannot preserve the higher-order history metadata.
Terms carrying charge or string-operator metadata must list their support
sites in increasing chain order so canonicalization cannot change their
virtual path convention.

An operator string or operator sequence describes a factorized product. To
compile a genuinely entangled local operator, pass its full square matrix on
two or more sites; Pepsy performs an exact operator-Schmidt decomposition and
inserts the resulting local MPO segment without densifying the full chain:

```python
from pepsy.operators import MPOLocalOperatorTerm, MPOBasis

basis = MPOBasis.from_local_terms(
    6,
    [MPOLocalOperatorTerm((1, 2, 4), local_eight_by_eight, coefficient=g)],
)
```

Site labels are strict integers: Boolean and fractional values are rejected.
For bosonic product terms, repeated sites are multiplied in the supplied local
order before site sorting. Compact Pauli input keeps the corresponding phase,
for example `X @ Y = 1j * Z`.

The compact Pauli mapping used elsewhere in Pepsy is accepted directly:

```python
terms = {"XX": (2, 3)}                    # coefficient defaults to 1
terms = {"XX": ((2, 3), J)}               # explicit coefficient
terms = {"xyz": (((0, 0), (1, 0), (0, 1)), J)}
```

The number of Pauli labels must match the number of supplied sites.

## MPO: parameterized chain

```python
import torch

from pepsy.operators import MPOBasis, MPOParameter

basis = MPOBasis.from_pauli_terms(
    8,
    [
        ((i, i + 1), "ZZ", MPOParameter("J"))
        for i in range(7)
    ] + [
        ((i,), "X", MPOParameter("h"))
        for i in range(8)
    ],
)

J = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
h = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)

# The result is a semantic FirstDegreeMPO. No dense 2**8 x 2**8 matrix is made.
U = basis.exp(
    -1j * tau,
    {"J": J, "h": h},
    order=4,
    mode="exact",
)
```

Use `coefficients=...` instead of `parameters=...` when the optimizer already
has one backend vector in term order:

```python
theta = torch.stack((J, h))  # one value per compiled term
U = basis.exp(-1j * tau, coefficients=theta, order=4, mode="exact")
```

`parameters` and `coefficients` are mutually exclusive. A parameter mapping
is resolved through `MPOParameter("name")` references; a coefficient vector
must have exactly `basis.num_terms` entries.

For a square-lattice Hamiltonian that should be executed as an MPO, use the
coordinate-aware compiler. It maps coordinates to a reusable 1D ordering with
`OneDMap`, canonicalizes reversed location/Pauli descriptions, and retains
the same autodiff coefficient path:

```python
basis = MPOBasis.from_square_lattice(
    4,
    4,
    [
        {
            "locations": ((0, 0), (1, 0)),
            "paulis": "ZZ",
            "parameter": "J",
        },
        {
            "locations": ((0, 0),),
            "paulis": "X",
            "parameter": "h",
        },
    ],
)
compiled = basis.compile_exp(order=4, mode="exact")
U = compiled.exp(-1j * tau, {"J": J, "h": h})
```

The default traversal is snake order; pass `map_mode=` or a configured
`OneDMap` to choose another ordering. Similar and duplicate terms share the
compiled MPO channel structure while retaining separate coefficient slots, so
independent autodiff parameters are summed on the shared path at build time.
If the desired output is a PEPO with local
square-lattice virtual legs rather than an MPO, use `PauliPEPOBasis` instead.

### Repeated MPO evaluations

Compile the order/mode policy once when only coefficients or `step` change:

```python
compiled = basis.compile_exp(order=4, mode="exact")

U = compiled.exp(-1j * tau, {"J": J, "h": h})
raw_tensors = compiled.exp_arrays(-1j * tau, {"J": J, "h": h})
```

`CompiledMPOExp` caches topology, history plans, coefficient-slot indices,
and static operator banks. It never caches coefficient-dependent arrays or
autodiff graphs. `compiled.exp(...)` returns the semantic MPO; use
`compiled.exp_arrays(...)` only when a numerical kernel needs the raw tensor
tuple.

### MPO step, order, and compression

All exponential methods use the same convention:

```text
real time:       step = -1j * tau
imaginary time:  step = -beta
custom operator: step = any backend scalar
```

`order` is the Taylor/history order. The named policies are:

| `mode` | Construction |
| --- | --- |
| `"base"` | Algorithms 1–2 |
| `"algorithm4"` | Algorithms 1, 2, and 4 |
| `"optimal"` | Algorithms 1–3, including the selected next-order extension |
| `"approximate"` | Algorithms 1–4 |

`max_bond` protects the temporary history construction. `chi` is a separate
final numerical MPO compression cap. With `chi=None`, the result remains a
semantic `FirstDegreeMPO`; with `chi` set, the default result is a Quimb MPO.
Use `differentiable=True` with `chi` for fixed-rank autodiff compression.
`chi=None` disables only this final numerical compression; the exact or
analytical history reductions selected by `mode` still run as part of the
construction.

Set `history_storage="reduced"` to stream local products directly into the
post-Algorithms-1/2 virtual space. This route supports all four modes,
preserves Torch/JAX coefficient and step gradients, and reports
`materialized_raw_virtual_tensors=False` in
`result.metadata["history_storage_blocks"]`. The default `"auto"` policy
remains unchanged so storage selection does not change silently.

Use one `MPOPhysicalSpace` when dimension, Abelian charges, and braiding need
to travel together. `MPOProductTerm(..., braiding="fermionic", parities=...)`
applies one minus sign for each odd-odd crossing during canonical site order.
This is construction-time graded ordering; native fermionic higher-order
history execution remains intentionally unsupported until its sector-aware
sign path is complete.

## PEPO: fixed Pauli channels on a square lattice

`PauliPEPOBasis` is for a qubit square lattice with translation-invariant
onsite and positive-direction edge Pauli slots. Its virtual channels are
fixed before coefficients are evaluated, so coefficient changes do not cause
rank-changing SVD decisions.

```python
import torch

from pepsy.operators import PauliPEPOBasis

basis = PauliPEPOBasis.compile(
    4,
    4,
    [("onsite", "X"), ("edge", "ZZ")],
    order=4,
    cyclic=True,
    symmetry="C4",
)

theta = torch.tensor(
    [0.5, 1.0],
    dtype=torch.float64,
    requires_grad=True,
)
tau = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)

# Default: keep the sparse active-sector representation.
active = basis.compile_exp().exp(
    -1j * tau,
    coefficients=theta,
)

# Explicit dense/interoperability boundary.
pepo = active.to_pepo()
```

The direct one-off form is `basis.exp(-1j * tau, coefficients=theta)`. The
compiled form returns the same `ActivePEPOBlocks` representation while
reusing the fixed lattice/channel topology. `materialize=True` is equivalent
to calling `.to_pepo()` for a single evaluation.

`ActivePEPOBlocks` contains only stored active sector blocks. It is the normal
autodiff and memory-saving result; dense PEPO tensors should be materialized
only when required by downstream code or a small-system validation.

The fixed Pauli implementation supports connected orders 1–9. Orders 1–4
use compact fixed Pauli channels; orders 5–9 use cached translated-shape
residuals and backend-native spanning-tree SVD channels. It does not perform
PEPO × PEPS contractions, expectation values, or symmetry-native block-space
calculations. For memory-controlled p≥5 runs, pass `max_tree_rank` to the
`PauliPEPOBasis` constructor; `None` retains exact local tree ranks.

## Dense cluster-expansion convenience API

For a fixed dense local Hamiltonian, use `ClusterExpansionPlan` or
`build_cluster_expansion_pepo`. That family uses the cluster convention
`exp(-beta * H)`:

```python
active = plan.build(beta=0.05, materialize=False)
pepo = active.to_pepo()
```

`ClusterExpansionReport` contains local residual and storage diagnostics. Its
residual norms are local factorization diagnostics, not a global PEPO error
bound. The dense implementation includes recursive generic order-five
through order-nine tree paths; orders above nine remain unsupported. Finite dense
model adapters are available through `ClusterModelAdapter` and
`build_model_cluster_expansion_pepo`.

For local coefficient lists and real-time evolution, use
`build_real_time_cluster_expansion_pepo`. It assembles the coefficient-weighted
one- and two-site terms first, then uses the cluster convention with
`beta=1j * time`, targeting `exp(-1j * time * H)` for the summed `H`.
With `fit_method="quimb"`, generic tree residuals use Quimb tree fitting and
generic loop residuals use complex ALS. This numerical path is not
coefficient-differentiable; its local fit diagnostics are returned in
`ClusterExpansionReport`.

For cyclic generic clusters, `adaptive_loop_rank=True` can grow the ALS loop
rank from `loop_rank_start` to `max_loop_rank` until the local residual meets
`fit_tol`. `fit_warm_start=True` (the default) reuses the previous rank's
fitted tensors. This improves local cluster fits; it is not the global PEPO
compression/environment step used by some infinite or finite-lattice time
evolution workflows.

## Caching and autodiff contract

- Compiling a basis caches structure only: topology, channels, histories, and
  static operator banks.
- Every `exp` call creates fresh backend-connected values, so Torch/JAX
  gradients see the current coefficients and `step`.
- `clear_history_cache()` releases reusable higher-order plans but leaves the
  compiled term topology intact.
- `to_mpo()` and `to_pepo()` are explicit interoperability/materialization
  boundaries; neither is needed to construct the exponential itself.

When an API returns a detailed construction or compression report, use
`report.api_info` for cross-family logging. It provides the stable
`family`/`algorithm`/`representation` vocabulary plus `order`,
`factor_count`, `truncated`, and `differentiable`; the concrete report retains
the algorithm-specific residual, rank, cutoff, and error fields.

## Compatibility names

New code should use `exp`, `compile_exp`, `CompiledMPOExp`, and
`CompiledPEPOExp`. The previous names `time_evolution`, `evolution_mpo`,
`compile_evolution`, `evaluate`, and `CompiledMPOEvolution` remain available
as compatibility shims so existing programs do not break. They are not the
recommended vocabulary for new code.
