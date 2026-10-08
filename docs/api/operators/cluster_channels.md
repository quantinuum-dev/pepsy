# Compact channels for repeated cluster evaluation

`prepare_cluster_channels` is an experimental, opt-in compact MPO/PEPO
construction path. It prepares constant virtual bases outside autodiff, then
contracts local residual factors directly into compact site tensors. Replay
performs no SVD, QR, numerical zero pruning or rank selection. The local
ordered exponentials and connected residuals keep their existing definitions.
Expanded sparse histories and unprojected MPO collection products are not
built during replay. Default preparation visits the reference channel topology;
MPOs can instead prepare directly from the active frontier described below.
PEPOs can reduce symbolic graph/square slices before evaluating references.
MPO preparation first shares formally equal transitions, independently of
the numerical reference, before choosing any approximate QR subspace.

```python
from pepsy.operators import prepare_cluster_channels

# builder: fixed MPO, or fixed graph/square PEPO cluster builder.
channels = prepare_cluster_channels(
    builder, step=-0.1j, parameters=reference_parameters,
    max_bond=12,
    samples=[{"parameters": another_reference}],
)
operator, report = channels.exp(
    -0.1j, parameters=current_parameters, return_report=True,
)
value = channels.trace_exp(-0.1j, current_parameters, normalized=True)
```

`exp` returns a Quimb MPO, square PEPO or graph TensorNetwork. `trace_exp`
constructs and measures that operator, including its projection. This differs
from the older MPO builder's scalar-partition `trace_exp`. `arrays` returns
only compact site tensors with physical `(output, input)` axes.

Square channel plans now use Pepsy's graph residuals and exact virtual-wire
routing. Their unprojected cluster target is preserved, but their virtual
gauge differs from the earlier square sparse-block channel implementation.
New capped plans can therefore give different approximations. No existing
ordinary `builder.exp()` construction policy is changed.

## MPO preparation without complete collection enumeration

```python
# Fixed MPO builder, with graph_assembly="exact" for a direct graph source.
channels = prepare_cluster_channels(
    builder, step, reference_parameters,
    preparation="frontier",
    max_bond=None,                 # no additional channel-projection error
    frontier_state_budget=4096,
)
mpo = channels.exp(step, current_parameters)
```

The frontier stores only selected clusters whose spans still cross the
current MPO cut. It forgets completed choices and shares their continuation.
At an unoccupied site it selects the singleton or starts a compatible cluster;
at an occupied site it advances the existing cluster. Each disjoint collection
has exactly one path. Crossing, nested and gapped higher-body supports are
handled by their actual occupied sites, not merely their spans.

This path builds reachable frontier transitions directly, without calling
the full-collection enumerator or constructing its reference MPO. It then
uses the same exact symbolic sharing, optional QR selection and fused replay.
It accepts fixed interval sources, exact direct graph sources without a
collection-order cap, and uncapped recursive sources. Bounded/auto graph
targets are rejected, so preparation cannot silently change their target.
Graph/square **PEPO** builders use `"reference"`, `"symbolic"`, or the
Pauli-specific `"algebraic"` mode below.

`frontier_state_budget` bounds active-cluster sets at each cut; when omitted,
it inherits the source's `assembly_state_budget` (normally 4096). It can be
set independently without changing the source. `collection_budget` does not
apply because complete collections are never listed. Exceeding the frontier
or memory budget raises instead of dropping contributions. Frontier dimensions
can still grow exponentially with the number of simultaneously crossing
clusters; this is not a guarantee of small exact chi or global minimality.

`preparation="reference"` remains the default for compatibility. The two
preparations can produce different capped approximations because their
virtual gauges differ. `pack` diagnostics use the selected preparation's
sparse transitions. See the [frontier scaling measurements](../../development/notes/2026-09-29-frontier-cluster-channels.md).

For an additional numerical reduction at the current parameters, pass the
resulting NumPy MPO to [delinearize_mpo](mpo_delinearize.md). It uses QR solves
without SVD and is separate from exact symbolic sharing and frozen QR
projection. It is not an autodiff operation or a minimum-rank guarantee.

## MPO preparation with operator-aware automaton reduction

```python
# Same fixed cluster builder: terms + geometry + cluster size.
channels = prepare_cluster_channels(
    builder, step, reference_parameters,
    preparation="automaton", max_bond=None,
)
mpo = channels.exp(step, current_parameters)
```

This explicit mode compiles a layered weighted automaton for the selected
cluster expansion. It starts with reachable frontier states, certifies the
Pauli algebra generated by the declared terms on each cluster, and sweeps
both directions to eliminate constant linear dependencies between complete
transition slices. Rational arithmetic proves the identities; the opposite
endpoint absorbs the transfer weights. The singleton rail stays separate.
Full channels require no SVD, QR, numerical rank tolerance, or complete
collection enumeration. Frozen maps fuse directly into replay contractions.

The mode currently accepts **dense qubit Pauli product terms** with fixed
finite NumPy local matrices and live scalar coefficients. The closed span
contains the local ordered exponentials, connected subtraction, and their
derivatives, including at zero parameters. A fixed Walsh projection is the
identity on this family. Arbitrary `bind_assembler` residuals outside that
span are projected; changing operator labels/supports requires a new plan.
Use `"frontier"` for native charge sectors or other local operator families.

`max_bond=None` preserves the selected cluster expansion. A smaller cap still
requests reference QR approximation; `structural_reuse=False` disables state
elimination but retains the Pauli projection. Frontier state/memory guards
also apply. Sparse frontier transitions and dense local residuals are still
prepared: this does not promise bounded memory as cluster order or graph
cutwidth grows. It does not guarantee a globally minimal automaton or
smaller bonds than frontier sharing for every model.

This construction targets `C_p`, whose finite cluster error is measured
against the desired exponential/product. The general
[`MPOAutomaton.from_product_terms`](automaton.md) targets a supplied linear
operator sum; its raw `to_mpo()` contract and defaults are unchanged.
See [implementation and measurements](../../development/notes/2026-09-29-mpo-automaton.md).

### Optional numerical delinearisation on evaluation

```python
mpo, report = channels.exp(
    step, current_parameters,
    delinearize=True,
    delinearize_opts={"rtol": 1e-12, "preserve_zeros": True, "max_sweeps": 4},
    return_report=True,
)
```

`ClusterChannelPlan.exp` integrates the existing QR-based
[`delinearize_mpo`](mpo_delinearize.md) for dense NumPy MPOs, including
frontier and automaton plans. Its default remains `delinearize=False`.
`trace_exp` accepts the same controls and traces the reduced operator.
PEPOs and native MPOs reject this option; Torch/JAX arrays reject without
detaching. `delinearize_opts` requires `delinearize=True` and accepts only
`rtol`, `preserve_zeros`, and `max_sweeps`.

The returned report and `mpo.pepsy_channel_report` describe the result's final
`bond_dimensions` and `output_dense_entries`, with `channel_bond_dimensions`
and a nested `delinearization` report recording the separate stages.
`channels.report` still describes the immutable channel plan. Its array
kernels and symbolic maps are unchanged; QR choices are recomputed at each
evaluation and do not become frozen autodiff maps. `replay_factorizations=0`
describes the channel assembly kernel; the explicit numerical postprocessing
does perform QR solves. Validate final accuracy and dimensions separately.

For a single call from Hamiltonian terms, both
[`exp_mpo_cluster` and `exp_mpo_cluster_product`](mpo_cluster.md#integrated-automaton-plus-qr-construction)
accept `preparation="automaton", delinearize=True`.

## PEPO preparation with exact edge sharing

```python
channels = prepare_cluster_channels(
    pepo_builder, step, reference_parameters,
    preparation="symbolic", max_bond=12,
    samples=[{"parameters": another_reference}],
)
pepo = channels.exp(step, current_parameters)
```

This opt-in path first represents every local residual matrix entry by an
independent formal atom. It removes formally zero fixed-identity blocks and
merges identical graph-edge slices. A slice includes **all** other virtual
indices and both physical indices. One endpoint selects a representative,
and the opposite endpoint sums the matching channels. This local tensor
identity holds on branching and looped networks without a one-dimensional
continuation assumption. The neutral singleton rail stays separate.

Graph reduction happens **before** square routing, reducing the dimensions
of wires whose products would otherwise be enumerated. The routed template
then shares identical square-edge slices. Routing computes lexicographic
channel labels arithmetically, without Cartesian lookup dictionaries.
Additional references and tangent snapshots evaluate static linear gathers
from fresh residuals; they do not rebuild the numerical expanded PEPO.
Graph maps are folded into the frozen replay constants before compilation.

The construction preserves the whole family of independently variable
residuals, including derivatives at zeros. It does not merge independent
parameters just because their current values agree. `max_bond=None` adds no
channel approximation, while any further QR cap acts on the smaller space.
`structural_reuse=False` disables slice sharing but still uses the reusable
template and removes formal identity zeros. `preparation="reference"`
remains the default and preserves the earlier preparation policy.

The exact quotient changes the virtual gauge. **At the same tight cap, its
operator or gradient error can be worse** than reference preparation; local
QR defects do not certify the global objective. The measurements include
this case. Full channels recover the same selected cluster expansion.
This is not a globally minimal PEPO: local factors and remaining routed
blocks are enumerated once, dense residual exponentials remain, and crossing
wires with independent contents cannot generally be merged. See the
[PEPO measurements and limits](../../development/notes/2026-09-29-symbolic-pepo-channels.md).

## PEPO reduction using the declared Pauli algebra

```python
channels = prepare_cluster_channels(
    pepo_builder, step, reference_parameters,
    preparation="algebraic", max_bond=None,
)
pepo = channels.exp(step, current_parameters)
```

For fixed qubit Pauli product terms, this mode proves more identities than
independent-entry sharing. Each cluster uses the Pauli span closed under
multiplication of **all declared terms contained in it**, across all ordered
factors. Terms remain in this inventory even when their coefficients are
zero. Local ordered exponentials, products and connected subtraction all
stay in that span. No commutation or factor reordering is assumed.

An independent formal coefficient represents each allowed Pauli direction.
Exact rational elimination finds constant linear dependencies between full
edge slices, including their other virtual indices. It keeps selected
original slices and transfers their linear combinations to the other
endpoint, protecting the singleton rail. This is a local tensor identity
valid on branches and loops. Reduction runs on graph edges before routing,
then on square edges. It is not global symbolic minimization.

With `max_bond=None`, **neither preparation nor replay uses SVD or QR**.
The same holds when `max_bond` is at least every structural bond dimension.
These exact paths validate reference and tangent specifications but skip all
numerical residual evaluations because no reference QR projection is needed.
Exact reference, frontier and automaton plans still need one base evaluation to
discover their numerical layout, but skip optional samples and tangents when the
cap cannot bind.
The proof has no numerical rank tolerance and never divides by a live
coefficient. Numerical endpoint maps introduce ordinary floating-point
roundoff. Shapes stay fixed at zero parameters and across repeated calls.
The real Weyl basis `X^x Z^z` keeps symbolic entries rational; the phase of
`Y = i XZ` stays in live coefficients. A full Pauli span automatically uses
independent matrix-entry coordinates to avoid a pointless basis expansion.

Replay projects residuals onto this fixed span with Walsh transforms formed
from fixed 2-by-2 contractions. This is the identity on the declared family;
it also defines what `bind_assembler` does to arbitrary supplied matrices
outside that span. `pack_residuals` still returns the original dense cluster
residuals. The local matrix exponentials have not been replaced by a reduced
algebra exponential evaluator. Public evaluation checks the static Pauli
inventory; changing operator labels/supports requires a new plan. Compiled
assembly kernels assume that prepared inventory is fixed.

The mode accepts fixed finite NumPy local matrices exactly proportional to
I, X, Y or Z. Non-Pauli, higher-dimensional, live local-operator, string and
native-array inputs are unsupported; use the existing preparation modes
where applicable. Live scalar coefficients, independent coefficient vectors,
Torch CPU/CUDA and JAX remain supported. No reference equality proves an
identity. `structural_reuse=False` disables edge elimination, retaining the
Pauli projection and reusable templates.

An optional smaller `max_bond` still uses the existing reference QR outside
autodiff and defines an approximation. Its quality depends on the changed
gauge, so a smaller exact representation need not improve a tight cap.
Rational elimination can cost more preparation time than identical-slice
sharing. There is no guaranteed chi reduction or speedup. Inspect
`report['residual_algebra_dimensions']`,
`report['unrestricted_residual_dimensions']`, `bond_dimensions` and
`structural_reuse`; see the [measurements and validation](../../development/notes/2026-09-29-algebraic-pepo-channels.md).

## Exact relabeling versus projection

Graph PEPO materialization now uses only the structural channel ids present
on each edge, matching square PEPO's existing edge-local relabeling. This
removes global padding exactly, with no change to derivatives. Inspect
`active.bond_dimensions`; `bond_dim` remains the global label-space size.
`to_tensor_network(compact_bonds=False)` retains padded diagnostic output.

For MPO channel plans, `structural_reuse=True` (the default) alternates exact
prefix and suffix sharing within charge sectors. It compares formal
expressions with an independent atom for every local residual matrix entry.
Equal current values or zero coefficients never justify a merge. One
endpoint selects a representative transition while its neighbor adds the
incoming histories, preserving path multiplicity. The rail stays separate.
This is a conservative transition quotient, not a globally minimal weighted
automaton; it does not infer algebraic relations between different residuals.
It reuses the package's sparse virtual-axis gather/merge machinery.

`max_bond=None` keeps the resulting exact representation. The cap applies
**after** this exact sharing, independently of `builder.cluster_size`.
This removes additional channel-projection error; the finite spatial cluster
cutoff can still differ from the full global exponential/product.
`structural_reuse=False` retains the earlier MPO gauge. New capped MPO plans
can differ numerically because reference projection is gauge dependent.
Graph/square PEPOs use the separate edge-slice quotient described above when
symbolic preparation is selected; they do not use the one-dimensional quotient.

A cap below the remaining structural dimensions is a
**reference-dependent approximation**, even if
training defects vanish. QR preparation chooses a subspace of one endpoint's
local unfolding, keeps the neutral singleton rail, and respects native MPO
charge groups. It selects the endpoint with the smaller local training
defect. This is not canonical/environment-optimal compression, and QR pivot
threshold `rtol` is not an operator error tolerance. `bases` represents
the effective first-endpoint maps; structural sharing requires separate
opposite-endpoint maps, so use `bind_assembler` / `bind_projector` instead of
manually assuming that `bases.conj()` describes the other endpoint.

Additional `samples` contain `step`, `parameters` and/or `coefficients`;
omitted arguments inherit the primary reference. Values of a parameter
mapping replace that mapping rather than merging individual keys. The
source's usual parameter/vector exclusivity rules still apply.
`tangent_pairs=[(plus_arguments, minus_arguments, denominator), ...]` adds
finite-difference block derivatives to the reference snapshots. Preparation
detaches these explicit references; neither their arrays nor their autodiff
graphs are retained. Only static bases and layout metadata survive.

Train on representative values and derivative directions, particularly near
zero initialization. A zero reference alone cannot identify all directions
needed later. Validate both the operator and objective gradient against the
uncompressed expansion at held-out parameters. Increasing `max_bond`, adding
references or refreshing the plan can improve accuracy, but changes the
projected objective. Keep one plan fixed during a differentiable evaluation
and any line search intended to measure that same objective.

## Autodiff and compilation

Torch CPU/CUDA and JAX preserve live coefficients and fixed shapes. The
checked JAX path includes native U1 MPOs and complete compact trace JIT.
Fixed projection supports derivatives of the projected operator at zeros;
it does not differentiate QR preparation or provide the gradient of the
uncompressed operator automatically.

The public numerical boundary permits separate kernel compilation:

```python
residuals = channels.pack_residuals(step, parameters)
assemble = channels.bind_assembler(residuals[0])
assemble = torch.compile(
    assemble, backend="aot_eager", fullgraph=True, dynamic=False,
)
small_arrays = assemble(residuals)
# Recompute residuals from fresh parameters on every evaluation.
```

`bind_assembler` captures constants only, including native charge gathers and
tree permutations. Backend transpose/einsum callables are resolved during
binding so lazy namespace caching does not mutate state during Torch capture.
Rebind after changing backend, dtype or device, and
compile separately for each fixed topology and chi. This complete assembly
kernel has passed full-graph capture and backward on dense/native MPO,
graph PEPO and square PEPO;
`aot_eager` is not an optimized-runtime benchmark. Complete Torch capture of
the exponential/residual evaluator is not claimed. JAX can trace a scalar closure
around `channels.trace_exp`; Python network objects are not JIT outputs.
The earlier `pack` / `bind_projector` pair remains available for diagnostic
comparison. It evaluates sparse blocks for the selected preparation and is not used
by `arrays`, `exp` or `trace_exp`.

## Scope and storage

- Requires `factorization="fixed"`. MPOs support direct and uncapped recursive
  source builders, local
  `cutoff=0` or `None`, and `max_bond=None` on the source builder. The channel
  plan supplies its own separate virtual cap.
- Supports dense spin MPO/PEPOs and native bosonic MPO sectors. Native PEPO
  projection and graded fermionic construction are unsupported.
- With reference preparation, direct graph collection budgets and
  approximation choices remain in force.
  A recursive source prepares an equivalent exact direct topology subject to
  its collection budget; exceeding that budget raises rather than silently
  dropping collections. Frontier preparation instead uses its active-state
  budget and requires an exact target as described above.
- Replay keeps only compact site accumulators, dense local residuals/factors,
  and the planned contraction intermediates. Gap wires, crossing MPO paths
  and square routing wires are fused into these contractions. Local clusters
  and preparation topology can still grow exponentially.
- `memory_budget` checks symbolic entry estimates, explicit preparation matrices, constant maps and
  output buffers. It excludes source enumeration, contraction workspace,
  backward storage and total process memory.

Reports identify exact structural preparation with `method` values such as
`exact-symbolic-channels` and `fixed-pauli-algebra`. `projection_applied`
and `rank_selection` distinguish those paths from `frozen-channel-qr`,
while `reference_evaluations` counts numerical residual builds used during
preparation. Exact uncapped or nonbinding symbolic/algebraic plans report zero.
Reports also include `assembly="fused-local-contractions"`,
`replay_history_blocks=0`, `peak_projection_entries`,
`projection_constant_entries`, and `local_residual_entries`, alongside the
original bond dimensions and local defects. Peak entries describe individual
planned projection operands/results, not total process or backward memory.
Native storage estimates describe dense equivalents. Local defects are not
global error bounds. `structural_reuse` reports the exact quotient's dimensions,
removed-channel count and symbolic identity policy, or is `None` when unused.
`preparation="frontier"` additionally reports `full_collections_enumerated=0`,
`frontier_state_counts`, `frontier_transition_count`, `frontier_state_budget`
and pre-sharing `frontier_bond_dimensions`. A frontier state can carry multiple
virtual channels from its local cluster factors.
`preparation="symbolic"` reports `expanded_reference_builds=0`, graph bond
dimensions before/after sharing, formally zero block removals, remaining
symbolic routed blocks, and the `unreduced_bond_dimensions` /
`unreduced_sparse_blocks` avoided by early sharing. Its `input_bond_dimensions`
describe the routed template after graph sharing but before square sharing
and QR. The nested structural report separates graph and routed quotients.
See the [symbolic sharing and chi sweep](../../development/notes/2026-09-29-symbolic-cluster-channels.md),
[fused replay evidence](../../development/notes/2026-09-29-fused-cluster-channels.md)
and the earlier [literature and approximation study](../../development/notes/2026-09-29-compact-cluster-channels.md).
