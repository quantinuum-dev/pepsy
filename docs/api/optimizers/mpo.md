# `pepsy.optimizers.mpo.optimizer`

`MpoOptimizer` accepts ordinary Quimb MPOs and Symmray block-sparse MPOs. For
a native graded fermion workflow, use the canonical
`Fermion.build_mpo(...)` entry point and replay the matching native gate
stream:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
edges = [(0, 1), (1, 2)]
hamiltonian = fermion.hamiltonian(edges, t=1.0, U=2.0, mu=0.1)
mpo = fermion.build_mpo(hamiltonian=hamiltonian, L=3)

opt = pepsy.MpoOptimizer(
    mpo,
    hamiltonian.trotter_gates(0.01),
    chi=16,
    mode="svd",
)
mpo = opt.run(progbar=False)
```

For dense MPOs, `mode="mpo"` accepts one- or multi-site dense gates, including
non-contiguous supports such as `(0, 1, 3)`. The gate matrix has dimension
`2**len(where)` by `2**len(where)` for qubit sites. `mode="svd"` and
`mode="dmrg"` retain their one-/two-site replay contracts. Native Symmray
`mode="mpo"` uses the symmetry-aware SVD route and therefore follows that
one-/two-site restriction.

For an explicit neutral term collection, arbitrary one- or multi-site support
is accepted:

```python
term = fermion.operator_term(
    [(1.0, ((0, "create_up"), (2, "annihilate_up")))],
    sites=(0, 2),
    add_hc=True,
)
mpo = fermion.build_mpo({(0, 2): term}, L=3)
```

The Jordan-Wigner compatibility path remains available by passing
`fermionic=False` to the same builder, together with the matching
`SymHamiltonian.jw_trotter_gates(...)` stream:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
edges = [(0, 1), (1, 2)]
hamiltonian = fermion.hamiltonian(edges, t=1.0, U=2.0, mu=0.1)
mpo = fermion.build_mpo(
    edges, L=3, t=1.0, U=2.0, mu=0.1, fermionic=False
)

opt = pepsy.MpoOptimizer(
    mpo,
    hamiltonian.jw_trotter_gates(0.01),
    chi=16,
    mode="svd",
)
mpo = opt.run(progbar=False)
```

Symmray gates keep their charge and dual metadata and are not coerced to dense
arrays. Native graded gates from `Fermion.strang_gate_stream(...)` are accepted
as well; even gates are adapted explicitly to the MPO's Jordan-Wigner
convention. `mode="svd"` and native `mode="mpo"` use symmetry-aware direct
compression, while native `mode="dmrg"` uses block-aware FIT. All three avoid
generic dense auxiliary MPO compression and bond padding, which do not support
multi-sector Symmray bonds reliably.

Native MPO tensors retain their Symmray graded metadata throughout replay and
compression. `mode="svd"` and `mode="mpo"` use the block-aware direct SVD path
for native Symmray MPOs, while `mode="dmrg"` uses native block-aware FIT; the
optimizer does not require a dense conversion of the input MPO.

For MPO DMRG, `fit_block_size=2` is the default native two-site FIT update.
`fit_block_size=3` enables the corresponding three-site effective tensor and
two direction-aware SVD splits; an interval containing only two sites
automatically uses the two-site update. The optimizer forwards `cutoff`,
`cutoff_mode`, `chi`, and `fit_sweep_sequence` to every output FIT split.
The first one or two sweeps can use the three-site block through
`fit_three_site_sweeps`; remaining sweeps use one-site refinement. Batched
targets accept `fit_max_span="auto"` to split disjoint gates before a wide
active window is formed. `cutoff="auto"` selects a dtype-aware cutoff.
`target_cutoff` controls only construction of the disposable target MPO, so
target construction and output compression remain separate choices. Use
`fit_block_size=1` to retain the legacy fixed-rank one-site path.

As with MPS DMRG, `n_iter` counts FIT sweeps. `mode="mpo"` applies each gate
with one direct MPO compression step and does not perform variational sweeps;
the two modes therefore have different one-iteration behavior for a
non-local gate even when they use the same `chi` and SVD cutoff.

`ham_tn.build_mpo(..., fermionic=True)` is also routed to the native
`Fermion.build_mpo(...)` entry point. `Fermion.to_mpo(...)` remains a
compatibility alias. Use `to_backend=...` on the model-facing builder when
the stored blocks must be moved to Torch or another supported backend.

Native MPO assembly/replay is also measurable with a native fermionic MPS.
`MpsEnergyOptimizer` applies the native MPO sitewise as a factorized graded
MPO-MPS network, preserving Symmray ordering while retaining MPO bond scaling.
Repeated evaluations reuse the optimizer's contraction paths. Optional
controlled truncation is available through
``native_mpo_compression={"max_bond": ..., "cutoff": ..., "method": "svd"}``;
the default remains exact and uncompressed.

To compress an existing MPO without replaying gates, use
`MpoOptimizer(mpo, gates=[], chi=...).compress()`. Symmray may retain a small
sector-multiplicity overshoot above the requested numeric bond cap.

## Parameterized higher-order MPO construction

`MPOBasis` is the reusable API for a Hamiltonian whose couplings change during
an optimization:

```python
basis = MPOBasis.from_pauli_terms(
    L,
    [((i, i + 1), "ZZ", MPOParameter("J")) for i in range(L - 1)]
    + [((i,), "X", MPOParameter("hx")) for i in range(L)],
)
U = basis.extensive_exponential(
    -1j * dt,
    {"J": J, "hx": hx},
    order=2,
    mode="optimal",
)
```

`mode="base"` applies Algorithms 1--2. `mode="optimal"` applies Algorithms
1--3, where Algorithm 3 adds selected order-`N + 1` terms without increasing
the analytical history bond dimension. `mode="approximate"` additionally
enables the paper's Algorithm 4. `MPOProductTerm` also accepts arbitrary
one-dimensional supports, for example `(0, 1, 3)` with operators `"XYZ"`.

The first build compiles the first-degree MPO topology and symbolic history
plans. Later calls reuse only level indices, reachability information, and
merge/insertion plans; local tensors are rebuilt from the supplied numerical
parameters. This separation keeps Torch and JAX autodiff graphs correct. The
plan state is visible through `basis.cache_info["history"]`, including
`compression_plan_orders`, `extension_plan_orders`, and
`extension_plan_batches`. `cache_history=False` avoids retaining a new raw
history topology for one-off large builds, while `history_storage="streaming"`
retains only adjacent history cuts during generation before assembling the
current MPO needed by Algorithms 1--3.

> API details are maintained as handwritten Markdown in this page.
