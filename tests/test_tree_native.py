"""Tree native-array regression tests."""


import numpy as np
import pytest
import quimb.tensor as qtn
import pepsy
from pepsy.optimizers.tree import TreeLayoutFinder
from pepsy.optimizers.tree import TreeOptimizer
from pepsy.optimizers.tree import TreePlan
from pepsy.optimizers.tree.ttn import _native_qr_block_scaled


from _tree_test_helpers import (
    _fidelity,
)


pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.tree]


def test_tree_native_fermionic_gate_stream_matches_mps():
    """Native (dim-4 Symmray) Fermi-Hubbard gates evolve correctly on a tree.

    The tree gate engine must apply block-sparse fermionic gates without
    reshaping them into base-2 sub-legs. At a bond dimension large enough to be
    exact, the tree real-time evolution must reproduce the MPS reference
    (identical seed) to numerical precision.
    """
    pytest.importorskip("symmray")
    tensors = pepsy.tensors

    Lx, Ly, L = 2, 2, 4
    t, U, dt = 1.0, 8.0, 0.05
    state_dtype = "complex128"

    fermion = pepsy.Fermion(spinful=True, symmetry="U1U1", dtype=state_dtype)
    setup = fermion.lattice_half_filling(Lx, Ly, pattern="checkerboard", cyclic=True)
    mapper = tensors.OneDMap(Lx, Ly, mode="snake")
    _, coo2idx = mapper.build()
    edges_1d = [
        tuple(sorted((coo2idx[a], coo2idx[b]))) for a, b in setup.edges
    ]
    sites = tuple(range(L))
    occ_1d = {coo2idx[coo]: c for coo, c in setup.occupations.items()}
    occupations = tuple(occ_1d[p] for p in range(L))

    def build_stream():
        half = dt / 2
        u_hop = fermion.hopping_gate(half, t=t, imaginary=False)
        onsite = [
            (fermion.onsite_gate(half, site=s, U=U, mu=0.0, imaginary=False), s)
            for s in sites
        ]
        layers = fermion.edge_coloring_layers(edges_1d)
        fwd = [(u_hop, e) for layer in layers for e in layer]
        rev = [(u_hop, e) for layer in reversed(layers) for e in reversed(layer)]
        return onsite + fwd + rev + onsite

    gates = build_stream() * 3
    native_hopping = fermion.hopping_gate(dt / 2, t=t, imaginary=False)
    native_submpo = qtn.MatrixProductOperator.from_dense(
        native_hopping, dims=(4, 4), sites=(0, 2), L=L,
    )
    assert all(
        type(tensor.data).__name__ == "U1U1FermionicArray"
        for tensor in native_submpo.tensors
    )

    seed_mps = pepsy.ps_to_mps(
        L, fermion=fermion, occupations=occupations,
        seed=1234, dtype=state_dtype, cyclic=False,
    )
    plan = TreeLayoutFinder(
        [(fermion.hopping_gate(0.1, t=t, imaginary=False), e) for e in edges_1d],
        n=L, chi=8, objective="hybrid",
    ).recommend_arities((2, 3, 4), seed=0)["plan"]
    seed_ttn = pepsy.ps_to_ttn(
        L, tree=plan, fermion=fermion, occupations=occupations, dtype=state_dtype
    )

    # The tree and MPS seeds must represent the identical fermionic state.
    assert float(tensors.tn_fidelity(seed_mps, seed_ttn)) > 1 - 1e-10

    # Large-chi references (no truncation for L=4: exact bond is 16).
    mps_exact = pepsy.MpsOptimizer(
        seed_mps.copy(), gates=gates, chi=256, mode="mpo", inplace=False,
    )
    mps_exact.run(cutoff=0.0)
    engine = TreeOptimizer(
        gates, n=L, tree=plan, state=seed_ttn.copy(), chi=256, cutoff=0.0,
        mode="mpo", run=False,
    )
    engine.run()

    # The two public modes are algebraically the same gate SVD: both defer
    # truncation until the whole path has been updated. They can only differ by
    # floating-point roundoff from the extra MPO factorisation/QR gauges.
    direct = TreeOptimizer(
        None, n=L, tree=plan, state=seed_ttn.copy(), chi=256, cutoff=0.0,
        mode="direct", run=False,
    )
    direct.run(gates)

    auto = TreeOptimizer(
        None, n=L, tree=plan, state=seed_ttn.copy(), chi=256, cutoff=0.0,
        mode="auto", run=False,
    )
    auto.run(gates)

    assert float(tensors.tn_fidelity(engine.p, direct.p)) > 1 - 1e-8
    assert float(tensors.tn_fidelity(auto.p, direct.p)) > 1 - 1e-10
    assert float(tensors.tn_fidelity(direct.p, mps_exact.p)) > 1 - 1e-9
    assert float(tensors.tn_fidelity(engine.p, mps_exact.p)) > 1 - 1e-8


@pytest.mark.filterwarnings(
    "ignore:TreeOptimizer is converting a gate/operator payload"
)
def test_native_complex64_threading_disables_zero_phase_stabilization(monkeypatch):
    """Native path threading keeps structural-zero QR sectors finite."""
    pytest.importorskip("symmray")
    L = 4
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex64",
    )
    plan = TreePlan.from_order(range(L), structure="balanced")
    state = pepsy.ps_to_ttn(
        L,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        dtype="complex64",
    )
    gate = fermion.hopping_gate(0.05, t=1.0, imaginary=False)

    threaded = False
    qr_stabilized = []
    original_hop = TreeOptimizer._fermionic_thread_hop
    original_split = qtn.Tensor.split

    def traced_hop(self, u, v):
        nonlocal threaded
        threaded = True
        try:
            return original_hop(self, u, v)
        finally:
            threaded = False

    def traced_split(self, *args, **kwargs):
        if threaded and kwargs.get("method") == "qr":
            qr_stabilized.append(kwargs.get("stabilized"))
        return original_split(self, *args, **kwargs)

    monkeypatch.setattr(TreeOptimizer, "_fermionic_thread_hop", traced_hop)
    monkeypatch.setattr(qtn.Tensor, "split", traced_split)

    optimizer = TreeOptimizer(
        None,
        n=L,
        tree=plan,
        state=state,
        chi=16,
        cutoff=0.0,
        mode="direct",
        run=False,
    )
    optimizer.apply_2q(gate, 0, 2)

    assert qr_stabilized
    assert qr_stabilized == [False] * len(qr_stabilized)
    assert all(
        np.isfinite(np.asarray(block)).all()
        for tensor in optimizer.tn.tensors
        for block in tensor.data.blocks.values()
    )


@pytest.mark.filterwarnings(
    "ignore:TreeOptimizer is converting a gate/operator payload"
)
def test_native_complex64_all_lossless_qr_routes_skip_zero_phase(monkeypatch):
    """All native tree QR routes avoid phase division on zero sectors."""
    pytest.importorskip("symmray")
    L = 4
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex64",
    )
    plan = TreePlan.from_order(range(L), structure="balanced")
    state = pepsy.ps_to_ttn(
        L,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        dtype="complex64",
    )
    hopping = fermion.hopping_gate(0.05, t=1.0, imaginary=False)
    routed_ops = [
        fermion.onsite_gate(
            0.01, site=site, U=8.0, mu=0.0, imaginary=False,
        )
        for site in (0, 1, 2)
    ]
    submpo = qtn.MPO_product_operator(
        routed_ops,
        sites=(0, 1, 2),
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )

    observed = []
    original_split = qtn.Tensor.split

    def traced_split(self, *args, **kwargs):
        if kwargs.get("method") == "qr" and hasattr(self.data, "blocks"):
            observed.append(kwargs.get("stabilized"))
        return original_split(self, *args, **kwargs)

    monkeypatch.setattr(qtn.Tensor, "split", traced_split)

    for operation in (
        lambda: TreeOptimizer(
            None, n=L, tree=plan, state=state.copy(), chi=16,
            cutoff=0.0, mode="direct", run=False,
        ).apply_2q(hopping, 0, 1),
        lambda: TreeOptimizer(
            None, n=L, tree=plan, state=state.copy(), chi=16,
            cutoff=0.0, mode="direct", run=False,
        ).apply_2q(hopping, 0, 2),
        lambda: TreeOptimizer(
            None, n=L, tree=plan, state=state.copy(), chi=16,
            cutoff=0.0, mode="submpo", run=False,
        ).apply_submpo(submpo, (0, 1, 2)),
    ):
        operation()

    assert observed
    assert observed == [False] * len(observed)


@pytest.mark.parametrize(
    ("symmetry", "occupations"),
    [
        ("U1", (1, 1, 1, 1)),
        ("U1U1", ((1, 0), (0, 1), (1, 0), (0, 1))),
    ],
)
def test_native_tree_direct_mpo_and_submpo_match_without_global_fidelity(
    symmetry, occupations,
):
    """Native gate kernels agree without Cotengra's process-based fidelity."""
    pytest.importorskip("symmray")
    L = 4
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(L), structure="balanced")
    seed = pepsy.ps_to_ttn(
        L,
        tree=plan,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
    )
    hopping = fermion.hopping_gate(0.05, t=1.0, imaginary=False)
    onsite = fermion.onsite_gate(
        0.03, site=1, U=8.0, mu=0.0, imaginary=False,
    )
    submpo = qtn.MatrixProductOperator.from_dense(
        hopping, dims=(4, 4), sites=(0, 2), L=L,
    )

    def dense_vector(opt):
        tensor = opt.tn.contract(all, optimize="greedy").transpose(
            *(opt.tn.site_ind(q) for q in range(L))
        )
        return np.asarray(tensor.data.to_dense()).reshape(-1)

    outputs = []
    for mode in ("direct", "mpo"):
        opt = TreeOptimizer(
            None,
            n=L,
            tree=plan,
            state=seed.copy(),
            chi=64,
            cutoff=0.0,
            mode=mode,
            run=False,
        )
        opt.apply_1q(onsite, 1)
        opt.apply_2q(hopping, 0, 2)
        outputs.append(dense_vector(opt))
        assert opt.tn.validate(check_canonical=True) is opt.tn

    submpo_opt = TreeOptimizer(
        None,
        n=L,
        tree=plan,
        state=seed.copy(),
        chi=64,
        cutoff=0.0,
        mode="submpo",
        run=False,
    )
    submpo_opt.apply_1q(onsite, 1)
    submpo_opt.apply_submpo(submpo, (0, 2))
    outputs.append(dense_vector(submpo_opt))
    assert submpo_opt.tn.validate(check_canonical=True) is submpo_opt.tn

    for output in outputs[1:]:
        assert _fidelity(outputs[0], output) > 1 - 1e-10

    before = dense_vector(submpo_opt)
    mpo_value = submpo_opt.expectation_mpo(
        submpo, (0, 2), max_bond=64,
    )
    direct_value = submpo_opt.tn.local_expectation(hopping, (0, 2))
    assert complex(mpo_value) == pytest.approx(complex(direct_value), abs=1e-5)
    assert np.allclose(dense_vector(submpo_opt), before)


def test_native_fermionic_submpo_keeps_graded_hub_recovery(monkeypatch):
    """Native routed Q metadata skips only already-proven graded QR."""
    pytest.importorskip("symmray")
    L = 4
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    occupations = ((1, 0), (0, 1), (1, 0), (0, 1))
    plan = TreePlan.from_order(range(L), structure="balanced")
    seed = pepsy.ps_to_ttn(
        L,
        tree=plan,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
    )
    sites = (0, 1, 2)
    local_ops = [
        fermion.onsite_gate(
            0.01, site=site, U=8.0, mu=0.0, imaginary=False
        )
        for site in sites
    ]
    submpo = qtn.MPO_product_operator(
        local_ops,
        sites=sites,
        L=L,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
    )
    candidate = TreeOptimizer(
        None,
        n=L,
        tree=plan,
        state=seed.copy(),
        chi=64,
        cutoff=0.0,
        run=False,
    )
    reference = TreeOptimizer(
        None,
        n=L,
        tree=plan,
        state=seed.copy(),
        chi=64,
        cutoff=0.0,
        run=False,
    )
    installs = []
    install_routed = candidate._install_routed_subtree
    compressions = []
    compress_edge = candidate._compress_edge_with_diagnostics

    def traced_install(local, snodes, hub):
        installs.append((frozenset(snodes), hub))
        assert candidate.tn.fermionic
        return install_routed(local, snodes, hub)

    def traced_compress_edge(
        u, v, *, max_bond=None, cutoff=None, reduced=True,
        reduction_proven=False,
    ):
        compressions.append(reduced)
        return compress_edge(
            u,
            v,
            max_bond=max_bond,
            cutoff=cutoff,
            reduced=reduced,
            reduction_proven=reduction_proven,
        )

    monkeypatch.setattr(candidate, "_install_routed_subtree", traced_install)
    monkeypatch.setattr(
        candidate, "_compress_edge_with_diagnostics", traced_compress_edge,
    )
    candidate.apply_submpo(submpo, sites)
    for site, op in zip(sites, local_ops):
        reference.apply_1q(op, site)

    def dense_vector(opt):
        tensor = opt.tn.contract(all, optimize="greedy").transpose(
            *(opt.tn.site_ind(q) for q in range(L))
        )
        return np.asarray(tensor.data.to_dense()).reshape(-1)

    assert installs
    assert compressions
    assert all(reduced in {True, "left"} for reduced in compressions)
    assert any(reduced == "left" for reduced in compressions)
    assert (
        _fidelity(dense_vector(candidate), dense_vector(reference))
        > 1 - 1e-10
    )
    assert candidate.validate_isometry_metadata() is candidate
    for nid, toward in candidate.isometry_map().items():
        if toward is not None:
            assert candidate.can_skip_canonize(nid, toward)
    assert candidate.tn.validate(check_canonical=True) is candidate.tn


@pytest.mark.parametrize(
    ("symmetry", "spinful", "occupations"),
    [
        ("U1", False, (1, 0, 1, 0)),
        ("U1U1", True, ((1, 0), (0, 1), (1, 0), (0, 1))),
    ],
)
def test_native_fermionic_left_inds_skips_lossless_qr(
    symmetry, spinful, occupations, monkeypatch,
):
    """Symmray U1 variants reuse native QR isometry metadata safely."""
    pytest.importorskip("symmray")
    L = 4
    fermion = pepsy.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(L), structure="balanced")
    ttn = pepsy.ps_to_ttn(
        L,
        tree=plan,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
    )
    target = plan.leaf_of_qubit[0]
    ttn.shift_orthogonality_center(target)

    source = next(
        nid for nid, toward in ttn.isometry_map().items()
        if toward == target and ttn.can_skip_canonize(nid, toward)
    )
    before = {
        nid: np.asarray(ttn.node_tensor(nid).data.to_dense()).copy()
        for nid in plan.nodes()
    }
    calls = []
    graded_qr = ttn._fermionic_canonize_edge_

    def traced_qr(*args, **kwargs):
        calls.append((args, kwargs))
        return graded_qr(*args, **kwargs)

    monkeypatch.setattr(ttn, "_fermionic_canonize_edge_", traced_qr)
    ttn.canonize_edge_(source, target)

    assert not calls
    assert ttn.orthogonality_center == target
    assert ttn.is_canonical_form(target)
    assert ttn.validate(check_canonical=True) is ttn
    for nid in plan.nodes():
        np.testing.assert_array_equal(
            np.asarray(ttn.node_tensor(nid).data.to_dense()), before[nid]
        )


def test_native_truncating_compression_keeps_explicit_svd(monkeypatch):
    """A positive cutoff never turns a native truncation into metadata-only."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    ttn = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        dtype="complex128",
    )
    target = plan.leaf_of_qubit[0]
    ttn.shift_orthogonality_center(target)
    source = next(
        nid for nid, toward in ttn.isometry_map().items()
        if toward == target and ttn.can_skip_canonize(nid, toward)
    )

    calls = []
    compress = ttn._fermionic_compress_edge_

    def traced_compress(*args, **kwargs):
        calls.append((args, kwargs))
        return compress(*args, **kwargs)

    monkeypatch.setattr(ttn, "_fermionic_compress_edge_", traced_compress)
    ttn.compress_edge_(
        source, target, max_bond=64, cutoff=1e-10, cutoff_mode="rel",
    )

    assert calls


def test_native_one_sided_compression_qr_reduces_before_svd(monkeypatch):
    """A proven native isometry sends only its reduced core to the SVD."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    ttn = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        dtype="complex128",
    )
    target = plan.leaf_of_qubit[0]
    ttn.shift_orthogonality_center(target)
    source = next(
        nid for nid, toward in ttn.isometry_map().items()
        if toward == target and ttn.can_skip_canonize(nid, toward)
    )

    split_methods = []
    original_split = qtn.Tensor.split

    def traced_split(self, *args, **kwargs):
        method = kwargs.get("method")
        if method in {"qr", "svd"} and hasattr(self.data, "blocks"):
            split_methods.append((method, tuple(self.shape)))
        return original_split(self, *args, **kwargs)

    monkeypatch.setattr(qtn.Tensor, "split", traced_split)
    ttn._fermionic_compress_edge_(
        target,
        source,
        max_bond=64,
        cutoff=1e-10,
        cutoff_mode="rel",
        absorb="right",
        reduced="left",
    )

    assert [method for method, _shape in split_methods] == ["qr", "svd"]
    assert ttn.validate(check_canonical=True) is ttn


def test_native_profile_reports_reduced_compression_routes():
    """Native profiling exposes reduced compression and no hidden fallback."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1U1",
        dtype="complex128",
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    state = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=((1, 0), (0, 1), (1, 0), (0, 1)),
        dtype="complex128",
    )
    hopping = fermion.hopping_gate(0.05, t=1.0, imaginary=False)
    optimizer = TreeOptimizer(
        None,
        n=4,
        tree=plan,
        state=state,
        chi=1,
        cutoff=0.0,
        mode="direct",
        profile=True,
        run=False,
    )

    optimizer.apply_2q(hopping, 0, 2)
    report = optimizer.profile_report()
    routes = report["native_compression_routes"]

    assert routes
    assert routes.get("full_svd_fallback", 0) == 0
    assert sum(
        count for route, count in routes.items()
        if route != "full_svd_fallback"
    ) == report["by_kind"]["native_compression_route"]["count"]
    assert optimizer.tn.validate(check_canonical=True) is optimizer.tn


@pytest.mark.parametrize(
    ("symmetry", "spinful", "occupations"),
    [
        ("U1", False, (1, 0, 1, 0)),
        ("U1U1", True, ((1, 0), (0, 1), (1, 0), (0, 1))),
    ],
)
@pytest.mark.parametrize("state_dtype", ["complex64", "complex128"])
def test_native_one_sided_and_two_sided_compression_fidelity(
    symmetry, spinful, occupations, state_dtype,
):
    """Native left/right/two-sided reductions preserve the state and gauge."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype=state_dtype,
    )
    plan = TreePlan.from_order(range(4), structure="balanced")
    base = pepsy.ps_to_ttn(
        4,
        tree=plan,
        fermion=fermion,
        occupations=occupations,
        dtype=state_dtype,
    )
    target = plan.leaf_of_qubit[0]
    base.shift_orthogonality_center(target)
    source = next(
        nid for nid, toward in base.isometry_map().items()
        if toward == target and base.can_skip_canonize(nid, target)
    )
    cutoff = 1e-10 if state_dtype == "complex128" else 1e-7
    fidelity_floor = 1 - (1e-10 if state_dtype == "complex128" else 1e-5)

    for a, b, reduced in (
        (source, target, "right"),
        (target, source, "left"),
        (target, source, True),
    ):
        candidate = base.copy()
        candidate._fermionic_compress_edge_(
            a,
            b,
            max_bond=64,
            cutoff=cutoff,
            cutoff_mode="rel",
            absorb="right",
            reduced=reduced,
        )
        assert float(pepsy.tn_fidelity(base, candidate)) > fidelity_floor
        assert candidate.validate(check_canonical=True) is candidate


def test_native_complex64_qr_scales_low_norm_rank_deficient_block():
    """Native QR keeps tiny complex64 charge blocks finite and exact."""
    torch = pytest.importorskip("torch")
    block = torch.zeros((16, 18), dtype=torch.complex64)
    generator = torch.Generator().manual_seed(17)
    left = torch.randn((16, 12), generator=generator) * 1e-9
    right = torch.randn((12, 18), generator=generator)
    block = (left @ right).to(torch.complex64)

    q, _, r = _native_qr_block_scaled(
        block,
        method="qr",
        absorb="right",
        stabilized=False,
    )

    assert torch.isfinite(q).all()
    assert torch.isfinite(r).all()
    torch.testing.assert_close(
        q @ r,
        block,
        rtol=2e-4,
        atol=1e-12,
    )


def test_native_complex64_qr_leaves_healthy_block_on_native_path(monkeypatch):
    """Healthy complex64 blocks use one unmodified native Torch QR call."""
    torch = pytest.importorskip("torch")

    generator = torch.Generator().manual_seed(4108)
    block = torch.randn((6, 4), generator=generator).to(torch.complex64)
    qr_calls = []
    original_qr = torch.linalg.qr

    def record_qr_call(x, *args, **kwargs):
        qr_calls.append(x.detach().clone())
        return original_qr(x, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "qr", record_qr_call)
    q, _, r = _native_qr_block_scaled(
        block,
        method="qr",
        absorb="right",
        stabilized=False,
    )
    expected_q, expected_r = original_qr(block)

    assert len(qr_calls) == 1
    torch.testing.assert_close(qr_calls[0], block)
    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(r, expected_r)


def test_native_complex64_qr_scales_dynamic_range_block(monkeypatch):
    """Native QR scales moderate-norm blocks with tiny charge entries."""
    torch = pytest.importorskip("torch")

    block = torch.zeros((4, 10), dtype=torch.complex64)
    for index, magnitude in enumerate((8.9e-3, 8.9e-11, 8.9e-25, 8.9e-41)):
        block[index, index] = complex(magnitude, magnitude)

    qr_input_maxes = []
    original_qr = torch.linalg.qr

    def record_qr_input(x, *args, **kwargs):
        qr_input_maxes.append(float(x.abs().amax().item()))
        return original_qr(x, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "qr", record_qr_input)
    q, _, r = _native_qr_block_scaled(
        block,
        method="qr",
        absorb="right",
        stabilized=False,
    )

    assert len(qr_input_maxes) == 2
    assert qr_input_maxes[0] == pytest.approx(8.9e-3 * 2**0.5, rel=1e-6)
    assert 0.5 <= qr_input_maxes[1] < 1.0
    assert torch.isfinite(q).all()
    assert torch.isfinite(r).all()
    torch.testing.assert_close(
        q @ r,
        block,
        rtol=2e-4,
        atol=1e-12,
    )


def test_native_complex64_qr_handles_structurally_rank_deficient_block():
    """Native QR keeps structural rank-deficient blocks in complex64."""
    torch = pytest.importorskip("torch")
    block = torch.zeros((4, 10), dtype=torch.complex64)
    for index, magnitude in enumerate((2e-2, 2e-10, 2e-24, 2e-40)):
        block[index, index] = complex(magnitude, magnitude)

    q, _, r = _native_qr_block_scaled(
        block,
        method="qr",
        absorb="right",
        stabilized=False,
    )

    assert torch.isfinite(q).all()
    assert torch.isfinite(r).all()
    assert q.dtype == torch.complex64
    assert r.dtype == torch.complex64
    torch.testing.assert_close(
        q @ r,
        block,
        rtol=2e-4,
        atol=1e-12,
    )


@pytest.mark.parametrize("backend_name", ["torch", "cupy"])
def test_native_complex64_gpu_qr_retries_same_device_double_precision(
    monkeypatch, backend_name,
):
    """Failed GPU QR uses an optimized same-device complex128 retry."""
    if backend_name == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        backend_module = torch
        block = torch.tensor(
            [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
            dtype=torch.complex64,
            device="cuda",
        )
        dtype32 = torch.complex64
    else:
        cupy = pytest.importorskip("cupy")
        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CUDA is unavailable")
        except cupy.cuda.runtime.CUDARuntimeError as exc:
            pytest.skip(f"CUDA is unavailable: {exc}")
        backend_module = cupy
        block = cupy.asarray(
            [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]],
            dtype=cupy.complex64,
        )
        dtype32 = cupy.complex64

    import pepsy.optimizers.tree.ttn as ttn_module

    original_do = ttn_module.ar.do
    qr_dtypes = []

    def fail_complex64_qr(fn, value, *args, **kwargs):
        result = original_do(fn, value, *args, **kwargs)
        if fn == "linalg.qr":
            qr_dtypes.append(value.dtype)
            if value.dtype == dtype32:
                result = tuple(
                    backend_module.full_like(
                        factor, complex(float("nan"), float("nan")),
                    )
                    for factor in result
                )
        return result

    monkeypatch.setattr(ttn_module.ar, "do", fail_complex64_qr)
    q, _, r = _native_qr_block_scaled(
        block,
        method="qr",
        absorb="right",
        stabilized=False,
    )

    assert qr_dtypes == [dtype32, backend_module.complex128]
    assert q.dtype == dtype32
    assert r.dtype == dtype32
    assert bool(backend_module.isfinite(q).all())
    assert bool(backend_module.isfinite(r).all())
    assert float(backend_module.max(backend_module.abs(q @ r - block))) < 1e-5
