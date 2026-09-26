"""MPS layout regression tests."""


import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn
import pepsy as py


from _mps_test_helpers import (
    _two_branch_flip_submpo,
)


pytestmark = [pytest.mark.core, pytest.mark.mps]


def _nonuniform_product_mps():
    """Return a non-translationally-invariant complex product state."""
    return qtn.MPS_product_state(
        [
            np.array([1.0, 0.0], dtype=complex),
            np.array([0.0, 1.0], dtype=complex),
            np.array([np.cos(0.3), np.sin(0.3)], dtype=complex),
            np.array([np.cos(0.5), 1j * np.sin(0.5)], dtype=complex),
        ]
    )


def test_mps_optimizer_layout_finder_api_is_separate_module():
    """Layout finder should live outside the optimizer implementation file."""
    from pepsy.optimizers.mps import MpsGateStreamLayoutFinder
    from pepsy.optimizers.mps.layout import MpsGateStreamLayoutFinder as LayoutFinder

    assert MpsGateStreamLayoutFinder is LayoutFinder
    assert py.MpsOptimizer.LayoutFinder is LayoutFinder


def test_layout_execution_preserves_subclass_state_hooks():
    """Extracted operations must dispatch through overridden optimizer hooks."""
    class ObservedOptimizer(py.MpsOptimizer):
        def _set_site_order(self, order):
            order = tuple(order)
            self.observed_orders = getattr(self, "observed_orders", []) + [order]
            return super()._set_site_order(order)

        def _product_site_vector(self, state, physical_site):
            self.observed_sites.append(physical_site)
            return super()._product_site_vector(state, physical_site)

    state = _nonuniform_product_mps()
    expected = state.to_dense().reshape(-1)
    opt = ObservedOptimizer(state, gates=[], chi=8, mode="svd")
    opt.observed_sites = []
    order = (0, 2, 3, 1)

    assert opt.apply_layout(order, layout_report=False) is opt
    assert opt.observed_orders[-1] == order
    assert opt.observed_sites == [0, 1, 2, 3]
    assert opt.qubits == opt.logical_order == list(order)
    np.testing.assert_allclose(opt.to_dense().reshape(-1), expected, atol=1e-12)
    np.testing.assert_array_equal(opt.remap_sample([0, 1, 2, 3]), [0, 3, 1, 2])

    copied = opt.copy()
    assert type(copied) is ObservedOptimizer
    assert copied.logical_order == opt.logical_order
    assert copied.logical_order is not opt.logical_order
    np.testing.assert_allclose(copied.to_dense().reshape(-1), expected, atol=1e-12)


def test_mps_optimizer_gate_stream_layout_remaps_long_range_path():
    """Gate-stream layout should find a short order without changing the stream."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]

    plan = py.MpsOptimizer.gate_stream_layout(gates, L=4)

    assert set(plan["site_order"]) == {0, 1, 2, 3}
    assert plan["stats"]["max_span"] == 1
    assert plan["stats"]["long_range_events"] == 0
    assert plan["input_stats"]["long_range_events"] == 2
    assert plan["stats"]["loss"] <= plan["input_stats"]["loss"]
    assert plan["score"] == plan["stats"]["loss"]
    assert plan["layout"] == plan["site_map"]
    assert "recursive_refined" in plan["candidate_scores"]
    assert "gate_stream" not in plan
    assert "gates" not in plan
    assert plan["where"] == tuple(where for _gate, where in gates)
    assert set(plan["inverse_site_map"]) == {0, 1, 2, 3}
    assert all(
        abs(where[0] - where[1]) == 1
        for where in plan["mapped_where"]
    )


def test_mps_quality_layout_includes_periodic_folded_candidate():
    """Quality search should remove the long wrap tail of a periodic grid."""
    Lx = Ly = 8

    def site(x, y):
        return (x % Lx) * Ly + (y % Ly)

    edge_builder = getattr(qtn, "edges_square", qtn.edges_2d_square)
    gates = tuple(
        (qu.CNOT(), (site(*left), site(*right)))
        for left, right in edge_builder(Lx, Ly, cyclic=True)
    )
    plan = py.MpsOptimizer.LayoutFinder(gates, L=Lx * Ly).run(
        order="quality",
        nevergrad_budget=0,
    )

    assert "folded_8" in plan["candidate_losses"]
    assert plan["selected_order"] == "folded_8"
    assert plan["stats"]["max_span"] < plan["input_stats"]["max_span"]
    assert plan["stats"]["loss"] < plan["input_stats"]["loss"]


def test_mps_quality_layout_can_exclude_input_candidate():
    """From-scratch search keeps the original order as diagnostics only."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]
    plan = py.MpsOptimizer.LayoutFinder(gates, L=4).run(
        order="quality",
        from_scratch=True,
        nevergrad_budget=0,
    )

    assert plan["from_scratch"] is True
    assert "input" not in plan["candidate_plans"]
    assert plan["input_stats"]["long_range_events"] == 2
    assert plan["stats"]["loss"] < plan["input_stats"]["loss"]


@pytest.mark.parametrize(
    "mode",
    [
        "row-major",
        "col-major",
        "snake",
        "snake-row-major",
        "folded-snake",
        "folded-snake-row-major",
        "hilbert",
        "hilbert-row-major",
    ],
)
def test_mps_layout_geometric_presets_match_onedmap(mode):
    """MPS geometric presets must use the shared OneDMap traversal exactly."""
    Lx, Ly = 3, 5
    mapping, _ = py.OneDMap.build(Lx, Ly, mode=mode)
    expected = tuple(x * Ly + y for x, y in mapping.values())

    finder = py.MpsOptimizer.LayoutFinder(
        [],
        L=Lx * Ly,
        lattice_shape=(Lx, Ly),
    )
    plan = finder.run(order=mode)

    assert plan["selected_order"] == mode
    assert plan["site_order"] == expected
    assert plan["mapped_where"] == ()


def test_mps_hilbert_layout_requires_lattice_shape():
    """Named MPS lattice orders reject ambiguous unshaped site sets."""
    finder = py.MpsOptimizer.LayoutFinder([], L=6)
    with pytest.raises(ValueError, match="lattice_shape"):
        finder.run(order="hilbert")


def test_mps_layout_accepts_explicit_fixed_site_order():
    """An explicit site permutation bypasses search and is preserved."""
    gates = [(qu.CNOT(), (0, 3)), (qu.CNOT(), (1, 2))]
    order = (2, 0, 3, 1)
    plan = py.MpsOptimizer.LayoutFinder(gates, L=4).run(order=order)

    assert plan["selected_order"] == "fixed"
    assert plan["site_order"] == order
    assert plan["mapped_where"] == ((1, 2), (3, 0))


def test_mps_compression_layout_reports_operator_cut_load():
    """Compression objective exposes cut-load diagnostics and rank bounds."""
    gate = np.eye(8, dtype=complex)
    plan = py.MpsOptimizer.gate_stream_layout(
        [(gate, (0, 1, 2))],
        L=3,
        objective="compression",
        max_operator_qubits=2,
    )

    assert plan["objective"] == "compression"
    assert plan["stats"]["compression_score"] == plan["score"]
    assert plan["rank_bounded_events"] > 0
    assert plan["rank_bound_reasons"]["max_operator_qubits"] > 0
    assert plan["candidate_plans"]

    exact = py.MpsOptimizer.gate_stream_layout(
        [(qu.CNOT(), (0, 1))],
        L=2,
        objective="compression",
    )
    assert exact["stats"]["rank_exact_events"] == 1
    assert exact["stats"]["total_operator_cut_load"] == pytest.approx(1.0)


def test_mps_compression_layout_pilot_is_non_mutating():
    """Pilot selection uses copied state and does not install a layout."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    gates = [(qu.CNOT(), (0, 3)), (qu.CNOT(), (3, 1))]
    opt = py.MpsOptimizer(
        p0, gates=gates, chi=2, mode="svd"
    )
    before = opt.to_dense()

    selected = opt.select_layout_for_compression(
        pilot_candidates=1,
        pilot_steps=1,
    )

    assert selected["pilot"]["selected_order"]
    assert selected["pilot"]["reports"]
    assert opt._persistent_layout_plan is None
    assert np.allclose(opt.to_dense(), before)


@pytest.mark.parametrize("bad_length", [-1, 1.5, True])
def test_mps_layout_finder_rejects_invalid_register_lengths(bad_length):
    """Layout register lengths must be explicit non-negative integers."""
    error = ValueError if bad_length == -1 else TypeError
    with pytest.raises(error, match="non-negative integer"):
        py.MpsOptimizer.LayoutFinder([], L=bad_length)


def test_mps_layout_finder_rejects_duplicate_support_sites():
    """Static plans must not silently accept repeated gate support labels."""
    with pytest.raises(ValueError, match="at most once"):
        py.MpsOptimizer.LayoutFinder(
            [(np.eye(4, dtype=complex), (1, 1))],
            L=3,
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"mode": "perm"}, "fixed-layout"),
        ({"layout": False}, "must not contain layout"),
    ],
)
def test_mps_compression_layout_pilot_rejects_conflicting_modes(kwargs, match):
    """Pilot selection should reject layouts it cannot execute up front."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=4,
        mode="direct",
    )
    with pytest.raises(ValueError, match=match):
        opt.select_layout_for_compression(
            pilot_candidates=1,
            pilot_steps=1,
            run_kwargs=kwargs,
        )


def test_mps_compression_layout_pilot_accepts_direct_cap_stream():
    """Pilot selection tracks a direct cap's shortened replay state."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000"),
        gates=[("cap", 1, [1.0, 1.0])],
        chi=4,
        mode="direct",
    )
    plan = opt.select_layout_for_compression(pilot_candidates=1, pilot_steps=1)
    assert plan["pilot"]["reports"]["input"]["status"] == "ok"


def test_mps_compression_layout_pilot_accepts_none_mode_override():
    """A ``mode=None`` run override keeps the optimizer's current mode."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[(qu.CNOT(), (0, 3))],
        chi=4,
        mode="direct",
    )
    plan = opt.select_layout_for_compression(
        pilot_candidates=1,
        pilot_steps=1,
        run_kwargs={"mode": None},
    )
    assert plan["pilot"]["reports"]


def test_mps_layout_finder_plot_draws_lattice_and_gate_order():
    """The MPS plot exposes the lattice, gate graph, and colored chain."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]
    finder = py.MpsOptimizer.LayoutFinder(gates, L=4)
    plan = finder.run(order="input")
    fig, ax = finder.plot(
        plan,
        site_coords={0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
    )

    assert fig is ax.figure
    assert ax.get_title() == ""
    assert len(ax.patches) == len(plan["site_order"]) - 1
    assert len(fig.axes) == 1  # no stream-order colorbar by default
    assert not ax.axison  # schematic-style presentation by default
    assert any(text.get_text() == "0" for text in ax.texts)
    assert any(text.get_text() == "3" for text in ax.texts)
    assert any(collection.get_offsets().shape[0] for collection in ax.collections)
    plt.close(fig)


def test_mps_optimizer_plot_layout_is_non_mutating():
    """The optimizer plotting wrapper does not install or alter a layout."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(
        p0,
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    before = tuple(opt.logical_order)
    fig, _ = opt.plot_layout(
        layout_kwargs={"order": "input"},
        site_coords={q: (q, 0) for q in range(4)},
    )

    assert tuple(opt.logical_order) == before
    assert opt._persistent_layout_plan is None
    plt.close(fig)


def test_mps_optimizer_gate_stream_layout_accepts_weight_fn():
    """User event weights should feed the weighted graph and report."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (1, 2)),
    ]

    def weight_fn(_payload, support, _event_type):
        return 10.0 if tuple(support) == (0, 3) else 1.0

    plan = py.MpsOptimizer.gate_stream_layout(
        gates,
        L=4,
        order="input",
        weight_fn=weight_fn,
    )

    assert plan["event_weights"] == (10.0, 1.0)
    assert plan["input_stats"]["total_edge_weight"] == pytest.approx(11.0)
    assert plan["input_stats"]["weighted_long_range_events"] == pytest.approx(10.0)


def test_mps_optimizer_gate_stream_layout_can_use_nevergrad():
    """Optional nevergrad candidate should be usable without touching streams."""
    pytest.importorskip("nevergrad")
    gates = [
        (qu.CNOT(), (0, 4)),
        (qu.CNOT(), (4, 1)),
        (qu.CNOT(), (1, 3)),
        (qu.CNOT(), (3, 2)),
    ]

    plan = py.MpsOptimizer.gate_stream_layout(
        gates,
        L=5,
        order="nevergrad",
        nevergrad_budget=8,
        refine_passes=1,
    )

    assert plan["selected_order"] == "nevergrad"
    assert "nevergrad" in plan["candidate_scores"]
    assert set(plan["site_order"]) == set(range(5))
    assert plan["where"] == tuple(where for _gate, where in gates)


def test_mps_optimizer_gate_stream_layout_kahypar_requires_config(monkeypatch):
    """Explicit KaHyPar layouts need a user-supplied config path."""
    monkeypatch.delenv("PEPSY_KAHYPAR_CONFIG", raising=False)
    gates = [(qu.CNOT(), (0, 3)), (qu.CNOT(), (3, 1))]

    with pytest.raises(ValueError, match="kahypar_config_path"):
        py.MpsOptimizer.gate_stream_layout(gates, L=4, order="kahypar")


def test_mps_optimizer_current_gate_stream_layout_uses_state_length():
    """Instance helper should include untouched MPS sites via ``p.L``."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    opt = py.MpsOptimizer(
        p0.copy(),
        gates=[(qu.CNOT(), (0, 2))],
        chi=8,
        mode="svd",
    )

    plan = opt.current_gate_stream_layout(order="input")

    assert plan["site_order"] == (0, 1, 2, 3)
    assert plan["where"] == ((0, 2),)
    assert plan["mapped_where"] == ((0, 2),)


def test_mps_optimizer_layout_finder_includes_conditional_action_support():
    """A possible conditional gate contributes its operator and support."""
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("0000", dtype="complex128"),
        gates=[
            ("measure", "Z", 0, +1),
            ("if", -1, 0, (qu.CNOT(), (0, 3))),
        ],
        chi=8,
        mode="svd",
    )

    plan = opt.current_gate_stream_layout(
        objective="compression",
        nevergrad_budget=0,
    )

    assert plan["event_types"] == ("conditional",)
    assert plan["where"] == ((0, 3),)
    assert plan["stats"]["max_span"] == 1
    assert plan["rank_exact_events"] == 1


def test_mps_optimizer_gate_stream_layout_preserves_submpo_events():
    """Layout planning should not rewrite explicit sub-MPO stream events."""
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    stream = [
        py.MpsOptimizer.submpo_event(mpo, (0, 3)),
        (qu.CNOT(), (3, 1)),
    ]

    plan = py.MpsOptimizer.gate_stream_layout(stream, L=4)

    assert plan["event_types"] == ("submpo", "gate")
    assert stream[0][1] is mpo
    assert stream[0][2] == (0, 3)
    assert plan["where"][0] == (0, 3)
    assert plan["mapped_where"][0] != plan["where"][0]


def test_mps_optimizer_layout_run_restores_original_mps_order_and_stream():
    """Layout-aware replay should be internal and return original site labels."""
    p0 = qtn.MPS_computational_state("0101", dtype="complex128")
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]
    ref = py.MpsOptimizer(
        p0.copy(),
        gates=gates,
        chi=16,
        mode="svd",
    ).run(progbar=False, cutoff=1e-12)

    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=16, mode="svd")
    out = opt.run(
        use_layout_finder=True,
        progbar=False,
        cutoff=1e-12,
    )

    inds = ["k0", "k1", "k2", "k3"]
    assert np.allclose(out.to_dense(inds), ref.to_dense(inds))
    assert out.outer_inds() == tuple(inds)
    assert opt.where == [(0, 3), (3, 1), (1, 2)]
    assert all(
        actual is expected
        for actual, (expected, _where) in zip(opt.G, gates)
    )
    assert opt.last_layout_plan is not None


@pytest.mark.parametrize("persistent", [False, True])
def test_mps_optimizer_layout_remaps_conditional_gate_action(persistent):
    """Conditional actions execute on their mapped physical layout sites."""
    stream = [
        ("measure", "Z", 0, +1),
        ("if", -1, 0, (qu.pauli("X"), 3)),
        (qu.CNOT(), (0, 3)),
    ]
    initial = qtn.MPS_computational_state("0000", dtype="complex128")
    reference = py.MpsOptimizer(
        initial.copy(),
        gates=stream,
        chi=8,
        mode="svd",
    )
    reference.run(progbar=False, cutoff=0.0)

    opt = py.MpsOptimizer(
        initial.copy(),
        gates=stream,
        chi=8,
        mode="svd",
    )
    order = (0, 3, 1, 2)
    if persistent:
        opt.apply_layout(order, layout_report=False)
        opt.run(progbar=False, cutoff=0.0)
        assert opt.logical_order == list(order)
    else:
        with pytest.warns(DeprecationWarning, match="temporary reorder"):
            opt.run(
                progbar=False,
                cutoff=0.0,
                use_layout_finder=opt._explicit_layout_plan(order),
                layout_report=False,
            )
        assert opt.logical_order == list(range(4))

    actual = np.asarray(opt.to_dense()).reshape(-1)
    expected = np.asarray(reference.to_dense()).reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.flatnonzero(np.abs(actual) > 1e-12).tolist() == [1]
    assert opt.measurements == [("Z", (0,), 1, 1.0)]


def test_mps_optimizer_layout_remaps_conditional_submpo_action():
    """Nested sub-MPO labels follow the conditional's mapped support."""
    mpo = _two_branch_flip_submpo(
        L=4,
        sites=(0, 3),
        targets=(0, 3),
        w0=0.0,
        w1=1.0,
    )
    stream = [
        ("measure", "Z", 1, +1),
        ("if", -1, 0, py.MpsOptimizer.submpo_event(mpo, (0, 3))),
    ]
    initial = qtn.MPS_computational_state("0000", dtype="complex128")
    reference = py.MpsOptimizer(
        initial.copy(),
        gates=stream,
        chi=8,
        mode="direct",
    )
    reference.run(progbar=False, cutoff=0.0)

    opt = py.MpsOptimizer(
        initial.copy(),
        gates=stream,
        chi=8,
        mode="direct",
    )
    plan = opt._explicit_layout_plan((0, 3, 1, 2))
    with pytest.warns(DeprecationWarning, match="temporary reorder"):
        opt.run(
            progbar=False,
            cutoff=0.0,
            use_layout_finder=plan,
            layout_report=False,
        )

    np.testing.assert_allclose(opt.to_dense(), reference.to_dense(), atol=1e-12)
    assert stream[1][3][1] is mpo


def test_mps_optimizer_apply_layout_relabels_product_state_without_swaps(monkeypatch):
    """A nonuniform bond-one state should relabel without any SVD swaps."""
    calls = []
    original = qtn.MatrixProductState.swap_site_to_

    def fail_swap(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductState, "swap_site_to_", fail_swap)

    opt = py.MpsOptimizer(
        _nonuniform_product_mps(),
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    opt._start_unitary_norm_tracking(opt.p)  # pylint: disable=protected-access
    assert opt._unitary_previous_norm is not None  # pylint: disable=protected-access
    opt.apply_layout((0, 2, 3, 1), layout_report=False)

    assert calls == []
    assert opt._unitary_previous_norm is None  # pylint: disable=protected-access
    assert opt.logical_order == [0, 2, 3, 1]
    assert opt.p.max_bond() == 1
    assert [opt.logical_site(pos) for pos in range(4)] == [0, 2, 3, 1]
    assert [opt.position(site) for site in range(4)] == [0, 3, 1, 2]


def test_mps_optimizer_persistent_layout_reuses_order_and_remaps_readout():
    """Persistent layout replay should agree with identity replay over repeats."""
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
    ]
    reference = py.MpsOptimizer(
        _nonuniform_product_mps(), gates=gates, chi=8, mode="svd"
    )
    reference.run(progbar=False, cutoff=1e-12)
    reference.run(progbar=False, cutoff=1e-12)

    laid_out = py.MpsOptimizer(
        _nonuniform_product_mps(), gates=gates, chi=8, mode="svd"
    )
    laid_out.apply_layout((0, 2, 3, 1), layout_report=False)
    laid_out.run(progbar=False, cutoff=1e-12)
    laid_out.run(progbar=False, cutoff=1e-12)

    reference_dense = np.asarray(reference.to_dense()).reshape(-1)
    laid_out_dense = np.asarray(laid_out.to_dense()).reshape(-1)
    overlap = np.vdot(reference_dense, laid_out_dense)
    assert abs(overlap) == pytest.approx(1.0, abs=1e-10)
    assert laid_out.logical_order == [0, 2, 3, 1]
    assert laid_out.p.max_bond() <= laid_out.chi
    assert laid_out.layout_plan is laid_out.last_layout_plan

    physical_configs = py.MpsSampler(laid_out.p, backend="quimb").sample(
        n_samples=24, seed=19
    ).configs_1d
    physical_dense = np.asarray(laid_out.p.to_dense()).reshape(-1)
    for physical_config in physical_configs:
        logical_config = laid_out.remap_sample(physical_config).tolist()
        physical_index = int("".join(map(str, physical_config)), 2)
        logical_index = int("".join(map(str, logical_config)), 2)
        assert abs(physical_dense[physical_index]) ** 2 == pytest.approx(
            abs(reference_dense[logical_index]) ** 2,
            abs=1e-10,
        )


def test_mps_optimizer_persistent_layout_rejects_entangled_state_by_default():
    """Entangled initialization needs explicit permission for one-time loss."""
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=23),
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    before = np.asarray(opt.p.to_dense()).copy()

    with pytest.raises(ValueError, match="initially product MPS"):
        opt.apply_layout((0, 2, 3, 1), layout_report=False)

    assert opt.logical_order == [0, 1, 2, 3]
    assert np.allclose(np.asarray(opt.p.to_dense()), before)


def test_mps_optimizer_persistent_layout_entangled_reorder_uses_cutoff(monkeypatch):
    """Lossy persistent initialization should use the caller's cutoff once."""
    calls = []
    original = qtn.MatrixProductState.swap_site_to_

    def counting(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductState, "swap_site_to_", counting)
    opt = py.MpsOptimizer(
        qtn.MPS_rand_state(4, bond_dim=2, dtype="complex128", seed=23),
        gates=[(qu.CNOT(), (0, 3))],
        chi=8,
        mode="svd",
    )
    opt.apply_layout(
        (0, 2, 3, 1),
        cutoff=1e-7,
        allow_lossy_reorder=True,
        layout_report=False,
    )

    assert opt.logical_order == [0, 2, 3, 1]
    assert calls
    assert all(call["cutoff"] == pytest.approx(1e-7) for call in calls)


def test_mps_optimizer_persistent_layout_controls_keep_logical_labels():
    """Persistent layout control events execute physically but record logically."""
    opt = py.MpsOptimizer(
        _nonuniform_product_mps(),
        gates=[
            (qu.hadamard(), (3,)),
            (qu.CNOT(), (0, 3)),
            ("measure", "Z", 3, +1),
        ],
        chi=8,
        mode="mpo",
    )
    opt.apply_layout((0, 2, 3, 1), layout_report=False)
    opt.run(progbar=False)

    assert opt.measurements[0][0:3] == ("Z", (3,), 1)
    assert np.isclose(
        py.MpsOptimizer._real_float(
            opt._state_expectation("Z", (opt.position(3),))
        ),
        1.0,
    )


def test_mps_optimizer_persistent_layout_remaps_submpo_without_mutating_stream():
    """Persistent layout should copy/remap each sub-MPO on every replay."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    stream = [py.MpsOptimizer.submpo_event(mpo, (0, 3))]
    reference = py.MpsOptimizer(p0.copy(), gates=stream, chi=16, mode="mpo")
    reference.run(progbar=False, cutoff=1e-12)
    reference.run(progbar=False, cutoff=1e-12)

    opt = py.MpsOptimizer(p0.copy(), gates=stream, chi=16, mode="mpo")
    opt.apply_layout((0, 2, 3, 1), layout_report=False)
    opt.run(progbar=False, cutoff=1e-12)
    opt.run(progbar=False, cutoff=1e-12)

    assert np.allclose(
        np.abs(np.asarray(opt.to_dense()).reshape(-1)),
        np.abs(np.asarray(reference.to_dense()).reshape(-1)),
    )
    assert stream[0][1] is mpo
    assert stream[0][2] == (0, 3)


def test_mps_optimizer_persistent_layout_tracks_direct_cap_events():
    """Persistent layouts update logical labels when a cap shortens the chain."""
    stream = [
        (qu.hadamard(), (0,)),
        ("cap", 1, [1.0, 0.0], "left"),
        (qu.CNOT(), (0, 1)),
    ]
    reference = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=stream,
        chi=16,
        mode="svd",
    )
    reference.run(progbar=False, cutoff=0.0)
    opt = py.MpsOptimizer(
        qtn.MPS_computational_state("000", dtype="complex128"),
        gates=stream,
        chi=16,
        mode="svd",
    )
    opt.apply_layout((2, 0, 1), layout_report=False)
    opt.run(progbar=False, cutoff=0.0)

    assert opt.mps_length_diagnostics()["length_history"] == (3, 2)
    assert opt.logical_order == [1, 0]
    assert np.allclose(
        np.asarray(opt.to_dense()).reshape(-1),
        np.asarray(reference.to_dense()).reshape(-1),
    )


def test_mps_optimizer_layout_run_reports_score_reduction(capsys):
    """Layout-aware replay should print a concise before/after report."""
    p0 = qtn.MPS_computational_state("0101", dtype="complex128")
    gates = [
        (qu.CNOT(), (0, 3)),
        (qu.CNOT(), (3, 1)),
        (qu.CNOT(), (1, 2)),
    ]
    opt = py.MpsOptimizer(p0.copy(), gates=gates, chi=16, mode="svd")

    opt.run(
        use_layout_finder=True,
        progbar=False,
        cutoff=1e-12,
        layout_report=True,
    )

    report = capsys.readouterr().out
    assert "MpsOptimizer layout finder:" in report
    assert "long-range events:" in report
    assert "score:" in report
    assert "graph span:" in report


def test_mps_optimizer_layout_run_copies_submpo_payloads():
    """Layout replay should remap sub-MPO copies without mutating the stream."""
    p0 = qtn.MPS_computational_state("0000", dtype="complex128")
    mpo = _two_branch_flip_submpo(L=4, sites=(0, 3), targets=(0, 3))
    stream = [py.MpsOptimizer.submpo_event(mpo, (0, 3))]
    ref = py.MpsOptimizer(
        p0.copy(),
        gates=stream,
        chi=16,
        mode="mpo",
    ).run(progbar=False, cutoff=1e-12)

    opt = py.MpsOptimizer(p0.copy(), gates=stream, chi=16, mode="mpo")
    out = opt.run(
        use_layout_finder=True,
        progbar=False,
        cutoff=1e-12,
    )

    inds = ["k0", "k1", "k2", "k3"]
    assert np.allclose(out.to_dense(inds), ref.to_dense(inds))
    assert out.outer_inds() == tuple(inds)
    assert out.site_inds == tuple(inds)
    assert stream[0][1] is mpo
    assert stream[0][2] == (0, 3)
    assert opt.where == [(0, 3)]
