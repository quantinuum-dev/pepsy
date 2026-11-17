"""Ordinary MPO DMT replay against independent physical density matrices."""

import numpy as np
import pytest
import quimb.tensor as qtn
from scipy.linalg import expm

from pepsy.operators import compress_mpo_dmt, square_lattice_dmt_regions
from pepsy.operators.pauli_mpo import _host_array
from pepsy.optimizers import MpoOptimizer


def mixture(n, rank=48, seed=321, dtype="complex128"):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(.2, 1., rank)
    weights /= weights.sum()
    v = rng.normal(size=(rank, n, 3))
    v *= .5 / np.linalg.norm(v, axis=-1, keepdims=True)
    local = np.empty((rank, n, 2, 2), dtype=complex)
    local[..., 0, 0] = (1 + v[..., 2]) / 2
    local[..., 1, 1] = (1 - v[..., 2]) / 2
    local[..., 0, 1] = (v[..., 0] - 1j * v[..., 1]) / 2
    local[..., 1, 0] = (v[..., 0] + 1j * v[..., 1]) / 2
    arrays = [weights[:, None, None] * local[:, 0]]
    for site in range(1, n - 1):
        core = np.zeros((rank, rank, 2, 2), dtype=complex)
        core[np.arange(rank), np.arange(rank)] = local[:, site]
        arrays.append(core)
    arrays.append(local[:, -1])
    return qtn.MatrixProductOperator([a.astype(dtype) for a in arrays]), weights, local


def product_marginal(weights, local, region):
    result = np.zeros((2 ** len(region),) * 2, dtype=complex)
    for w, factors in zip(weights, local):
        value = np.ones((1, 1))
        for site in region:
            value = np.kron(value, factors[site])
        result += w * value
    return result


def mpo_marginal(mpo, region):
    """Independent host contraction of only the requested ket/bra legs."""
    value = np.ones(1, dtype=complex)
    for i in range(mpo.L):
        bonds = (mpo.bond(i - 1, i),) if i else ()
        bonds += (mpo.bond(i, i + 1),) if i + 1 < mpo.L else ()
        tensor = mpo[i].transpose(*(bonds + (mpo.upper_ind(i), mpo.lower_ind(i))))
        core = _host_array(tensor.data).reshape(value.shape[-1], -1, 2, 2)
        if i in region:
            value = np.moveaxis(np.tensordot(value, core, axes=(-1, 0)), -3, -1)
        else:
            value = value @ np.trace(core, axis1=2, axis2=3)
    k = len(region)
    panel = value[..., 0].transpose(tuple(range(0, 2 * k, 2)) + tuple(range(1, 2 * k, 2)))
    return panel.reshape(2**k, 2**k) * 10. ** mpo.exponent


def dense(mpo):
    return mpo_marginal(mpo, tuple(range(mpo.L)))


def marginal(matrix, n, region):
    value = matrix.reshape((2,) * (2 * n))
    remaining = n
    for i in reversed(range(n)):
        if i not in region:
            value = np.trace(value, axis1=i, axis2=i + remaining)
            remaining -= 1
    return value.reshape(2**remaining, 2**remaining)


def embed(gate, where, n):
    """Binary-basis oracle that preserves the caller's support order."""
    out = np.zeros((2**n, 2**n), dtype=complex)
    for col in range(2**n):
        bits = [(col >> (n - 1 - i)) & 1 for i in range(n)]
        local_col = sum(bits[site] << (len(where) - 1 - i) for i, site in enumerate(where))
        for local_row in range(2**len(where)):
            row_bits = list(bits)
            for i, site in enumerate(where):
                row_bits[site] = (local_row >> (len(where) - 1 - i)) & 1
            row = sum(b << (n - 1 - i) for i, b in enumerate(row_bits))
            out[row, col] = gate[local_row, local_col]
    return out


def gate(seed=15):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    return expm(-.13j * (h + h.conj().T))


@pytest.mark.parametrize("kind,cap", [("plaquettes", 40), ("l_shapes", 24)])
def test_2d_replay_checks_every_local_density_of_each_untruncated_target(kind, cap):
    initial, _, _ = mixture(9, rank=64)
    regions = square_lattice_dmt_regions(3, 3, map_mode="snake-row-major", kind=kind)
    supports = [(0, 5), (8, 1), (3, 4), (2, 1), (7, 8), (0, 1), (4, 5), (6, 2), (6, 7)]
    gates = [(gate(20 + i), w) for i, w in enumerate(supports)]
    saved_gates = [g.copy() for g, _ in gates]
    before = dense(initial)
    reference = before
    checked = []

    def check(candidate):
        nonlocal reference
        g, where = gates[len(checked)]
        # Preserve MpoOptimizer's documented payload convention: effective G.T.
        full = embed(g.T, where, 9)
        target = full @ reference @ full.conj().T
        actual = dense(candidate)
        for region in regions:
            np.testing.assert_allclose(marginal(actual, 9, region), marginal(target, 9, region), atol=2e-12)
        assert abs(np.trace(actual) - np.trace(target)) < 2e-12
        reference = actual
        checked.append(1)
        return True

    opt = MpoOptimizer(initial, gates, chi=cap, mode="dmt", contraction_opt="greedy")
    with pytest.warns(RuntimeWarning, match="finite_check"):
        result = opt.run(cutoff=0., dmt_regions=regions, dmt_diagnostics=True, finite_check=check,
                         timing=True, fidelity_samples=0)
    assert len(checked) == len(gates)
    assert result.max_bond() <= cap
    assert max(b.discarded_weight for r in opt.get_dmt_reports() for b in r.bond_reports) > 1e-8
    assert opt.info_c["cur_orthog"] == (7, 7)
    assert len(opt.get_dmt_reports()[-1].bond_reports) == 1
    assert opt.norm_diagnostics()["cumulative_fidelity"] is None
    assert opt.get_norm_events() == []
    assert "dmt.target" in opt.get_run_timing()["stages"]
    assert opt.get_run_timing()["stages"]["dmt.local"]["calls"] == 6
    np.testing.assert_array_equal(dense(initial), before)
    for (g, _), saved in zip(gates, saved_gates):
        np.testing.assert_array_equal(g, saved)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("variant", ["density", "operator"])
@pytest.mark.parametrize("exponent", [0., 2.])
def test_untruncated_replay_matches_complex_ket_bra_oracle_and_repeats(inplace, variant, exponent):
    initial, _, _ = mixture(4, rank=12)
    initial.exponent = exponent
    g, b = gate(), gate(32)
    queue = [((g, b), (3, 0)), ((np.diag([1., 1j]), None), (1,))]
    expected = dense(initial)
    opt = MpoOptimizer(initial, queue, chi=16, mode="dmt", inplace=inplace, contraction_opt="greedy")
    for _ in range(2):
        expected = embed(g.T, (3, 0), 4) @ expected @ embed(b.T, (3, 0), 4).conj().T
        expected = embed(np.diag([1., 1j]), (1,), 4) @ expected
        result = opt.run(cutoff=0., dmt_variant=variant)
        np.testing.assert_allclose(dense(result), expected, atol=2e-13)
    assert (result is initial) == inplace


def test_manual_dmt_compression_matches_operator_api():
    initial, _, _ = mixture(9)
    regions = square_lattice_dmt_regions(3, 3)
    expected = compress_mpo_dmt(initial, max_bond=30, regions=regions)
    opt = MpoOptimizer(initial, [], chi=30, contraction_opt="greedy")
    result, report = opt.compress_dmt(regions=regions, return_report=True)
    np.testing.assert_allclose(dense(result), dense(expected), atol=2e-13)
    assert opt.get_dmt_reports() == (report,)
    assert opt.mode == "quimb-direct"


@pytest.mark.parametrize("shape,cap", [
    ((4, 3), 48), ((4, 4), 64),
    pytest.param((6, 6), 96, marks=pytest.mark.slow),
    pytest.param((8, 8), 128, marks=pytest.mark.slow),
])
def test_larger_2d_computational_basis_mpo_against_positive_product_oracle(shape, cap):
    initial, weights, local = mixture(np.prod(shape), rank=max(72, cap + 8))
    regions = square_lattice_dmt_regions(*shape, map_mode="snake-row-major")
    result, report = compress_mpo_dmt(initial, max_bond=cap, regions=regions, return_report=True)
    assert result.max_bond() <= cap
    # Hilbert-Schmidt weights shrink exponentially with chain length for
    # trace-one densities. Use the maximally mixed norm as a physical scale.
    assert max(b.discarded_weight for b in report.bond_reports) > 1e-8 * 2. ** (-initial.L / 2)
    for region in regions:
        np.testing.assert_allclose(mpo_marginal(result, region), product_marginal(weights, local, region), atol=2e-11)
    np.testing.assert_allclose(mpo_marginal(result, ()), [[1.]], atol=2e-11)


def test_atomic_failure_and_transactional_steps_restore_dmt_state():
    initial, _, _ = mixture(9)
    opt = MpoOptimizer(initial, [(gate(), (0, 5))], chi=4, mode="dmt", contraction_opt="greedy")
    before = dense(opt.p)
    with pytest.raises(ValueError, match="protected budget"):
        opt.run(cutoff=0., dmt_regions=square_lattice_dmt_regions(3, 3))
    np.testing.assert_array_equal(dense(opt.p), before)
    assert opt.get_dmt_reports() == ()
    assert not opt._dmt_compression_seen
    assert opt.last_run_status == "failed"


@pytest.mark.parametrize("options,error", [
    ({"target_cutoff": 1e-8}, ValueError), ({"submpo_method": "src"}, ValueError),
    ({"fit_fallback": "svd"}, ValueError), ({"layout": [0, 1, 2, 3]}, ValueError),
    ({"dmt_regions": []}, ValueError), ({"dmt_variant": "bad"}, ValueError),
])
def test_incompatible_controls_fail_before_any_replay(options, error):
    opt = MpoOptimizer(qtn.MPO_identity(4), [(gate(), (0, 3))], chi=16, mode="dmt", contraction_opt="greedy")
    before = [a.copy() for a in opt.p.arrays]
    with pytest.raises(error):
        opt.run(**options)
    for a, b in zip(opt.p.arrays, before):
        np.testing.assert_array_equal(a, b)


def test_target_gate_factorization_has_no_hidden_cutoff(monkeypatch):
    original = qtn.MatrixProductOperator.from_dense.__func__
    cutoffs = []

    def checked(cls, *args, **kwargs):
        cutoffs.append(kwargs.get("cutoff"))
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(qtn.MatrixProductOperator, "from_dense", classmethod(checked))
    initial, _, _ = mixture(4, rank=12)
    opt = MpoOptimizer(initial, [(gate(), (0, 3))], chi=16, mode="dmt", contraction_opt="greedy")
    opt.run(cutoff=0.)
    assert cutoffs and all(c == 0. for c in cutoffs)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_native_backend_replay_and_numpy_gate_constants(backend, dtype, request):
    module = pytest.importorskip(backend)
    initial, _, _ = mixture(4, rank=16, dtype=dtype)
    if backend == "torch":
        initial.apply_to_arrays(lambda a: module.as_tensor(a))
    elif backend == "jax":
        previous = module.config.x64_enabled
        request.addfinalizer(lambda: module.config.update("jax_enable_x64", previous))
        module.config.update("jax_enable_x64", True)
        jnp = pytest.importorskip("jax.numpy")
        initial.apply_to_arrays(jnp.asarray)
    g = gate().astype(dtype)
    before = dense(initial)
    queue = [(g, (0, 3)), (g, (2, 1)), (g, (3, 0))]
    reference = before
    checked = []

    def check(candidate):
        nonlocal reference
        _, where = queue[len(checked)]
        full = embed(g.T, where, 4)
        target = full @ reference @ full.conj().T
        for r in [(0, 1, 2), (1, 2, 3)]:
            np.testing.assert_allclose(mpo_marginal(candidate, r), marginal(target, 4, r),
                                       atol=2e-6 if dtype == "complex64" else 1e-12)
        reference = dense(candidate)
        checked.append(1)
        return True

    opt = MpoOptimizer(initial, queue, chi=10, mode="dmt", contraction_opt="greedy")
    with pytest.warns(RuntimeWarning, match="finite_check"):
        result = opt.run(finite_check=check, timing=True)
    assert len(checked) == 3
    assert opt.get_run_timing()["stages"]["dmt.local"]["calls"] == 1
    assert opt.get_run_timing()["stages"]["dmt.routed"]["calls"] == 1
    assert opt.get_run_timing()["stages"]["dmt.target"]["calls"] == 1
    assert result[0].data.dtype == initial[0].data.dtype
    assert result[0].data.device == initial[0].data.device


def test_completed_steps_survive_failure_without_whole_run_atomicity():
    initial, _, _ = mixture(4, rank=16)
    g = gate()
    opt = MpoOptimizer(initial, [(g, (0, 3)), (g, (1, 2))], chi=10, mode="dmt", contraction_opt="greedy")
    checks = []

    def check(p):
        checks.append(dense(p))
        return len(checks) == 1

    with pytest.warns(RuntimeWarning, match="finite_check"):
        with pytest.raises(FloatingPointError, match="non-finite"):
            opt.run(cutoff=0., dmt_diagnostics=True, finite_check=check, atomic=False)
    np.testing.assert_allclose(dense(opt.p), checks[0], atol=1e-13)
    assert len(opt.get_dmt_reports()) == 1


def test_previous_fidelity_ledger_is_suppressed_after_dmt_and_resets_with_new_state():
    initial, _, _ = mixture(4, rank=16)
    opt = MpoOptimizer(initial, [(gate(), (0, 3))], chi=10, mode="svd", contraction_opt="greedy")
    opt.run(cutoff=0.)
    old_events = opt.get_norm_events()
    assert opt.norm_diagnostics()["cumulative_fidelity"] is not None
    opt.compress_dmt(variant="operator")
    assert opt.get_norm_events() == old_events
    assert opt.norm_diagnostics()["cumulative_fidelity"] is None
    opt.set_gates([(gate(2), (1, 2))])
    opt.run(mode="svd", cutoff=0.)
    assert opt.norm_diagnostics()["cumulative_fidelity"] is None
    opt.set_mpo(initial)
    opt.run(mode="svd", cutoff=0.)
    assert opt.norm_diagnostics()["cumulative_fidelity"] is not None


def test_atomic_rollback_restores_backend_dtype_after_complex_promotion():
    initial = qtn.MPO_identity(4, dtype="float64") / 16
    opt = MpoOptimizer(initial, [(gate(), (0, 3)), (gate(), (1, 2))],
                       chi=16, mode="dmt", contraction_opt="greedy")
    old_dtype = opt.backend_dtype
    count = []

    def check(p):
        count.append(1)
        return len(count) == 1

    with pytest.warns(RuntimeWarning, match="finite_check"):
        with pytest.raises(FloatingPointError):
            opt.run(cutoff=0., finite_check=check)
    assert opt.backend_dtype == old_dtype
    assert all(a.dtype == np.float64 for a in opt.p.arrays)


def test_torch_full_replay_gradient_of_protected_local_observable():
    torch = pytest.importorskip("torch")
    from pepsy.backends import TorchLinalgConfig

    initial, _, _ = mixture(4, rank=16)
    rho = dense(initial)
    rng = np.random.default_rng(33)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = h + h.conj().T
    observable = embed(np.diag([1., -1.]), (0,), 4)

    def reference(angle):
        u = embed(expm(-1j * angle * h).T, (0, 3), 4)
        v = embed(expm(-1j * angle * h).T, (2, 1), 4)
        w = embed(expm(-1j * angle * h).T, (3, 1), 4)
        return np.trace(observable @ w @ v @ u @ rho @ u.conj().T @ v.conj().T @ w.conj().T).real

    initial.apply_to_arrays(lambda a: torch.as_tensor(a))
    angle = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
    with TorchLinalgConfig(stabilized=True, quimb_split_drivers=True, svd_gap_rtol=1e-9).activated():
        g = torch.matrix_exp(-1j * angle * torch.as_tensor(h))
        opt = MpoOptimizer(initial, [(g, (0, 3)), (g, (2, 1)), (g, (3, 1))],
                           chi=10, mode="dmt", contraction_opt="greedy")
        result = opt.run(cutoff=0., timing=True)
        assert opt.get_run_timing()["stages"]["dmt.target"]["calls"] == 1
        # Native contraction of a one-site observable, without host readout.
        value = torch.ones(1, dtype=torch.complex128)
        for i in range(result.L):
            bonds = (result.bond(i - 1, i),) if i else ()
            bonds += (result.bond(i, i + 1),) if i + 1 < result.L else ()
            t = result[i].transpose(*(bonds + (result.upper_ind(i), result.lower_ind(i))))
            a = t.data.reshape(value.shape[0], -1, 2, 2)
            diagonal = torch.diagonal(a, dim1=2, dim2=3)
            traced = diagonal[..., 0] - diagonal[..., 1] if i == 0 else diagonal.sum(-1)
            value = value @ traced
        value.real.sum().backward()
    expected = (reference(.130001) - reference(.129999)) / .000002
    assert abs(expected) > 1e-8
    np.testing.assert_allclose(angle.grad, expected, rtol=3e-3, atol=2e-7)


@pytest.mark.parametrize("where", [(2,), (0, 3)])
def test_nonunitary_gate_changes_trace_physically_without_hidden_normalization(where):
    initial, _, _ = mixture(4, rank=16)
    operator = np.diag(np.linspace(.2, .8, 2**len(where)))
    full = embed(operator.T, where, 4)
    expected = full @ dense(initial)
    opt = MpoOptimizer(initial, [((operator, None), where)], chi=10, mode="dmt", contraction_opt="greedy")
    result = opt.run(cutoff=0.)
    assert abs(np.trace(expected) - 1.) > .1
    for r in [(0, 1, 2), (1, 2, 3)]:
        np.testing.assert_allclose(mpo_marginal(result, r), marginal(expected, 4, r), atol=1e-12)


def test_fixed_layout_preserves_logical_queue_and_physical_regions():
    local = [np.diag([.55 + .02 * i, .45 - .02 * i]) for i in range(6)]
    source = qtn.MPO_product_operator(local)
    g = gate()
    opt = MpoOptimizer(source, [(g, (0, 5))], chi=64, mode="dmt", contraction_opt="greedy")
    order = [3, 0, 4, 1, 5, 2]
    opt.apply_layout(order)
    regions = square_lattice_dmt_regions(3, 2, site_order=opt.logical_order)
    before = dense(opt.p)
    positions = tuple(order.index(i) for i in (0, 5))
    full = embed(g.T, positions, 6)
    expected = full @ before @ full.conj().T
    result = opt.run(cutoff=0., dmt_regions=regions)
    assert opt.logical_order == order
    for r in regions:
        np.testing.assert_allclose(mpo_marginal(result, r), marginal(expected, 6, r), atol=1e-12)


def test_larger_2d_identity_density_is_stationary_under_unitary_replay():
    source = qtn.MPO_identity(16, dtype="complex128") / 2**16
    regions = square_lattice_dmt_regions(4, 4, map_mode="snake-row-major")
    queue = [(gate(i), w) for i, w in enumerate([(0, 15), (1, 14), (6, 9)])]
    opt = MpoOptimizer(source, queue, chi=32, mode="dmt", contraction_opt="greedy")
    result = opt.run(cutoff=0., dmt_regions=regions)
    assert result.max_bond() <= 32
    for r in regions:
        np.testing.assert_allclose(mpo_marginal(result, r), np.eye(16) / 16, atol=1e-12)
    np.testing.assert_allclose(opt.norm_diagnostics()["state_norm"], 2.**-8, rtol=1e-11)


@pytest.mark.parametrize("dtype,expected", [("complex64", 1e-6), ("complex128", 1e-12)])
def test_ordinary_compression_defaults_resolve_auto_and_preserve_explicit_zero(dtype, expected):
    initial, _, _ = mixture(6, rank=24, dtype=dtype)
    result, report = compress_mpo_dmt(initial, max_bond=14, return_report=True)
    opt = MpoOptimizer(initial, [], chi=14, contraction_opt="greedy")
    manual, class_report = opt.compress_dmt(return_report=True)
    assert report.cutoff == class_report.cutoff == expected
    assert report.cutoff_mode == class_report.cutoff_mode == "rel"
    np.testing.assert_allclose(dense(manual), dense(result), atol=1e-6 if dtype == "complex64" else 1e-14)
    _, zero = compress_mpo_dmt(initial, max_bond=14, cutoff=0., return_report=True)
    assert zero.cutoff == 0.


def test_local_nonunitary_density_update_keeps_physical_trace_and_norm():
    initial, _, _ = mixture(4, rank=16)
    loss = np.diag([.2, .4, .6, .8])
    queue = [(gate(), (0, 3)), ((loss, None), (2, 1))]
    reference = dense(initial)
    checked = []

    def check(candidate):
        nonlocal reference
        if not checked:
            full = embed(queue[0][0].T, queue[0][1], 4)
            target = full @ reference @ full.conj().T
        else:
            target = embed(loss, (2, 1), 4) @ reference
        for r in [(0, 1, 2), (1, 2, 3)]:
            np.testing.assert_allclose(mpo_marginal(candidate, r), marginal(target, 4, r), atol=1e-12)
        reference = dense(candidate)
        checked.append(1)
        return True

    opt = MpoOptimizer(initial, queue, chi=10, mode="dmt", contraction_opt="greedy")
    with pytest.warns(RuntimeWarning, match="finite_check"):
        result = opt.run(finite_check=check, timing=True)
    assert opt.get_run_timing()["stages"]["dmt.local"]["calls"] == 1
    assert abs(np.trace(reference) - 1) > .1
    np.testing.assert_allclose(opt.norm_diagnostics()["state_norm"], np.linalg.norm(reference), atol=1e-13)
    saved = dense(result)
    opt.set_gates([(gate(), (0, 3)), ((np.zeros((4, 4)), None), (1, 2))])
    with pytest.raises(ValueError, match="resolved nonzero trace"):
        opt.run()
    np.testing.assert_array_equal(dense(opt.p), saved)


def test_routed_failure_preserves_source_for_full_protected_fallback(monkeypatch):
    import pepsy.operators.mpo_dmt as kernel

    original, calls = kernel._apply_pair_exact, []

    def fail_after_move(*args, **kwargs):
        original(*args, **kwargs)
        calls.append(1)
        raise ValueError("DMT temporary protected budget exceeds the cap")

    monkeypatch.setattr(kernel, "_apply_pair_exact", fail_after_move)
    initial, _, _ = mixture(5, rank=12)
    queue = [(gate(), (0, 1)), (gate(32), (4, 0))]
    expected = dense(initial)
    for payload, where in queue:
        full = embed(payload.T, where, 5)
        expected = full @ expected @ full.conj().T
    opt = MpoOptimizer(initial, queue, chi=16, mode="dmt", contraction_opt="greedy")
    result = opt.run(cutoff=0., timing=True)
    assert calls == [1]
    assert opt.get_run_timing()["stages"]["dmt.target"]["calls"] == 2
    np.testing.assert_allclose(dense(result), expected, atol=2e-13)


def test_routing_preserves_regions_with_temporary_budget_above_final_cap(monkeypatch):
    from pepsy.operators._pauli_dmt import _PauliDMTWorkspace, _ProtectedBudgetError

    initial, _, _ = mixture(6, rank=32)
    regions = [(0, 1, 2, 3), (2, 3, 4, 5)]
    cap = 18
    original, injected = _PauliDMTWorkspace.split_pair, []

    def tighter_temporary_budget(self, bond, transfer=None, **kwargs):
        if (not injected and self.regions != tuple(regions) and bond == 2
                and transfer is not None and self.max_bond == cap):
            # Exercise a temporary conservative bound above the final cap;
            # all actual decompositions and marginal checks remain active.
            injected.append(1)
            raise _ProtectedBudgetError(bond, cap, 1, cap)
        return original(self, bond, transfer, **kwargs)

    monkeypatch.setattr(_PauliDMTWorkspace, "split_pair", tighter_temporary_budget)
    queue = [(gate(7), (1, 2)), (gate(8), (0, 5))]
    reference, checked = dense(initial), []

    def check(candidate):
        nonlocal reference
        payload, where = queue[len(checked)]
        full = embed(payload.T, where, 6)
        target = full @ reference @ full.conj().T
        for r in regions:
            np.testing.assert_allclose(mpo_marginal(candidate, r), marginal(target, 6, r), atol=2e-12)
        assert candidate.max_bond() <= cap
        reference = dense(candidate)
        checked.append(1)
        return True

    opt = MpoOptimizer(initial, queue, chi=cap, mode="dmt", contraction_opt="greedy")
    with pytest.warns(RuntimeWarning, match="finite_check"):
        opt.run(cutoff=0., dmt_regions=regions, finite_check=check, dmt_diagnostics=True, timing=True)
    assert opt.get_run_timing()["stages"]["dmt.target"]["calls"] == 1
    assert opt.get_run_timing()["stages"]["dmt.routed"]["calls"] == 1
    assert injected == [1]
    assert max(b.final_bond for b in opt.get_dmt_reports()[-1].bond_reports) > cap


@pytest.mark.parametrize("where", [(0, 5), (4, 1)])
def test_benchmark_reduced_density_gate_oracle_matches_dense_target(where):
    from examples.mpo_dmt_benchmark import unitary_target_panels

    initial, _, _ = mixture(6, rank=12)
    payload = gate()
    full = embed(payload.T, where, 6)
    expected = full @ dense(initial) @ full.conj().T
    regions = [(), (0, 1), (2, 3), (0, 2, 3, 5)]
    panels = unitary_target_panels(initial, payload, where, regions)
    for region, panel in panels.items():
        np.testing.assert_allclose(panel, marginal(expected, 6, region), atol=3e-14)


def test_benchmark_global_compression_error_matches_dense_difference():
    from examples.mpo_dmt_benchmark import compression_relative_error

    initial, weights, local = mixture(6, rank=24)
    result = compress_mpo_dmt(initial, max_bond=12)
    reference = dense(initial)
    expected = np.linalg.norm(dense(result) - reference) / np.linalg.norm(reference)
    assert expected > 1e-4
    assert compression_relative_error(result, weights, local) == pytest.approx(expected, rel=2e-10)


@pytest.mark.parametrize("side", ["ket", "bra"])
def test_routed_density_trace_stabilization_keeps_nonunitary_target_scale(side):
    initial, _, _ = mixture(5, rank=16)
    initial.exponent = 2.
    loss = np.diag([.2, .4, .6, .8])
    payload = (loss, None) if side == "ket" else (None, loss)
    queue = [(gate(), (1, 2)), (payload, (4, 0))]
    first = embed(queue[0][0].T, (1, 2), 5)
    expected = first @ dense(initial) @ first.conj().T
    full = embed(loss, (4, 0), 5)
    expected = full @ expected if side == "ket" else expected @ full
    opt = MpoOptimizer(initial, queue, chi=16, mode="dmt", contraction_opt="greedy")
    result = opt.run(cutoff=0., timing=True)
    assert opt.get_run_timing()["stages"]["dmt.target"]["calls"] == 1
    assert abs(np.trace(expected) - 100.) > 10.
    np.testing.assert_allclose(dense(result), expected, atol=2e-12)


@pytest.mark.parametrize("exponent,inplace", [(0., False), (2., True)])
@pytest.mark.parametrize("kind", ["pair", "ket", "bra", "tensor"])
def test_local_updates_match_dense_ket_bra_and_actual_canonical_gauge(exponent, inplace, kind):
    initial, _, _ = mixture(5, rank=20)
    initial.exponent = exponent
    g, b = gate(), gate(31)
    payload = {"pair": (g, b), "ket": (g, None), "bra": (None, b),
               "tensor": (g.reshape((2,) * 4), b.reshape((2,) * 4))}[kind]
    queue = [(gate(20), (0, 4)), (payload, (3, 2)), (payload, (0, 1)),
             (payload, (2, 1)), (gate(33), (4, 0)), (payload, (2, 1))]
    reference = dense(initial)
    checked = []

    def check(candidate):
        nonlocal reference
        p, where = queue[len(checked)]
        k, br = p if isinstance(p, tuple) else (p, p)
        ket = np.eye(4) if k is None else k.reshape(4, 4).T
        bra = np.eye(4) if br is None else br.reshape(4, 4).T
        target = embed(ket, where, 5) @ reference @ embed(bra, where, 5).conj().T
        np.testing.assert_allclose(dense(candidate), target, atol=3e-11)
        reference = dense(candidate)
        checked.append(1)
        return True

    opt = MpoOptimizer(initial, queue, chi=16, mode="dmt", inplace=inplace, contraction_opt="greedy")
    with pytest.warns(RuntimeWarning, match="finite_check"):
        result = opt.run(cutoff=0., dmt_variant="operator", finite_check=check,
                         dmt_diagnostics=True, timing=True)
    assert opt.get_run_timing()["stages"]["dmt.local"]["calls"] == 4
    assert opt.get_run_timing()["stages"]["dmt.target"]["calls"] == 1
    assert opt.get_run_timing()["stages"]["dmt.routed"]["calls"] == 1
    assert (result is initial) == inplace
    center = opt.info_c["cur_orthog"][0]
    assert center == 2
    for site in range(result.L):
        t = result[site]
        if site == center:
            assert t.left_inds is None
            continue
        bond = result.bond(site, site + 1) if site < center else result.bond(site - 1, site)
        matrix = _host_array(t.transpose(*(tuple(i for i in t.inds if i != bond) + (bond,))).data)
        matrix = matrix.reshape(-1, t.ind_size(bond))
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[1]), atol=2e-12)
