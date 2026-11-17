"""Density-operator gate replay for the dedicated PEPO optimizer."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import PepoOptimizer


@pytest.mark.parametrize("mode", ["svd", "simple-update", "full-update"])
def test_complex_sandwich_and_replay(mode):
    state = qtn.PEPO.rand(2, 2, bond_dim=1, phys_dim=2, dtype="complex128", seed=17)
    rng = np.random.default_rng(14)
    g = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    where = ((0, 0), (0, 1))
    if mode == "full-update":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(torch.tensor)
    before = np.asarray(state.to_dense())
    unitary = np.kron(g, np.eye(4))
    expected = unitary @ before @ unitary.conj().T
    optimizer = PepoOptimizer(state, [(g, where)], chi=16, mode=mode,
                              boundary_chi=64, cutoff=0.)
    output = optimizer.run()
    actual = output.to_dense()
    if hasattr(actual, "detach"):
        actual = actual.detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(state.to_dense()), before)
    assert output.max_bond() <= 16


def test_dmt_exact_target_all_l_shapes():
    state = qtn.PEPO.rand(2, 2, bond_dim=1, phys_dim=2, dtype="complex128", seed=1)
    rng = np.random.default_rng(12)
    g = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    u = np.kron(g, np.eye(4))
    target = u @ state.to_dense() @ u.conj().T
    opt = PepoOptimizer(state, [(g, ((0, 0), (0, 1)))], chi=16, mode="dmt")
    output = opt.run().to_dense()
    np.testing.assert_allclose(output, target, atol=1e-10)
    assert opt.get_step_records()[0]["regions"] == 4


def test_dmt_failed_compression_is_transactional():
    state = qtn.PEPO.rand(2, 2, bond_dim=1, phys_dim=2, dtype="complex128", seed=3)
    rng = np.random.default_rng(4)
    g = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    opt = PepoOptimizer(state, [(g, ((0, 0), (0, 1)))], chi=1, mode="dmt", dmt_max_nfev=20)
    before = opt.state.to_dense().copy()
    with pytest.raises(ValueError, match="could not satisfy"):
        opt.run()
    np.testing.assert_allclose(opt.state.to_dense(), before)


def _reduced(matrix, keep, n):
    a = matrix.reshape((2,) * (2 * n))
    current = n
    for site in reversed(range(n)):
        if site not in keep:
            a = np.trace(a, axis1=site, axis2=site + current)
            current -= 1
    return a.reshape(2 ** len(keep), 2 ** len(keep))


def test_dmt_lossy_split_preserves_selected_l():
    from pepsy.optimizers._pepo_dmt import _act

    rng = np.random.default_rng(19)
    # A product density and an entangling gate on the right-hand column.
    rho = np.array([[0.7, 0.13j], [-0.13j, 0.3]], dtype=complex)
    state = qtn.PEPO_product_operator([[rho] * 3, [rho] * 3])
    g = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    before = state.to_dense().reshape((2,) * 12)
    target = _act(_act(before, g, (2, 5)), g.conj(), (8, 11)).reshape(64, 64)
    region = ((0, 0), (0, 1), (1, 0))
    opt = PepoOptimizer(state, [(g, ((0, 2), (1, 2)))], chi=1, mode="dmt",
                        regions=[region], dmt_max_nfev=150)
    output = opt.run().to_dense()
    assert np.linalg.norm(output - target) > 1e-5
    np.testing.assert_allclose(_reduced(output, (0, 1, 3), 6),
                               _reduced(target, (0, 1, 3), 6), atol=1e-9)
    assert opt.get_step_records()[0]["max_density_residual"] < 1e-8


def test_exact_single_layer_map_matches_dense_partial_trace():
    from pepsy.optimizers._pepo_dmt import _map

    state = qtn.PEPO.rand(2, 2, bond_dim=2, phys_dim=2, dtype="complex128", seed=34)
    sites = ((0, 0), (0, 1))
    a, b = (state[s] for s in sites)
    bond = next(iter(a.bonds(b)))
    inds = tuple(i for i in a.inds if i != bond) + tuple(i for i in b.inds if i != bond)
    pair = qtn.tensor_contract(a, b, output_inds=inds)
    region = ((0, 0), (1, 0), (1, 1))
    op = _map(state, sites, inds, pair.shape, region, "greedy")
    result = (op @ pair.data.ravel()).reshape((2,) * 6)
    # Map output alternates upper/lower physical indices by site.
    result = result.transpose(0, 2, 4, 1, 3, 5).reshape(8, 8)
    np.testing.assert_allclose(result, _reduced(state.to_dense(), (0, 2, 3), 4), atol=1e-10)


def test_dmt_lossy_split_preserves_all_l_shapes():
    state = qtn.PEPO.rand(2, 2, bond_dim=2, phys_dim=2, dtype="complex128", seed=5)
    for site in state.gen_site_coos():
        tensor = state[site]
        upper, lower = state.upper_ind(*site), state.lower_ind(*site)
        virtual = [i for i in tensor.inds if i not in (upper, lower)]
        data = np.zeros(tensor.transpose(*virtual, upper, lower).shape, dtype=complex)
        data[(0,) * len(virtual) + (slice(None), slice(None))] = np.eye(2) / 2
        data[(1,) * len(virtual) + (slice(None), slice(None))] = (
            (0.2 if site == (0, 0) else 1) * np.diag([1., -1.]) / 2
        )
        tensor.modify(data=data, inds=(*virtual, upper, lower))
    target = state.to_dense()
    opt = PepoOptimizer(state, [(np.eye(4), ((0, 0), (0, 1)))], chi=1, mode="dmt")
    output = opt.run().to_dense()
    assert np.linalg.norm(output - target) > 0.01
    for missing in range(4):
        keep = tuple(i for i in range(4) if i != missing)
        np.testing.assert_allclose(_reduced(output, keep, 4), _reduced(target, keep, 4), atol=1e-10)
    assert opt.get_step_records()[0]["regions"] == 4


@pytest.mark.parametrize("method", ["direct", "dmrg"])
def test_4x4_boundary_map_converges_to_exact(method):
    from pepsy.optimizers._pepo_dmt import _boundary_map, _map

    state = qtn.PEPO.rand(4, 4, bond_dim=2, phys_dim=2, dtype="complex128", seed=4)
    state.exponent = 0.3  # Both map constructions omit the common PEPO scale.
    sites = ((0, 0), (0, 1))
    a, b = (state[s] for s in sites)
    bond = next(iter(a.bonds(b)))
    inds = tuple(i for i in a.inds if i != bond) + tuple(i for i in b.inds if i != bond)
    pair = qtn.tensor_contract(a, b, output_inds=inds)
    region = ((0, 0), (1, 0), (1, 1))
    exact = _map(state, sites, inds, pair.shape, region, "greedy")
    low = _boundary_map(state, sites, inds, pair.shape, region, "greedy",
                        chi=1, method=method, cutoff=0.)
    medium = _boundary_map(state, sites, inds, pair.shape, region, "greedy",
                           chi=2, method=method, cutoff=0.)
    high = _boundary_map(state, sites, inds, pair.shape, region, "greedy",
                         chi=8, method=method, cutoff=0.)
    assert np.linalg.norm(low - exact) / np.linalg.norm(exact) > .01
    assert np.linalg.norm(medium - exact) / np.linalg.norm(exact) < .2
    np.testing.assert_allclose(high, exact, atol=1e-9, rtol=1e-9)


def _density_on_region(state, region):
    tensors = []
    outputs = []
    for site in state.gen_site_coos():
        t = state[site]
        upper, lower = state.upper_ind(*site), state.lower_ind(*site)
        if site in region:
            tensors.append(t)
            outputs.extend((upper, lower))
        else:
            tensors.append(t.trace(upper, lower, preserve_tensor=True))
    return qtn.tensor_contract(*tensors, output_inds=outputs, optimize="greedy").data


@pytest.mark.parametrize("method", ["direct", "dmrg"])
def test_4x4_lossy_dmt_keeps_every_l_density(method):
    from pepsy.optimizers._pepo_dmt import l_regions

    state = qtn.PEPO.rand(4, 4, bond_dim=2, phys_dim=2, dtype="complex128", seed=5)
    for site in state.gen_site_coos():
        tensor = state[site]
        upper, lower = state.upper_ind(*site), state.lower_ind(*site)
        virtual = [i for i in tensor.inds if i not in (upper, lower)]
        data = np.zeros(tensor.transpose(*virtual, upper, lower).shape, dtype=complex)
        data[(0,) * len(virtual) + (slice(None), slice(None))] = np.eye(2) / 2
        data[(1,) * len(virtual) + (slice(None), slice(None))] = (
            (0.2 if site == (0, 0) else 1) * np.diag([1., -1.]) / 2
        )
        tensor.modify(data=data, inds=(*virtual, upper, lower))
    opt = PepoOptimizer(state, [(np.eye(4), ((0, 0), (0, 1)))], chi=1, mode="dmt",
                        boundary_chi=4, fit_mode=method)
    output = opt.run()
    assert opt.get_step_records()[0]["pair_error"] > 1e-4
    assert opt.get_step_records()[0]["regions"] == 36
    assert opt.get_step_records()[0]["environment"] == "boundary-mps"
    for region in l_regions((4, 4)):
        np.testing.assert_allclose(_density_on_region(output, region),
                                   _density_on_region(state, region), atol=1e-10)


@pytest.mark.parametrize("method", ["direct", "dmrg"])
def test_4x4_nontrivial_gate_preserves_all_l_target_densities(method):
    from pepsy.operators.gates import gate_simple
    from pepsy.optimizers._pepo_dmt import l_regions

    local = np.array([[.7, .1j], [-.1j, .3]], dtype=complex)
    state = qtn.PEPO_product_operator([[local] * 4 for _ in range(4)])
    rng = np.random.default_rng(79)
    gate = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    where = ((0, 0), (0, 1))
    target, gauges = state.copy(), {}
    gate_simple(target, gate, where=where, gauges=gauges, max_bond=16,
                cutoff=0., renorm=False)
    target.gauge_simple_insert(gauges)
    opt = PepoOptimizer(state, [(gate, where)], chi=4, mode="dmt", boundary_chi=4,
                        fit_mode=method)
    output = opt.run()
    assert opt.get_step_records()[0]["regions"] == 36
    for region in l_regions((4, 4)):
        np.testing.assert_allclose(_density_on_region(output, region),
                                   _density_on_region(target, region), atol=1e-10)


def test_factor_jacobian_matches_complex_finite_differences():
    from pepsy.optimizers._pepo_dmt import _factor_jacobian

    rng = np.random.default_rng(73)
    complex_random = lambda shape: rng.normal(size=shape) + 1j * rng.normal(size=shape)
    a, b = complex_random((3, 2)), complex_random((2, 4))
    op = complex_random((7, 12))
    da, db = complex_random(a.shape), complex_random(b.shape)
    dv = np.concatenate((da.ravel(), db.ravel()))
    direction = np.concatenate((dv.real, dv.imag))
    epsilon = 1e-6
    change = (op @ ((a + epsilon * da) @ (b + epsilon * db)).ravel()
              - op @ ((a - epsilon * da) @ (b - epsilon * db)).ravel()) / (2 * epsilon)
    expected = np.concatenate((change.real, change.imag))
    np.testing.assert_allclose(_factor_jacobian(op, a, b) @ direction, expected,
                               atol=2e-8, rtol=1e-8)


def test_dmt_rejects_nonfinite_gate_without_updating_state():
    state = qtn.PEPO.rand(2, 2, bond_dim=1, phys_dim=2, dtype="complex128", seed=7)
    before = state.to_dense().copy()
    bad_gate = np.eye(4, dtype=complex)
    bad_gate[0, 0] = np.nan
    opt = PepoOptimizer(state, [(bad_gate, ((0, 0), (0, 1)))], chi=4, mode="dmt")
    with pytest.raises(ValueError, match="nonfinite"):
        opt.run()
    np.testing.assert_allclose(opt.state.to_dense(), before)


def test_direct_dmt_boundary_uses_quimb_1d_direct(monkeypatch):
    import quimb.tensor.tn1d.compress as compress
    from pepsy.optimizers._pepo_dmt import _boundary_map

    state = qtn.PEPO.rand(4, 4, bond_dim=2, phys_dim=2, dtype="complex128", seed=8)
    sites = ((0, 0), (0, 1))
    a, b = (state[s] for s in sites)
    bond = next(iter(a.bonds(b)))
    inds = tuple(i for i in a.inds if i != bond) + tuple(i for i in b.inds if i != bond)
    pair = qtn.tensor_contract(a, b, output_inds=inds)
    calls = []
    original = compress.tensor_network_1d_compress

    def record(*args, **kwargs):
        calls.append(kwargs.get("method"))
        return original(*args, **kwargs)

    monkeypatch.setattr(compress, "tensor_network_1d_compress", record)
    _boundary_map(state, sites, inds, pair.shape, ((0, 0), (1, 0), (1, 1)),
                  "greedy", chi=2, method="direct", cutoff=0.)
    assert calls and set(calls) == {"direct"}


@pytest.mark.parametrize("method", ["direct", "dmrg"])
def test_dmt_cache_reuses_same_pair_and_invalidates_exterior(method):
    local = np.array([[.7, .1j], [-.1j, .3]], dtype=complex)
    state = qtn.PEPO_product_operator([[local] * 3 for _ in range(3)])
    rng = np.random.default_rng(58)
    gates = [np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
             for _ in range(4)]
    pair = ((0, 0), (0, 1))
    other = ((1, 0), (1, 1))
    stream = list(zip(gates, (pair, pair, other, pair)))
    options = dict(chi=16, mode="dmt", regions=[((0, 0), (0, 1), (1, 0))],
                   boundary_chi=8, fit_mode=method)
    cached = PepoOptimizer(state, stream, **options)
    fresh = PepoOptimizer(state, stream, dmt_cache=False, **options)
    actual, reference = cached.run(), fresh.run()
    np.testing.assert_allclose(actual.to_dense(), reference.to_dense(), atol=1e-9, rtol=1e-9)
    records = cached.get_step_records()
    assert records[1]["map_cache"]["hits"] == 1
    assert records[2]["map_cache"]["invalidations"] >= 1
    assert records[3]["map_cache"]["misses"] >= 3


def test_dmt_cache_restarts_between_runs_after_external_mutation():
    state = qtn.PEPO.rand(2, 2, bond_dim=1, phys_dim=2, dtype="complex128", seed=8)
    opt = PepoOptimizer(state, [(np.eye(4), ((0, 0), (0, 1)))], chi=4, mode="dmt",
                        regions=[((0, 0), (0, 1), (1, 0))], boundary_chi=4)
    opt.run()
    opt.state[1, 1].modify(data=2 * opt.state[1, 1].data)
    expected = opt.state.to_dense().copy()
    actual = opt.run().to_dense()
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert opt.get_step_records()[0]["map_cache"]["hits"] == 0


@pytest.mark.parametrize("method", ["direct", "dmrg"])
@pytest.mark.parametrize("chi", [1, 4])
def test_6x6_all_l_shapes_with_two_gates_and_cache(method, chi):
    from pepsy.optimizers._pepo_dmt import l_regions

    local = np.array([[.7, .1j], [-.1j, .3]], dtype=complex)
    state = qtn.PEPO_product_operator([[local] * 6 for _ in range(6)])
    gate = np.diag([1, 1, 1, -1]).astype(complex) if chi == 4 else np.eye(4, dtype=complex)
    pair = ((0, 0), (0, 1))
    opt = PepoOptimizer(state, [(gate, pair), (gate, pair)], chi=chi, mode="dmt",
                        boundary_chi="auto", fit_mode=method)
    result = opt.run()
    records = opt.get_step_records()
    assert records[0]["regions"] == 100
    assert records[0]["boundary_chi"] == max(4, 2 * chi)
    assert records[1]["map_cache"]["hits"] == 100
    # Both gates square to the identity, so every final L equals the input L.
    for region in l_regions((6, 6)):
        np.testing.assert_allclose(_density_on_region(result, region),
                                   _density_on_region(state, region), atol=1e-9, rtol=1e-9)


def test_6x6_lossy_bond_two_to_one_all_l_shapes():
    from pepsy.optimizers._pepo_dmt import l_regions

    state = qtn.PEPO.rand(6, 6, bond_dim=2, phys_dim=2, dtype="complex128", seed=5)
    for site in state.gen_site_coos():
        tensor = state[site]
        upper, lower = state.upper_ind(*site), state.lower_ind(*site)
        virtual = [i for i in tensor.inds if i not in (upper, lower)]
        data = np.zeros(tensor.transpose(*virtual, upper, lower).shape, dtype=complex)
        data[(0,) * len(virtual) + (slice(None), slice(None))] = np.eye(2) / 2
        data[(1,) * len(virtual) + (slice(None), slice(None))] = (
            (0.2 if site == (0, 0) else 1) * np.diag([1., -1.]) / 2
        )
        tensor.modify(data=data, inds=(*virtual, upper, lower))
    pair = ((0, 0), (0, 1))
    opt = PepoOptimizer(state, [(np.eye(4), pair)], chi=1, mode="dmt", boundary_chi=4)
    result = opt.run()
    assert result[pair[0]].bonds_size(result[pair[1]]) == 1
    assert opt.get_step_records()[0]["pair_error"] > 1e-4
    for region in l_regions((6, 6)):
        np.testing.assert_allclose(_density_on_region(result, region),
                                   _density_on_region(state, region), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("method", ["direct", "dmrg"])
def test_6x6_random_boundary_maps_converge(method):
    from pepsy.optimizers._pepo_dmt import _boundary_map, _map

    state = qtn.PEPO.rand(6, 6, bond_dim=2, phys_dim=2, dtype="complex128", seed=41)
    sites = ((0, 0), (0, 1))
    a, b = (state[s] for s in sites)
    bond = next(iter(a.bonds(b)))
    inds = tuple(i for i in a.inds if i != bond) + tuple(i for i in b.inds if i != bond)
    pair = qtn.tensor_contract(a, b, output_inds=inds)
    region = ((0, 0), (1, 0), (1, 1))
    exact = _map(state, sites, inds, pair.shape, region, "greedy")
    errors = []
    for chi in (1, 4, 8):
        approximate = _boundary_map(state, sites, inds, pair.shape, region, "greedy",
                                    chi=chi, method=method, cutoff=0.)
        errors.append(np.linalg.norm(approximate - exact) / np.linalg.norm(exact))
    assert errors[0] > .1
    assert errors[1] < .1
    assert errors[2] < 1e-10


def test_regional_cache_evicts_and_invalidates_safely():
    from pepsy.optimizers._pepo_dmt import RegionalMapCache

    cache = RegionalMapCache(6)
    pair_a = frozenset(((0, 0), (0, 1)))
    pair_b = frozenset(((1, 0), (1, 1)))
    cache.put((pair_a, "a"), np.ones(4))
    cache.put((pair_b, "b"), np.ones(3))
    assert cache.get((pair_a, "a")) is None
    assert cache.report()["evictions"] == 1
    cache.invalidate(((0, 0),))
    assert cache.report()["invalidations"] == 1
    assert cache.report()["entries"] == 0


@pytest.mark.parametrize("mode", ["svd", "simple-update", "full-update"])
def test_lossy_stream_respects_cap_and_replays(mode):
    state = qtn.PEPO.rand(2, 2, bond_dim=1, phys_dim=2, dtype="complex128", seed=2)
    if mode == "full-update":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(torch.tensor)
    rng = np.random.default_rng(22)
    g = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    sites = ((0, 0), (0, 1))
    original = np.asarray(state.to_dense())
    exact = np.kron(g, np.eye(4))
    expected = exact @ original @ exact.conj().T
    optimizer = PepoOptimizer(state, [(g, sites)], chi=1, mode=mode,
                              boundary_chi=16, cutoff=0.)
    first = np.asarray(optimizer.run().to_dense())
    assert optimizer.state.max_bond() <= 1
    assert np.linalg.norm(first - expected) > 1e-5
    second = np.asarray(optimizer.run().to_dense())
    assert optimizer.state.max_bond() <= 1
    assert np.linalg.norm(second - first) > 1e-5


def test_invalid_gate_rejected_before_state_change():
    state = qtn.PEPO.rand(2, 2, bond_dim=1, phys_dim=2, dtype="complex128", seed=3)
    optimizer = PepoOptimizer(state, [(np.eye(4), ((0, 0), (1, 1)))], chi=2)
    before = np.asarray(optimizer.state.to_dense())
    with pytest.raises(ValueError, match="adjacent"):
        optimizer.run()
    np.testing.assert_allclose(optimizer.state.to_dense(), before)


@pytest.mark.parametrize("mode", ["svd", "simple-update", "full-update"])
def test_one_site_gate(mode):
    state = qtn.PEPO.rand(2, 2, bond_dim=1, phys_dim=2, dtype="complex128", seed=9)
    if mode == "full-update":
        torch = pytest.importorskip("torch")
        state.apply_to_arrays(torch.tensor)
    before = np.asarray(state.to_dense())
    g = np.array([[0., 1j], [1., 0.]], dtype=complex)
    exact = np.kron(g, np.eye(8))
    optimizer = PepoOptimizer(state, [(g, (0, 0))], chi=2, mode=mode)
    np.testing.assert_allclose(optimizer.run().to_dense(), exact @ before @ exact.conj().T,
                               atol=1e-10)


def test_dmrg_boundary_lossy_full_update():
    torch = pytest.importorskip("torch")
    state = qtn.PEPO.rand(2, 2, bond_dim=1, phys_dim=2, dtype="complex128", seed=6)
    state.apply_to_arrays(torch.tensor)
    g = np.diag([1, 1, 1, -1]).astype(complex)
    optimizer = PepoOptimizer(state, [(g, ((0, 0), (0, 1)))], chi=1,
                              mode="full-update", fit_mode="dmrg", boundary_chi=8,
                              boundary_kwargs={"n_iter": 2})
    assert optimizer.run().max_bond() <= 1
    assert optimizer.get_step_records()
