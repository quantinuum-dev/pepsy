"""Physical-region DMT checks against dense partial traces."""

from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import PauliMPO, square_lattice_dmt_regions
from pepsy.operators._pauli_dmt import _PauliDMTWorkspace
from pepsy.optimizers import PauliDMT
from pepsy.tensors import OneDMap
from test_pauli_dmt_large import _P, product_mixture


def dense_marginal(matrix, n, region):
    tensor = matrix.reshape((2,) * (2 * n))
    remaining = n
    for site in reversed(range(n)):
        if site not in region:
            tensor = np.trace(tensor, axis1=site, axis2=site + remaining)
            remaining -= 1
    return tensor.reshape(2**remaining, 2**remaining)


@pytest.mark.parametrize('kind,cap', [('l_shapes', 20), ('plaquettes', 30)])
def test_every_split_preserves_physical_regions_and_truncates(kind, cap):
    initial, _, _ = product_mixture(9, rank=48, strength=.5)
    regions = square_lattice_dmt_regions(3, 3, kind=kind)
    workspace = _PauliDMTWorkspace(initial, max_bond=cap, regions=regions, diagnostics=True)
    discarded = []
    for bond in range(8):
        before = PauliMPO.from_pauli_cores(workspace.cores).to_dense()
        report = workspace.split_pair(bond)
        after = PauliMPO.from_pauli_cores(workspace.cores).to_dense()
        for region in regions:
            np.testing.assert_allclose(dense_marginal(after, 9, region),
                                       dense_marginal(before, 9, region), atol=2e-12, rtol=2e-12)
        discarded.append(report.discarded_weight)
    assert max(discarded) > 1e-8
    assert max(PauliMPO.from_pauli_cores(workspace.cores).pauli_bond_dimensions) <= cap
    # At least one physical region contains sites distant along the MPO.
    assert any(max(r) - min(r) >= 5 for r in regions)


def test_explicit_regions_validate_and_failure_is_transactional():
    initial, _, _ = product_mixture(9, rank=48, strength=.5)
    regions = square_lattice_dmt_regions(3, 3)
    before = initial.to_dense()
    with pytest.raises(ValueError, match='protected budget'):
        initial.compress_dmt(max_bond=12, regions=regions, inplace=True)
    np.testing.assert_array_equal(initial.to_dense(), before)
    for invalid in ([], [()], [(0, 0)], [(-1, 1)], [(0, 9)], [(True, 1)]):
        with pytest.raises(ValueError, match='region'):
            initial.compress_dmt(max_bond=20, regions=invalid)


def test_lattice_inventory_respects_mapper_and_geometry():
    import pepsy.operators as operators

    assert 'square_lattice_dmt_regions' in operators.__all__
    mapper = OneDMap(3, 3, mode='row-major')
    _, mapping = mapper.build()
    plaquettes = square_lattice_dmt_regions(3, 3, mapper=mapper)
    assert tuple(sorted(mapping[p] for p in ((0, 0), (1, 0), (0, 1), (1, 1)))) in plaquettes
    assert len(plaquettes) == 4
    assert len(square_lattice_dmt_regions(3, 3, kind='l_shapes', mapper=mapper)) == 16
    assert len(square_lattice_dmt_regions(3, 3, kind='triples', mapper=mapper)) == 22
    with pytest.raises(ValueError, match='mapper'):
        square_lattice_dmt_regions(2, 3, mapper=mapper)


@pytest.mark.parametrize('mode,site_grid', [
    ('snake', ((0, 1), (3, 2), (4, 5))),
    ('snake-row-major', ((0, 5), (1, 4), (2, 3))),
    ('row-major', ((0, 1), (2, 3), (4, 5))),
])
def test_mapping_modes_match_expected_physical_regions_and_mpo_basis(mode, site_grid):
    from pepsy.operators import MPOBasis

    mapper = OneDMap(3, 2, mode=mode)
    _, mapping = mapper.build()
    assert tuple(tuple(mapping[(x, y)] for y in range(2)) for x in range(3)) == site_grid
    expected = tuple(sorted(tuple(sorted((site_grid[x][0], site_grid[x][1],
                                         site_grid[x + 1][0], site_grid[x + 1][1])))
                            for x in range(2)))
    regions = square_lattice_dmt_regions(3, 2, mapper=mapper)
    assert regions == expected
    assert regions == square_lattice_dmt_regions(3, 2, map_mode=mode)
    assert regions == square_lattice_dmt_regions(3, 2, mapper=mapper, map_mode='col-major')
    from itertools import combinations

    expected_l_shapes = {
        tuple(sorted(mapping[c] for c in triangle))
        for x in range(2)
        for triangle in combinations(((x, 0), (x, 1), (x + 1, 0), (x + 1, 1)), 3)
    }
    expected_straight = {
        tuple(sorted(mapping[(x, y)] for x in range(3))) for y in range(2)
    }
    assert set(square_lattice_dmt_regions(3, 2, mapper=mapper, kind='l_shapes')) == expected_l_shapes
    assert set(square_lattice_dmt_regions(3, 2, mapper=mapper, kind='triples')) == (
        expected_l_shapes | expected_straight
    )
    basis = MPOBasis.from_square_lattice(
        3, 2, [(((0, 0), (1, 0)), 'ZZ', 1.)], mapper=mapper,
    )
    assert basis.lattice_to_chain == mapping
    assert basis.terms[0].sites == tuple(sorted((site_grid[0][0], site_grid[1][0])))


@pytest.mark.parametrize('mode,site_order', [
    ('snake', None), ('snake-row-major', None), ('row-major', None),
    ('row-major', (0, 3, 6, 1, 4, 7, 2, 5, 8)),
])
def test_same_physical_density_plaquettes_with_each_mapping(mode, site_order):
    _, weights, logical = product_mixture(9, rank=48, strength=.5)
    mapper = OneDMap(3, 3, mode=mode)
    coords, mapping = mapper.build()
    if site_order is not None:
        coords = {position: coords[logical] for position, logical in enumerate(site_order)}
        mapping = {coordinate: position for position, coordinate in coords.items()}
    order = [coords[j][0] * 3 + coords[j][1] for j in range(9)]
    local = logical[:, order, :]
    rank = len(weights)
    cores = [(weights[:, None] * local[:, 0])[None, :, :]]
    for site in range(1, 8):
        core = np.zeros((rank, rank, 4))
        core[np.arange(rank), np.arange(rank)] = local[:, site]
        cores.append(core)
    cores.append(local[:, -1][:, None, :])
    rho = PauliMPO.from_pauli_cores(cores)
    regions = square_lattice_dmt_regions(3, 3, mapper=mapper, site_order=site_order)
    out, report = rho.compress_dmt(max_bond=30, regions=regions, return_report=True)
    assert max(b.discarded_weight for b in report.bond_reports) > 1e-8
    for x in range(2):
        for y in range(2):
            physical = ((x, y), (x, y + 1), (x + 1, y), (x + 1, y + 1))
            sites = tuple(sorted(mapping[c] for c in physical))
            actual = out.partial_trace(sites, keep=True).to_dense().reshape((2,) * 8)
            source_coords = [coords[site] for site in sites]
            permutation = [source_coords.index(c) for c in physical]
            actual = actual.transpose(permutation + [p + 4 for p in permutation]).reshape(16, 16)
            expected = np.zeros((16, 16), dtype=complex)
            for k, weight in enumerate(weights):
                matrix = np.ones((1, 1), dtype=complex)
                for a, b in physical:
                    matrix = np.kron(matrix, np.einsum('p,pij->ij', logical[k, a * 3 + b], _P))
                expected += weight * matrix
            np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


def test_mps_logical_order_maps_regions_to_storage_positions():
    import quimb.tensor as qtn
    from pepsy.optimizers import MpsOptimizer

    state = qtn.MPS_computational_state('000010', dtype='complex128')
    opt = MpsOptimizer(state, gates=[], chi=8, mode='svd')
    order = (0, 2, 4, 1, 3, 5)
    opt.apply_layout(order, layout_report=False)
    assert tuple(opt.logical_order) == order
    mapper = OneDMap(3, 2, mode='row-major')
    regions = square_lattice_dmt_regions(3, 2, mapper=mapper, site_order=opt.logical_order)
    assert regions == ((0, 1, 3, 4), (1, 2, 4, 5))
    assert regions != square_lattice_dmt_regions(3, 2, mapper=mapper)
    physical = np.asarray(opt.p.to_dense()).reshape(-1)
    bits = format(int(np.argmax(np.abs(physical))), '06b')
    cores = [np.array([.5, 0., 0., .5 if bit == '0' else -.5]).reshape(1, 1, 4)
             for bit in bits]
    rho = PauliMPO.from_pauli_cores(cores)
    first_plaquette = rho.partial_trace(regions[0], keep=True).to_dense()
    expected = np.zeros((16, 16))
    expected[0, 0] = 1.
    np.testing.assert_allclose(first_plaquette, expected, atol=1e-14)
    wrong_region = square_lattice_dmt_regions(3, 2, mapper=mapper)[0]
    assert np.linalg.norm(rho.partial_trace(wrong_region, keep=True).to_dense() - expected) > 1.
    for invalid in ((0, 1), (0, 0, 2, 3, 4, 5), (0, 1, 2, 3, 4, True), 123):
        with pytest.raises(ValueError, match='site_order'):
            square_lattice_dmt_regions(3, 2, mapper=mapper, site_order=invalid)


def test_gate_preserves_regions_of_evolved_target():
    initial, _, _ = product_mixture(9, rank=32, strength=.5)
    regions = square_lattice_dmt_regions(3, 3)
    x = np.array([[0, 1], [1, 0]])
    z = np.diag([1, -1])
    gate = expm(-.2j * (np.kron(x, z) + .4 * np.kron(z, x)))
    bond = 3
    full_gate = np.kron(np.kron(np.eye(2**bond), gate), np.eye(2**(9-bond-2)))
    dense = initial.to_dense()
    target = full_gate @ dense @ full_gate.conj().T
    run = PauliDMT(initial, max_bond=40, regions=regions, diagnostics=True)
    run.apply_gate(gate, (bond, bond + 1))
    actual = run.state.to_dense()
    for region in regions:
        np.testing.assert_allclose(dense_marginal(actual, 9, region),
                                   dense_marginal(target, 9, region), atol=2e-12, rtol=2e-12)
    direct = initial.apply_gate_dmt(gate, (bond, bond + 1), max_bond=40, regions=regions)
    np.testing.assert_allclose(direct.to_dense(), actual, atol=2e-12, rtol=2e-12)


def test_geometric_gate_environments_follow_nonmonotonic_sweeps():
    initial, _, _ = product_mixture(6, rank=24, strength=.5)
    regions = square_lattice_dmt_regions(3, 2, kind='plaquettes')
    x = np.array([[0, 1], [1, 0]])
    z = np.diag([1, -1])
    gate = expm(-.17j * (np.kron(x, z) + .37 * np.kron(z, x)))
    run = PauliDMT(initial, max_bond=24, regions=regions, diagnostics=True)
    reports = []
    for bond in (2, 4, 1, 3, 0, 4, 2, 1):
        before = run.state.to_dense()
        full_gate = np.kron(np.kron(np.eye(2**bond), gate), np.eye(2**(4 - bond)))
        target = full_gate @ before @ full_gate.conj().T
        reports.append(run.apply_gate(gate, (bond, bond + 1)))
        actual = run.state.to_dense()
        for region in regions:
            np.testing.assert_allclose(dense_marginal(actual, 6, region),
                                       dense_marginal(target, 6, region), atol=4e-12, rtol=4e-12)
    assert any(b.input_rank > b.final_bond for b in reports)


def test_reused_geometric_environments_match_fresh_contractions(monkeypatch):
    import pepsy.operators._pauli_dmt as dmt

    initial, _, _ = product_mixture(9, rank=48, strength=.5)
    regions = square_lattice_dmt_regions(3, 3, kind='plaquettes')
    optimized, report = initial.compress_dmt(max_bond=30, regions=regions, return_report=True)

    def fresh_vectors(cores, bond, patterns, eye, environments):
        del environments
        left, right = [], []
        for pattern in patterns[0]:
            labels = dict(pattern)
            v = np.ones(1, dtype=cores[0].dtype)
            for j in range(bond):
                v = dmt._unit(v @ cores[j][:, :, labels.get(j, 0)])
            left.append(np.outer(v.conj(), eye[:, labels.get(bond, 0)]).reshape(-1))
        for pattern in patterns[1]:
            labels = dict(pattern)
            v = np.ones(1, dtype=cores[0].dtype)
            for j in range(len(cores) - 1, bond + 1, -1):
                v = dmt._unit(cores[j][:, :, labels.get(j, 0)] @ v)
            right.append(np.outer(eye[:, labels.get(bond + 1, 0)], v).reshape(-1))
        return np.stack(left, axis=1), np.stack(right, axis=1)

    monkeypatch.setattr(dmt, '_region_vectors', fresh_vectors)
    reference, fresh_report = initial.compress_dmt(max_bond=30, regions=regions, return_report=True)
    np.testing.assert_allclose(optimized.to_dense(), reference.to_dense(), atol=2e-12, rtol=2e-12)
    assert report.final_bond_dimensions == fresh_report.final_bond_dimensions
    assert any(b.discarded_weight > 0 for b in report.bond_reports)


@pytest.mark.parametrize('scale', [1e-150, 1., 1e150])
def test_scaled_region_panels_match_independent_partial_trace(scale):
    from pepsy.operators._pauli_dmt import _region_trace_data

    operator = PauliMPO.from_terms(6, [
        (scale, (0, 5), 'XZ'), (.3j * scale, (1, 3), 'YZ'),
    ])
    regions = ((2, 4), (0, 5), (0, 2, 5), (0, 1, 3, 5))
    for region, (panel, log_scale) in zip(
        regions, _region_trace_data(operator.to_pauli_cores(), regions),
    ):
        matrix = np.zeros((2**len(region), 2**len(region)), dtype=complex)
        if np.isfinite(log_scale):
            coefficients = np.asarray(panel) * np.exp(log_scale) * 2**(6 - len(region))
            for value, labels in zip(coefficients, product(range(4), repeat=len(region))):
                basis = np.ones((1, 1), dtype=complex)
                for label in labels:
                    basis = np.kron(basis, _P[label])
                matrix += value * basis
        expected = operator.partial_trace(region, keep=True).to_dense()
        np.testing.assert_allclose(matrix / scale, expected / scale, atol=2e-12, rtol=2e-12)


def test_torch_geometric_gate_gradient_matches_finite_difference():
    torch = pytest.importorskip('torch')
    from pepsy.backends import TorchLinalgConfig
    from test_pauli_dmt import _density

    initial = _density(5)
    cores = [np.asarray(a).real for a in initial.to_pauli_cores()]
    regions = ((0, 2, 4),)
    probe = PauliMPO.from_terms(5, [(1., 'XYZXI')])
    generator = PauliMPO(2, [(1., 'ZX'), (.3, 'XI')]).to_dense()

    def value(angle):
        gate = expm(-1j * angle * generator)
        out = PauliMPO.from_pauli_cores(cores).apply_gate_dmt(
            gate, (1, 2), max_bond=10, regions=regions,
        )
        return float(np.real(probe.inner(out)))

    theta = torch.tensor(.13, dtype=torch.float64, requires_grad=True)
    tcores = [torch.tensor(a, dtype=torch.float64) for a in cores]
    tgen = torch.tensor(generator, dtype=torch.complex128)
    with TorchLinalgConfig(stabilized=True, svd_gap_rtol=1e-9).activated():
        gate = torch.matrix_exp(-1j * theta * tgen)
        out = PauliMPO.from_pauli_cores(tcores).apply_gate_dmt(
            gate, (1, 2), max_bond=10, regions=regions,
        )
        loss = probe.inner(out).real
        loss.backward()
    expected = (value(.130001) - value(.129999)) / .000002
    assert abs(expected) > 1e-8
    assert np.isfinite(theta.grad.item())
    assert theta.grad.item() == pytest.approx(expected, rel=2e-3, abs=2e-7)


@pytest.mark.parametrize('backend', ['numpy', 'torch', 'jax'])
def test_complex_backends_preserve_regions(backend, request):
    initial, _, _ = product_mixture(6, rank=16, strength=.5)
    cores = [np.asarray(a, dtype='complex128') for a in initial.to_pauli_cores()]
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        cores = [torch.tensor(a) for a in cores]
        host = lambda a: a.detach().cpu().numpy()
    elif backend == 'jax':
        jax = pytest.importorskip('jax')
        previous_x64 = jax.config.x64_enabled
        request.addfinalizer(lambda: jax.config.update('jax_enable_x64', previous_x64))
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        cores = [jnp.asarray(a) for a in cores]
        host = np.asarray
    else:
        host = np.asarray
    rho = PauliMPO.from_pauli_cores(cores)
    regions = square_lattice_dmt_regions(3, 2)
    result = rho.compress_dmt(max_bond=16, regions=regions, cutoff='auto')
    assert str(result.to_pauli_cores()[0].dtype).split('.')[-1] == 'complex128'
    before, after = host(rho.to_dense()), host(result.to_dense())
    for region in regions:
        np.testing.assert_allclose(dense_marginal(after, 6, region),
                                   dense_marginal(before, 6, region), atol=3e-12, rtol=3e-12)


def test_nonlocal_gate_followed_by_geometry_aware_sweep():
    initial, _, _ = product_mixture(6, rank=12, strength=.5)
    regions = square_lattice_dmt_regions(3, 2)
    z = np.diag([1., -1.])
    gate = expm(-.2j * np.kron(z, z))
    # Sites 0 and 3 are a physical nearest-neighbor pair in this snake order.
    evolved = initial.apply_gate(gate, (0, 3), atol=0.)
    before = evolved.to_dense()
    full_zz = np.kron(np.kron(np.kron(z, np.eye(2**2)), z), np.eye(2**2))
    full_gate = np.cos(.2) * np.eye(2**6) - 1j * np.sin(.2) * full_zz
    np.testing.assert_allclose(before, full_gate @ initial.to_dense() @ full_gate.conj().T,
                               atol=2e-12, rtol=2e-12)
    result = evolved.compress_dmt(max_bond=24, regions=regions)
    for region in regions:
        np.testing.assert_allclose(dense_marginal(result.to_dense(), 6, region),
                                   dense_marginal(before, 6, region), atol=3e-12, rtol=3e-12)


def test_traceless_operator_regions():
    rho = PauliMPO.from_terms(6, [(1., (0, 5), 'ZX'), (.2j, (1,), 'Y')])
    regions = square_lattice_dmt_regions(3, 2, kind='triples')
    result = rho.compress_dmt(max_bond=20, variant='operator', regions=regions)
    for region in regions:
        np.testing.assert_allclose(dense_marginal(result.to_dense(), 6, region),
                                   dense_marginal(rho.to_dense(), 6, region), atol=2e-12)


def test_complex_virtual_gauges_with_actual_truncation():
    initial, _, _ = product_mixture(12, rank=72, strength=.5)
    cores = [np.asarray(a, dtype='complex128') for a in initial.to_pauli_cores()]
    rng = np.random.default_rng(47)
    for j in range(len(cores) - 1):
        phases = np.exp(1j * rng.uniform(-3, 3, cores[j].shape[1]))
        cores[j] = cores[j] * phases[None, :, None]
        cores[j + 1] = cores[j + 1] / phases[:, None, None]
    gauged = PauliMPO.from_pauli_cores(cores)
    regions = square_lattice_dmt_regions(4, 3)
    result, report = gauged.compress_dmt(max_bond=40, regions=regions, return_report=True)
    assert max(b.discarded_weight for b in report.bond_reports) > 1e-8
    for region in regions:
        np.testing.assert_allclose(result.partial_trace(region, keep=True).to_dense(),
                                   gauged.partial_trace(region, keep=True).to_dense(),
                                   atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('lx,ly,cap', [(4, 3, 40), (4, 4, 48)])
def test_larger_plaquettes_and_ising_energy_against_product_oracle(lx, ly, cap):
    rho, weights, local = product_mixture(lx * ly, rank=72, strength=.5)
    mapper = OneDMap(lx, ly, mode='snake')
    regions = square_lattice_dmt_regions(lx, ly, mapper=mapper)
    result, report = rho.compress_dmt(max_bond=cap, regions=regions, return_report=True)
    assert max(b.discarded_weight for b in report.bond_reports) > 1e-8
    assert max(result.pauli_bond_dimensions) <= cap
    for region in regions:
        exact = np.zeros((16, 16), dtype=complex)
        for k, weight in enumerate(weights):
            factors = [np.einsum('p,pij->ij', local[k, site], _P) for site in region]
            matrix = factors[0]
            for factor in factors[1:]:
                matrix = np.kron(matrix, factor)
            exact += weight * matrix
        actual = result.partial_trace(region, keep=True).to_dense()
        np.testing.assert_allclose(actual, exact, atol=2e-11, rtol=2e-11)
    _, mapping = mapper.build()
    edges = [(mapping[(x, y)], mapping[(a, b)])
             for x in range(lx) for y in range(ly)
             for a, b in ((x + 1, y), (x, y + 1)) if a < lx and b < ly]
    exact_energy = sum(.7 * np.dot(weights, 4 * local[:, a, 3] * local[:, b, 3])
                       for a, b in edges)
    exact_energy += sum(.3 * np.dot(weights, 2 * local[:, j, 1]) for j in range(lx * ly))
    zz = np.kron(_P[3], _P[3])
    energy = sum(.7 * np.trace(result.partial_trace((a, b), keep=True).to_dense() @ zz)
                 for a, b in edges)
    energy += sum(.3 * np.trace(result.partial_trace((j,), keep=True).to_dense() @ _P[1])
                  for j in range(lx * ly))
    np.testing.assert_allclose(energy, exact_energy, atol=2e-11, rtol=2e-11)


@pytest.mark.slow
@pytest.mark.parametrize('lx,ly,cap,dtype', [
    (5, 5, 72, 'float64'), (6, 6, 88, 'complex128'), (8, 8, 92, 'float64'),
])
def test_large_square_plaquettes_trace_and_energy(lx, ly, cap, dtype):
    rho, weights, local = product_mixture(lx * ly, rank=96, strength=.5, dtype=dtype)
    if dtype == 'complex128':
        cores = list(rho.to_pauli_cores())
        rng = np.random.default_rng(456)
        for j in range(len(cores) - 1):
            phases = np.exp(1j * rng.uniform(-3, 3, cores[j].shape[1]))
            cores[j] = cores[j] * phases[None, :, None]
            cores[j + 1] = cores[j + 1] / phases[:, None, None]
        rho = PauliMPO.from_pauli_cores(cores)
    mapper = OneDMap(lx, ly, mode='snake-row-major')
    regions = square_lattice_dmt_regions(lx, ly, mapper=mapper)
    out, report = rho.compress_dmt(max_bond=cap, cutoff='auto', regions=regions, return_report=True)
    assert max(out.pauli_bond_dimensions) <= cap
    assert any(b.input_rank > b.final_bond for b in report.bond_reports)
    assert max(b.discarded_weight for b in report.bond_reports) > 0.
    np.testing.assert_allclose(out.trace(), 1., atol=2e-10, rtol=2e-10)
    for region in regions:
        exact = np.zeros((16, 16), dtype=complex)
        for k, weight in enumerate(weights):
            matrix = np.ones((1, 1), dtype=complex)
            for site in region:
                matrix = np.kron(matrix, np.einsum('p,pij->ij', local[k, site], _P))
            exact += weight * matrix
        actual = out.partial_trace(region, keep=True).to_dense()
        np.testing.assert_allclose(actual, exact, atol=2e-10, rtol=2e-10)
    _, mapping = mapper.build()
    edges = [(mapping[(x, y)], mapping[(a, b)])
             for x in range(lx) for y in range(ly)
             for a, b in ((x + 1, y), (x, y + 1)) if a < lx and b < ly]
    exact_energy = sum(.7 * np.dot(weights, 4 * local[:, a, 3] * local[:, b, 3])
                       for a, b in edges)
    exact_energy += sum(.3 * np.dot(weights, 2 * local[:, j, 1]) for j in range(lx * ly))
    actual_energy = sum(.7 * np.trace(out.partial_trace((a, b), keep=True).to_dense()
                                      @ np.kron(_P[3], _P[3])) for a, b in edges)
    actual_energy += sum(.3 * np.trace(out.partial_trace((j,), keep=True).to_dense() @ _P[1])
                         for j in range(lx * ly))
    np.testing.assert_allclose(actual_energy, exact_energy, atol=2e-9, rtol=2e-9)


@pytest.mark.slow
def test_large_square_l_shapes_against_product_oracle():
    rho, weights, local = product_mixture(25, rank=64, strength=.5)
    regions = square_lattice_dmt_regions(5, 5, kind='l_shapes', map_mode='snake-row-major')
    out, report = rho.compress_dmt(max_bond=40, regions=regions, return_report=True)
    assert len(regions) == 64
    assert max(b.discarded_weight for b in report.bond_reports) > 1e-12
    for region in regions:
        expected = np.zeros((8, 8), dtype=complex)
        for k, weight in enumerate(weights):
            matrix = np.ones((1, 1), dtype=complex)
            for site in region:
                matrix = np.kron(matrix, np.einsum('p,pij->ij', local[k, site], _P))
            expected += weight * matrix
        actual = out.partial_trace(region, keep=True).to_dense()
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
