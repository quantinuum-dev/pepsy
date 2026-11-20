"""Protected directions against independent contractions and partial traces."""

import numpy as np
import pytest

from pepsy.operators import PauliMPO, square_lattice_dmt_regions
from pepsy.operators import _pauli_dmt as dmt
from pepsy.operators.mpo_dmt import _extract_cores
from test_pauli_dmt_large import product_mixture
from test_pauli_dmt_lattice import dense_marginal


def reference_vectors(cores, bond, patterns, eye, trace):
    """Contract every string independently from the chain boundary."""
    trace = eye[:, 0] if trace is None else trace

    def unit(v):
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    outputs = []
    for side, strings in enumerate(patterns):
        columns = []
        for string in strings:
            labels = dict(string)

            def physical(j):
                return eye[:, labels[j]] if j in labels else trace

            v = np.ones(1, dtype=cores[0].dtype)
            if side == 0:
                for j in range(bond):
                    v = unit(v @ np.einsum('abp,p->ab', cores[j], physical(j)))
                columns.append(np.outer(v.conj(), physical(bond)).reshape(-1))
            else:
                for j in range(len(cores) - 1, bond + 1, -1):
                    v = unit(np.einsum('abp,p->ab', cores[j], physical(j)) @ v)
                columns.append(np.outer(physical(bond + 1), v).reshape(-1))
        outputs.append(np.stack(columns, axis=1))
    return outputs


def random_cores(n=16, rank=5):
    rng = np.random.default_rng(607)
    bonds = [1] + [rank] * (n - 1) + [1]
    return [rng.normal(size=(a, b, 4)) + 1j*rng.normal(size=(a, b, 4))
            for a, b in zip(bonds, bonds[1:])]


@pytest.mark.parametrize('selection', ['smallest', 'min_cover'])
@pytest.mark.parametrize('ordinary', [False, True])
@pytest.mark.parametrize('kind', ['plaquettes', 'six_spin'])
def test_shared_directions_match_independent_strings_after_tensor_changes(selection, ordinary, kind):
    cores = random_cores()
    regions = (square_lattice_dmt_regions(4, 4) if kind == 'plaquettes' else
               tuple((i, i+1, i+2, i+5, i+6, i+7) for i in range(9)))
    trace = np.array([1., 0., 0., 1.]) if ordinary else None
    eye = np.eye(4)
    patterns = dmt._region_pattern(regions, 7, selection=selection)
    env = dmt._TraceEnvironments(cores, trace)
    before = None
    for changed in (False, True):
        if changed:
            cores[5] = cores[5] + .3*np.roll(cores[5], 1, axis=0)
            cores[10] = cores[10] + .2*np.roll(cores[10], 1, axis=1)
            env.invalidate(5, 11)
        actual = dmt._region_vectors(cores, 7, patterns, eye, env)
        expected = reference_vectors(cores, 7, patterns, eye, trace)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=2e-14, rtol=2e-14)
        if before is not None:
            assert max(np.linalg.norm(a-b) for a, b in zip(actual, before)) > .01
        before = actual


def test_overlapping_strings_share_partial_contractions(monkeypatch):
    cores = random_cores()
    patterns = dmt._region_pattern(((i, i+1, i+2, i+5, i+6, i+7)
                                   for i in range(9)), 7)
    env = dmt._TraceEnvironments(cores)
    env.at(7, 9)
    independent_steps = sum(7 - min((s for s, _ in p), default=7) for p in patterns[0])
    independent_steps += sum(max((s for s, _ in p), default=8) - 8 for p in patterns[1])
    calls = []
    original = dmt._unit

    def observe(value):
        calls.append(value.shape)
        return original(value)

    monkeypatch.setattr(dmt, '_unit', observe)
    dmt._region_vectors(cores, 7, patterns, np.eye(4), env)
    assert len(calls) < independent_steps / 2


@pytest.mark.parametrize('ordinary', [False, True])
def test_shared_torch_graph_matches_independent_complex_finite_difference(ordinary):
    torch = pytest.importorskip('torch')
    cores = random_cores(n=10)
    patterns = dmt._region_pattern(((0, 2, 4, 7), (1, 3, 6, 9)), 4)
    trace = np.array([1., 0., 0., 1.]) if ordinary else None
    delta = np.zeros_like(cores[2])
    delta[1, 2, 0] = .7 + .2j

    def reference(theta):
        values = list(cores)
        values[2] = values[2] + theta*delta
        vectors = reference_vectors(values, 4, patterns, np.eye(4), trace)
        return sum(np.sum(np.abs(v)**2 * np.arange(1, v.shape[0]+1)[:, None]) for v in vectors)

    theta = torch.tensor(.17, dtype=torch.float64, requires_grad=True)
    arrays = [torch.tensor(c, dtype=torch.complex128) for c in cores]
    arrays[2] = arrays[2] + theta*torch.tensor(delta)
    trace_tensor = None if trace is None else torch.tensor(trace, dtype=torch.complex128)
    vectors = dmt._region_vectors(arrays, 4, patterns, torch.eye(4, dtype=torch.complex128),
                                 dmt._TraceEnvironments(arrays, trace_tensor))
    loss = sum((v.abs()**2 * torch.arange(1, v.shape[0]+1)[:, None]).sum() for v in vectors)
    loss.backward()
    expected = (reference(.170001) - reference(.169999)) / .000002
    assert abs(expected) > 1e-4
    assert loss.item() == pytest.approx(reference(.17), abs=2e-12)
    assert theta.grad.item() == pytest.approx(expected, rel=2e-6, abs=2e-7)


@pytest.mark.parametrize('ordinary', [False, True])
def test_lossy_split_preserves_joint_six_spin_marginal_and_its_subsets(ordinary):
    from scipy.linalg import expm
    from pepsy.operators.mpo_dmt import _apply_pair_exact

    rho, _, _ = product_mixture(8, rank=96, strength=.7)
    # The gate touches one protected site and one gap site. Compare with its
    # exact target, including correlations between the two three-spin blocks.
    gate = expm(-.23j * np.kron(np.diag([1., -1.]), [[0., 1.], [1., 0.]]))
    exact = dmt._PauliDMTWorkspace(rho, max_bond=256)
    transfer = dmt._gate_transfer(gate, exact.cores[0], 'conjugate')
    _apply_pair_exact(exact, 2, transfer)
    target = PauliMPO.from_pauli_cores(exact.cores)
    region = (0, 1, 2, 5, 6, 7)
    options = dict(max_bond=70, regions=[region], region_selection='min_cover', diagnostics=True)
    if ordinary:
        cores, _, _ = _extract_cores(target.to_mpo())
        workspace = dmt._PauliDMTWorkspace(_cores=cores, _trace_vector=np.eye(2).reshape(-1),
                                          **options)
    else:
        workspace = dmt._PauliDMTWorkspace(target, **options)
    report = workspace.split_pair(3)
    if ordinary:
        from pepsy.operators.mpo_dmt import _write_workspace

        output = target.to_mpo()
        _, layouts, dimension = _extract_cores(output)
        workspace.mpo_layouts, workspace.mpo_dimension = layouts, dimension
        _write_workspace(output, workspace)
        actual = np.asarray(output.to_dense())
    else:
        actual = PauliMPO.from_pauli_cores(workspace.cores).to_dense()
    expected = target.to_dense()
    assert report.final_bond <= 70
    assert report.discarded_weight > 1e-10
    for sites in (region, (0, 1, 2), (5, 6, 7), (1, 2)):
        np.testing.assert_allclose(dense_marginal(actual, 8, sites),
                                   dense_marginal(expected, 8, sites), atol=2e-12, rtol=2e-12)
