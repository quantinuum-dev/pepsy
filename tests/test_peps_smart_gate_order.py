"""Dependency scheduling must preserve the circuit and its per-gate provenance."""

from types import SimpleNamespace

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps._gate_order import (
    _commute, _fuse_singles, _pauli_support, fixed_gate_sites, order_gates,
)

PAULI = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]),
         'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.diag([1, -1])}


def rotation(word, angle=.17):
    p = PAULI[word[0]]
    for letter in word[1:]:
        p = np.kron(p, PAULI[letter])
    return np.cos(angle) * np.eye(len(p)) - 1j * np.sin(angle) * p


def queue(gates):
    return SimpleNamespace(gates=[(g, where, None) for g, where in gates], which=None)


def dense_circuit(entries, sites, initial=None):
    """Independent dense operator reference, retaining supplied operand order."""
    n = len(sites)
    operator = (np.eye(2**n, dtype=complex) if initial is None else initial).reshape((2,) * n + (-1,))
    for gate, where, _ in entries:
        gate = ar.to_numpy(gate)
        support = (where,) if gate.shape == (2, 2) else where
        axes = [sites.index(tuple(site)) for site in support]
        rest = [i for i in range(n+1) if i not in axes]
        operator = np.tensordot(gate.reshape((2,) * (2 * len(axes))), operator,
                                axes=(list(range(len(axes), 2*len(axes))), axes))
        operator = operator.transpose(np.argsort(axes + rest))
    return operator.reshape(2**n, -1)


def converter(backend):
    if backend == 'torch':
        return pytest.importorskip('torch').tensor
    if backend == 'cupy':
        cp = pytest.importorskip('cupy')
        try:
            if cp.cuda.runtime.getDeviceCount() == 0:
                pytest.skip('CUDA unavailable')
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip('CUDA unavailable')
        return cp.asarray
    return np.asarray


@pytest.mark.parametrize('backend', ['numpy', 'torch', pytest.param('cupy', marks=pytest.mark.optional)])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_smart_schedule_and_fusion_match_independent_dense_circuit(backend, dtype, monkeypatch):
    convert = converter(backend)
    sites = [(i, j) for i in range(2) for j in range(3)]
    bonds = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 2), (1, 2)),
             ((0, 0), (0, 1)), ((0, 1), (0, 2)), ((1, 0), (1, 1)), ((1, 1), (1, 2))]
    rng = np.random.default_rng(72)
    gates = [(rotation(word, rng.uniform(-.8, .8)), bonds[i % len(bonds)][::(-1 if i % 2 else 1)])
             for i, word in enumerate(['XX', 'YY', 'ZZ', 'XY', 'YZ', 'ZX'] * 4)]
    for i in range(15):
        gates.insert(2*i, (rotation('XYZ'[i % 3], rng.uniform(-.8, .8)), sites[i % len(sites)]))
    gates.extend([(rotation('X'), (0, 0)), (rotation('Y'), (0, 0)), (rotation('Z'), (0, 0))])
    opt = queue([(convert(g.astype(dtype)), where) for g, where in gates])
    original = list(opt.gates)
    with monkeypatch.context() as patch:
        to_numpy = ar.to_numpy
        def forbidden(value, *args, **kwargs):
            assert ar.infer_backend(value) not in {'torch', 'cupy'}, 'scheduler transferred an array to NumPy'
            return to_numpy(value, *args, **kwargs)
        patch.setattr(ar, 'to_numpy', forbidden)
        if backend == 'cupy':
            import cupy as cp
            patch.setattr(cp, 'asnumpy', forbidden)
        ordered = order_gates(opt, 'smart')
        compiled, groups = _fuse_singles(opt, ordered)
    assert [i for i, _ in ordered] != list(range(len(original)))
    assert len(compiled) < len(original)
    assert sorted(i for group in groups for i in group) == list(range(1, len(original)+1))
    assert all(ar.infer_backend(g) == backend and str(g.dtype).endswith(dtype) for _, (g, _, _) in compiled)
    np.testing.assert_allclose(dense_circuit(original, sites), dense_circuit([e for _, e in compiled], sites),
                               atol=2e-6 if dtype == 'complex64' else 1e-12)
    assert opt.gates == original


def test_pauli_commutation_distinguishes_same_bond_and_single_shared_site():
    a, b, c = (0, 0), (0, 1), (0, 2)
    def descriptor(word, sites):
        return sites, _pauli_support(rotation(word), len(sites), {})
    for x in ('XX', 'YY', 'ZZ'):
        for y in ('XX', 'YY', 'ZZ'):
            assert _commute(descriptor(x, (a, b)), descriptor(y, (b, a)))
            assert _commute(descriptor(x, (a, b)), descriptor(y, (b, c))) == (x == y)
    assert _commute(descriptor('Z', (a,)), descriptor('ZZ', (a, b)))
    assert not _commute(descriptor('X', (a,)), descriptor('ZZ', (a, b)))
    almost_xx = rotation('XX')
    almost_xx[0, 1] = 1e-30
    assert _pauli_support(almost_xx, 2, {}) is None
    assert not _commute(((a, b), None), descriptor('XX', (b, c)))


def test_smart_scheduler_keeps_opaque_entries_as_barriers_and_refreshes_mutated_gates():
    torch = pytest.importorskip('torch')
    g = rotation('XX')
    trainable = torch.tensor(g, requires_grad=True)
    opt = queue([(g, ((0, 0), (0, 1))), (trainable, ((0, 1), (0, 2))),
                 (g, ((0, 1), (1, 1))), (g, ((1, 0), (1, 1)))])
    assert order_gates(opt, 'smart')[1][0] == 1
    opt = queue([(g, ((0, 0), (0, 1))), (rotation('XX'), ((0, 1), (1, 1)))])
    assert [i for i, _ in order_gates(opt, 'smart')] == [1, 0]
    g[:] = rotation('ZZ')
    assert [i for i, _ in order_gates(opt, 'smart')] == [0, 1]


def test_smart_scheduler_finishes_commuting_strips_despite_local_rotations():
    gates = []
    for i in range(3):
        for j in range(4):
            gates += [(rotation('X'), (i, j)), (rotation('XX'), ((i, j), (i+1, j)))]
    opt = queue(gates)
    ordered = order_gates(opt, 'smart')
    columns = [entry[1][0][1] for _, entry in ordered if len(fixed_gate_sites(opt, entry)) == 2]
    assert sum(a != b for a, b in zip(columns, columns[1:])) == 3
    # All X rotations commute, so they are deferred rather than splitting strips.
    assert all(len(fixed_gate_sites(opt, entry)) == 2 for _, entry in ordered[:12])


def test_smart_scheduler_unlocks_pairs_through_chained_local_dependencies():
    a, b, c, d = (0, 0), (0, 1), (1, 0), (1, 1)
    gates = [(rotation('ZZ'), (a, b)), (rotation('X'), b), (rotation('Z'), b),
             (rotation('XX'), (b, d)), (rotation('Y'), d), (rotation('X'), d),
             (rotation('ZZ'), (c, d)), (rotation('Y'), c), (rotation('ZZ'), (a, c))]
    opt = queue(gates)
    ordered = order_gates(opt, 'smart')
    assert [i for i, _ in ordered] == list(range(len(gates)))
    np.testing.assert_allclose(dense_circuit([e for _, e in ordered], [a, b, c, d]),
                               dense_circuit(opt.gates, [a, b, c, d]), atol=1e-12)


@pytest.mark.parametrize('backend', ['torch', pytest.param('cupy', marks=pytest.mark.optional)])
def test_full_update_smart_refinement_spans_required_local_rotations(backend, monkeypatch):
    convert = converter(backend)
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype='complex128', seed=24)
    state.apply_to_arrays(convert)
    state /= state.norm()
    gates = [(convert(rotation('ZZ')), ((1, 0), (1, 1))),
             (convert(rotation('X')), (1, 1)), (convert(rotation('Y')), (1, 1)),
             (convert(rotation('ZZ')), ((1, 1), (1, 2)))]
    opt = PepsOptimizer(state, gates, chi=2, mode='full-update', gate_order='smart',
                        normalize_initial=False, contraction_opt='greedy', fit_mode='direct',
                        boundary_chi=32, normalize_chi=32, boundary_kwargs={'cutoff': 0.},
                        boundary_convergence=False,
                        full_update_kwargs={'refine_sweeps': 1, 'max_iterations': 4})
    refine, targets = opt._full_update_refine, []
    def capture(target, *args, **kwargs):
        targets.append(target.copy())
        return refine(target, *args, **kwargs)
    monkeypatch.setattr(opt, '_full_update_refine', capture)
    original = opt.gates
    out = opt.run(measure_infidelity=False, measure_final_infidelity=False, accept_if_improved=False)
    assert opt.gates is original
    assert opt.last_gate_order['original_step_groups'] == [[1], [2, 3], [4]]
    assert opt.last_gate_order['input_gate_count'] == 4
    assert opt.last_gate_order['compiled_gate_count'] == 3
    assert opt.last_gate_order['single_qubit_gates_fused'] == 1
    records = opt.get_step_records()
    assert [r['original_steps'] for r in records] == [[1], [4]]
    assert records[-1]['accumulated_local_gate_count'] == 2
    assert records[0]['strip_refinement'] is None
    assert records[1]['strip_refinement']['start_step'] == 1
    assert records[1]['strip_refinement']['end_step'] == 3
    assert records[1]['strip_refinement']['original_steps'] == [1, 2, 3, 4]
    assert len(targets) == 1
    exact = state.copy()
    for gate, where in gates:
        exact.gate_(gate, where, contract='split', cutoff=0.)
    np.testing.assert_allclose(ar.to_numpy(targets[0].to_dense()), ar.to_numpy(exact.to_dense()), atol=1e-12)
    assert float(out.norm().real) == pytest.approx(1., abs=1e-10)
    assert all(ar.infer_backend(t.data) == backend for t in out)


def test_failed_smart_run_restores_the_unfused_queue():
    state = qtn.PEPS.rand(2, 2, bond_dim=1, dtype='complex128', seed=12)
    state.apply_to_arrays(pytest.importorskip('torch').tensor)
    gates = [(rotation('X'), (0, 0)), (rotation('Y'), (0, 0)),
             (rotation('ZZ'), ((0, 0), (0, 1)))]
    opt = PepsOptimizer(state, gates, chi=2, mode='full-update', gate_order='smart',
                        normalize_initial=False, contraction_opt='greedy', boundary_convergence=False)
    original = opt.gates
    with pytest.raises(ValueError, match='gate-by-gate'):
        opt.run(k_2q_batch=3)
    assert opt.gates is original and len(original) == 3
    assert opt.last_gate_order['original_step_groups'] == [[1, 2], [3]]


def test_nonunitary_single_site_normalization_preserves_active_strip_target(monkeypatch):
    torch = pytest.importorskip('torch')
    state = qtn.PEPS.rand(2, 3, bond_dim=1, dtype='complex128', seed=24)
    state.apply_to_arrays(torch.tensor)
    state /= state.norm()
    pair = torch.tensor(rotation('ZZ'))
    single = torch.tensor(np.cosh(.2) * PAULI['I'] + np.sinh(.2) * PAULI['X'])
    gates = [(pair, ((0, 0), (0, 1))), (single, (0, 1)),
             (pair, ((0, 1), (0, 2)))]
    opt = PepsOptimizer(
        state, gates, chi=2, mode='full-update', gate_order='smart',
        normalize_initial=False, contraction_opt='greedy', fit_mode='direct',
        boundary_chi=16, normalize_chi=16, boundary_convergence=False,
        full_update_kwargs={'refine_sweeps': 1, 'max_iterations': 4},
    )
    expected = dense_circuit(opt.gates[:2], list(state.gen_site_coos()), ar.to_numpy(state.to_dense()))
    expected /= np.linalg.norm(expected)
    build = opt._build_target
    observed = []
    def check_target(target, gate, where, *args, **kwargs):
        if where == gates[-1][1]:
            # Observe both states after the intervening nonunitary single and
            # before the next pair can normalize or refine either one again.
            observed.append(True)
            np.testing.assert_allclose(ar.to_numpy(target.to_dense()), expected, atol=1e-11)
            np.testing.assert_allclose(ar.to_numpy(opt.state.to_dense()), expected, atol=1e-11)
        return build(target, gate, where, *args, **kwargs)
    monkeypatch.setattr(opt, '_build_target', check_target)
    out = opt.run(non_unitary=True, normalize_final=False, measure_infidelity=False,
                  measure_final_infidelity=False, accept_if_improved=False)
    assert observed == [True]
    exact = dense_circuit(opt.gates, list(state.gen_site_coos()), ar.to_numpy(state.to_dense()))
    exact /= np.linalg.norm(exact)
    np.testing.assert_allclose(ar.to_numpy(out.to_dense()), exact, atol=1e-11)


@pytest.mark.parametrize('shape', [(3, 4), (4, 3)])
@pytest.mark.parametrize('backend', ['torch', pytest.param('cupy', marks=pytest.mark.optional)])
@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_rectangular_full_update_mixed_axes_repeats_and_edge_rotations(shape, backend, dtype):
    convert = converter(backend)
    lx, ly = shape
    state = qtn.PEPS.rand(lx, ly, bond_dim=1, dtype=dtype, seed=82)
    state.apply_to_arrays(convert)
    state /= state.norm()
    corner = (lx-1, ly-1)
    vertical = (corner, (lx-2, ly-1))
    specs = [('ZZ', ((0, 0), (0, 1))), ('X', (0, 1)), ('Y', (0, 1)),
             ('ZZ', ((0, 2), (0, 1))), ('XX', vertical), ('YY', vertical[::-1]),
             ('Z', corner), ('XX', (corner, (lx-1, ly-2))),
             ('XX', ((0, 1), (0, 0))), ('Y', (lx-1, 0))]
    gates = [(convert(rotation(word, .23).astype(dtype)), where) for word, where in specs]
    opt = PepsOptimizer(state, gates, chi=2, mode='full-update', gate_order='smart',
                        normalize_initial=False, contraction_opt='greedy', fit_mode='direct',
                        boundary_chi=32, normalize_chi=32, boundary_kwargs={'cutoff': 0.},
                        boundary_convergence=False,
                        full_update_kwargs={'refine_sweeps': 2, 'max_iterations': 6})
    sites = list(state.gen_site_coos())
    initial = ar.to_numpy(state.to_dense())
    ordered, groups = _fuse_singles(opt, order_gates(opt, 'smart'))
    exact = dense_circuit(opt.gates, sites, initial)
    tolerance = 1e-5 if dtype == 'complex64' else 1e-10
    np.testing.assert_allclose(dense_circuit([e for _, e in ordered], sites, initial), exact, atol=tolerance)
    out = opt.run(measure_infidelity=False, measure_final_infidelity=False, accept_if_improved=False)
    records = opt.get_step_records()
    original_pairs = [i+1 for i, (word, _) in enumerate(specs) if len(word) == 2]
    assert sorted(r['original_steps'][0] for r in records) == original_pairs
    assert records[-1]['accumulated_local_gate_count'] == len(original_pairs)
    assert opt.last_gate_order['original_step_groups'] == groups
    reports = [r['strip_refinement'] for r in records if r['strip_refinement'] is not None]
    assert {r['axis'] for r in reports} == {'x', 'y'}
    assert all(len(r['sites']) == (lx if r['axis'] == 'y' else ly) for r in reports)
    assert all(r['target_max_bond'] <= 4 * opt.chi for r in reports)
    assert all(r['infidelity_after'] <= r['infidelity_before'] for r in reports if r['accepted'])
    assert all(r['local_fidelity'] is not None for r in records)
    assert out.max_bond() <= 2 and (out.Lx, out.Ly) == shape
    for site in sites:
        assert out[site].inds == state[site].inds
        # Public gate absorption/normalization add G/KET bookkeeping tags.
        assert set(state[site].tags) <= set(out[site].tags)
        assert set(out[site].tags) - set(state[site].tags) <= {'G', 'KET'}
        assert out[site].data.dtype == state[site].data.dtype
        assert out[site].data.device == state[site].data.device
        assert ar.infer_backend(out[site].data) == backend
    vector = ar.to_numpy(out.to_dense())
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1., abs=tolerance)
    fidelity = abs(np.vdot(exact, vector))**2 / (np.vdot(exact, exact).real * np.vdot(vector, vector).real)
    assert 0. < fidelity <= 1. + tolerance


@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('backend', ['torch', pytest.param('cupy', marks=pytest.mark.optional)])
def test_rectangular_smart_empty_or_single_site_only_queue(empty, backend):
    state = qtn.PEPS.rand(3, 4, bond_dim=1, dtype='complex128', seed=11)
    state.apply_to_arrays(converter(backend))
    state /= state.norm()
    gates = [] if empty else [(rotation('X'), (2, 3)), (rotation('Y'), (0, 0)), (rotation('Z'), (2, 3))]
    opt = PepsOptimizer(state, gates, chi=2, mode='full-update', gate_order='smart',
                        normalize_initial=False, contraction_opt='greedy', fit_mode='direct',
                        boundary_chi=16, normalize_chi=16, boundary_convergence=False,
                        full_update_kwargs={'refine_sweeps': 1})
    exact = dense_circuit(opt.gates, list(state.gen_site_coos()), ar.to_numpy(state.to_dense()))
    out = opt.run(measure_infidelity=False, measure_final_infidelity=False, accept_if_improved=False)
    np.testing.assert_allclose(ar.to_numpy(out.to_dense()), exact, atol=1e-12)
    assert opt.get_step_records() == []


@pytest.mark.parametrize('transform', ['dagger', 'transpose'])
def test_single_gate_transforms_keep_product_order_on_rectangular_state(transform):
    torch = pytest.importorskip('torch')
    state = qtn.PEPS.rand(3, 4, bond_dim=1, dtype='complex128', seed=11)
    state.apply_to_arrays(torch.tensor)
    state /= state.norm()
    gates = [(torch.tensor(rotation('X')), (2, 3)), (torch.tensor(rotation('Y')), (2, 3))]
    transformed = [(ar.to_numpy(g).conj().T if transform == 'dagger' else ar.to_numpy(g).T, where, None)
                   for g, where in gates]
    exact = dense_circuit(transformed, list(state.gen_site_coos()), ar.to_numpy(state.to_dense()))
    opt = PepsOptimizer(state, gates, chi=2, mode='sweep', gate_order='smart',
                        normalize_initial=False, contraction_opt='greedy', fit_mode='direct',
                        boundary_chi=16, normalize_chi=16, boundary_convergence=False)
    out = opt.run(gate_kwargs={transform: True}, measure_infidelity=False,
                  measure_final_infidelity=False, accept_if_improved=False)
    np.testing.assert_allclose(ar.to_numpy(out.to_dense()), exact, atol=1e-12)
    assert opt.last_gate_order['single_qubit_gates_fused'] == 0
