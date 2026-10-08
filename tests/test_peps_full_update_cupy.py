"""Optional real-GPU full-update, native-array and cache-invalidation checks."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.boundary._reuse import StripEnvironmentCache, _array_stamp
from pepsy.optimizers.peps import PepsOptimizer
from pepsy.optimizers.peps._full_update import ReducedPair

pytestmark = [pytest.mark.optional, pytest.mark.peps]


@pytest.fixture
def cp():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('CuPy needs a CUDA device')
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f'CUDA unavailable: {exc}')
    return cp


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('solver', ['quimb', 'qr'])
@pytest.mark.parametrize('adaptive', [False, True])
def test_cupy_full_update_matches_torch_without_host_tensor_transfers(cp, monkeypatch, dtype, solver, adaptive):
    torch = pytest.importorskip('torch')
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype=dtype, seed=24)
    state /= state.norm()
    gate = np.diag(np.exp(np.array([-.21j, .21j, .21j, -.21j]))).astype(dtype)
    outputs = []
    for backend in ('torch', 'cupy'):
        convert = torch.tensor if backend == 'torch' else cp.asarray
        native = state.copy()
        native.apply_to_arrays(convert)
        native_gate = convert(gate)
        gates = [(native_gate, where) for where in
                 [((0, 0), (0, 1)), ((0, 1), (1, 1))]]
        opt = PepsOptimizer(native, gates, chi=2, mode='full-update',
                            contraction_opt='greedy', fit_mode='direct', boundary_chi=16,
                            boundary_kwargs={'cutoff': 0.},
                            boundary_convergence={'max_chi': 16} if adaptive else False,
                            full_update_kwargs={'solver': solver, 'max_iterations': 10})
        with monkeypatch.context() as patch:
            original = ar.to_numpy
            original_asnumpy = cp.asnumpy
            def scalar_only(value, *args, **kwargs):
                if isinstance(value, cp.ndarray):
                    assert value.size == 1, 'CuPy tensor transferred to host'
                return original(value, *args, **kwargs)
            def scalar_asnumpy(value, *args, **kwargs):
                assert not isinstance(value, cp.ndarray) or value.size == 1
                return original_asnumpy(value, *args, **kwargs)
            patch.setattr(ar, 'to_numpy', scalar_only)
            patch.setattr(cp, 'asnumpy', scalar_asnumpy)
            out = opt.run(measure_infidelity=False, measure_final_infidelity=False,
                          accept_if_improved=False)
        if backend == 'cupy':
            assert all(isinstance(t.data, cp.ndarray) and t.data.dtype == dtype
                       and t.data.device == native[0, 0].data.device for t in out)
            if adaptive:
                assert all(r['boundary_convergence']['converged'] for r in opt.get_step_records())
        outputs.append(ar.to_numpy(out.to_dense()).ravel())
        assert all(r['optimizer_result']['solver'] == solver for r in opt.get_step_records())
        assert np.linalg.norm(outputs[-1]) == pytest.approx(1., abs=2e-5)
    a, b = outputs
    fidelity = abs(np.vdot(a, b))**2 / (np.vdot(a, a).real * np.vdot(b, b).real)
    assert fidelity == pytest.approx(1., abs=2e-5 if dtype == 'complex64' else 2e-9)


def test_cupy_stamps_detect_alias_writes_and_release_snapshots(cp):
    import gc
    import weakref
    from pepsy.boundary._reuse import _cupy_versions
    data = cp.ones((3, 3), dtype=cp.complex128)
    initial = _array_stamp(data)
    assert initial == _array_stamp(data)
    view = data[:, 0]
    view[0] = 2j
    assert initial != _array_stamp(data)
    ref, key = weakref.ref(data), id(data)
    del data, view
    gc.collect()
    assert ref() is None and key not in _cupy_versions


@pytest.mark.parametrize('fit_mode', ['eff', 'dmrg2'])
def test_cupy_full_update_with_dmrg_boundary_compression(cp, monkeypatch, fit_mode):
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype='complex128', seed=45)
    state.apply_to_arrays(cp.asarray)
    state /= state.norm()
    gate = cp.diag(cp.exp(cp.asarray([-.21j, .21j, .21j, -.21j])))
    where = ((1, 0), (1, 1))
    target = state.gate(gate, where, contract='split', cutoff=0.)
    opt = PepsOptimizer(state, [(gate, where)], chi=2, mode='full-update',
                        contraction_opt='greedy', fit_mode=fit_mode, boundary_chi=16,
                        boundary_kwargs={'cutoff': 0., 'n_iter': 2},
                        normalize_chi=16, boundary_convergence=False,
                        full_update_kwargs={'max_iterations': 4})
    original = ar.to_numpy
    def scalar_only(value):
        assert not isinstance(value, cp.ndarray) or value.size == 1
        return original(value)
    monkeypatch.setattr(ar, 'to_numpy', scalar_only)
    out = opt.run(measure_infidelity=False, measure_final_infidelity=False,
                  accept_if_improved=False)
    assert all(isinstance(t.data, cp.ndarray) for t in out)
    a, b = out.to_dense().ravel(), target.to_dense().ravel()
    fidelity = float(abs(cp.vdot(a, b))**2 / (cp.vdot(a, a).real * cp.vdot(b, b).real))
    report = opt.get_step_records()[0]['optimizer_result']
    assert float(cp.linalg.norm(a)) == pytest.approx(1., abs=1e-10)
    assert report['fidelity'] == pytest.approx(fidelity, abs=1e-9)


@pytest.mark.parametrize('axis', ['column', 'row'])
def test_cupy_reuses_both_environment_levels_and_invalidates_inplace_changes(cp, axis):
    state = qtn.PEPS.rand(4, 4, bond_dim=2, dtype='complex128', seed=24)
    state.apply_to_arrays(cp.asarray)
    gate = cp.diag(cp.exp(cp.asarray([-.21j, .21j, .21j, -.21j])))
    cache, boundary = StripEnvironmentCache(), None
    def environment(i, boundary=None, cache=None):
        where = ((i, 1), (i + 1, 1)) if axis == 'column' else ((1, i), (1, i + 1))
        pair = ReducedPair(state, gate, where, chi=2)
        guess, _ = pair.initial_states()
        norm, handle = pair.environment(guess, boundary=boundary, chi=16,
                                        contraction_opt='greedy',
                                        boundary_kwargs={'fit_mode': 'direct', 'cutoff': 0.},
                                        strip_cache=cache)
        return norm / cp.linalg.norm(norm), handle
    for i in (0, 1, 2):
        actual, boundary = environment(i, boundary, cache)
        expected, _ = environment(i)
        cp.testing.assert_allclose(actual, expected, atol=1e-10)
    assert cache.hits > 0
    assert boundary.mps_b.environment_cache.hits > 0
    for site in ((0, 1) if axis == 'column' else (1, 0), (0, 0)):
        rebuilds = cache.rebuilds
        state[site].data[..., 0] *= 1.3 + .2j
        actual, boundary = environment(2, boundary, cache)
        expected, _ = environment(2)
        assert cache.rebuilds > rebuilds
        cp.testing.assert_allclose(actual, expected, atol=1e-10)
