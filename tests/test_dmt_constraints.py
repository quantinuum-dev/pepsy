"""Rank-aware covers, contained regions, and explicit small-chi relaxation."""

import warnings

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.operators import PauliMPO, compress_mpo_dmt, square_lattice_dmt_regions
from pepsy.operators._pauli_dmt import (
    _PauliDMTWorkspace, _region_trace_data, _relative_density_defect, _split,
)
from pepsy.operators.mpo_dmt import _extract_cores
from pepsy.operators.pauli_dmt_geometry import _maximal_regions
from pepsy.optimizers import MpoOptimizer, PauliDMT
from test_mpo_dmt_replay import dense, embed, gate, marginal, mixture
from test_pauli_dmt import _density


def rebuilt_mixture(weights, local):
    rank, n = local.shape[:2]
    arrays = [weights[:,None,None] * local[:,0]]
    for i in range(1,n-1):
        core = np.zeros((rank,rank,2,2),complex)
        core[np.arange(rank),np.arange(rank)] = local[:,i]
        arrays.append(core)
    arrays.append(local[:,-1])
    return qtn.MatrixProductOperator(arrays)


def test_rank_aware_cover_fits_a_cap_that_the_string_cover_rejects():
    _, weights, local = mixture(6,rank=24)
    local[:,3:5] = np.eye(2)/2
    rho = rebuilt_mixture(weights, local)
    cores, _, _ = _extract_cores(rho)
    region = (0,1,3,4)
    options = dict(_cores=cores, _trace_vector=np.eye(2).reshape(-1),
                   max_bond=2, regions=[region], diagnostics=True)
    old = _PauliDMTWorkspace(**options, region_selection='min_cover')
    with pytest.raises(ValueError, match='protected budget'):
        old.split_pair(2)
    new = _PauliDMTWorkspace(**options, region_selection='rank_aware')
    before = _region_trace_data(new.cores,[region],new.trace_vector)
    report = new.split_pair(2)
    after = _region_trace_data(new.cores,[region],new.trace_vector)
    assert report.protected_left_rank + report.protected_right_rank == 2
    assert report.final_bond <= 2 and report.discarded_weight > 1e-5
    assert _relative_density_defect(before,after) < 2e-12


@pytest.mark.parametrize('pauli', [False, True])
def test_contained_region_panels_equal_independent_contractions(pauli):
    rho = _density()
    if pauli:
        cores, trace = rho.to_pauli_cores(), None
    else:
        cores, _, _ = _extract_cores(rho.to_mpo())
        trace = np.eye(2).reshape(-1)
    regions = ((0,1,3,4),(0,3),(1,),(2,5),(5,))
    assert _maximal_regions(regions) == ((0,1,3,4),(2,5))
    combined = _region_trace_data(cores,regions,trace)
    separate = [_region_trace_data(cores,[r],trace)[0] for r in regions]
    assert _relative_density_defect(separate,combined) < 2e-12
    # Keep every requested region in public checks while deriving children.
    out = (rho.compress_dmt(max_bond=20,regions=regions,region_selection='rank_aware') if pauli
           else compress_mpo_dmt(rho.to_mpo(),max_bond=20,regions=regions,region_selection='rank_aware'))
    actual = out.to_dense() if pauli else dense(out)
    for region in regions:
        np.testing.assert_allclose(marginal(actual,6,region), marginal(rho.to_dense(),6,region),atol=2e-12)


def test_contained_panels_remain_log_scaled_on_long_chains():
    cores = [np.array([[[.5,0.,0.,0.]]])] * 1200
    regions = ((0,100,900,1199),(0,900),(100,))
    combined = _region_trace_data(cores,regions)
    separate = [_region_trace_data(cores,[r])[0] for r in regions]
    # Independently grouped sums of 1200 log factors accumulate roundoff.
    assert _relative_density_defect(separate,combined) < 64*len(cores)*np.finfo(float).eps
    assert all(log < -700 and np.isfinite(log) for _,log in combined)


@pytest.mark.parametrize('pauli', [False,True])
@pytest.mark.parametrize('cap', [1,2,4])
def test_small_cap_relaxation_is_explicit_and_reports_real_density_error(pauli,cap):
    rho = _density()
    regions = square_lattice_dmt_regions(3,2)
    before = rho.to_dense()
    options = dict(max_bond=cap,regions=regions,region_selection='rank_aware')
    with pytest.raises(ValueError,match='protected budget'):
        if pauli:
            rho.compress_dmt(**options)
        else:
            compress_mpo_dmt(rho.to_mpo(),**options)
    with pytest.warns(RuntimeWarning, match='constraints were relaxed'):
        if pauli:
            out,report = rho.compress_dmt(**options,constraint_mode='relaxed',return_report=True)
            after = out.to_dense()
            assert max(out.pauli_bond_dimensions) <= cap
        else:
            out,report = compress_mpo_dmt(rho.to_mpo(),**options,constraint_mode='relaxed',return_report=True)
            after = dense(out)
            assert out.max_bond() <= cap
    expected = max(np.linalg.norm(marginal(after,6,r)-marginal(before,6,r)) /
                   np.linalg.norm(marginal(before,6,r)) for r in regions)
    assert expected > 1e-8 and report.relaxed_bonds
    assert report.constraint_mode == 'relaxed'
    np.testing.assert_allclose(report.relative_density_error,expected,rtol=2e-9,atol=2e-13)
    np.testing.assert_allclose(np.trace(after),np.trace(before),atol=2e-12)
    assert all(b.constraint_error is not None and b.relative_density_error is not None
               for b in report.bond_reports)
    np.testing.assert_array_equal(rho.to_dense(),before)


def test_relaxed_mode_keeps_exact_route_when_it_fits():
    rho = _density()
    with warnings.catch_warnings(record=True) as caught:
        out,report = rho.compress_dmt(max_bond=12,constraint_mode='relaxed',return_report=True)
    assert not caught and not report.relaxed_bonds
    assert report.relative_density_error < 2e-12
    assert np.linalg.norm(out.to_dense()-rho.to_dense()) > 1e-7


@pytest.mark.parametrize('backend,dtype', [('numpy','complex128'),('numpy','complex64'),
                                         ('torch','complex128'),('jax','complex128'),('cupy','complex128')])
def test_relaxed_native_backend_dtype_trace_and_cap(backend,dtype):
    rho,_,_ = mixture(6,rank=16,dtype=dtype)
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        rho.apply_to_arrays(lambda x:torch.tensor(x))
    elif backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64',True)
        rho.apply_to_arrays(jax.numpy.asarray)
    elif backend == 'cupy':
        cupy = pytest.importorskip('cupy')
        try:
            if not cupy.cuda.runtime.getDeviceCount():
                pytest.skip('no CUDA device')
        except cupy.cuda.runtime.CUDARuntimeError:
            pytest.skip('CUDA runtime unavailable')
        rho.apply_to_arrays(cupy.asarray)
    with pytest.warns(RuntimeWarning,match='constraints were relaxed'):
        out = compress_mpo_dmt(rho,max_bond=1,regions=square_lattice_dmt_regions(3,2),
                               constraint_mode='relaxed')
    assert out.max_bond() == 1 and out[0].data.dtype == rho[0].data.dtype
    if backend != 'numpy':
        assert out[0].data.device == rho[0].data.device
    assert out.compression_report.relaxed_bonds
    assert out.compression_report.relative_density_error > 0
    np.testing.assert_allclose(np.trace(dense(out)),1.,atol=3e-6 if dtype=='complex64' else 2e-12)


def test_relaxed_torch_trace_gradient():
    torch = pytest.importorskip('torch')
    from pepsy.backends import TorchLinalgConfig

    rng = np.random.default_rng(8)
    value = rng.normal(size=(8,8)) + 1j*rng.normal(size=(8,8))
    value[0,0] += 10
    m = torch.tensor(value,dtype=torch.complex128,requires_grad=True)
    v = torch.eye(8,dtype=torch.complex128)[:,:3]
    with TorchLinalgConfig(stabilized=True).activated():
        l,r,report = _split(m,v,v,max_bond=1,cutoff=0.,variant='density',absorb='left',
                            bond=0,diagnostics=True,constraint_mode='relaxed')
        (l@r)[0,0].real.backward()
    assert report.constraints_relaxed and report.final_bond == 1
    expected = np.zeros((8,8),complex)
    expected[0,0] = 1.
    np.testing.assert_allclose(m.grad.detach().numpy(),expected,atol=2e-8)


def test_2d_fixed_small_cap_replay_records_every_gate_error():
    rho,_,_ = mixture(6,rank=24)
    regions = square_lattice_dmt_regions(3,2)
    supports = [(1,2),(0,5),(3,4),(0,3),(4,5),(1,4)]
    queue = [(gate(40+i),w) for i,w in enumerate(supports)]
    queue[1] = queue[1][0]@np.diag([1.,.8,.7,.5]), queue[1][1]
    reference, errors = dense(rho), []

    def check(candidate):
        nonlocal reference
        payload,where = queue[len(errors)]
        full = embed(payload.T,where,6)
        target = full@reference@full.conj().T
        actual = dense(candidate)
        error = max(np.linalg.norm(marginal(actual,6,r)-marginal(target,6,r)) /
                    np.linalg.norm(marginal(target,6,r)) for r in regions)
        np.testing.assert_allclose(np.trace(actual),np.trace(target),atol=2e-12)
        assert candidate.max_bond() <= 2
        errors.append(error)
        reference = actual
        return True

    opt = MpoOptimizer(rho,queue,chi=2,mode='dmt',contraction_opt='greedy')
    with pytest.warns(RuntimeWarning):
        opt.run(dmt_regions=regions,dmt_region_selection='rank_aware',
                dmt_constraint_mode='relaxed',finite_check=check,fidelity_samples=0)
    reports = opt.get_dmt_reports()  # retained without dmt_diagnostics=True
    assert len(reports) == len(queue) and any(r.relaxed_bonds for r in reports)
    assert np.trace(reference).real < .9
    np.testing.assert_allclose([r.relative_density_error for r in reports],errors,rtol=2e-9,atol=2e-12)
    out = opt.compress_dmt(constraint_mode='relaxed',regions=regions)
    assert out is opt.p and len(opt.get_dmt_reports()) == 1


def test_pauli_small_cap_gate_and_long_sequence():
    rho = _density()
    regions = square_lattice_dmt_regions(3,2)
    run = PauliDMT(rho,max_bond=1,regions=regions,constraint_mode='relaxed')
    with pytest.warns(RuntimeWarning,match='constraints were relaxed'):
        for i in range(12):
            report = run.apply_gate(gate(i), (2,3))
            assert report.final_bond == 1 and report.relative_density_error >= 0
    np.testing.assert_allclose(run.state.trace(),rho.trace(),atol=2e-12)


def test_exact_plaquette_protection_does_not_certify_future_local_dynamics():
    regions = square_lattice_dmt_regions(3,2)
    rho = PauliMPO.from_terms(6,[(1/64,'IIIIII'),(.5/64,'ZZZZZZ')])
    compressed = rho.compress_dmt(max_bond=1,regions=regions,region_selection='rank_aware',
                                   truncation='optimized',max_refinement_sweeps=0)
    original = rho.to_dense()
    for r in regions:
        np.testing.assert_allclose(marginal(compressed.to_dense(),6,r),marginal(original,6,r),atol=2e-13)
    cnot = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.],[0.,0.,1.,0.]])
    run = PauliDMT(compressed,max_bond=1,regions=regions,region_selection='rank_aware',
                   truncation='optimized',diagnostics=True)
    for i in range(5):
        report = run.apply_gate(cnot,(i,i+1))
        assert not report.constraints_relaxed
        assert report.constraint_error < 2e-12
        full = embed(cnot,(i,i+1),6)
        original = full@original@full.T
    # CNOT conjugation turns the discarded Z0...Z5 correlation into Z5.
    np.testing.assert_allclose(marginal(original,6,(5,)),np.diag([.75,.25]),atol=2e-13)
    np.testing.assert_allclose(marginal(run.state.to_dense(),6,(5,)),np.eye(2)/2,atol=2e-13)


@pytest.mark.parametrize('options', [dict(constraint_mode='automatic'),dict(constraint_weight=-1),
                                    dict(constraint_weight=float('inf')),dict(constraint_weight=True)])
def test_invalid_modes_do_not_mutate_input(options):
    rho,_,_ = mixture(4,rank=8)
    before = dense(rho)
    with pytest.raises(ValueError):
        compress_mpo_dmt(rho,max_bond=1,inplace=True,**options)
    np.testing.assert_array_equal(dense(rho),before)


def test_relative_density_defect_handles_zero_references():
    zero = [(np.zeros(4),-np.inf)]
    assert _relative_density_defect(zero,zero) == 0.
    assert np.isinf(_relative_density_defect(zero,[(np.array([1.,0.,0.,0.]),0.)]))


@pytest.mark.parametrize('pauli', [False, True])
def test_fast_region_checks_keep_strict_density_guarantee(pauli):
    rho = _density()
    regions = square_lattice_dmt_regions(3,2)
    if pauli:
        out, report = rho.compress_dmt(max_bond=20, regions=regions, check_regions=False,
                                       return_report=True, region_selection='rank_aware')
        actual = out.to_dense()
    else:
        out, report = compress_mpo_dmt(rho.to_mpo(), max_bond=20, regions=regions,
                                       check_regions=False, return_report=True,
                                       region_selection='rank_aware')
        actual = dense(out)
    assert report.relative_density_error is None
    for r in regions:
        np.testing.assert_allclose(marginal(actual,6,r), marginal(rho.to_dense(),6,r), atol=2e-12)


def test_fast_relaxed_replay_reports_unmeasured_density_error():
    rho,_,_ = mixture(6,rank=16)
    regions = square_lattice_dmt_regions(3,2)
    opt = MpoOptimizer(rho, [(gate(11),(1,2)), (gate(12),(0,5))], chi=1, mode='dmt', contraction_opt='greedy')
    with pytest.warns(RuntimeWarning,match='constraints were relaxed'):
        opt.run(dmt_regions=regions,dmt_constraint_mode='relaxed',dmt_check_regions=False)
    report = opt.get_dmt_reports()[0]
    assert report.relaxed_bonds and report.relative_density_error is None
    assert report.max_relative_density_error is None
    assert opt.p.max_bond() == 1
    assert len(opt.get_dmt_reports()) == 2
