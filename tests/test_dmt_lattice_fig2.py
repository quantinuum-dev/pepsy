"""Energy transport normalization, storage mapping and independent oracles."""

import numpy as np
import pytest
from scipy.linalg import expm

from examples import dmt_lattice_fig2 as bench


def test_initial_packet_normalization_and_interleaving():
    pytest.importorskip('torch')
    from pepsy.optimizers import PauliDMT

    m=bench.problem(5,3)
    assert all(b-a==3 for a,b in m['edges'])
    assert (0,3,6) in m['regions'] and (0,1,3) in m['regions']
    run=PauliDMT(bench.initial_operator(m),max_bond=64,variant='operator',trace_guard=False)
    row=bench.measure(run,m)
    norm=bench.GX**2+bench.GZ**2+1
    expected=np.array([0,.25,norm-.5,.25,0])/norm
    np.testing.assert_allclose(row['correlator'],expected,atol=1e-14)
    assert row['total_weight']==pytest.approx(1.,abs=1e-14)
    assert row['msd']==pytest.approx(.5/norm,abs=1e-14)
    assert row['identity_coefficient']<1e-14


@pytest.mark.parametrize('dtype',['float64','complex128'])
def test_interleaved_operator_matches_independent_exact_circuit(dtype):
    pytest.importorskip('torch')
    n=6;eye=np.eye(2);x=np.array([[0,1],[1,0]]);z=np.diag([1,-1])
    def word(choices):
        a=np.ones((1,1))
        for j in range(n):a=np.kron(a,choices.get(j,eye))
        return a
    fields=sum(bench.GX*word({j:x})+bench.GZ*word({j:z}) for j in range(n))
    edges=[(0,2),(2,4),(1,3),(3,5)]
    zz=sum(word({a:z,b:z}) for a,b in edges)
    packet=sum(bench.GX*word({j:x})+bench.GZ*word({j:z}) for j in (2,3))+.5*zz
    u=expm(-.05j*fields)@expm(-.1j*zz)@expm(-.05j*fields)
    u=u@u
    actual=u.conj().T@packet@u
    m=bench.problem(3,2);cx=[]
    for column in m['columns']:
        hx=sum(c*word({s:{'X':x,'Z':z}[p] for s,p in zip(sites,labels)})
               for c,sites,labels in column)
        cx.append(np.trace(hx@actual).real/2**n/m['normalization'])
    result=bench.run_benchmark(length=3,width=2,chi=64,dt=.1,time=.2,dtype=dtype,
                               check_regions=True,max_seconds=60)
    assert result['status']=='completed' and not result['preservation_errors']
    np.testing.assert_allclose(result['samples'][-1]['correlator'],cx,atol=2e-13)


def test_timestep_convergence_against_exact_continuous_operator():
    pytest.importorskip('torch')
    expected=bench.exact_reference(5,[1.])[0]
    errors=[]
    for dt in (.2,.1):
        result=bench.run_benchmark(length=5,width=1,chi=16,dt=dt,time=1.,max_seconds=60)
        assert result['status']=='completed' and not result['preservation_errors']
        actual=result['samples'][-1]
        errors.append(np.linalg.norm(np.asarray(actual['correlator'])-expected['correlator']))
        assert actual['total_weight']==pytest.approx(1.,abs=.02)
    assert 3.5<errors[0]/errors[1]<4.5


def test_independent_sparse_reference_actions_match_dense_spin_construction():
    n=5;eye=np.eye(2);x=np.array([[0,1],[1,0]]);z=np.diag([1,-1])
    def word(choices):
        a=np.ones((1,1))
        for j in range(n):a=np.kron(a,choices.get(j,eye))
        return a
    h=sum(bench.GX*word({j:x})+bench.GZ*word({j:z}) for j in range(n))
    h+=sum(word({j:z,j+1:z}) for j in range(n-1))
    np.testing.assert_allclose(bench.sparse_chain(n).toarray(),h,atol=1e-15)
    rng=np.random.default_rng(18);v=rng.normal(size=2**n)+1j*rng.normal(size=2**n)
    for j in range(n):
        local=bench.GX*word({j:x})+bench.GZ*word({j:z})
        for k in (j-1,j+1):
            if 0<=k<n:local+=.5*word({j:z,k:z})
        np.testing.assert_allclose(bench.energy_action(v,j,n),local@v,atol=2e-15)


def test_published_vector_fixture_calibration_and_duplicate_reference():
    import json
    from pathlib import Path

    data=json.loads((Path(__file__).parents[1]/'examples/data/lattice_dmt_fig2.json').read_text())
    refs=[c for c in data['curves'] if c['reference']]
    assert len(refs)==2
    np.testing.assert_allclose(refs[0]['times'],refs[1]['times'],atol=1e-12)
    np.testing.assert_allclose(refs[0]['msd'],refs[1]['msd'],atol=1e-12)
    assert refs[0]['times'][0]==pytest.approx(0.) and refs[0]['times'][-1]==pytest.approx(20.)
    expected=.5/(bench.GX**2+bench.GZ**2+1.)
    for row in data['curves']:
        if not row['reference']:
            np.testing.assert_allclose(row['times'],np.arange(21),atol=1e-12)
            assert row['msd'][0]==pytest.approx(expected,abs=2e-4)


def test_traceless_lossy_interval_keeps_all_physical_three_site_panels():
    torch=pytest.importorskip('torch')
    from pepsy.backends import TorchLinalgConfig
    from pepsy.operators import PauliMPO
    from pepsy.optimizers import PauliDMT
    from examples import dmt_2d_dynamics as exact

    rng=np.random.default_rng(183)
    bonds=[1,4,16,48,16,4,1]
    op=PauliMPO.from_pauli_cores([.2*rng.normal(size=(a,b,4)) for a,b in zip(bonds,bonds[1:])])
    op=op-PauliMPO.identity(6,coefficient=op.trace()/64)
    op=PauliMPO.from_pauli_cores([torch.as_tensor(a.real,dtype=torch.float64) for a in op.to_pauli_cores()])
    source=op.to_dense().numpy()
    m=bench.problem(3,2)
    with TorchLinalgConfig(cpu_svd='scipy_gesdd',cpu_qr='scipy_pivoted',quimb_split_drivers=True).activated():
        run=PauliDMT(op,max_bond=32,variant='operator',trace_guard=False,
                     regions=m['regions'],region_selection='min_cover',check_regions=True,diagnostics=True)
        transfer=[torch.tensor(a) for a in bench.zz_transfer(4,-.12)]
        report=run.apply_transfer_mpo(1,transfer)
        actual=run.state.to_dense().numpy()
    z=np.diag([1,-1]);u=expm(.12j*np.kron(z,z))
    target=exact.apply_density(source,u,(1,4),6)
    assert not report.preservation_errors and max(report.final_bond_dimensions)<=32
    assert max(b.discarded_weight for b in report.bond_reports)>1e-8
    assert np.linalg.norm(actual-target)>1e-6
    for region in m['regions']:
        np.testing.assert_allclose(exact.ref.partial_trace(actual,region,6),
                                   exact.ref.partial_trace(target,region,6),atol=2e-12)
    assert abs(np.trace(actual))<2e-12
