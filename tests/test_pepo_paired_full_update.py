"""Both physical legs share one exact target and one full-update solve."""
import numpy as np
import pytest
from pepsy.optimizers import PepsOptimizer
from test_pepo_full_update import fixture,unitary,dense,action,fidelity

pytestmark=[pytest.mark.optional,pytest.mark.peps]


@pytest.mark.parametrize('policy',['input','column','smart'])
@pytest.mark.parametrize('reverse',[False,True])
def test_paired_exact_operator_scale_and_original_steps(policy,reverse):
    import torch
    state=fixture();state.exponent=.3
    pair=((0,0),(0,1)); other=tuple(reversed(pair)) if reverse else pair
    a=(1.2+.3j)*unitary(4,14);b=(.8-.2j)*unitary(4,29)
    entries=[(torch.tensor(a),pair,'upper'),(torch.tensor(b),other,'lower')]
    expected=action(state,action(state,dense(state),a,pair,'upper'),b,other,'lower')
    opt=PepsOptimizer(state,entries,chi=32,mode='full-update',gate_order=policy,
        normalize_initial=False,boundary_convergence=False,contraction_opt='greedy')
    out=opt.run(fuse_pepo_legs=True,normalize_final=False)
    np.testing.assert_allclose(dense(out),expected,atol=2e-10,rtol=2e-10)
    records=opt.get_step_records();assert len(records)==1
    assert records[0]['joint_pepo_legs'] and records[0]['operator_gate_count']==2
    assert set(records[0]['original_steps'])=={1,2}


@pytest.mark.parametrize('fit_mode',['direct','dmrg'])
def test_paired_fit_one_environment_and_dense_fidelity(fit_mode,monkeypatch):
    import torch
    import pepsy.bp.reduced_update as reduced
    state=fixture(shape=(2,3));pair=((0,0),(0,1))
    a,b=unitary(4,13),unitary(4,17)
    expected=action(state,action(state,dense(state),a,pair,'upper'),b,pair,'lower')
    opt=PepsOptimizer(state,[(torch.tensor(a),pair,'upper'),(torch.tensor(b),pair,'lower')],
        chi=2,mode='full-update',boundary_chi=64,fit_mode=fit_mode,
        contraction_opt='greedy',boundary_convergence=False,
        full_update_kwargs={'solver':'quimb','max_iterations':4,'skip_exact':False})
    calls=[];environment=opt._full_update_environment
    def track(*args,**kwargs):calls.append(1);return environment(*args,**kwargs)
    def forbidden(*args,**kwargs):pytest.fail('Native paired solve expanded the full physical metric')
    monkeypatch.setattr(opt,'_full_update_environment',track)
    monkeypatch.setattr(reduced,'_dense_problem_metric',forbidden)
    out=opt.run(fuse_pepo_legs=True,normalize_final=False)
    assert len(calls)==1
    report=opt.get_step_records()[0]['optimizer_result']
    assert report['which']=='both' and report['solver']=='quimb'
    assert report['fidelity']==pytest.approx(fidelity(dense(out),expected),abs=3e-9)


def test_pairing_preserves_same_leg_dependencies_and_does_not_cross_barriers():
    import torch
    pair=((0,0),(0,1));state=fixture();g=torch.tensor(unitary(4))
    for entries in ([(g,pair,'upper'),(g,pair,'upper')],
                    [(g,pair,'upper'),(torch.eye(2,dtype=torch.complex128),(0,0),'lower'),(g,pair,'lower')]):
        opt=PepsOptimizer(state,entries,chi=32,mode='full-update',gate_order='input',
                          contraction_opt='greedy',boundary_convergence=False)
        opt.run(fuse_pepo_legs=True)
        assert not any(r['joint_pepo_legs'] for r in opt.get_step_records())


def test_paired_unequal_physical_legs_and_reversed_sites():
    import torch
    state=fixture();rng=np.random.default_rng(33)
    for site in state.gen_site_coos():
        t=state[site];shape=[3 if i==state.upper_ind(*site) else s for i,s in zip(t.inds,t.shape)]
        t.modify(data=torch.tensor(rng.normal(size=shape)+1j*rng.normal(size=shape)))
    a,b=unitary(9),unitary(4,27);pair=((0,0),(0,1));rev=tuple(reversed(pair))
    expected=action(state,action(state,dense(state),a,pair,'upper'),b,rev,'lower')
    opt=PepsOptimizer(state,[(torch.tensor(a),pair,'upper'),(torch.tensor(b),rev,'lower')],
        chi=32,mode='full-update',normalize_initial=False,contraction_opt='greedy',boundary_convergence=False)
    np.testing.assert_allclose(dense(opt.run(fuse_pepo_legs=True)),expected,atol=2e-9,rtol=2e-9)


@pytest.mark.parametrize('options,run_options',[
    ({},{'warmstart_method':'simple-update'}),
    ({'full_update_kwargs':{'refine_sweeps':1}},{}),
    ({'full_update_kwargs':{'tensor_mode':'full'}},{})])
def test_joint_modes_share_fitting_and_refinement(options,run_options):
    import torch
    pair=((0,0),(0,1));g=torch.tensor(unitary(4))
    state=fixture()
    target=action(state,action(state,dense(state),g.numpy(),pair,'upper'),g.numpy(),pair,'lower')
    opt=PepsOptimizer(state,[(g,pair,'upper'),(g,pair,'lower')],chi=2,
        mode='full-update',contraction_opt='greedy',**options)
    out=opt.run(fuse_pepo_legs=True,measure_infidelity=True,measure_final_infidelity=True,**run_options)
    assert len(opt.get_step_records())==1
    assert opt.get_step_records()[0]['post_infidelity']==pytest.approx(1-fidelity(dense(out),target),abs=3e-9)


@pytest.mark.parametrize('fuse',[False,True])
def test_driver_avoids_unused_target_network(fuse,monkeypatch):
    import torch
    from pepsy.optimizers.peps._full_update import ReducedPair
    prepare=ReducedPair.initial_states;observed=[]
    def track(pair,*args,**kwargs):
        result=prepare(pair,*args,**kwargs)
        observed.append((kwargs.get('build_target',True),result[1] is None))
        return result
    monkeypatch.setattr(ReducedPair,'initial_states',track)
    pair=((0,0),(0,1));g=torch.tensor(unitary(4))
    opt=PepsOptimizer(fixture(),[(g,pair,'upper'),(g,pair,'lower')],chi=2,
        mode='full-update',boundary_chi=16,contraction_opt='greedy',
        full_update_kwargs={'solver':'quimb','max_iterations':2})
    opt.run(fuse_pepo_legs=fuse)
    assert observed and all(not built and missing for built,missing in observed)
    assert all(not r['target_network_built'] for r in opt.get_step_records())
