"""Centered loop-union patches and their SU-closed reduced environments."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.bp import (
    ContractionPlanCache, ReducedLoopClusterCache, apply_reduced_loop_cluster_gate,
    loop_cluster_reduced_update_problem, prepare_reduced_bond_pair,
    su_cluster_reduced_update_problem,
)
from pepsy.operators import gate_loop_cluster
from pepsy.tensors import build_optimizer


def test_contraction_cache_preserves_operand_order():
    rng = np.random.default_rng(998)
    tensors = [qtn.Tensor(rng.normal(size=(2, 2)), inds=inds)
               for inds in (("a", "b"), ("b", "c"), ("c", "a"))]
    cache = ContractionPlanCache()
    for order in ((0, 1, 2), (1, 0, 2), (1, 0, 2)):
        network = qtn.TensorNetwork([tensors[i] for i in order])
        tree, _, _ = cache.plan_for(network, output_inds=(), optimize="greedy")
        expected = network.contract(output_inds=(), optimize="greedy")
        assert network.contract(output_inds=(), optimize=tree) == pytest.approx(expected)
    assert cache.hits == 1


@pytest.mark.parametrize("where,expected", [
    (((3, 3), (3, 4)), {(i, j) for i in (2, 3, 4) for j in (3, 4)}),
    (((3, 3), (4, 3)), {(i, j) for i in (3, 4) for j in (2, 3, 4)}),
    (((0, 2), (0, 3)), {(i, j) for i in (0, 1) for j in (2, 3)}),
])
def test_loopunion_selects_both_plaquettes_and_clips_at_edge(where, expected):
    tn = qtn.PEPS.rand(7, 7, bond_dim=1, seed=1)
    pair = prepare_reduced_bond_pair(tn, where=where, run_bp=False)
    tids = {pair.left_tid, pair.right_tid, *pair._cluster_tids(4, mode="loopunion", fillin=True)}
    sites = {site for site in tn.gen_site_coos() if set(tn.tag_map[tn.site_tag(site)]) & tids}
    assert sites == expected


def test_graph_fillin_and_loopunion_are_separate_choices():
    tn = qtn.PEPS.rand(7, 7, bond_dim=1, seed=1)
    pair = prepare_reduced_bond_pair(tn, where=((3, 3), (3, 4)), run_bp=False)
    assert len(pair._cluster_tids(1)) + 2 == 8
    assert len(pair._cluster_tids(1, fillin=True)) + 2 == 12
    assert len(pair._cluster_tids(4, mode="loopunion", fillin=True)) + 2 == 6
    with pytest.raises(ValueError, match="cluster_mode"):
        pair._cluster_tids(1, mode="rectangle")
    with pytest.raises(ValueError, match="fillin"):
        pair._cluster_tids(1, fillin=-1)


@pytest.mark.parametrize("kind", ["peps", "pepo"])
def test_patch_metric_matches_independent_weighted_cluster_and_reuses_paths(kind, monkeypatch):
    factory = qtn.PEPS if kind == "peps" else qtn.PEPO
    tn = factory.rand(4, 3, bond_dim=2, dtype="complex128", seed=83)
    gauges = {index: np.array([0.6, 1.2]) for index in tn.inner_inds()}
    where = ((1, 0), (1, 1))
    optimize = build_optimizer(parallel=False, progbar=False, max_repeats=4)
    plans, loops = ContractionPlanCache(), ReducedLoopClusterCache()
    original_do = ar.do

    def without_environment_spectrum(fn, *args, **kwargs):
        if fn in {"linalg.eigh", "linalg.eigvalsh"}:
            raise AssertionError("Hermitian-only environment must not compute eigenvalues")
        return original_do(fn, *args, **kwargs)

    monkeypatch.setattr(ar, "do", without_environment_spectrum)
    norms = []
    for scale in (1.0, 1.1):
        if scale != 1.0:
            tn[tn.site_tag(0, 0)].modify(data=tn[tn.site_tag(0, 0)].data * scale)
        pair = prepare_reduced_bond_pair(tn, gauges, where=where)
        problem = loop_cluster_reduced_update_problem(
            pair, None, cluster_mode="loopunion", base_radius=4, fillin=True,
            psd_project=False, optimize=optimize, plan_cache=plans, loop_cache=loops,
        )
        assert len(problem.terms) == 1
        assert len(problem.terms[0].region_tids) == 6
        assert problem.cluster_mode == "loopunion"
        assert not problem.psd_projected and problem.raw_min_eigenvalue is None
        assert problem.clipped_eigenvalues == 0
        assert problem.contraction_cost["flops_log10"] > 0
        # Independent single-layer cluster: SU insertion gives full lambda on
        # cut legs and half weights internally. Sum its squared amplitudes over
        # physical AND cut legs, without using QR/LQ or the environment builder.
        patch = tn.select(tuple(tn.site_tag(i, j) for i in range(3) for j in range(2)),
                          which="any", virtual=False)
        patch.gauge_simple_insert(gauges)
        amplitudes = patch.contract(output_inds=patch.outer_inds(), optimize="greedy").data
        expected = float(np.vdot(amplitudes, amplitudes).real)
        assert problem.target_norm == pytest.approx(expected, rel=2e-11)
        # Compare the quadratic error for an independent candidate with the
        # actual weighted single-layer cluster amplitudes, not another builder.
        rng = np.random.default_rng(132)
        left = pair.r_left.data + .1 * (rng.normal(size=pair.r_left.shape)
                                      + 1j * rng.normal(size=pair.r_left.shape))
        right = pair.l_right.data + .1 * (rng.normal(size=pair.l_right.shape)
                                        + 1j * rng.normal(size=pair.l_right.shape))
        trial = pair.reconstruct_tn(left, right).select(
            tuple(tn.site_tag(i, j) for i in range(3) for j in range(2)),
            which="any", virtual=False,
        )
        for index in trial.outer_inds():
            if index in gauges:
                tid = next(iter(trial.ind_map[index]))
                trial.tensor_map[tid].multiply_index_diagonal_(
                    index, gauges[index] ** .5,
                )
        candidate = trial.contract(output_inds=patch.outer_inds(), optimize="greedy").data
        delta = candidate - amplitudes
        expected_error = float(np.vdot(delta, delta).real)
        assert problem.cost(np.einsum("aps,sqb->apqb", left, right)) == pytest.approx(
            expected_error, rel=3e-11,
        )
        norms.append(problem.target_norm)
        direct = su_cluster_reduced_update_problem(
            pair, None, radius=4, cluster_mode="loopunion", fillin=True,
            psd_project=False, optimize="greedy",
        )
        np.testing.assert_allclose(direct.environment.data, problem.environment.data, rtol=2e-11, atol=1e-11)
        assert direct.raw_min_eigenvalue is None
        d = pair.theta_shape[0] * pair.theta_shape[3]
        matrix = problem.environment.data.transpose(2, 3, 0, 1).reshape(d, d)
        np.testing.assert_allclose(matrix, matrix.conj().T, rtol=2e-11, atol=1e-11)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-11 * np.linalg.norm(matrix)
    assert norms[1] == pytest.approx(norms[0] * 1.1**2, rel=2e-11)
    assert plans.hits >= 1


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_pepo_gate_stream_preserves_spectator_legs_and_matches_dense(side):
    tn = qtn.PEPO.rand(3, 2, bond_dim=1, dtype="complex128", seed=89)
    gauges = {index: np.ones(1) for index in tn.inner_inds()}
    where = ((1, 0), (1, 1))
    rng = np.random.default_rng(31)
    gate, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    # Contract an independently gate-decorated operator network. A nonsymmetric
    # gate detects an accidental lower-leg transpose or conjugation.
    inds = tuple(getattr(tn, side + "_ind")(*site) for site in where)
    one_site = np.array([[1., 1j], [1j, 1.]]) / np.sqrt(2)
    target = tn.gate_inds(one_site, (tn.upper_ind(0, 0),), contract=False)
    target = target.gate_inds(gate, inds, contract=False)
    core, results = gate_loop_cluster(
        tn, ((one_site, ((0, 0),), "upper"), (gate, where, side)), gauges=gauges,
        cluster_mode="loopunion", base_radius=4, fillin=True, max_bond=4,
        optimize="greedy", als_opts={"max_iterations": 2},
        inplace=False, return_results=True,
    )
    rebuilt = core.copy()
    rebuilt.gauge_simple_insert(gauges)
    np.testing.assert_allclose(rebuilt.to_dense(), target.to_dense(), rtol=1e-10, atol=1e-10)
    assert len(results[0].problem.terms[0].region_tids) == 6
    assert results[0].problem.which == side
    for site in where:
        assert tn.lower_ind(*site) in rebuilt[rebuilt.site_tag(site)].inds
        assert tn.upper_ind(*site) in rebuilt[rebuilt.site_tag(site)].inds


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_patch_gate_matches_cpu_and_uses_native_dense_solve(device, monkeypatch):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    tn = qtn.PEPO.rand(3, 2, bond_dim=2, dtype="complex128", seed=91)
    where = ((1, 0), (1, 1))
    gate_numpy = np.diag(np.exp(0.1j * np.array([1., -1., -1., 1.])))
    reference = apply_reduced_loop_cluster_gate(
        tn, {index: np.array([0.8, 1.1]) for index in tn.inner_inds()}, gate_numpy,
        which="upper", where=where, max_bond=2,
        cluster_mode="loopunion", base_radius=4, fillin=True,
        optimize="greedy", als_opts={"max_iterations": 2, "solver": "quimb"},
    )
    for tensor in tn.tensors:
        tensor.modify(data=torch.as_tensor(tensor.data, device=device))
    gauges = {index: torch.tensor([0.8, 1.1], device=device, dtype=torch.float64) for index in tn.inner_inds()}
    gate = torch.as_tensor(gate_numpy, device=device)
    original = qtn.tensor_network_fit_als
    seen = []

    def checked(*args, **kwargs):
        seen.append(kwargs["dense_solve"])
        return original(*args, **kwargs)

    monkeypatch.setattr(qtn, "tensor_network_fit_als", checked)
    result = apply_reduced_loop_cluster_gate(
        tn, gauges, gate, which="upper", where=where, max_bond=2,
        cluster_mode="loopunion", base_radius=4, fillin=True,
        optimize="greedy", als_opts={"max_iterations": 2, "solver": "quimb"},
    )
    assert seen and all(value is True for value in seen)
    assert all(t.data.device.type == device for t in result.core.tensors)
    assert np.isfinite(result.solution.costs[-1])
    assert result.solution.costs[-1] <= result.solution.costs[0] + 1e-8
    rebuilt = result.core.copy()
    rebuilt.gauge_simple_insert(result.gauges)
    np.testing.assert_allclose(
        ar.to_numpy(rebuilt.to_dense()), ar.to_numpy(result.physical_tn.to_dense()),
        rtol=2e-10, atol=2e-10,
    )
    np.testing.assert_allclose(
        ar.to_numpy(result.physical_tn.to_dense()), reference.physical_tn.to_dense(),
        rtol=2e-7, atol=2e-7,
    )


def test_physical_input_two_gate_stream_switches_to_su_core():
    tn = qtn.PEPS.rand(3, 2, bond_dim=1, dtype="complex128", seed=97)
    where = ((1, 0), (1, 1))
    first = np.diag(np.exp(0.1j * np.array([1., -1., -1., 1.])))
    second = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2))
    expected = tn.gate(first, where=where, contract=False).gate(second, where=where, contract=False)
    gauges = {}
    core = gate_loop_cluster(
        tn, ((first, where), (second, where)), gauges=gauges, input_mode="physical",
        cluster_mode="loopunion", base_radius=4, fillin=True,
        max_bond=2, optimize="greedy", als_opts={"max_iterations": 2}, inplace=False,
    )
    rebuilt = core.copy()
    rebuilt.gauge_simple_insert(gauges)
    np.testing.assert_allclose(rebuilt.to_dense(), expected.to_dense(), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('kind', ['peps', 'pepo'])
def test_gate_loop_cluster_stream_keeps_torch_tensors_and_gauges_on_device(device, kind):
    torch = pytest.importorskip('torch')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    factory = qtn.PEPS if kind == 'peps' else qtn.PEPO
    tn = factory.rand(3, 2, bond_dim=2, dtype='complex128', seed=103)
    tn.apply_to_arrays(lambda x: torch.as_tensor(x, device=device))
    gauges = {ix:torch.ones(tn.ind_size(ix), dtype=torch.float64, device=device) for ix in tn.inner_inds()}
    one = torch.diag(torch.exp(0.13j*torch.tensor([1.,-1.], dtype=torch.float64, device=device)))
    two = torch.diag(torch.exp(0.07j*torch.tensor([1.,-1.,-1.,1.], dtype=torch.float64, device=device)))
    entries = [(one, ((0,0),)), (two, ((1,0),(1,1)))]
    if kind == 'pepo':
        entries = [(g,where,'upper') for g,where in entries]
    out, results = gate_loop_cluster(tn, entries, gauges=gauges,
        max_bond=2, cluster_mode='loopunion', base_radius=4, fillin=True,
        optimize='greedy', als_opts={'max_iterations':2}, inplace=False, return_results=True)
    assert all(isinstance(t.data,torch.Tensor) and t.data.device.type==device for t in out.tensors)
    assert all(isinstance(g,torch.Tensor) and g.device.type==device for g in gauges.values())
    assert results[0].problem.target.device.type==device
    rebuilt=out.copy()
    rebuilt.gauge_simple_insert(gauges)
    np.testing.assert_allclose(ar.to_numpy(rebuilt.to_dense()),
        ar.to_numpy(results[0].physical_tn.to_dense()),rtol=2e-10,atol=2e-10)
