"""Global optimizer ownership, normalization and best-iterate regressions."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import GlobalOptimizer, PepsOptimizer


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("normalize_final", [False, True])
def test_global_multibatch_preserves_backend_and_normalizes_once(monkeypatch, backend, normalize_final):
    torch = pytest.importorskip("torch")
    pytest.importorskip("nlopt")
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=311)
    state.multiply_(1 / np.linalg.norm(state.to_dense()))
    if backend == "torch":
        state.apply_to_arrays(lambda data: torch.tensor(data, dtype=torch.complex128))
    original = np.asarray(state.to_dense()).copy()
    gates = []
    for angle, where in [(.37, ((0, 0), (0, 1))), (.23, ((1, 1), (2, 1)))]:
        gate = np.diag(np.exp(-1j * angle * np.array([1., -1., -1., 1.])))
        if backend == "torch":
            gate = torch.tensor(gate)
        gates.append((gate, where))

    def unexpected(*args, **kwargs):
        pytest.fail("Delegated normalization must be disabled in the PEPS driver")

    monkeypatch.setattr(GlobalOptimizer, "_normalize_state", unexpected)
    optimizer = PepsOptimizer(
        state, gates, chi=2, mode="global", contraction_opt="greedy",
        normalize_chi=(32, 48), evaluation_chi=(32, 48),
        global_optimize_kwargs={"n": 8},
    )
    output = optimizer.run(
        k_2q_batch=1, infidelity_tol=0., normalize_final=normalize_final,
        accept_if_improved=False, progress=False,
    )
    assert len(optimizer.normalizations) == (5 if normalize_final else 3)
    records = optimizer.get_step_records()
    assert len(records) == 2 and all(record["optimized"] for record in records)
    assert all(record["final_infidelity"] < record["pre_infidelity"] for record in records)
    assert output.max_bond() <= 2
    np.testing.assert_array_equal(np.asarray(state.to_dense()), original)
    for tensor in output:
        if backend == "torch":
            assert isinstance(tensor.data, torch.Tensor)
            assert tensor.data.dtype == torch.complex128
            assert tensor.data.device == state.tensor_map[next(iter(state.tensor_map))].data.device
        else:
            assert tensor.data.dtype == np.complex128
            assert isinstance(tensor.data, np.ndarray)
    if normalize_final:
        vector = np.asarray(output.to_dense()).reshape(-1)
        assert np.vdot(vector, vector).real == pytest.approx(1., abs=1e-10)


@pytest.mark.parametrize("entrypoint", ["optimize", "optimize_nlopt"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_global_entrypoints_restore_torch_dtype(entrypoint, dtype):
    torch = pytest.importorskip("torch")
    pytest.importorskip("nlopt")
    states = []
    for seed in (43, 47):
        state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype=dtype, seed=seed)
        state.multiply_(1 / np.linalg.norm(state.to_dense()))
        state.apply_to_arrays(lambda data: torch.tensor(data, dtype=getattr(torch, dtype)))
        states.append(state)
    states[1].mangle_inner_("_target")
    optimizer = GlobalOptimizer(
        *states, loss_kwargs={"mode": "exact", "contraction_opt": "greedy"},
        normalize_kwargs={"mode": "exact", "contraction_opt": "greedy"},
    )
    before = float(optimizer.loss())
    output = getattr(optimizer, entrypoint)(
        n=3, optimizer="adam" if entrypoint == "optimize" else "LD_LBFGS", progbar=False,
    )
    assert float(optimizer.loss()) < before
    assert all(tensor.data.dtype == getattr(torch, dtype) for tensor in output)
    assert all(isinstance(tensor.data, torch.Tensor) for tensor in output)


@pytest.mark.parametrize("fail", [False, True])
def test_nlopt_restores_best_evaluated_vector_and_preserves_history(monkeypatch, fail):
    nlopt = pytest.importorskip("nlopt")
    pytest.importorskip("torch")
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=43)
    state.multiply_(1 / np.linalg.norm(state.to_dense()))
    target = state.copy()
    target.mangle_inner_("_target")
    optimizer = GlobalOptimizer(
        state, target, normalize_kwargs={},
        loss_kwargs={"mode": "exact", "contraction_opt": "greedy"},
    )

    def trials(tnopt, **kwargs):
        for perturb in (False, True):
            if perturb:
                tnopt.vectorizer.vector[:] += np.random.default_rng(71).normal(size=tnopt.d)
            value = tnopt.handler.value(tnopt.vectorizer.unpack())
            tnopt.losses.append(float(value))
        if fail:
            raise nlopt.RoundoffLimited("controlled worse trial")
        return tnopt.get_tn_opt()

    monkeypatch.setattr(qtn.TNOptimizer, "optimize_nlopt", trials)
    if fail:
        with pytest.warns(RuntimeWarning, match="stopped early"):
            output = optimizer.optimize_nlopt(n=2, progbar=False)
    else:
        output = optimizer.optimize_nlopt(n=2, progbar=False)
    assert optimizer.losses[-1] > .9
    assert float(optimizer.loss(state=output)) < 1e-12
    assert optimizer.final_loss == pytest.approx(optimizer.losses[0], abs=1e-12)
    assert optimizer.optimization_info["best_restored"]
    assert optimizer.optimization_info["stopped_early"] is fail


@pytest.mark.parametrize("invalid_trial", [False, True])
def test_nlopt_without_valid_iterate_retains_input(monkeypatch, invalid_trial):
    nlopt = pytest.importorskip("nlopt")
    pytest.importorskip("torch")
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=43)
    original = state.to_dense().copy()
    optimizer = GlobalOptimizer(
        state, state.copy(), normalize_kwargs={},
        loss_kwargs={"mode": "exact", "contraction_opt": "greedy"},
    )

    def fail(tnopt, **kwargs):
        tnopt.vectorizer.vector[:] = np.nan
        if invalid_trial:
            # A finite score cannot validate nonfinite parameters.
            tnopt.losses.append(0.)
        raise nlopt.RoundoffLimited("no valid iterate")

    monkeypatch.setattr(qtn.TNOptimizer, "optimize_nlopt", fail)
    with pytest.warns(RuntimeWarning, match="stopped early"):
        output = optimizer.optimize_nlopt(n=2, progbar=False)
    np.testing.assert_array_equal(output.to_dense(), original)
    assert optimizer.optimization_info["success"] is False
    assert optimizer.final_loss is None


def test_standalone_normalize_uses_its_own_options(monkeypatch):
    state = qtn.PEPS.rand(2, 2, bond_dim=2, seed=8)
    optimizer = GlobalOptimizer(
        state, norm_kwargs={"chi": 32, "mode": "exact"},
        normalize_kwargs={"chi": 7, "mode": "mps"},
    )
    captured = {}

    def normalize(state, **kwargs):
        captured.update(kwargs)
        return state

    monkeypatch.setattr(optimizer, "_normalize_state", normalize)
    assert optimizer.normalize(cutoff=1e-8) is state
    assert captured == {"chi": 7, "mode": "mps", "cutoff": 1e-8}
    assert optimizer.norm_kwargs == {"chi": 32, "mode": "exact"}


@pytest.mark.parametrize("evaluate", [False, True])
def test_driver_reports_recovered_loss_or_failed_cleanup(monkeypatch, evaluate):
    nlopt = pytest.importorskip("nlopt")
    pytest.importorskip("torch")
    state = qtn.PEPS.rand(2, 2, bond_dim=2, dtype="complex128", seed=43)
    gate = np.diag(np.exp(-.2j * np.array([1., -1., -1., 1.])))
    optimizer = PepsOptimizer(
        state, [(gate, ((0, 0), (0, 1)))], chi=1,
        mode="global", contraction_opt="greedy",
    )

    def fail(tnopt, **kwargs):
        if evaluate:
            tnopt.losses.append(float(tnopt.handler.value(tnopt.vectorizer.unpack())))
            tnopt.vectorizer.vector[:] += np.random.default_rng(71).normal(size=tnopt.d)
            tnopt.losses.append(float(tnopt.handler.value(tnopt.vectorizer.unpack())))
        raise nlopt.RoundoffLimited("driver recovery check")

    monkeypatch.setattr(qtn.TNOptimizer, "optimize_nlopt", fail)
    with pytest.warns(RuntimeWarning, match="stopped early"):
        optimizer.run(
            infidelity_tol=0., accept_if_improved=False,
            measure_final_infidelity=False, progress=False,
        )
    record = optimizer.get_step_records()[0]
    info = record["optimizer_result"]
    assert info["stopped_early"]
    assert info["termination_reason"] == "nlopt_error"
    assert not info["fallback_used"]
    if evaluate:
        assert record["optimizer_infidelity"] == pytest.approx(info["returned_loss"])
        assert info["returned_loss"] == pytest.approx(info["loss_initial"])
        assert info["returned_loss"] < info["loss_final"]
        assert record["final_infidelity"] == pytest.approx(record["pre_infidelity"], abs=1e-10)
    else:
        assert record["reason"] == "optimizer_failed"
        assert not record["optimized"]
        assert record["optimizer_infidelity"] is None
        assert record["final_infidelity"] == record["pre_infidelity"]


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_global_native_fermionic_output_remains_torch(symmetry):
    torch = pytest.importorskip("torch")
    pytest.importorskip("symmray")
    pytest.importorskip("nlopt")
    from pepsy.backends import backend_torch, TorchLinalgConfig
    from pepsy.tensors import SymPEPS, site_charge_alternating

    TorchLinalgConfig(mode="complex", stabilized=True, quimb_split_drivers=True).register()
    options = dict(
        symmetry=symmetry, bond_dim=2, phys_dim=4 if symmetry == "U1U1" else 2,
        fermionic=True, dtype="complex128", contraction_opt="greedy",
        to_backend=backend_torch(dtype=torch.complex128),
    )
    if symmetry == "U1U1":
        options["site_charge"] = site_charge_alternating((1, 0), (0, 1))
    state = SymPEPS.random(2, 2, seed=7, **options).peps
    target = SymPEPS.random(2, 2, seed=107, **options).peps
    target.mangle_inner_("_target")
    target_norm = (target.H & target).contract(all, optimize="greedy")
    optimizer = GlobalOptimizer(
        state, target,
        loss_kwargs={"target_norm": target_norm, "mode": "mps", "contraction_opt": "greedy"},
        normalize_kwargs={"mode": "mps", "contraction_opt": "greedy"},
    )
    before = float(optimizer.loss())
    output = optimizer.optimize_nlopt(n=5, optimizer="LD_LBFGS", progbar=False)
    assert float(optimizer.loss()) < before
    for tid, tensor in output.tensor_map.items():
        assert tensor.data.indices == state.tensor_map[tid].data.indices
        assert tensor.data.blocks.keys() == state.tensor_map[tid].data.blocks.keys()
        for block in tensor.data.blocks.values():
            assert isinstance(block, torch.Tensor)
            assert block.dtype == torch.complex128
            assert torch.isfinite(block).all()
