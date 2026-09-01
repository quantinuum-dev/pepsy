"""Tests for the opt-in fixed-shape Torch VMC execution path."""

import pytest


@pytest.mark.smoke
def test_export_compile_matches_eager_and_keeps_parameter_gradients():
    """Export/vmap/compile must preserve values, logs, and derivatives."""
    torch = pytest.importorskip("torch")
    qtn = pytest.importorskip("quimb.tensor")

    from pepsy.vmc.torch import TorchPEPSAmplitude

    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=401,
        dtype="float64",
    )
    model = TorchPEPSAmplitude(
        peps,
        contraction="exact",
        dtype=torch.float64,
        amplitude_batching="vmap",
    )
    configs = torch.tensor(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]],
        dtype=torch.long,
    )

    eager = model.forward(configs, params=list(model.params))
    eager_log = model.forward_log(configs, params=list(model.params))
    model.export_and_compile(configs, backend="eager")

    compiled = model.forward(configs)
    assert model.last_amplitude_batching == "export-vmap-compile"
    assert torch.allclose(compiled, eager)

    compiled_log = model.forward_log(configs)
    assert model.last_amplitude_batching == "export-vmap-compile-log"
    assert torch.allclose(compiled_log[0], eager_log[0])
    assert torch.allclose(compiled_log[1], eager_log[1])

    model.zero_grad()
    compiled.sum().backward()
    compiled_grads = [param.grad.detach().clone() for param in model.params]
    model.zero_grad()
    model.forward(configs, params=list(model.params)).sum().backward()
    for param, compiled_grad in zip(model.params, compiled_grads):
        assert torch.allclose(param.grad, compiled_grad)

    # Parameters are explicit exported-graph inputs, so an optimizer update
    # changes the compiled result without requiring another export.
    with torch.no_grad():
        model.params[0].add_(0.01)
    eager_after = model.forward(configs, params=list(model.params))
    compiled_after = model.compiled_forward(configs)
    assert torch.allclose(compiled_after, eager_after)

    with pytest.raises(ValueError, match="exactly 3 configurations"):
        model.compiled_forward(configs[:2])


@pytest.mark.smoke
def test_compiled_metropolis_proposals_use_the_full_fixed_batch():
    """Changing subsets are padded before compiled amplitude evaluation."""
    torch = pytest.importorskip("torch")
    qtn = pytest.importorskip("quimb.tensor")

    from pepsy.vmc.torch import TorchPEPSAmplitude
    from pepsy.vmc.torch.proposals import metropolis_exchange_sweep

    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=402,
        dtype="float64",
    )
    model = TorchPEPSAmplitude(
        peps,
        contraction="exact",
        dtype=torch.float64,
        amplitude_batching="vmap",
    )
    configs = torch.tensor(
        [[0, 0, 1, 1], [0, 1, 0, 1]],
        dtype=torch.long,
    )
    model.export_and_compile(configs, backend="eager")

    amplitude_batch_sizes = []
    log_batch_sizes = []
    compiled_amplitude = model._compiled_amplitude_forward
    compiled_log = model._compiled_log_amplitude_forward

    def record_amplitude(batch, *params):
        amplitude_batch_sizes.append(int(batch.shape[0]))
        return compiled_amplitude(batch, *params)

    def record_log(batch, *params):
        log_batch_sizes.append(int(batch.shape[0]))
        return compiled_log(batch, *params)

    model._compiled_amplitude_forward = record_amplitude
    model._compiled_log_amplitude_forward = record_log
    result = metropolis_exchange_sweep(
        configs,
        model,
        [(0, 1)],
        proposal="spin",
        generator=torch.Generator().manual_seed(7),
    )

    assert result.configs.shape == configs.shape
    assert amplitude_batch_sizes
    assert log_batch_sizes
    assert set(amplitude_batch_sizes) == {2}
    assert set(log_batch_sizes) == {2}

    # An explicitly supplied log function remains authoritative and must not
    # be replaced by the model's compiled forward_log method.
    log_batch_sizes.clear()

    def custom_log(batch):
        return torch.ones(batch.shape[0]), torch.zeros(batch.shape[0])

    metropolis_exchange_sweep(
        configs,
        model,
        [(0, 1)],
        proposal="spin",
        log_amplitude_fn=custom_log,
        generator=torch.Generator().manual_seed(7),
    )
    assert not log_batch_sizes


@pytest.mark.smoke
def test_export_compile_supports_flat_symmray_z2_peps():
    """The paper's flat-Z2 PEPS representation is exportable in Torch."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("symmray")

    from pepsy.vmc.netket import fermionic_peps_rand
    from pepsy.vmc.torch import TorchPEPSAmplitude

    peps = fermionic_peps_rand(
        "Z2",
        2,
        2,
        2,
        n_fermions_per_spin=(2, 2),
        seed=403,
        dtype="float64",
        flat=True,
    )
    model = TorchPEPSAmplitude(
        peps,
        contraction="exact",
        dtype=torch.float64,
        amplitude_batching="vmap",
    )
    configs = torch.tensor(
        [[0, 1, 2, 3], [3, 2, 1, 0]],
        dtype=torch.long,
    )
    eager = model.forward(configs, params=list(model.params))
    model.export_and_compile(configs, backend="eager")

    assert torch.allclose(model.compiled_forward(configs), eager)
    assert all("Flat" in type(peps[site].data).__name__ for site in peps.sites)


@pytest.mark.smoke
def test_compiled_boundary_reuse_batches_connected_local_energy():
    """Boundary environments and same-row targets share one compiled class."""
    torch = pytest.importorskip("torch")
    qtn = pytest.importorskip("quimb.tensor")

    from pepsy.vmc.torch import (
        TorchConnections,
        TorchPEPSBoundaryAmplitude,
        local_energy_from_connections,
    )

    peps = qtn.PEPS.rand(
        Lx=3,
        Ly=3,
        bond_dim=2,
        phys_dim=2,
        seed=404,
        dtype="float64",
    )
    model = TorchPEPSBoundaryAmplitude(
        peps,
        contraction="boundary",
        chi=2,
        dtype=torch.float64,
        amplitude_batching="serial",
    )
    configs = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=torch.long,
    )
    targets = configs.clone()
    targets[0, 3:5] = torch.tensor([0, 1])
    targets[1, 3:5] = torch.tensor([1, 0])
    connections = TorchConnections(
        configs=targets,
        coeffs=torch.ones(2, dtype=torch.float64),
        batch_ids=torch.arange(2),
    )

    model.export_and_compile_boundary_reuse(
        configs,
        backend="eager",
        directions=("x",),
        widths=(1,),
        compile_log=True,
    )
    with torch.no_grad():
        amplitudes = model.forward(configs)
        connected = model.connected_amplitudes(configs, amplitudes, connections)
        first_stats = dict(model.last_connected_reuse_stats)
        local_values = local_energy_from_connections(
            configs,
            amplitudes,
            connections,
            model,
            deduplicate_targets=True,
        )

    stats = first_stats
    assert stats["num_compiled_groups"] == 1
    assert stats["num_compiled_connections"] == 2
    assert stats["num_environment_compiled"] == 2
    assert torch.allclose(local_values, connected / amplitudes)

    # The compiled environment cache is reusable on the next estimator call.
    with torch.no_grad():
        second = local_energy_from_connections(
            configs,
            amplitudes,
            connections,
            model,
            deduplicate_targets=True,
        )
    assert model.last_connected_reuse_stats["num_environment_compiled"] == 0
    assert torch.allclose(second, local_values)

    # The same geometry-class cache is also used by the local Metropolis
    # proposal path, including its stable-log acceptance values.
    with torch.no_grad():
        proposal = model.proposal_amplitudes(configs, targets, amplitudes)
        proposal_phase, proposal_log_abs = model.proposal_log_amplitudes(
            configs,
            targets,
        )
    assert model.last_proposal_cache_stats["num_compiled_connections"] == 2
    assert torch.allclose(proposal, connected)
    assert torch.isfinite(proposal_phase).all()
    assert torch.isfinite(proposal_log_abs).all()

    # Compare the compiled result to the assembled boundary networks, which
    # is the eager oracle for the static reuse path.
    import quimb.tensor as qtn_local

    with torch.no_grad():
        tn = model._unpack_tn()
        for parent, target, value in zip(configs, targets, connected):
            envs = model._boundary_environment_cache[
                ("x", model._configuration_key(parent))
            ]
            selected = model._select_config(tn, target)
            reuse = (
                envs[("xmin", 1)]
                | selected.select([tn.x_tag(1)], which="any")
                | envs[("xmax", 1)]
            )
            reuse.view_as_(qtn_local.PEPS, **model._boundary_geometry["view_kwargs"])
            oracle = reuse.contract(all)
            assert torch.allclose(value, oracle)
