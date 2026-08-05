def test_transition_plan_matches_live_connections_and_slices():
    torch = __import__("torch")
    from pepsy.vmc.torch import TorchConnections, TorchVMCDriver

    class Amplitude(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, configs):
            return self.scale * (configs.to(torch.float64).sum(dim=1) + 1.0)

    def connections(configs, graph):
        del graph
        flipped = configs.clone()
        flipped[:, 0] = 1 - flipped[:, 0]
        batch_ids = torch.arange(configs.shape[0], device=configs.device)
        return TorchConnections(
            configs=torch.cat((configs, flipped)),
            coeffs=torch.ones(2 * configs.shape[0], device=configs.device),
            batch_ids=torch.cat((batch_ids, batch_ids)),
        )

    configs = torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.long)
    driver = TorchVMCDriver(Amplitude(), None, configs, connection_fn=connections)
    plan = driver.compile_fock_plan(configs, observables={"obs": None})
    live = driver.measure_samples(configs, observables={"obs": None})
    planned = driver.measure_samples(
        configs, observables={"obs": None}, connection_plan=plan,
    )
    assert torch.allclose(live["obs"].local_energies, planned["obs"].local_energies)

    sliced = plan.slice(1, 3)
    sliced_result = driver.measure_samples(
        configs[1:], observables={"obs": None}, connection_plan=sliced,
    )
    assert torch.allclose(
        planned["obs"].local_energies[:, 1:], sliced_result["obs"].local_energies,
    )


def test_amplitude_cache_reuses_targets_and_invalidates_on_parameter_update():
    torch = __import__("torch")
    from pepsy.vmc.torch import TorchAmplitudeCache, TorchConnections, TorchVMCDriver

    class Amplitude(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, configs):
            return self.scale * (configs.to(torch.float64).sum(dim=1) + 1.0)

    configs = torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=torch.long)
    driver = TorchVMCDriver(
        Amplitude(), None, configs,
        connection_fn=lambda rows, graph: TorchConnections(
            configs=rows,
            coeffs=torch.ones(rows.shape[0]),
            batch_ids=torch.arange(rows.shape[0]),
        ),
    )
    cache = TorchAmplitudeCache(max_entries=32)
    with torch.no_grad():
        first = cache.evaluate(driver.model, configs)
        second = cache.evaluate(driver.model, configs)
    assert torch.equal(first, second)
    assert cache.snapshot()["hits"] == 3
    with torch.no_grad():
        driver.model.scale.mul_(2.0)
    with torch.no_grad():
        updated = cache.evaluate(driver.model, configs)
    assert torch.allclose(updated, 2.0 * first)
    assert cache.snapshot()["misses"] == 3
