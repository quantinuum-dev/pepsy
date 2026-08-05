"""Regression tests for reusable Torch-VMC local-estimator amplitudes."""

import pytest


def test_local_energy_reuses_matching_parent_amplitudes_across_walkers():
    """A target already retained by another walker needs no PEPS call."""
    torch = pytest.importorskip("torch")
    from pepsy.vmc.torch import TorchConnections, local_energy_from_connections

    class Amplitude:
        def __init__(self):
            self.connections = None

        def connected_amplitudes(self, configs, amplitudes, connections, **kwargs):
            del configs, amplitudes, kwargs
            self.connections = connections
            return connections.configs.sum(dim=1, dtype=torch.float64) + 2.0

    configs = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
    amplitudes = torch.tensor([2.0, 3.0], dtype=torch.float64)
    connections = TorchConnections(
        configs=torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
        coeffs=torch.ones(2, dtype=torch.float64),
        batch_ids=torch.zeros(2, dtype=torch.long),
    )
    amplitude = Amplitude()

    values = local_energy_from_connections(
        configs,
        amplitudes,
        connections,
        amplitude,
        deduplicate_targets=True,
    )

    assert amplitude.connections.configs.tolist() == [[0, 1]]
    assert amplitude.connections.batch_ids.tolist() == [0]
    assert torch.allclose(
        values,
        torch.tensor([3.0, 0.0], dtype=torch.float64),
    )


def test_boundary_connected_fallbacks_are_dispatched_as_one_cached_batch():
    """Unresolved boundary targets use the cache-aware forward route."""
    torch = pytest.importorskip("torch")
    from pepsy.vmc.torch import TorchConnections, TorchPEPSBoundaryAmplitude

    class BoundaryProbe(TorchPEPSBoundaryAmplitude):
        def __init__(self):
            self.contraction = "boundary"
            self._boundary_geometry = object()
            self.amplitude_batching = "serial"
            self._connection_vmap_enabled = False
            self.forward_batches = []

        def _ensure_boundary_cache_current(self):
            pass

        def _unpack_tn(self):
            return object()

        def _reference_tensor(self):
            return torch.tensor(0.0)

        def _changed_axis_windows(self, parent_config, target_config):
            del parent_config, target_config
            return ()

        def forward(self, configs, params=None, *, chunk_size=None):
            del params, chunk_size
            self.forward_batches.append(configs.clone())
            return configs.sum(dim=1, dtype=torch.float64)

    model = BoundaryProbe()
    configs = torch.tensor([[0], [1]], dtype=torch.long)
    amplitudes = torch.tensor([1.0, 2.0], dtype=torch.float64)
    connections = TorchConnections(
        configs=torch.tensor([[2], [3]], dtype=torch.long),
        coeffs=torch.ones(2, dtype=torch.float64),
        batch_ids=torch.tensor([0, 1], dtype=torch.long),
    )

    values = model.connected_amplitudes(configs, amplitudes, connections)

    assert len(model.forward_batches) == 1
    assert model.forward_batches[0].tolist() == [[2], [3]]
    assert torch.allclose(
        values,
        torch.tensor([2.0, 3.0], dtype=torch.float64),
    )
    assert model.last_connected_reuse_stats["num_fallback"] == 2


def test_boundary_workers_parallelize_no_grad_primary_windows():
    """Independent cached boundary windows can run in parallel on CPU."""
    torch = pytest.importorskip("torch")
    from pepsy.vmc.torch import TorchConnections, TorchPEPSBoundaryAmplitude

    class BoundaryProbe(TorchPEPSBoundaryAmplitude):
        def __init__(self):
            self.contraction = "boundary"
            self._boundary_geometry = object()
            self.amplitude_batching = "serial"
            self.graded_torch = False
            self._connection_vmap_enabled = False
            self.boundary_workers = 2
            self._boundary_environment_cache = {}
            self._boundary_strip_cache = {}
            self.last_amplitude_cache_stats = {"stale": 1}

        def _ensure_boundary_cache_current(self):
            pass

        def _unpack_tn(self):
            return object()

        def _select_config(self, tn, config):
            del tn, config
            return object()

        def _reference_tensor(self):
            return torch.tensor(0.0)

        @staticmethod
        def _configuration_key(config):
            return tuple(int(value) for value in config.tolist())

        def _changed_axis_windows(self, parent_config, target_config):
            del parent_config, target_config
            return (("x", (0,)), ("y", (0,)))

        def _cached_boundary_environments(self, *args, **kwargs):
            del args, kwargs
            return object(), False

        def _cached_boundary_strip(self, *args, **kwargs):
            del args, kwargs
            return object(), False

        def _contract_cached_axis_window(
            self,
            tn,
            parent_config,
            target_config,
            axis,
            indices,
            envs,
            strip_tn,
            reference,
        ):
            del tn, parent_config, axis, indices, envs, strip_tn, reference
            return target_config.sum(dtype=torch.float64)

    model = BoundaryProbe()
    configs = torch.tensor([[0, 0], [1, 0]], dtype=torch.long)
    amplitudes = torch.ones(2, dtype=torch.float64)
    connections = TorchConnections(
        configs=torch.tensor([[1, 1], [0, 1]], dtype=torch.long),
        coeffs=torch.ones(2, dtype=torch.float64),
        batch_ids=torch.tensor([0, 1], dtype=torch.long),
    )

    with torch.no_grad():
        values = model.connected_amplitudes(configs, amplitudes, connections)

    assert torch.allclose(values, torch.tensor([2.0, 1.0], dtype=torch.float64))
    assert model.last_amplitude_cache_stats is None
    assert model.last_connected_reuse_stats["num_requests"] == 2
    assert model.last_connected_reuse_stats["num_parallel"] == 2
    assert model.last_connected_reuse_stats["num_reused"] == 2


def test_amplitude_benchmark_records_chunks_and_restores_fast_path_state():
    """Benchmarking vmap candidates does not alter the production policy."""
    torch = pytest.importorskip("torch")
    from pepsy.vmc.torch import benchmark_torch_amplitudes

    class Amplitude:
        amplitude_batching = "auto"
        _vmap_forward_enabled = True
        last_amplitude_batching = None

        def __init__(self):
            self.boundary_cache_size = 7
            self._boundary_amplitude_cache = {"retained": torch.tensor(1.0)}

        def __call__(self, configs):
            self.last_amplitude_batching = self.amplitude_batching
            return configs.sum(dim=1, dtype=torch.float64)

    amplitude = Amplitude()
    run = benchmark_torch_amplitudes(
        amplitude,
        torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long),
        chunk_sizes=(None, 2),
        amplitude_batchings=("serial", "vmap"),
        warmup=0,
        repeats=1,
    )

    assert len(run.entries) == 4
    assert run.best in run.entries
    assert {entry.executed_batching for entry in run.entries} == {"serial", "vmap"}
    assert amplitude.amplitude_batching == "auto"
    assert amplitude._vmap_forward_enabled
    assert amplitude.boundary_cache_size == 7
    assert list(amplitude._boundary_amplitude_cache) == ["retained"]
