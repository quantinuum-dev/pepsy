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


@pytest.mark.parametrize("failure,workers", [
    (None, 2), ("primary", 1), ("primary", 2),
    ("context", 1), ("all", 2), ("compiled", 1),
])
def test_boundary_workers_parallelize_no_grad_primary_windows(failure, workers):
    """Serial and threaded reuse preserve values and expose retry causes."""
    torch = pytest.importorskip("torch")
    from pepsy.vmc.torch import TorchConnections, TorchPEPSBoundaryAmplitude

    class BoundaryProbe(TorchPEPSBoundaryAmplitude):
        def __init__(self):
            self.contraction = "boundary"
            self._boundary_geometry = object()
            self.amplitude_batching = "serial"
            self.graded_torch = False
            self._connection_vmap_enabled = False
            self.boundary_workers = workers
            self.failure = failure
            self._compiled_boundary_reuse = (
                {("x", (0,)): object()} if failure == "compiled" else {}
            )
            self._compiled_boundary_batch_size = 8
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

        def _cached_boundary_environments(self, tn, config, axis, **kwargs):
            del tn, config, kwargs
            if self.failure == "context" and axis == "x":
                raise RuntimeError("injected environment failure")
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
            del tn, parent_config, indices, envs, strip_tn, reference
            assert not torch.is_grad_enabled()
            if self.failure == "all" or (self.failure == "primary" and axis == "x"):
                raise RuntimeError("injected window failure")
            return target_config.sum(dtype=torch.float64)

        def _populate_compiled_boundary_environments(self, configs, axis):
            raise RuntimeError("injected compiled environment failure")

        def _compiled_boundary_reuse_batch(self, *args):
            raise RuntimeError("injected compiled reuse failure")

        def forward(self, configs, *, chunk_size=None):
            return configs.sum(dim=1, dtype=torch.float64)

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
    stats = model.last_connected_reuse_stats
    assert stats["num_parallel"] == (2 if workers == 2 else 0)
    assert stats["num_reused"] == (0 if failure == "all" else 2)
    assert stats["num_fallback"] == (2 if failure == "all" else 0)
    expected_stages = {
        None: [],
        "primary": ["primary_window"] * 2,
        "context": ["context"] * 2,
        "all": ["primary_window"] * 2 + ["alternative_window"] * 2,
        "compiled": ["compiled_environment", "compiled_reuse"],
    }[failure]
    assert stats["num_fallback_errors"] == len(expected_stages)
    assert [entry["stage"] for entry in stats["fallback_errors"]] == expected_stages
    assert all("RuntimeError: injected" in entry["error"] for entry in stats["fallback_errors"])

    # Diagnostics describe this call, including a clean call after a failure.
    model.failure = None
    model._compiled_boundary_reuse = {}
    with torch.no_grad():
        model.connected_amplitudes(configs, amplitudes, connections)
    assert model.last_connected_reuse_stats["fallback_errors"] == []
    assert model.last_connected_reuse_stats["num_fallback_errors"] == 0


def test_boundary_fallback_diagnostics_have_bounded_storage():
    from pepsy.vmc.torch.amplitude import _connected_reuse_stats, _record_boundary_fallback
    from pepsy.vmc.torch.results import _accumulate_cache_profile

    stats = _connected_reuse_stats()
    for _ in range(20):
        _record_boundary_fallback(stats, "context", RuntimeError("x" * 1000))
    assert stats["num_fallback_errors"] == 20
    assert len(stats["fallback_errors"]) == 8
    assert all(len(entry["error"]) == 400 for entry in stats["fallback_errors"])
    profile = _accumulate_cache_profile({}, {"connected": stats})
    assert profile["connected"]["num_fallback_errors"] == 20


@pytest.mark.parametrize("route", ["eager", "compiled", "compiled_log"])
def test_boundary_proposal_retry_records_cause_without_changing_value(monkeypatch, route):
    torch = pytest.importorskip("torch")
    import quimb.tensor as qtn
    from pepsy.vmc.torch import TorchPEPSBoundaryAmplitude

    model = TorchPEPSBoundaryAmplitude(
        qtn.PEPS.rand(2, 2, bond_dim=2, seed=17, dtype="float64"),
        chi=4, dtype=torch.float64, amplitude_batching="serial",
        proposal_batching="cache",
    )
    parents = torch.tensor([[0, 0, 0, 0]])
    targets = torch.tensor([[1, 0, 0, 0]])

    def fail(*args, **kwargs):
        raise RuntimeError("injected proposal failure")

    if route == "eager":
        monkeypatch.setattr(model, "_contract_axis_window", fail)
    else:
        key = model._changed_axis_windows(parents[0], targets[0])[0]
        model._compiled_boundary_reuse = {key: {"log_fn": object()}}
        model._compiled_boundary_batch_size = 1
        monkeypatch.setattr(model, "_populate_compiled_boundary_environments", lambda *a: 0)
        monkeypatch.setattr(model, "_compiled_boundary_reuse_batch", fail)

    with torch.no_grad():
        expected = model(targets)
        if route == "compiled_log":
            phase, log_abs = model.proposal_log_amplitudes(parents, targets)
            actual = phase * torch.exp(log_abs)
        else:
            actual = model.proposal_amplitudes(parents, targets, model(parents))
    assert torch.allclose(actual, expected)
    stats = model.last_proposal_cache_stats
    assert stats["num_fallback_errors"] == (2 if route == "eager" else 1)
    assert all("injected proposal failure" in entry["error"] for entry in stats["fallback_errors"])


def test_sparse_cutoff_retry_records_original_error_and_preserves_options():
    from types import SimpleNamespace
    from pepsy.vmc.torch import TorchPEPSAmplitude

    model = SimpleNamespace(cutoff=1e-8, symmray_tensor_ids=(0,), cutoff_fallbacks=0)
    options = {"method": "cholesky"}
    calls = []

    def contract(**kwargs):
        calls.append(kwargs)
        if kwargs["cutoff"]:
            raise ValueError("empty sparse sector")
        return 7.0

    value = TorchPEPSAmplitude._contract_approximate(model, contract, compress_opts=options)
    assert value == 7.0
    assert model.cutoff_fallbacks == 1
    assert model.cutoff_fallback_error == "ValueError: empty sparse sector"
    assert [call["cutoff"] for call in calls] == [1e-8, 0.0]
    assert calls[1]["compress_opts"] == {"method": "svd"}
    assert options == {"method": "cholesky"}


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
