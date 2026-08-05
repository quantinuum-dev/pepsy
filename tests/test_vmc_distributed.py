"""Focused checks for optional rank-sharded Torch-VMC reductions."""

from types import SimpleNamespace

import pytest


def test_rank_sharded_statistics_all_reduce_only_compact_moments():
    torch = pytest.importorskip("torch")
    from pepsy.vmc.torch.distributed import distributed_unweighted_statistics

    class FakeDistributed:
        class ReduceOp:
            SUM = "sum"
            MAX = "max"

        def all_reduce(self, tensor, *, op, group):
            del group
            assert op == self.ReduceOp.SUM
            # The second rank owns values 2 and 3, with local ESS two.
            tensor.add_(torch.tensor([5.0, 0.0, 13.0, 2.0, 2.0]))

    runtime = SimpleNamespace(
        module=FakeDistributed(),
        group=None,
        world_size=2,
    )
    (
        mean,
        variance,
        stderr,
        naive_stderr,
        effective_sample_size,
        n_samples,
    ) = distributed_unweighted_statistics(
        torch.tensor([0.0, 1.0]),
        local_effective_sample_size=2.0,
        runtime=runtime,
    )

    assert mean.item() == pytest.approx(1.5)
    assert variance.item() == pytest.approx(1.25)
    assert stderr.item() == pytest.approx((1.25 / 4.0) ** 0.5)
    assert naive_stderr.item() == pytest.approx((1.25 / 4.0) ** 0.5)
    assert effective_sample_size.item() == pytest.approx(4.0)
    assert n_samples == 4


def test_driver_measurement_returns_global_rank_sharded_estimate(monkeypatch):
    torch = pytest.importorskip("torch")
    import pepsy.vmc.torch.driver as driver_module
    from pepsy.vmc.torch import TorchConnections, TorchVMCDriver

    class FakeDistributed:
        class ReduceOp:
            SUM = "sum"
            MAX = "max"

        def __init__(self):
            self.scalar_sum_calls = 0

        def all_reduce(self, tensor, *, op, group):
            del group
            if op == self.ReduceOp.MAX:
                return
            if tensor.numel() == 5:
                tensor.add_(torch.tensor([5.0, 0.0, 13.0, 2.0, 2.0]))
            else:
                self.scalar_sum_calls += 1
                if self.scalar_sum_calls == 1:
                    # The remote rank has the same two local chains.
                    tensor.add_(2)

    class Amplitude:
        def __call__(self, configs):
            return configs[:, 0].to(dtype=torch.float64) + 1.0

    def connections(configs, graph):
        del graph
        return TorchConnections(
            configs=configs.clone(),
            coeffs=configs[:, 0].to(dtype=torch.float64),
            batch_ids=torch.arange(configs.shape[0]),
        )

    runtime = SimpleNamespace(
        module=FakeDistributed(),
        group=None,
        rank=0,
        world_size=2,
        backend="gloo",
    )
    monkeypatch.setattr(driver_module, "resolve_torch_distributed", lambda _: runtime)
    driver = TorchVMCDriver(
        Amplitude(),
        object(),
        torch.tensor([[0], [1]]),
        connection_fn=connections,
        proposal="spin",
    )

    result = driver.measure_samples(
        torch.tensor([[[0], [1]]]),
        distributed=True,
    )

    assert result.energy_mean.item() == pytest.approx(1.5)
    assert result.energy_variance.item() == pytest.approx(1.25)
    assert result.n_samples == 4
    assert result.samples_per_second > 0
    assert result.chain_diagnostics is None
    assert result.distributed.global_n_chains == 4
    assert result.distributed.local_n_chains == 2
    assert result.distributed.global_n_samples == 4


def test_rank_shard_counts_cover_global_chains_without_empty_ranks():
    from pepsy.vmc.torch.distributed import shard_chain_count

    class Runtime:
        world_size = 3

        def __init__(self, rank):
            self.rank = rank

    assert [shard_chain_count(8, Runtime(rank)) for rank in range(3)] == [3, 3, 2]
    with pytest.raises(ValueError, match="at least"):
        shard_chain_count(2, Runtime(0))


def test_fermion_lazy_setup_uses_rank_local_seed(monkeypatch):
    import pepsy.vmc.torch.fermion as fermion_module
    from pepsy.vmc import SamplingConfig
    from pepsy.vmc.torch import TorchFermionVMC

    runtime = SimpleNamespace(rank=2, world_size=3)
    monkeypatch.setattr(
        fermion_module,
        "resolve_torch_distributed",
        lambda _: runtime,
    )
    sampling = SamplingConfig(
        n_samples_per_chain=2,
        n_chains=8,
        seed=11,
    )

    returned_runtime, local_sampling = TorchFermionVMC._rank_sharded_sampling_config(
        sampling,
        True,
    )

    assert returned_runtime is runtime
    assert local_sampling.n_chains == 2
    assert local_sampling.seed == 11 + 2 * 104_729


def test_distributed_sample_metadata_survives_portable_conversion():
    torch = pytest.importorskip("torch")
    from pepsy.vmc.torch import TorchDistributedMetadata, TorchMCMCSamples

    distributed = TorchDistributedMetadata(
        rank=1,
        world_size=2,
        backend="gloo",
        global_n_chains=4,
        local_n_chains=2,
        global_n_samples=12,
        local_n_samples=6,
    )
    native = TorchMCMCSamples(
        configs=torch.zeros((3, 2, 1), dtype=torch.long),
        amplitudes=torch.ones((3, 2), dtype=torch.float64),
        n_samples=6,
        n_samples_per_chain=3,
        n_chains=2,
        n_discard_per_chain=0,
        sweep_size=1,
        acceptance_rate=0.5,
        n_proposed=12,
        n_accepted=6,
        elapsed_seconds=1.0,
        samples_per_second=6.0,
        distributed=distributed,
    )

    portable = native.to_common()

    assert portable.n_chains == 2
    assert portable.diagnostics["distributed"] == {
        "rank": 1,
        "world_size": 2,
        "backend": "gloo",
        "global_n_chains": 4,
        "local_n_chains": 2,
        "global_n_samples": 12,
        "local_n_samples": 6,
    }
