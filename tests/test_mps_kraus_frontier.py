"""Bounded GPU probability batches across live trajectory branches."""

from types import SimpleNamespace

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import MpsOptimizer, noise

pytestmark = [pytest.mark.optional, pytest.mark.integration]


@pytest.fixture(params=["cuda", "cupy"])
def convert(request):
    if request.param == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        return lambda x, dtype: torch.tensor(np.array(x), device="cuda", dtype=getattr(torch, dtype))
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    return lambda x, dtype: cp.asarray(x, dtype=dtype)


def parents(convert, dtype="complex128", bonds=(2, 2, 2, 2, 2)):
    result = []
    for i, bond in enumerate(bonds):
        state = qtn.MPS_rand_state(4, bond, dtype=dtype, seed=701 + i)
        state.apply_to_arrays(lambda x: convert(x, dtype))
        result.append(SimpleNamespace(optimizer=MpsOptimizer(state, chi=4)))
    return result


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_frontier_probabilities_match_dense_and_bound_downloads(convert, dtype, monkeypatch):
    nodes = parents(convert, dtype)
    channel = noise.TrajectoryChannel.amplitude_damping(.31)
    before = [ar.to_numpy(n.optimizer.to_dense()).ravel() for n in nodes]
    expected = []
    for vector in before:
        block = vector.reshape((2,) * 4).transpose(1, 0, 2, 3).reshape(2, -1)
        row = np.array([np.linalg.norm(o.gate @ block)**2 for o in channel.outcomes])
        expected.append(row / row.sum())
    downloads = []
    original = ar.to_numpy

    def small_probabilities(value):
        downloads.append(tuple(value.shape))
        assert tuple(value.shape) == (5, 2), "State amplitudes were downloaded"
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(ar, "to_numpy", small_probabilities)
        result = list(noise._coalesced_kraus_probabilities(nodes, channel, (1,)))
    assert downloads == [(5, 2)]
    tolerance = 5e-7 if dtype == "complex64" else 1e-12
    np.testing.assert_allclose(result, expected, atol=tolerance, rtol=tolerance)
    for node, vector in zip(nodes, before):
        np.testing.assert_allclose(ar.to_numpy(node.optimizer.to_dense()).ravel(), vector,
                                   atol=tolerance, rtol=tolerance)
        assert node.optimizer.info_c["cur_orthog"] == (1, 1)
        assert all(ar.get_dtype_name(t.data) == dtype for t in node.optimizer.p.tensors)


@pytest.mark.parametrize("limit", ["states", "bytes", "disabled"])
def test_frontier_chunk_budget_preserves_distributions(convert, monkeypatch, limit):
    nodes = parents(convert)
    channel = noise.TrajectoryChannel.amplitude_damping(.31)
    reference = list(noise._coalesced_kraus_probabilities(nodes, channel, (1,), batch=False))
    if limit == "states":
        monkeypatch.setattr(noise, "_KRAUS_FRONTIER_MAX_STATES", 2)
    elif limit == "bytes":
        # All prepared blocks have two bond-two legs and one physical leg.
        per_parent = 16 * (8 * (4 + 6 * 2) + 4 * 2)
        monkeypatch.setattr(noise, "_KRAUS_FRONTIER_MAX_BYTES", 2 * per_parent)
    else:
        monkeypatch.setattr(noise, "_KRAUS_FRONTIER_MAX_BYTES", 1)
    downloads = []
    original = ar.to_numpy

    def record(value):
        downloads.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(ar, "to_numpy", record)
    result = list(noise._coalesced_kraus_probabilities(nodes, channel, (1,)))
    assert downloads == ([(2,)] * 5 if limit == "disabled" else [(2, 2), (2, 2), (2,)])
    np.testing.assert_allclose(result, reference, atol=1e-12)


def test_frontier_handles_unequal_bonds_and_physical_layouts(convert):
    nodes = parents(convert, bonds=(1, 2, 2, 3, 1))
    channel = noise.TrajectoryChannel.amplitude_damping(.31)
    for node in nodes:
        if node.optimizer.p.max_bond() == 1:
            node.optimizer.apply_layout([3, 2, 1, 0])
    reference = list(noise._coalesced_kraus_probabilities(nodes, channel, (1,), batch=False))
    result = list(noise._coalesced_kraus_probabilities(nodes, channel, (1,)))
    np.testing.assert_allclose(result, reference, atol=1e-12)


def test_frontier_preserves_rare_and_zero_complex64_probabilities(convert):
    nodes = []
    for amplitude in (1e-30, 0., 2e-30):
        p = qtn.MPS_computational_state("0", dtype="complex64")
        p[0].modify(data=convert([1., amplitude], "complex64"))
        nodes.append(SimpleNamespace(optimizer=MpsOptimizer(p, chi=2)))
    channel = noise.TrajectoryChannel.kraus([
        ("common", np.diag([1., 0.])), ("rare", np.diag([0., 1.])),
    ])
    result = np.array(list(noise._coalesced_kraus_probabilities(nodes, channel, (0,))))
    np.testing.assert_allclose(result[:, 1], [1e-60, 0., 4e-60], rtol=5e-6, atol=0.)


def test_frontier_keeps_real_and_complex_dtype_groups_separate(convert):
    nodes = parents(convert, "float64", bonds=(2, 2))
    nodes += parents(convert, "complex128", bonds=(2, 2))
    channel = noise.TrajectoryChannel.amplitude_damping(.31)
    reference = list(noise._coalesced_kraus_probabilities(nodes, channel, (1,), batch=False))
    result = list(noise._coalesced_kraus_probabilities(nodes, channel, (1,)))
    np.testing.assert_allclose(result, reference, atol=1e-12)
    for i, node in enumerate(nodes):
        assert node.optimizer.backend_dtype == ("float64" if i < 2 else "complex128")


def test_frontier_still_rejects_zero_norm_parent(convert):
    nodes = parents(convert, bonds=(2, 2))
    nodes[1].optimizer.p[0].modify(data=nodes[1].optimizer.p[0].data * 0.)
    nodes[1].optimizer.info_c["cur_orthog"] = (0, 0)
    with pytest.raises(ValueError, match="zero- or invalid-norm"):
        list(noise._coalesced_kraus_probabilities(
            nodes, noise.TrajectoryChannel.amplitude_damping(.2), (1,),
        ))


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_batched_frontier_replay_matches_serial_parent_evaluation(convert, monkeypatch, workers, dtype):
    initial = qtn.MPS_rand_state(4, 2, dtype=dtype, seed=891)
    initial.apply_to_arrays(lambda x: convert(x, dtype))
    stream = [("x_error", .5, 0), ("z_error", .5, 1)]
    stream += [("amplitude_damping", .2, q) for q in (0, 1, 2)]
    sim = MpsOptimizer(initial, stream, chi=4)
    kwargs = dict(shots=64, seed=771, strategy="coalesced", workers=workers,
                  progress=False, progbar=False, cutoff=0.)
    with monkeypatch.context() as patch:
        patch.setattr(noise, "_KRAUS_FRONTIER_MAX_BYTES", 1)
        reference = sim.run(**kwargs)
    result = sim.run(**kwargs)
    assert result.counts == reference.counts
    assert result.diagnostics.max_kraus_parent_batch > 1
    assert reference.diagnostics.max_kraus_parent_batch == 1
    assert [[r.label for r in row] for row in result.records] == [
        [r.label for r in row] for row in reference.records
    ]
    tolerance = 4e-6 if dtype == "complex64" else 1e-11
    for actual, wanted in zip(result.optimizers, reference.optimizers):
        np.testing.assert_allclose(ar.to_numpy(actual.to_dense()), ar.to_numpy(wanted.to_dense()),
                                   atol=tolerance, rtol=tolerance)
        assert actual.backend_info() == sim.backend_info()
    assert result.diagnostics.max_kraus_probability_residual < tolerance


def test_callable_proposal_keeps_serial_parent_evaluation(convert, monkeypatch):
    initial = qtn.MPS_computational_state("10", dtype="complex128")
    initial.apply_to_arrays(lambda x: convert(x, "complex128"))
    sim = MpsOptimizer(initial, [("x_error", .5, 1), ("amplitude_damping", .2, 0)], chi=2)
    calls = []

    def proposal(event_index, labels, target, optimizer):
        calls.append(event_index)
        return target

    def forbidden(*args):
        raise AssertionError("Callable proposal enabled frontier pre-evaluation")

    monkeypatch.setattr(noise, "_kraus_frontier_key", forbidden)
    result = sim.run(shots=16, strategy="coalesced", workers=1, seed=71,
                     importance_sampling=proposal, progress=False, progbar=False)
    assert result.shots == 16 and calls


def test_frontier_workspace_released_before_following_gates(convert, monkeypatch):
    initial = qtn.MPS_computational_state("10", dtype="complex128")
    initial.apply_to_arrays(lambda x: convert(x, "complex128"))
    sim = MpsOptimizer(initial, [
        ("x_error", .5, 1), ("amplitude_damping", .2, 0), ("h", 0),
    ], chi=2)
    original = noise._coalesced_kraus_probabilities
    run_entries = noise._run_coalesced_entries
    closed = []

    def distributions(*args, **kwargs):
        try:
            yield from original(*args, **kwargs)
        finally:
            closed.append(True)

    def after_channel(nodes, entries, *args, **kwargs):
        if any(index > 1 for index, _ in entries):
            assert closed, "Probability workspace retained during subsequent gates"
        return run_entries(nodes, entries, *args, **kwargs)

    monkeypatch.setattr(noise, "_coalesced_kraus_probabilities", distributions)
    monkeypatch.setattr(noise, "_run_coalesced_entries", after_channel)
    result = sim.run(shots=16, seed=77, strategy="coalesced", workers=1,
                     retain="none", progress=False, progbar=False)
    assert len(closed) == 1
    assert result.diagnostics.max_kraus_parent_batch == 2
    assert not result.optimizers


def test_frontier_probability_evaluation_preserves_torch_gradient():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sources, nodes = [], []
    for _ in range(2):
        v = torch.tensor([.6, .8], dtype=torch.complex128, device="cuda", requires_grad=True)
        p = qtn.MPS_computational_state("0", dtype="complex128")
        p[0].modify(data=v)
        sources.append(v)
        nodes.append(SimpleNamespace(optimizer=MpsOptimizer(p, chi=2)))
    channel = noise.TrajectoryChannel.amplitude_damping(.2)
    rows = list(noise._coalesced_kraus_probabilities(nodes, channel, (0,)))
    np.testing.assert_allclose(rows, [[.872, .128]] * 2, atol=1e-12)
    for source, node in zip(sources, nodes):
        value = node.optimizer.to_dense().real.sum()
        gradient, = torch.autograd.grad(value, source)
        torch.testing.assert_close(gradient, torch.ones_like(source))
