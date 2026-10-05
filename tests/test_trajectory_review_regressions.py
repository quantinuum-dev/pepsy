"""Exact controls, rare multi-site branches and generated-gate backend contracts."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import MpsOptimizer, noise


@pytest.mark.parametrize("mode", ["exact", "exact-batch"])
@pytest.mark.parametrize("strategy", ["auto", "coalesced"])
@pytest.mark.parametrize("control", ["measure", "reset"])
def test_exact_coalesced_controls_match_bell_branches(mode, strategy, control):
    stream = [("h", 0), ("cnot", 0, 1)]
    stream += ([("measure", "Z", 0), ("if", -1, 1, ("x", 1))]
               if control == "measure" else [("reset", 0)])
    initial = qtn.MPS_computational_state("00", dtype="complex128")
    initial.exponent = .5
    simulator = MpsOptimizer(initial, stream, chi=4, mode=mode)
    result = simulator.run(shots=128, seed=1, strategy=strategy, workers=1,
                           progress=False, progbar=False, cutoff=0.)
    assert result.coalesced and result.branches == 2
    assert sum(result.counts) == 128
    expected = {0, 2} if control == "measure" else {0, 1}
    assert {int(np.argmax(abs(opt.to_dense()))) for opt in result.optimizers} == expected
    for leaf in result.raw.leaves:
        np.testing.assert_allclose(np.linalg.norm(leaf.optimizer.to_dense()), 1., atol=1e-12)
        assert leaf.measurements[0].probability == pytest.approx(.5)
    np.testing.assert_allclose(simulator.to_dense(), initial.to_dense())


def test_exact_coalesced_postselection_retains_rare_schmidt_direction():
    initial = qtn.MatrixProductState.from_dense(
        np.array([1., 0., 0., 1e-20], dtype=complex), [2, 2], cutoff=0.,
    )
    result = MpsOptimizer(initial, [("z", 0), ("measure", "Z", 0, -1)],
                          chi=2, mode="exact").run(
        shots=4, strategy="coalesced", workers=1, progress=False, progbar=False,
        cutoff=0.,
    )
    assert result.branches == 1
    record = result.raw.leaves[0].measurements[0]
    assert record.probability == pytest.approx(1e-40, rel=1e-12, abs=0.)
    np.testing.assert_allclose(abs(result.optimizers[0].to_dense().ravel()), [0., 0., 0., 1.])


@pytest.mark.parametrize("mode", ["direct", "dmrg2", "svd", "exact", "exact-batch"])
@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
def test_rare_two_site_kraus_records_use_projected_amplitudes(mode, strategy):
    u = np.ones(4, dtype=complex) / 2
    v = np.array([1., -1., -1., 1.], dtype=complex) / 2
    initial = qtn.MatrixProductState.from_dense(u + 1e-10 * v, [2, 2], cutoff=0.)
    rare = np.outer(v, v.conj())
    channel = noise.TrajectoryChannel.kraus([
        ("common", np.eye(4) - rare), ("rare", rare),
    ])
    template = MpsOptimizer(initial, chi=4, mode=mode)
    vector = template.to_dense().ravel()
    expected = np.linalg.norm(rare @ vector)**2 / np.linalg.norm(vector)**2
    result = noise.run_trajectory_shots(
        lambda: MpsOptimizer(initial, chi=4, mode=mode),
        [noise.TrajectoryEvent(channel, (0, 1))], 16, seed=6, strategy=strategy,
        importance_sampling={"common": .5, "rare": .5},
        run_kwargs={"progbar": False, "cutoff": 0., "n_iter": 4},
    )
    records = result.records if strategy == "independent" else [x.records for x in result.leaves]
    rare_records = [records[0] for records in records if records[0].label == "rare"]
    assert rare_records
    for record in rare_records:
        assert record.probability == pytest.approx(expected, rel=1e-5, abs=0.)
        assert record.likelihood_ratio == pytest.approx(2 * expected, rel=1e-5, abs=0.)
    for opt in result.optimizers:
        assert np.linalg.norm(opt.to_dense()) == pytest.approx(1., abs=1e-12)


@pytest.mark.parametrize("where", [(0, 1), (1, 0), (0, 3), (3, 0), (3, 1, 0)])
@pytest.mark.parametrize("backend", ["numpy", pytest.param("torch", marks=pytest.mark.optional)])
def test_multisite_kraus_support_order_matches_dense_reference(where, backend):
    rng = np.random.default_rng(713)
    vector = rng.normal(size=16) + 1j * rng.normal(size=16)
    vector /= np.linalg.norm(vector)
    initial = qtn.MatrixProductState.from_dense(vector, [2]*4, cutoff=0.)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        initial.apply_to_arrays(lambda x: torch.tensor(np.array(x), dtype=torch.complex128))
    opt = MpsOptimizer(initial, chi=4)
    channel = noise.TrajectoryChannel.amplitude_damping(.37)
    outcomes = [(x.label, np.kron(x.gate, np.eye(2**(len(where)-1))))
                for x in channel.outcomes]
    channel = noise.TrajectoryChannel.kraus(outcomes)
    order = where + tuple(i for i in range(4) if i not in where)
    block = vector.reshape((2,)*4).transpose(order).reshape(2**len(where), -1)
    expected = [np.linalg.norm(gate @ block)**2 for _, gate in outcomes]
    observed = noise._kraus_probabilities(opt, channel, where)
    np.testing.assert_allclose(observed, expected, atol=1e-12)
    np.testing.assert_allclose(ar.to_numpy(opt.to_dense()).ravel(), vector, atol=1e-12)
    assert not opt._trajectory_diagnostics.get("used_kraus_copy_fallback", False)


@pytest.mark.optional
@pytest.mark.integration
@pytest.mark.parametrize("strategy", ["independent", "coalesced"])
@pytest.mark.parametrize("event", ["x_error", "amplitude_damping"])
def test_real_torch_generated_noise_preserves_state_dtype(strategy, event):
    torch = pytest.importorskip("torch")
    initial = qtn.MPS_computational_state("10", dtype="float64")
    initial.apply_to_arrays(lambda x: torch.tensor(np.array(x), dtype=torch.float64))
    result = MpsOptimizer(initial, [(event, .3, 0)], chi=4).run(
        shots=64, seed=1, strategy=strategy, workers=1, progress=False, progbar=False,
    )
    seen = set()
    records = (result.raw.records if strategy == "independent"
               else [leaf.records for leaf in result.raw.leaves])
    for opt, record in zip(result.optimizers, records):
        assert all(t.data.dtype == torch.float64 for t in opt.p.tensors)
        assert all(t.data.device.type == "cpu" for t in opt.p.tensors)
        vector = ar.to_numpy(opt.to_dense()).ravel()
        bit = int(np.argmax(abs(vector)))
        seen.add(bit)
        assert np.linalg.norm(vector) == pytest.approx(1., abs=1e-12)
        assert record[0].probability == pytest.approx(.3 if bit == 0 else .7)
    assert seen == {0, 2}


@pytest.mark.optional
@pytest.mark.integration
def test_real_torch_rejects_genuinely_complex_trajectory_operator():
    torch = pytest.importorskip("torch")
    initial = qtn.MPS_computational_state("0", dtype="float64")
    initial.apply_to_arrays(lambda x: torch.tensor(np.array(x), dtype=torch.float64))
    channel = noise.TrajectoryChannel.mixture([("S", 1., np.diag([1., 1j]))])
    opt = MpsOptimizer(initial, [noise.TrajectoryEvent(channel, 0)], chi=2)
    with pytest.raises(TypeError, match="complex trajectory operator requires a complex MPS"):
        opt.run(shots=2, workers=1, progress=False, progbar=False)
    np.testing.assert_allclose(ar.to_numpy(opt.to_dense()).ravel(), [1., 0.])


@pytest.mark.optional
@pytest.mark.integration
@pytest.mark.parametrize("mode", ["direct", "exact"])
def test_jax_trajectory_precision_is_scoped(mode):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    initial = qtn.MPS_computational_state("1", dtype="complex64")
    initial.apply_to_arrays(lambda x: jnp.asarray(x))
    with jax.default_matmul_precision("bfloat16"):
        result = MpsOptimizer(initial, [("amplitude_damping", .3, 0)], chi=2, mode=mode).run(
            shots=64, strategy="coalesced", seed=1, workers=1, progress=False, progbar=False,
        )
        assert jax.config.jax_default_matmul_precision == "bfloat16"
    assert result.branches == 2
    for leaf in result.raw.leaves:
        record = leaf.records[0]
        assert record.probability == pytest.approx(.3 if record.label == "jump" else .7, abs=2e-7)
        assert all(t.data.dtype == jnp.complex64 for t in leaf.optimizer.p.tensors)
