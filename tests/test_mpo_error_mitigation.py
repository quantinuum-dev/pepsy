"""Dense references for middle-out mitigation and randomized Pauli readout."""

from itertools import product

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from pepsy.optimizers import MpoErrorMit, PauliShots


_PAULIS = tuple(np.asarray(qu.pauli(axis)) for axis in "IXYZ")
_TWO_PAULIS = np.array([np.kron(a, b) for a, b in product(_PAULIS, repeat=2)])


def _transfer(action):
    images = np.array([action(pauli) for pauli in _TWO_PAULIS])
    return np.einsum("aij,bji->ab", _TWO_PAULIS, images).real / 4


def _conjugate(unitary, rho):
    return unitary @ rho @ unitary.conj().T


@pytest.mark.parametrize("mode,chi", [
    ("direct", None), ("direct", 16), ("dmrg", 16), ("dmrg2", 16), ("dmrg3", 16), ("src", 16),
])
def test_middle_out_map_matches_dense_for_ordered_nonsymmetric_channels(mode, chi):
    identity, x, _, _ = _PAULIS
    r1 = np.kron(qu.Ry(0.67), identity)
    swap = np.asarray(qu.swap())
    r2 = swap @ qu.CNOT() @ swap @ np.kron(identity, qu.Rx(0.31))
    k0 = np.kron(np.diag([1, np.sqrt(0.8)]), identity)
    k1 = np.kron(np.array([[0, np.sqrt(0.2)], [0, 0]]), identity)
    flip = np.kron(x, identity)

    def noise1(rho):
        decayed = _conjugate(k0, rho) + _conjugate(k1, rho)
        return 0.87 * decayed + 0.13 * _conjugate(flip, decayed)

    def noise2(rho):
        return 0.84 * rho + 0.16 / 15 * sum(_conjugate(pauli, rho) for pauli in _TWO_PAULIS[1:])

    ideal = _transfer(lambda rho: _conjugate(r2 @ r1, rho))
    noisy = _transfer(lambda rho: noise2(_conjugate(r2, noise1(_conjugate(r1, rho)))))
    expected = ideal @ np.linalg.inv(noisy)
    gates = [("ry", 0.67, 0)]
    noise = [("amplitude_damping", 0.2, 0), ("x_error", 0.13, 0)]
    later = [("rx", 0.31, 1), ("cnot", 1, 0)]
    later_noise = [("depolarize2", 0.16, 0, 1)]
    tem = MpoErrorMit(gates + noise + later + later_noise, chi=chi, mode=mode)
    mpo = tem.build_mpo()
    np.testing.assert_allclose(mpo.to_dense(), expected, atol=2e-13)
    assert chi is None or mpo.max_bond() <= chi

    shots = PauliShots(
        ["ZY", "ZX", "YX", "ZZ", "ZZ"],
        ["01", "10", "11", "00", "11"],
        counts=[3, 1, 6, 2, 5],
        basis_probs=[[0.2, 0.3, 0.5], [0.4, 0.35, 0.25]],
    )
    observed = tem.estimate(["ZI", "ZZ", "YZ", "II"], shots, chunk_size=2)
    duals = []
    for bases, bits in zip(shots.bases, shots.bits, strict=True):
        factors = [
            (identity + (1 - 2 * bit) * _PAULIS[axis + 1] / shots.basis_probs[site, axis]) / 2
            for site, (axis, bit) in enumerate(zip(bases, bits, strict=True))
        ]
        duals.append(np.einsum("aij,ji->a", _TWO_PAULIS, np.kron(*factors)).real)
    empirical_duals = np.repeat(duals, shots.counts, axis=0)
    for word, value in observed.items():
        axis = 4 * "IXYZ".index(word[0]) + "IXYZ".index(word[1])
        raw = empirical_duals[:, axis]
        scores = empirical_duals @ expected[axis]
        assert value.raw == pytest.approx(raw.mean())
        assert value.mitigated == pytest.approx(scores.mean())
        assert value.raw_stderr == pytest.approx(raw.std(ddof=1) / np.sqrt(shots.shots))
        assert value.mitigated_stderr == pytest.approx(scores.std(ddof=1) / np.sqrt(shots.shots))
    assert observed["II"].mitigated == pytest.approx(1)


@pytest.mark.parametrize("strategy,mps_mode", [
    ("auto", "direct"), ("coalesced", "dmrg2"), ("independent", "exact"),
])
def test_collect_shots_uses_seeded_mps_trajectories_before_building_mpo(strategy, mps_mode):
    tem = MpoErrorMit([("ry", 0.3, 0), ("depolarize1", 0.2, 0)], n_qubits=1)
    state = qtn.MPS_computational_state("0")
    kwargs = dict(seed=10, strategy=strategy, mps_mode=mps_mode, batch_size=11)
    shots = tem.collect_shots(state, 32, **kwargs)
    repeat = tem.collect_shots(state, 32, **kwargs)
    assert "_mitigation" not in vars(tem)
    assert shots.shots == 32
    for key in ("bases", "bits", "counts"):
        np.testing.assert_array_equal(getattr(shots, key), getattr(repeat, key))
    assert tem.estimate("I", shots)["I"].mitigated == pytest.approx(1)
    np.testing.assert_allclose(state.to_dense().ravel(), [1, 0], atol=1e-13)


def test_reuses_mitigation_map_for_new_observables_and_shots(monkeypatch):
    from pepsy.optimizers.mpo import error_mitigation

    fits = []
    fit = error_mitigation._fit_lazy_mpo_target

    def count_fit(*args, **kwargs):
        fits.append(kwargs["chi"])
        return fit(*args, **kwargs)

    monkeypatch.setattr(error_mitigation, "_fit_lazy_mpo_target", count_fit)
    tem = MpoErrorMit(layers=[
        ([("ry", 0.67, 0), ("rx", 0.25, 1)], [("depolarize1", 0.11, 0)]),
        ([("ry", 0.54, 1), ("rzz", 0.73, 0, 1)], [("depolarize2", 0.14, 0, 1)]),
    ], chi=16)
    shots = tem.collect_shots(qtn.MPS_computational_state("00"), 8192, seed=7, strategy="coalesced")
    assert fits == []
    public_map = tem.build_mpo()
    assert fits == [16, 16]
    public_map[0].data[...] = 0
    value = tem.estimate("ZZ", shots)["ZZ"]
    ideal = np.cos(0.67) * np.cos(0.25) * np.cos(0.54)
    assert abs(value.raw - ideal) > 0.1
    assert abs(value.mitigated - ideal) < 3 * value.mitigated_stderr
    another_batch = PauliShots(["XY", "ZZ"], ["00", "01"])
    assert tem.estimate("II", another_batch)["II"].mitigated == pytest.approx(1)
    assert fits == [16, 16]


@pytest.mark.parametrize("gate,dtype", [
    (np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.float32),
    (np.asarray(qu.Rx(0.67)), np.complex64),
])
def test_explicit_single_precision_gates_respect_unitarity_at_input_precision(gate, dtype):
    noise = ("amplitude_damping", 0.2, 0)
    rounded_gate = gate.astype(dtype)
    expected = MpoErrorMit([noise, (gate, 0)], chi=None).build_mpo().to_dense()
    actual = MpoErrorMit([noise, (rounded_gate, 0)], chi=None).build_mpo().to_dense()
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)

    with pytest.raises(ValueError, match="require unitary gates"):
        MpoErrorMit([noise, (0.99 * rounded_gate, 0)])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["dmrg", "dmrg2"])
def test_torch_mitigation_and_pauli_readout_match_numpy(device, mode, monkeypatch):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from pepsy._internal.quimb import quimb_1d_compression_method_supports_seed
    from pepsy.backends import backend_torch
    from pepsy.operators import mpo_product

    stream = [("ry", 0.62, 0), ("cnot", 0, 1), ("depolarize2", 0.12, 0, 1), ("rx", 0.31, 1)]
    reference = MpoErrorMit(stream, chi=None)
    tem = MpoErrorMit(stream, chi=16, mode=mode, to_backend=backend_torch(device, torch.float64))
    state = qtn.MPS_computational_state("00")
    state.apply_to_arrays(backend_torch(device, torch.complex64))
    shots = tem.collect_shots(state, 24, seed=14, strategy="coalesced", mps_mode="perm")
    assert shots.shots == 24

    if not quimb_1d_compression_method_supports_seed("src"):
        with pytest.raises(NotImplementedError, match="needs a newer build"):
            tem.build_mpo()
        return

    compress = mpo_product.run_seeded_quimb
    seeds = []

    def record_src(seed, function, target, **kwargs):
        assert kwargs["method"] == "src"
        guess = compress(seed, function, target, **kwargs)
        assert all(array.device.type == device for array in guess.arrays)
        seeds.append(seed)
        return guess

    monkeypatch.setattr(mpo_product, "run_seeded_quimb", record_src)
    mpo = tem.build_mpo()
    assert seeds == [0, 1]
    assert all(array.device.type == device for array in mpo.arrays)
    np.testing.assert_allclose(mpo.to_dense().cpu().numpy(), reference.build_mpo().to_dense(), atol=1e-13)
    expected = reference.estimate(["II", "ZI", "YZ"], shots, chunk_size=4)
    actual = tem.estimate(expected, shots, chunk_size=4)
    for word in expected:
        assert vars(actual[word]) == pytest.approx(vars(expected[word]), abs=1e-13)


def test_rejects_singular_channels_and_invalid_readouts():
    with pytest.raises(ValueError, match="singular"):
        MpoErrorMit([("depolarize1", 0.75, 0)])
    with pytest.raises(ValueError, match="basis_probs"):
        PauliShots(["X"], ["1"], basis_probs=[0.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="bits"):
        PauliShots(["XYZ"], ["012"])


@pytest.mark.parametrize("mps_mode", ["perm", "exact", "exact-batch"])
def test_trajectory_readout_preserves_logical_qubit_order(mps_mode):
    tem = MpoErrorMit([("cnot", 0, 2)], n_qubits=3)
    shots = tem.collect_shots(
        qtn.MPS_computational_state("100"), 32, mps_mode=mps_mode,
        seed=11, basis_probs=(0.1, 0.1, 0.8),
    )
    on_z = shots.bases == 2
    assert on_z.any(axis=0).all()
    expected = np.broadcast_to([1, 0, 1], shots.bits.shape)
    np.testing.assert_array_equal(shots.bits[on_z], expected[on_z])
