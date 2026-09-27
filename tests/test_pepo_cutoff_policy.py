"""Public PEPO construction honors its requested truncation policy."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.operators import build_pepo_from_gates
from pepsy.tensors import contract_hypercompressed_tn


def test_pepo_builder_absolute_and_relative_cutoffs_differ_numerically():
    z = np.diag([1., -1.])
    gate = 10 * np.eye(4) + np.kron(z, z)
    kwargs = {"where": ((0, 0), (1, 0)), "cutoff": .15, "max_bond": 4}
    absolute = build_pepo_from_gates(gate, cutoff_mode="abs", **kwargs)
    relative = build_pepo_from_gates(gate, cutoff_mode="rel", **kwargs)
    assert absolute.max_bond() == 2
    assert relative.max_bond() == 1
    np.testing.assert_allclose(absolute.to_dense(), gate, atol=1e-12)
    np.testing.assert_allclose(relative.to_dense(), 10 * np.eye(4), atol=1e-12)


def test_pepo_builder_safety_compression_uses_requested_cutoff(monkeypatch):
    target = qtn.PEPO.rand(2, 1, bond_dim=4, dtype="complex128", seed=351)
    original = qtn.PEPO.compress_all
    calls = []

    def record(self, *args, **kwargs):
        calls.append((kwargs["cutoff"], kwargs["cutoff_mode"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.PEPO, "compress_all", record)
    result = build_pepo_from_gates(
        np.eye(2), where=((0, 0),), pepo_=target, max_bond=1,
        cutoff=.02, cutoff_mode="abs",
    )
    assert result.max_bond() == 1
    assert calls == [(.02, "abs")]


@pytest.mark.parametrize("cutoff_mode", [None, "abs", "rsum2"])
def test_hypercompression_forwards_cutoff_policy(monkeypatch, cutoff_mode):
    network = qtn.MPS_rand_state(5, bond_dim=2, seed=157)
    original = qtn.TensorNetwork.contract_compressed_
    calls = []

    def record(self, *args, **kwargs):
        calls.append(kwargs["compress_opts"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.TensorNetwork, "contract_compressed_", record)
    result = contract_hypercompressed_tn(
        network, "greedy", max_bond=4, cutoff_mode=cutoff_mode,
        do_full_simplify=False, output_inds=network.site_inds,
    )
    np.testing.assert_allclose(
        result.to_dense(network.site_inds).reshape(-1), network.to_dense().reshape(-1),
        atol=1e-12,
    )
    assert calls == [{} if cutoff_mode is None else {"cutoff_mode": cutoff_mode}]
