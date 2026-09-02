"""Tests for the native Trotter-to-MPO facade."""

import numpy as np
import pytest

import pepsy
import quimb as qu
from pepsy.operators import MPOParameter, MPOProductTerm, TrotterMPOReport, exp_trotter


def _dense_gate_product(mpo, length):
    """Replay the attached native gate schedule as dense left actions."""
    result = np.eye(2**length, dtype=complex)
    for trotter_gate in mpo.pepsy_trotter_gates:
        gate, where = trotter_gate
        result = qu.ikron(gate, [2] * length, inds=where) @ result
    return result


@pytest.mark.parametrize("order", [1, 2, 4])
def test_exp_trotter_is_public_and_replays_quimb_schedule(order):
    """The MPO has the same ordered product as Quimb's native schedule."""
    terms = [
        (("ZZ", 0.7), (0, 1)),
        (("XX", -0.2), (1, 2)),
    ]
    mpo, report = exp_trotter(
        terms,
        -0.03j,
        shape=3,
        order=order,
        steps=2,
        fuse_adjacent=False,
        chi=64,
        cutoff=0.0,
        return_report=True,
    )

    assert exp_trotter is pepsy.operators.exp_trotter
    assert isinstance(report, TrotterMPOReport)
    assert report.gate_count == len(mpo.pepsy_trotter_gates)
    assert report.order == order
    assert report.layers == (((0, 1),), ((1, 2),))
    np.testing.assert_allclose(
        mpo.to_dense(),
        _dense_gate_product(mpo, 3),
        atol=1.0e-12,
    )


def test_exp_trotter_handles_isolated_one_site_terms_exactly():
    """Onsite terms not incident to a pair are still included exactly."""
    mpo = exp_trotter(
        [(("X", 0.3), 0), (("Z", -0.2), 2)],
        -0.04j,
        shape=3,
        chi=8,
        cutoff=0.0,
    )

    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    expected = qu.ikron(
        qu.expm(-0.04j * 0.3 * x),
        [2, 2, 2],
        inds=(0,),
    ) @ qu.ikron(
        qu.expm(0.04j * 0.2 * z),
        [2, 2, 2],
        inds=(2,),
    )
    np.testing.assert_allclose(mpo.to_dense(), expected, atol=1.0e-12)
    assert mpo.pepsy_trotter_report.isolated_sites == (0, 2)


def test_exp_trotter_handles_a_single_site_chain():
    """A one-site MPO uses Quimb's bondless operator layout."""
    mpo = exp_trotter(
        [(("X", 0.3), 0)],
        -0.04j,
        shape=1,
        cutoff=0.0,
    )
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(mpo.to_dense(), qu.expm(-0.04j * 0.3 * x))
    assert mpo.bond_sizes() == []


def test_exp_trotter_accepts_exp_mpo_parameter_and_dt_surface():
    """Parameter binding and the ``dt`` spelling share the MPO parser."""
    mpo = exp_trotter(
        [MPOProductTerm.from_pauli((0, 1), "ZZ", coefficient=MPOParameter("J"))],
        dt=-0.02j,
        shape=2,
        parameters={"J": 0.4},
        order=1,
        steps=2,
        mode="svd",
        chi=8,
        cutoff="auto",
        cutoff_mode="auto",
    )

    assert mpo.pepsy_trotter_report.order == 1
    assert mpo.pepsy_trotter_report.steps == 2
    assert mpo.pepsy_trotter_report.mode == "svd"
    assert mpo.pepsy_trotter_report.cutoff_mode == "rsum2"
    assert max(mpo.bond_sizes()) <= 8


def test_exp_trotter_to_backend_reaches_mpo_and_gate_payloads():
    """The explicit Autoray converter is respected through MPO replay."""
    torch = pytest.importorskip("torch")
    backend = pepsy.backend_torch(dtype=torch.complex128)
    coefficient = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    step = torch.tensor(-0.01j, dtype=torch.complex128, requires_grad=True)

    mpo = exp_trotter(
        [(("ZZ", coefficient), (0, 1))],
        step,
        shape=2,
        chi=8,
        cutoff=0.0,
        to_backend=backend,
    )

    assert all(isinstance(tensor.data, torch.Tensor) for tensor in mpo)
    assert all(isinstance(trotter_gate.U, torch.Tensor) for trotter_gate in mpo.pepsy_trotter_gates)
    loss = sum(tensor.data.real.sum() for tensor in mpo)
    grad_step, grad_coefficient = torch.autograd.grad(loss, (step, coefficient))
    assert torch.isfinite(grad_step)
    assert torch.isfinite(grad_coefficient)


def test_exp_trotter_honors_optimizer_physical_index_formats():
    """Optimizer index customizations also configure the identity MPO."""
    mpo = exp_trotter(
        [(("ZZ", 0.2), (0, 1))],
        0.01,
        shape=2,
        chi=8,
        cutoff=0.0,
        optimizer_kwargs={"ind_id_k": "K{}", "ind_id_b": "B{}"},
    )

    assert set(mpo.outer_inds()) == {"K0", "K1", "B0", "B1"}


def test_exp_trotter_rejects_unsupported_many_site_terms():
    """The native Quimb local Hamiltonian boundary is explicit."""
    with pytest.raises(NotImplementedError, match="one- and two-site"):
        exp_trotter(
            [(("XXX", 0.2), (0, 1, 2))],
            0.01,
            shape=3,
        )


def test_exp_trotter_validates_explicit_ordering_coverage():
    """Explicit layer plans cannot silently drop Hamiltonian terms."""
    terms = [(("ZZ", 0.2), (0, 1)), (("XX", 0.1), (1, 2))]
    with pytest.raises(ValueError, match="every two-site"):
        exp_trotter(
            terms,
            0.01,
            shape=3,
            ordering=(((0, 1),),),
        )
    with pytest.raises(ValueError, match="non-overlapping"):
        exp_trotter(
            terms,
            0.01,
            shape=3,
            ordering=(((0, 1), (1, 2)),),
        )

    mpo = exp_trotter(
        terms,
        0.01,
        shape=3,
        ordering=(((1, 2),), ((0, 1),)),
        cutoff=0.0,
    )
    assert mpo.pepsy_trotter_report.layers == (((1, 2),), ((0, 1),))
