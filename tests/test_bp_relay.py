"""Tests for pepsy.bp relay-BP (disordered-memory 1-norm belief propagation)."""

from __future__ import annotations

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from pepsy.bp import RelayBPResult, one_norm_bp, relay_bp  # noqa: E402


def _ising_tn(length: int = 3, beta: float = 0.3):
    return qtn.TN2D_classical_ising_partition_function(length, length, beta=beta)


def _scalar_two_site_tree():
    """A minimal scalar tree, whose exact contraction is 13."""
    return qtn.TensorNetwork(
        [
            qtn.Tensor(data=np.array([1.0, 2.0]), inds=("bond",)),
            qtn.Tensor(data=np.array([3.0, 5.0]), inds=("bond",)),
        ]
    )


def test_one_norm_bp_close_to_exact_on_small_grid():
    tn = _ising_tn(3, 0.2)
    exact = float(tn.contract())
    res = one_norm_bp(tn, method="l1bp", tol=1e-10)
    assert res.converged
    z_bp = float(res.contract())
    # Loopy BP is a good (not exact) approximation on this weakly-correlated grid.
    assert abs(z_bp - exact) / exact < 0.05


def test_relay_bp_returns_plain_fixed_point_when_easy():
    tn = _ising_tn(3, 0.3)
    plain = float(one_norm_bp(tn, tol=1e-10).contract())
    relay = relay_bp(tn, num_relays=4, gamma_range=(-0.3, 0.9), tol=1e-10, seed=0)
    assert relay.converged
    # First leg is plain BP, so best-of returns the exact plain-BP fixed point.
    assert abs(float(relay.contract()) - plain) / abs(plain) < 1e-8


def test_relay_bp_result_api():
    tn = _ising_tn(3, 0.25)
    res = relay_bp(tn, num_relays=2, seed=1)
    assert isinstance(res, RelayBPResult)
    assert res.num_legs_run == 2
    assert res.messages is res.bp.messages
    assert res.iterations >= 1
    assert np.isfinite(float(res.contract()))


def test_relay_bp_memory_legs_run_and_stay_valid():
    tn = _ising_tn(3, 0.4)
    # Force every leg to use memory; the result must still be a finite estimate.
    res = relay_bp(
        tn, num_relays=3, memory_first_leg=True, gamma_range=(-0.5, 0.8), seed=2
    )
    assert np.isfinite(float(res.contract()))
    assert res.num_legs_run == 3


def test_relay_bp_method_validation():
    tn = _ising_tn(3)
    with pytest.raises(ValueError):
        relay_bp(tn, method="not-a-method")


def test_message_reuse_warm_start_reaches_same_fixed_point():
    tn = _ising_tn(3, 0.3)
    first = relay_bp(tn, num_relays=2, seed=0)
    msgs = first.snapshot()
    assert isinstance(msgs, dict) and len(msgs) == len(first.messages)

    # Warm-start a fresh (identical-topology) run from the previous fixed point.
    second = relay_bp(_ising_tn(3, 0.3), num_relays=1, init_messages=msgs, seed=1)
    assert second.converged
    ref = float(first.contract())
    assert abs(float(second.contract()) - ref) / abs(ref) < 1e-6

    # one_norm_bp also accepts the warm start.
    third = one_norm_bp(_ising_tn(3, 0.3), init_messages=msgs, tol=1e-10)
    assert third.converged


def test_d1bp_message_snapshot_warm_start_and_relay_memory():
    """D1BP uses bare arrays, unlike L1BP's Tensor message objects."""
    first = one_norm_bp(_scalar_two_site_tree(), method="d1bp", tol=1e-12)
    assert first.converged
    msgs = first.snapshot()
    assert all(isinstance(message, np.ndarray) for message in msgs.values())

    second = one_norm_bp(
        _scalar_two_site_tree(), method="d1bp", init_messages=msgs, tol=1e-12
    )
    assert second.converged
    assert np.isclose(float(second.contract()), 13.0)

    # Exercise relay's disordered-memory replacement path for bare arrays.
    relay = relay_bp(
        _scalar_two_site_tree(),
        method="d1bp",
        num_relays=2,
        memory_first_leg=True,
        gamma_range=(0.1, 0.2),
        max_iterations=100,
        tol=1e-10,
        seed=0,
    )
    assert relay.converged
    assert np.isclose(float(relay.contract()), 13.0)


def test_parallel_update_runs():
    tn = _ising_tn(3, 0.3)
    res = relay_bp(tn, num_relays=2, update="parallel", seed=0)
    assert np.isfinite(float(res.contract()))
