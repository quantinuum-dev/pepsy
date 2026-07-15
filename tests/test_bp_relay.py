"""Tests for pepsy.bp relay-BP (disordered-memory 1-norm belief propagation)."""

from __future__ import annotations

import numpy as np
import pepsy as py
import pytest
from pathlib import Path
import runpy

qtn = pytest.importorskip("quimb.tensor")

from pepsy.bp import RelayBPResult, one_norm_bp, relay_bp  # noqa: E402
from pepsy.bp.relay import _relay_message_sources  # noqa: E402


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


def _scalar_three_site_chain():
    """A D1BP chain whose middle tensor sends on two different bonds."""
    return qtn.TensorNetwork(
        [
            qtn.Tensor(data=np.array([1.0, 2.0]), inds=("left",)),
            qtn.Tensor(data=np.ones((2, 2)), inds=("left", "right")),
            qtn.Tensor(data=np.array([3.0, 5.0]), inds=("right",)),
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


def test_top_level_one_norm_bp_runs_d1bp():
    result = py.one_norm_bp(_scalar_two_site_tree(), method="d1bp", tol=1e-12)

    assert result.converged
    assert np.isclose(float(result.contract()), 13.0)


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


def test_d1bp_relay_groups_disorder_by_source_tensor_not_bond():
    from quimb.tensor.belief_propagation import D1BP

    bp = D1BP(_scalar_three_site_chain())
    sources = _relay_message_sources(bp, "d1bp")
    left_tids = set(bp.tn.ind_map["left"])
    right_tids = set(bp.tn.ind_map["right"])
    (middle,) = left_tids & right_tids
    (left,) = left_tids - {middle}
    (right,) = right_tids - {middle}

    # These two directed messages leave the same middle tensor. Their relay
    # memory strength must therefore be shared even though their bonds differ.
    assert sources["left", left] == middle
    assert sources["right", right] == middle


def test_hv1bp_plain_api_snapshots_and_warm_starts():
    first = one_norm_bp(_ising_tn(3, 0.25), method="hv1bp", tol=1e-10)
    assert first.converged
    messages = first.snapshot()
    assert isinstance(messages, tuple) and len(messages) == 2

    second = one_norm_bp(
        _ising_tn(3, 0.25),
        method="hv1bp",
        init_messages=messages,
        tol=1e-10,
    )
    assert second.converged
    assert np.isclose(float(second.contract()), float(first.contract()))


def test_relay_rejects_hv1bp_and_invalid_controls():
    tn = _ising_tn(3)
    with pytest.raises(ValueError, match="max_iterations"):
        one_norm_bp(tn, max_iterations=0)
    with pytest.raises(ValueError, match="only 'l1bp' and 'd1bp'"):
        relay_bp(tn, method="hv1bp")
    with pytest.raises(ValueError, match="positive integer"):
        relay_bp(tn, num_relays=0)
    with pytest.raises(ValueError, match="max < 1"):
        relay_bp(tn, gamma_range=(0.0, 1.0))


def test_d1bp_rejects_open_tensor_networks_cleanly():
    open_tn = qtn.TensorNetwork([qtn.Tensor(np.array([1.0, 2.0]), inds=("x",))])
    with pytest.raises(ValueError, match="closed graph"):
        one_norm_bp(open_tn, method="d1bp")
    with pytest.raises(ValueError, match="closed graph"):
        relay_bp(open_tn, method="d1bp")


def test_warm_start_rejects_a_different_message_topology():
    messages = one_norm_bp(_ising_tn(3), tol=1e-10).snapshot()
    with pytest.raises(ValueError, match="topology"):
        one_norm_bp(_ising_tn(4), init_messages=messages, tol=1e-10)


def test_parallel_update_runs():
    tn = _ising_tn(3, 0.3)
    res = relay_bp(tn, num_relays=2, update="parallel", seed=0)
    assert np.isfinite(float(res.contract()))


def test_relay_d1bp_odd_cycle_stress_cases_converge_strictly():
    """Relay damps deterministic parallel D1BP stalls on odd parity cycles."""
    example_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "RelayBP"
        / "odd_cycle_stress.py"
    )
    records = runpy.run_path(str(example_path))["run_stress_cases"]()

    assert len(records) == 2
    for record in records:
        assert record["exact"] > 0.0
        assert record["plain_converged"] is False
        assert record["plain_max_mdiff"] > 1e-3
        assert record["relay_converged"] is True
        assert record["relay_num_legs"] == 5
        assert record["relay_iterations"] < 100
        assert record["relay_max_mdiff"] < 1e-10
        # The exact reference measures the uncontrolled loopy-BP error;
        # convergence alone intentionally makes no accuracy guarantee.
        assert np.isfinite(record["relay_relative_error"])
