"""A branch cap must not discard sampled histories or replay shared work."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import MpsOptimizer, noise


def test_prefix_runs_once_and_full_record_indices_survive(monkeypatch):
    p = qtn.MPS_computational_state("0000", dtype="complex128")
    stream = [("h", 2), ("cnot", 2, 3)] + [("x_error", .01, 0)] * 4
    sim = MpsOptimizer(p, stream, chi=4)
    original = noise._run_trajectory_entries
    prefix_calls = []

    def count(opt, entries, kwargs, **options):
        if len(entries) == 2:
            prefix_calls.append(opt)
        return original(opt, entries, kwargs, **options)

    monkeypatch.setattr(noise, "_run_trajectory_entries", count)
    result = sim.run(shots=16, seed=1134, max_branches=4,
                     workers=1, progress=False)
    assert result.diagnostics.continued_from_cap
    assert len(prefix_calls) == 1
    assert sum(result.counts) == 16
    for leaf in result.raw.leaves:
        assert [record.event_index for record in leaf.records] == [2, 3, 4, 5]
        parity = sum(record.label == "X" for record in leaf.records) % 2
        expected = np.zeros(16)
        expected[8 * parity] = expected[8 * parity + 3] = 2**-.5
        np.testing.assert_allclose(leaf.optimizer.to_dense().ravel(), expected, atol=1e-12)


@pytest.mark.parametrize("retain", ["all", "final", "none"])
def test_continuation_preserves_controls_and_retention(retain):
    p = qtn.MPS_computational_state("00", dtype="complex128")
    stream = [("h", 0), ("cnot", 0, 1), ("measure", "Z", 0),
              ("if", -1, 1, ("x", 1)), ("reset", 0)]
    result = noise.run_trajectory_shots(
        lambda: MpsOptimizer(p.copy(), chi=2), stream, 64, seed=41,
        strategy="coalesced", max_branches=1, _continue_on_cap=True, retain=retain,
    )
    assert result.diagnostics.continued_from_cap
    assert result.shots == 64
    if retain == "none":
        assert result.leaves == ()
    else:
        assert len(result.leaves) == 64
        for leaf in result.leaves:
            np.testing.assert_allclose(abs(leaf.optimizer.to_dense().ravel()),
                                       [1., 0., 0., 0.], atol=1e-12)


def test_importance_weights_and_sampling_law_survive_continuation():
    p = qtn.MPS_computational_state("0", dtype="complex128")
    result = noise.run_trajectory_shots(
        lambda: MpsOptimizer(p.copy(), chi=2), [("x_error", .1, 0)] * 2,
        1500, seed=56, strategy="coalesced", max_branches=2,
        importance_sampling={"I": .5, "X": .5}, _continue_on_cap=True,
    )
    assert result.diagnostics.continued_from_cap
    weighted_one = 0.
    for leaf in result.leaves:
        expected_weight = np.prod([record.likelihood_ratio for record in leaf.records])
        assert leaf.weight == pytest.approx(expected_weight)
        parity = sum(record.label == "X" for record in leaf.records) % 2
        weighted_one += leaf.count * leaf.weight * parity
    # Exact two-bit-flip probability. This is a statistical observable check,
    # independent of the implementation's choice of seeded draw ordering.
    assert weighted_one / result.shots == pytest.approx(.18, abs=.025)


@pytest.mark.parametrize("workers", [1, 2])
def test_pauli_macro_continues_and_preserves_fault_history(workers):
    p = qtn.MPS_computational_state("0", dtype="complex128")
    sim = MpsOptimizer(p, [("x", 0)] * 4, chi=2)
    result = sim.run(shots=16, seed=1134, workers=workers, progress=False,
                     max_branches=2, error_model=noise.PauliErrorModel(p_x=.01))
    assert result.diagnostics.continued_from_cap
    assert result.shots == 16
    for leaf in result.raw.leaves:
        parity = len(leaf.faults) % 2
        assert all(0 <= fault.gate_index < 4 for fault in leaf.faults)
        np.testing.assert_allclose(abs(leaf.optimizer.to_dense().ravel()),
                                   [1 - parity, parity], atol=1e-12)


def test_continuation_copies_existing_leakage_state():
    p = qtn.MPS_computational_state("0", dtype="complex128")
    node = noise._CoalescedNode(MpsOptimizer(p, chi=2), 8)
    node.leakage_state.leaked.add(0)
    result = noise.run_coalesced_trajectory_shots(
        lambda: pytest.fail("recreated initial state"),
        [("measure_leaked", 0), ("reset", 0), ("x", 0), ("measure_leaked", 0)],
        8, max_branches=2, _continue_on_cap=True, _nodes=[node],
        _rng=np.random.default_rng(94),
    )
    assert result.diagnostics.continued_from_cap
    for leaf in result.leaves:
        assert leaf.leakage_records[0].measurement == 2
        assert leaf.leakage_records[-1].measurement == 1
        assert not leaf.leakage_records[-1].finally_leaked
        np.testing.assert_allclose(abs(leaf.optimizer.to_dense().ravel()), [0, 1], atol=1e-12)
