"""Serialization compatibility across responsibility-based module moves."""

import pickle

import numpy as np
import pytest


pytestmark = pytest.mark.core


def test_legacy_mps_stream_snapshot_restores_an_independent_cache(monkeypatch):
    """Old snapshots resolve the moved class and recreate its private lock."""
    from pepsy.optimizers.mps import optimizer
    from pepsy.optimizers.mps._streams import _MpsStreamPlan

    plan = _MpsStreamPlan(entries=(("H", 0),), event_types=("gate",),
                          has_trajectory_events=False)
    source = object()
    plan.get_or_create_backend_payload("sample", source, lambda: np.eye(2))
    assert optimizer._MpsStreamPlan is _MpsStreamPlan
    with monkeypatch.context() as patch:
        patch.setattr(_MpsStreamPlan, "__module__", optimizer.__name__)
        legacy = pickle.dumps(plan)

    restored = pickle.loads(legacy)
    assert type(restored) is _MpsStreamPlan
    assert restored == plan
    assert restored._backend_cache == {}
    assert restored._backend_cache_lock is not plan._backend_cache_lock
    calls = []

    def factory():
        calls.append(True)
        return np.eye(2)

    payload = restored.get_or_create_backend_payload("sample", source, factory)
    assert restored.get_or_create_backend_payload("sample", source, factory) is payload
    assert calls == [True]


@pytest.mark.optional
@pytest.mark.symmetry
@pytest.mark.parametrize("spinful", [False, True])
def test_legacy_fermion_model_roundtrip_preserves_native_cache(monkeypatch, spinful):
    """Existing checkpoints retain model metadata and native cached arrays."""
    pytest.importorskip("symmray")
    from pepsy.tensors import symmetric
    from pepsy.tensors.symm_fermions import Fermion

    model = Fermion(spinful=spinful, dtype="complex64")
    number = model.observable("number")
    gate = model.hopping_gate(dt=0.02, t=0.5)
    with monkeypatch.context() as patch:
        patch.setattr(Fermion, "__module__", symmetric.__name__)
        legacy = pickle.dumps(model)

    restored = pickle.loads(legacy)
    assert type(restored) is Fermion
    assert restored.spinful == spinful
    assert restored.symmetry == model.symmetry
    assert restored.dtype == model.dtype
    for old, new in ((number, restored.observable("number")),
                     (gate, restored.hopping_gate(dt=0.02, t=0.5))):
        assert type(new) is type(old)
        assert new.charge == old.charge
        assert new.duals == old.duals
        np.testing.assert_array_equal(new.to_dense(), old.to_dense())
    assert restored.observable("number") is restored.observable("number")
