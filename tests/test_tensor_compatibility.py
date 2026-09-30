"""Legacy tensor hooks remain scoped to compatibility entry points."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.tensors import contractions, core, observables

pytestmark = pytest.mark.core


@pytest.mark.parametrize("fail", [False, True])
def test_legacy_fidelity_hook_is_restored(monkeypatch, fail):
    """Legacy hooks affect that call and cannot leak into canonical callers."""
    original = observables.build_optimizer
    calls = []

    def optimizer_hook(**kwargs):
        calls.append(kwargs)
        if fail:
            raise RuntimeError("legacy optimizer unavailable")
        return "greedy"

    monkeypatch.setattr(core, "build_optimizer", optimizer_hook)
    state = qtn.MPS_rand_state(3, 2, dtype="complex128", seed=12)
    reference = qtn.MPS_rand_state(3, 2, dtype="complex128", seed=13)
    left, right = state.to_dense().ravel(), reference.to_dense().ravel()
    expected = abs(np.vdot(left, right)) ** 2 / (
        np.vdot(left, left).real * np.vdot(right, right).real
    )
    if fail:
        with pytest.raises(RuntimeError, match="legacy optimizer unavailable"):
            core.tn_fidelity(state, reference)
    else:
        assert core.tn_fidelity(state, reference) == pytest.approx(expected)

    assert calls == [{"progbar": False}]
    assert observables.build_optimizer is original
    assert observables.tn_fidelity(
        state, reference, contraction_opt="greedy"
    ) == pytest.approx(expected)


@pytest.mark.parametrize("name", ["build_optimizer", "build_compressed_optimizer"])
@pytest.mark.parametrize("fail", [False, True])
def test_legacy_acceleration_hook_is_restored(monkeypatch, name, fail):
    """Compatibility builders restore their temporary hooks even on failure."""
    original = contractions._ensure_cotengrust
    calls = []

    def acceleration_hook():
        calls.append(True)
        if fail:
            raise RuntimeError("legacy acceleration unavailable")
        return None

    monkeypatch.setattr(core, "_ensure_cotengrust", acceleration_hook)
    build = getattr(core, name)
    options = {"max_time": 0, "max_repeats": 1, "progbar": False}
    if name == "build_optimizer":
        options.update(parallel=False, optlib="sbplx")
    if fail:
        with pytest.raises(RuntimeError, match="legacy acceleration unavailable"):
            build(**options)
    else:
        assert build(**options) is not None

    assert calls == [True]
    assert contractions._ensure_cotengrust is original
