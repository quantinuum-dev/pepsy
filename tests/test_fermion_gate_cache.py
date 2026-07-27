"""Regression tests for content-addressed ``Fermion.operator_gate`` caching.

``operator_gate`` memoises gate exponentials.  Historically the cache key for a
raw (already-built) operator used ``id(operator)``.  A freshly built operator
can be garbage collected and have its memory address reused, so a later,
unrelated operator could alias a stale cache entry and return the wrong gate.
These tests pin the content-addressed behaviour that fixes that.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from pepsy.tensors.symmetric import (
    Fermion,
    _gate_from_term,
    _operator_content_fingerprint,
)


def _dense(gate):
    return np.asarray(gate.to_dense())


def test_fingerprint_is_content_addressed_not_identity():
    """Equal contents share a fingerprint; different contents do not."""
    fermion = Fermion(spinful=True, symmetry="U1", t=1.0, U=8.0)

    up_a = fermion.hopping_operator(spin="up")
    up_b = fermion.hopping_operator(spin="up")
    down = fermion.hopping_operator(spin="down")
    inter = fermion.interaction_operator()

    # Freshly built objects (distinct ids) with identical contents must agree.
    assert up_a is not up_b
    assert _operator_content_fingerprint(up_a) == _operator_content_fingerprint(up_b)

    # Different operators must produce different fingerprints.
    assert _operator_content_fingerprint(up_a) != _operator_content_fingerprint(down)
    assert _operator_content_fingerprint(up_a) != _operator_content_fingerprint(inter)

    # A non-operator object cannot be fingerprinted and must not be cached.
    assert _operator_content_fingerprint(object()) is None


def test_operator_gate_does_not_alias_distinct_operators_under_id_reuse():
    """Interleaving fresh operators at a fixed angle must stay correct.

    This reproduces the real failure mode: building many short-lived operators
    of different kinds at the same ``theta`` used to let a recycled ``id`` return
    a stale cached gate.  Every gate must match an independent exponential.
    """
    fermion = Fermion(spinful=True, symmetry="U1", t=1.0, U=8.0)
    theta = 0.10667747

    references = {
        "up": _dense(_gate_from_term(fermion.hopping_operator(spin="up"), theta)),
        "down": _dense(_gate_from_term(fermion.hopping_operator(spin="down"), theta)),
        "inter": _dense(_gate_from_term(fermion.interaction_operator(), theta)),
    }

    for _ in range(64):
        for kind, ref in references.items():
            if kind == "up":
                operator = fermion.hopping_operator(spin="up")
            elif kind == "down":
                operator = fermion.hopping_operator(spin="down")
            else:
                operator = fermion.interaction_operator()
            gate = _dense(fermion.operator_gate(operator, theta))
            np.testing.assert_allclose(gate, ref, atol=1e-12)
            del operator
        gc.collect()  # encourage id recycling between iterations


def test_operator_gate_cache_hits_return_equivalent_gate():
    """Repeated calls with equal contents reuse a single correct gate."""
    fermion = Fermion(spinful=True, symmetry="U1", t=1.0, U=8.0)
    theta = 0.37

    reference = _dense(_gate_from_term(fermion.hopping_operator(spin="up"), theta))
    first = fermion.operator_gate(fermion.hopping_operator(spin="up"), theta)
    second = fermion.operator_gate(fermion.hopping_operator(spin="up"), theta)

    # Content-addressed cache: equal contents collapse to one stored gate.
    assert first is second
    np.testing.assert_allclose(_dense(first), reference, atol=1e-12)


def test_operator_gate_skips_cache_for_autodiff_operators():
    """Blocks that carry an autodiff graph must not be fingerprinted/cached."""
    torch = pytest.importorskip("torch")

    class _StubOperator:
        symmetry = "U1"
        charge = 0
        duals = (False, True)
        shape = (1, 1)

        def __init__(self, blocks):
            self.blocks = blocks

    grad_tracked = _StubOperator(
        {(0, 0): torch.zeros((1, 1), dtype=torch.complex128, requires_grad=True)}
    )
    assert _operator_content_fingerprint(grad_tracked) is None

    plain = _StubOperator(
        {(0, 0): torch.zeros((1, 1), dtype=torch.complex128)}
    )
    assert _operator_content_fingerprint(plain) is not None
