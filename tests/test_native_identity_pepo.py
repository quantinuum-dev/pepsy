"""Native fermionic identity-PEPO construction checks."""

import numpy as np
import pytest

import pepsy


@pytest.mark.parametrize(
    ("spinful", "symmetry"),
    [
        (False, "U1"),
        (False, "Z2"),
        (True, "U1"),
        (True, "U1U1"),
        (True, "Z2"),
        (True, "Z2Z2"),
    ],
)
def test_native_id_to_pepo_is_identity_on_half_filled_state(spinful, symmetry):
    """The safe native identity handles every advertised fermion space."""
    pytest.importorskip("symmray")

    fermion = pepsy.Fermion(
        spinful=spinful,
        symmetry=symmetry,
        dtype="complex128",
    )
    operator = pepsy.id_to_pepo(
        (2, 3),
        fermion=fermion,
        cyclic=True,
    )
    state = pepsy.ps_to_peps(
        (2, 3),
        fermion=fermion,
        dtype="complex128",
        seed=17,
    )

    assert all(
        type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in operator
    )
    acted = operator.apply(state, contract=True, compress=False)
    numerator = complex(
        np.asarray((state.H & acted).contract(all, optimize="auto-hq")).item()
    )
    denominator = complex(
        np.asarray((state.H & state).contract(all, optimize="auto-hq")).item()
    )
    assert numerator / denominator == pytest.approx(1.0, abs=1.0e-10)


def test_native_id_to_pepo_rejects_state_sector_controls():
    """A full identity must not be reduced to a half-filled sector."""
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=True, symmetry="U1")

    with pytest.raises(ValueError, match="full local identity"):
        pepsy.id_to_pepo(
            2,
            3,
            fermion=fermion,
            occupations=[1] * 6,
        )


def test_fermion_to_pepo_identity_delegates_to_safe_constructor(monkeypatch):
    """The model API shares the same native identity implementation."""
    pytest.importorskip("symmray")
    from pepsy.tensors import constructors

    fermion = pepsy.Fermion(spinful=True, symmetry="U1")
    called = {}

    def fake_identity(model, lx, ly, **options):
        called.update(model=model, lx=lx, ly=ly, options=options)
        return "native-identity"

    monkeypatch.setattr(
        constructors,
        "_native_fermion_identity_pepo",
        fake_identity,
    )
    result = fermion.to_pepo(
        {((0, 0),): fermion.observable("identity")},
        Lx=2,
        Ly=3,
        cyclic=True,
    )

    assert result == "native-identity"
    assert called["model"] is fermion
    assert (called["lx"], called["ly"]) == (2, 3)
    assert called["options"]["cyclic"] is True
