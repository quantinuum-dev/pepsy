"""Serialization and subclass contracts for the extracted state wrappers."""

import pickle

import numpy as np
import pytest

from pepsy.tensors import SymMPS, SymPEPS

pytestmark = [pytest.mark.core, pytest.mark.optional, pytest.mark.symmetry]
pytest.importorskip("symmray")


@pytest.mark.parametrize("state_type, shape", [(SymMPS, (2,)), (SymPEPS, (2, 2))])
@pytest.mark.parametrize("model", ["heisenberg", "fermi_hubbard_spinless"])
def test_legacy_state_pickle_retains_native_state_and_hamiltonian(
    monkeypatch, state_type, shape, model,
):
    state = state_type.for_model(
        model, *shape, bond_dim=1, seed=17, contraction_opt="greedy",
    )
    # Materialize the user-selectable charge mapping rather than serializing
    # a local closure returned by the default charge-pattern factory.
    state.site_charge = state.site_charges()
    state.build_hamiltonian()
    state.gauges = {"saved": np.array([1.0])}
    with monkeypatch.context() as patch:
        # Reproduce the class path written by releases before extraction.
        patch.setattr(state_type, "__module__", "pepsy.tensors.symmetric")
        payload = pickle.dumps(state)

    restored = pickle.loads(payload)
    assert type(restored) is state_type
    assert restored.overall_charge() == state.overall_charge()
    assert restored.fermionic == state.fermionic
    assert restored.site_ind_id == state.site_ind_id
    assert restored.phys_sectors == state.phys_sectors
    assert restored.hamiltonian.model == state.hamiltonian.model
    assert restored.hamiltonian.edges == state.hamiltonian.edges
    np.testing.assert_allclose(
        restored.psi.to_dense().to_dense(), state.psi.to_dense().to_dense(),
    )
    np.testing.assert_allclose(restored.gauges["saved"], state.gauges["saved"])
    for left, right in zip(restored.psi.tensors, state.psi.tensors):
        assert type(left.data) is type(right.data)
        assert left.inds == right.inds
        assert left.tags == right.tags
        assert left.data.charge == right.data.charge
        assert left.data.duals == right.data.duals


def test_symmetric_state_copy_preserves_subclass_and_configuration():
    class CustomMPS(SymMPS):
        def norm(self, **kwargs):
            return 2 * super().norm(**kwargs)

    source = SymMPS.for_model("heisenberg", 2, bond_dim=1, seed=4)
    state = CustomMPS(
        psi=source.psi, symmetry=source.symmetry, edges=source.edges,
        contraction_opt="greedy", phys_sectors=source.phys_sectors,
        site_charge=source.site_charges(), gauges={"bond": 1.0},
    )
    clone = state.copy()
    assert type(clone) is CustomMPS
    assert clone.psi is not state.psi
    assert clone.phys_sectors == state.phys_sectors
    assert clone.phys_sectors is not state.phys_sectors
    assert clone.gauges == state.gauges
    assert clone.gauges is not state.gauges
    assert clone.overall_charge() == state.overall_charge()
    assert clone.norm() == pytest.approx(state.norm())
