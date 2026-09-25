"""Tree update modes must apply the same represented operator and scale."""

import numpy as np
import pytest

from pepsy import TreeMPO, TreeOptimizer, TreePlan


def _operator():
    plan = TreePlan.from_order((4, 1, 3, 0, 2), structure="balanced", top_arity=2)
    gate = np.eye(4, dtype=complex)
    gate[np.ix_((0, 3), (0, 3))] = ((0.8, -0.6j), (-0.6j, 0.8))
    return TreeMPO.from_gate(plan, gate, (1, 2))


@pytest.mark.parametrize("mode", [
    "direct", "dm", "src", "sdc", "zipup", "dmrg1", "dmrg2", "dmrg3",
])
@pytest.mark.parametrize("representation", ["exponent", "canonical", "distributed"])
def test_modes_preserve_operator_scale_and_exterior_factors(mode, representation):
    operator = _operator()
    reference = operator.to_dense()[:, 0]
    if representation == "exponent":
        operator.exponent = 400.
    elif representation == "canonical":
        operator.canonicalize()
    else:
        operator.multiply_each_(1.1)
        reference *= 1.1 ** operator.num_tensors
    opt = TreeOptimizer(
        None, tree=operator.plan, chi=32, mode=mode, cutoff=0.,
        fit_rtol=None, compression_seed=4, run=False,
    )
    opt.tn.exponent = 3.
    opt.apply_sub_mpotree(operator, operator.operator_support, track_norm=False)
    assert opt.tn.exponent == (403. if representation == "exponent" else 3.)
    # Compare working amplitudes without materializing 10**403.
    work = opt.tn.copy()
    work.exponent = 0.
    np.testing.assert_allclose(work.to_dense().reshape(-1), reference, atol=1e-11)
    assert opt.tn.is_canonical_form(opt.center)


def test_scaled_operator_copies_readouts_and_arithmetic_preserve_exponent():
    operator = _operator()
    reference = operator.to_dense()
    operator.exponent = 2.
    state = TreeOptimizer(None, tree=operator.plan, run=False)
    for copied, expected in (
        (operator, reference), (operator.copy(), reference),
        (TreeMPO(operator), reference), (operator.conj(), reference.conj()),
        (operator.H, reference.conj().T),
    ):
        assert copied.exponent == 2.
        assert copied.expectation(state) == pytest.approx(100 * reference[0, 0])
        assert copied.amplitude((0,) * 5) == pytest.approx(100 * reference[0, 0])
        np.testing.assert_allclose(copied.to_dense(), 100 * expected, atol=1e-11)
    other = _operator()
    other.exponent = 1.
    np.testing.assert_allclose((operator + other).to_dense(), 110 * reference, atol=1e-11)
    np.testing.assert_allclose((operator - other).to_dense(), 90 * reference, atol=1e-11)
    np.testing.assert_allclose((operator @ other).to_dense(), 1000 * reference @ reference, atol=1e-11)
    np.testing.assert_allclose((operator * 2).to_dense(), 200 * reference, atol=1e-11)
    np.testing.assert_allclose(operator.to_dense(), 100 * reference, atol=1e-11)


def test_identity_shortcut_survives_copies_but_not_exterior_edits():
    operator = _operator()
    assert operator._identity_exterior_unchanged()
    for copied in (operator.copy(deep=True), TreeMPO(operator), operator.H, operator * 2):
        assert copied._identity_exterior_unchanged()
    node, _, _ = operator._identity_exterior[0]
    tensor = operator.node_tensor(node)
    tensor.modify(data=tensor.data * 2)
    assert not operator._identity_exterior_unchanged()
    assert not operator.copy()._identity_exterior_unchanged()
    assert not TreeMPO(operator)._identity_exterior_unchanged()
    assert not operator.conj()._identity_exterior_unchanged()


def test_raw_exterior_edit_requires_explicit_metadata_invalidation():
    operator = _operator()
    node, _, _ = operator._identity_exterior[0]
    operator.node_tensor(node).data[...] *= 2
    operator.invalidate_canonical_form()
    opt = TreeOptimizer(None, tree=operator.plan, chi=32, cutoff=0., run=False)
    opt.apply_sub_mpotree(operator, operator.operator_support, track_norm=False)
    np.testing.assert_allclose(opt.to_dense(), operator.to_dense()[:, 0], atol=1e-11)


def test_rejected_nonfinite_operator_exponent_preserves_state_and_rng():
    operator = _operator()
    opt = TreeOptimizer(None, tree=operator.plan, seed=4, run=False)
    before = opt.to_dense().copy()
    rng = dict(opt.rng.bit_generator.state)
    center = opt.center
    operator.exponent = np.inf
    with pytest.raises(ValueError, match="exponent must be finite"):
        opt.apply_sub_mpotree(operator)
    np.testing.assert_array_equal(opt.to_dense(), before)
    assert opt.center == center and opt.rng.bit_generator.state == rng


@pytest.mark.parametrize("mode", ["direct", "zipup", "dmrg2"])
def test_native_operator_exponent_is_preserved_during_application(mode):
    pytest.importorskip("symmray")
    from pepsy import Fermion, ps_to_ttn

    fermion = Fermion(spinful=True, symmetry="U1", dtype="complex128")
    plan = TreePlan.from_order(range(4), structure="balanced", top_arity=2)
    identity = fermion.operator_term([(1., ())], sites=(0, 3))
    operator = TreeMPO.from_gate(plan, identity, (0, 3), fermionic=True)
    state = ps_to_ttn(
        4, tree=plan, fermion=fermion,
        occupations=((1, 1), (0, 0), (1, 1), (0, 0)),
    )
    opt = TreeOptimizer(None, state=state, mode=mode, chi=16, cutoff=0., run=False)
    reference = opt.to_dense().copy()
    assert reference.dtype != object and reference.shape == (256,)
    assert np.flatnonzero(np.abs(reference) > 1e-12).tolist() == [204]
    np.testing.assert_allclose(
        state.to_statevector((3, 2, 1, 0)),
        reference.reshape((4,) * 4).transpose(3, 2, 1, 0).reshape(-1), atol=1e-12,
    )
    operator.exponent = 2.
    assert operator.expectation(state) == pytest.approx(100.)
    np.testing.assert_allclose(operator.to_dense(), 100 * np.eye(256), atol=1e-11)
    if mode == "direct":
        other = operator.copy()
        other.exponent = 1.
        np.testing.assert_allclose((operator + other).to_dense(), 110 * np.eye(256), atol=1e-11)
    opt.apply_sub_mpotree(operator, track_norm=False)
    assert opt.tn.exponent == 2.
    np.testing.assert_allclose(opt.to_dense(), 100 * reference, atol=1e-10)


@pytest.mark.parametrize("mode", ["direct", "dmrg2"])
def test_external_support_hint_does_not_hide_modified_physical_sites(mode):
    source = _operator()
    network = source.tree_networks[0].copy()
    site = next(q for q in source.sites if q not in source.operator_support)
    tensor = network[source.node_tag(source.plan.node_of_qubit[site])]
    tensor.gate_(np.array([[0., 1.], [1., 0.]], dtype=complex), source.upper_ind(site))
    operator = TreeMPO(source.plan, network, operator_support=source.operator_support)
    assert not operator._identity_exterior_unchanged()
    opt = TreeOptimizer(None, tree=source.plan, mode=mode, chi=32, cutoff=0., run=False)
    opt.apply_sub_mpotree(operator, operator.operator_support, track_norm=False)
    np.testing.assert_allclose(opt.to_dense(), operator.to_dense()[:, 0], atol=1e-11)


def test_sector_network_exponents_keep_relative_scale():
    base = _operator()
    first = base.tree_networks[0].copy()
    second = base.tree_networks[0].copy()
    first.exponent, second.exponent = 1., 2.
    combined = TreeMPO(base.plan, (first, second))
    combined.exponent += 2.
    reference = 11000 * base.to_dense()
    for operator in (combined, combined.copy(), TreeMPO(combined)):
        np.testing.assert_allclose(operator.to_dense(), reference, atol=1e-10)
        assert operator.amplitude((0,) * 5) == pytest.approx(reference[0, 0])
    np.testing.assert_allclose((combined + combined).to_dense(), 2 * reference, atol=1e-10)
    assert first.exponent == 1. and second.exponent == 2.


@pytest.mark.parametrize("backend,mode", [("torch", "src"), ("jax", "dmrg2")])
def test_backend_operator_conversion_keeps_local_identity_proof(backend, mode):
    lib = pytest.importorskip(backend)
    operator = _operator()
    opt = TreeOptimizer(None, tree=operator.plan, mode=mode, chi=32, cutoff=0., run=False)
    if backend == "torch":
        convert = lambda x: lib.as_tensor(x, dtype=lib.complex128)
    else:
        import jax.numpy as jnp

        if not lib.config.x64_enabled:
            pytest.skip("requires JAX x64")
        convert = lambda x: jnp.asarray(x, dtype=jnp.complex128)
    for tensor in opt.tn:
        tensor.modify(data=convert(tensor.data))
    converted = opt._coerce_tree_mpo_backend(operator)
    assert converted._identity_exterior_unchanged()
    converted.exponent = 2.
    opt.apply_sub_mpotree(converted, track_norm=False)
    np.testing.assert_allclose(opt.to_dense(), 100 * operator.to_dense()[:, 0], atol=1e-10)
