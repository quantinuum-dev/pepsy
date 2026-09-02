"""Tests for purified finite-temperature MPS preparation."""

import numpy as np
import pytest
import quimb.tensor as qtn
from scipy.linalg import expm

from pepsy import GibbsMps, bell_to_mps
from pepsy.operators import MPOBasis


def test_bell_to_mps_is_a_quimb_interleaved_identity_purification():
    """The reusable Bell constructor has the same layout as GibbsMps."""
    state = bell_to_mps(2)

    assert state.L == 4
    assert state.cyclic is False
    np.testing.assert_allclose(
        np.asarray(state.partial_trace_to_mpo(keep=(0, 2)).to_dense()),
        np.eye(4) / 4.0,
        atol=1.0e-12,
    )

    cyclic = bell_to_mps(2, cyclic=True)
    assert cyclic.cyclic is True


def test_gibbs_mps_uses_interleaved_bell_pairs_and_traces_ancillas():
    """The beta-zero purification reduces to the normalized identity."""
    state = GibbsMps([(("Z", 0.3), 0)], shape=2)

    assert state.mps.L == 4
    assert state.physical_sites == (0, 2)
    assert state.ancilla_sites == (1, 3)
    assert state.mps.cyclic is False
    assert state.mps.norm() == pytest.approx(1.0)

    dense = np.asarray(state.to_dense())
    np.testing.assert_allclose(dense, np.eye(4) / 4.0, atol=1.0e-12)
    assert state.trace() == pytest.approx(1.0)
    assert state.partition_function() == pytest.approx(4.0)


def test_gibbs_mps_trace_options_keep_the_quimb_mpo_path():
    """Explicit Quimb contraction options do not densify the purification."""
    state = GibbsMps([(("ZZ", 0.3), (0, 1))], shape=2)
    state.prepare(0.2, n_steps=2, cutoff=0.0)

    native = state.raw_mpo
    configured = state.to_mpo(
        normalized=False,
        contract_opts={"optimize": "greedy"},
    )
    np.testing.assert_allclose(
        np.asarray(configured.to_dense()),
        np.asarray(native.to_dense()),
        atol=1.0e-12,
    )
    assert configured.L == state.length


def test_gibbs_mps_uses_quimb_native_trace_with_stored_scale(monkeypatch):
    """Scaled readout stays on Quimb's native partial-trace path."""
    state = GibbsMps([(("ZZ", 0.3), (0, 1))], shape=2)
    state.prepare(0.2, n_steps=1, normalize_every=True, cutoff=0.0)
    calls = []
    native = state.mps.partial_trace_to_mpo

    def traced(*args, **kwargs):
        calls.append((args, kwargs))
        return native(*args, **kwargs)

    monkeypatch.setattr(state.mps, "partial_trace_to_mpo", traced)
    state.to_mpo(normalized=False)

    assert calls
    assert calls[0][1]["keep"] == state.physical_sites


def test_gibbs_mps_reuses_one_resolved_quimb_ordering(monkeypatch):
    """Metadata and execution share the same graph-layer schedule."""
    state = GibbsMps(
        [
            (("ZZ", 0.7), (0, 1)),
            (("XX", 0.2), (1, 2)),
        ],
        shape=3,
    )
    native = qtn.LocalHamGen.get_trotter_gates
    native_ordering = qtn.LocalHamGen.get_auto_ordering
    seen = []
    ordering_calls = []

    def traced(self, x, **kwargs):
        seen.append(kwargs["ordering"])
        return native(self, x, **kwargs)

    def traced_ordering(self, *args, **kwargs):
        ordering_calls.append((args, kwargs))
        return native_ordering(self, *args, **kwargs)

    monkeypatch.setattr(qtn.LocalHamGen, "get_trotter_gates", traced)
    monkeypatch.setattr(qtn.LocalHamGen, "get_auto_ordering", traced_ordering)
    state.prepare(0.2, n_steps=1, cutoff=0.0)
    state.prepare(0.2, n_steps=1, cutoff=0.0)

    assert seen == [state.trotter_layers, state.trotter_layers]
    assert len(ordering_calls) == 1


def test_gibbs_mps_random_metadata_matches_executable_schedule():
    """Randomized layer metadata and gate replay use one draw."""
    state = GibbsMps(
        [
            (("ZZ", 0.7), (0, 1)),
            (("XX", 0.2), (1, 2)),
            (("YY", -0.1), (2, 3)),
            (("ZZ", 0.3), (3, 4)),
        ],
        shape=5,
    )
    state.prepare(
        0.2,
        n_steps=1,
        trotter_order=1,
        trotter_ordering="random",
        trotter_fuse_adjacent=False,
        cutoff=0.0,
    )

    assert all(
        trotter_gate.where in state.trotter_layers[trotter_gate.layer]
        for trotter_gate in state.trotter_gates
    )


def test_gibbs_mps_second_order_trotter_matches_small_exact_reference():
    """Second-order imaginary-time replay converges to the exact Gibbs state."""
    terms = [
        (("ZZ", 0.7), (0, 1)),
        (("X", -0.2), 0),
    ]
    state = GibbsMps(terms, shape=2)
    state.prepare(0.4, n_steps=40, chi=32, mode="mpo", cutoff=0.0)

    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = 0.7 * np.kron(z, z) - 0.2 * np.kron(x, np.eye(2))
    exact = expm(-0.4 * hamiltonian)
    exact /= np.trace(exact)

    np.testing.assert_allclose(
        np.asarray(state.to_dense()),
        exact,
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    assert state.optimizer is not None
    assert state.optimizer.norm_diagnostics()["tracking"] is True
    assert state.optimizer._unitary_previous_norm is None


def test_gibbs_mps_uses_quimb_trotter_schedule_and_physical_mapping():
    """Quimb's schedule metadata survives the purification site mapping."""
    state = GibbsMps(
        [
            (("ZZ", 0.7), (0, 1)),
            (("XX", 0.2), (1, 2)),
        ],
        shape=3,
    )
    state.prepare(
        0.2,
        n_steps=2,
        trotter_order=2,
        trotter_fuse_adjacent=False,
        cutoff=0.0,
    )

    assert state.trotter_layers == (((0, 1),), ((1, 2),))
    assert len(state.trotter_gates) == 6
    assert all(
        hasattr(gate, attribute)
        for gate in state.trotter_gates
        for attribute in ("frac", "layer", "step")
    )
    assert [gate.step for gate in state.trotter_gates] == [0] * 3 + [1] * 3
    assert all(
        where == tuple(2 * site for site in logical_where)
        for (gate, where), logical_where in zip(
            state.gates,
            (trotter_gate.where for trotter_gate in state.trotter_gates),
        )
    )


@pytest.mark.parametrize("order", (1, 4))
def test_gibbs_mps_accepts_quimb_first_and_fourth_order_schedules(order):
    """The public order control follows Quimb's supported product formulas."""
    state = GibbsMps([(("ZZ", 0.2), (0, 1))], shape=2)
    state.prepare(0.1, n_steps=1, trotter_order=order, cutoff=0.0)

    assert state.trotter_order == order
    assert state.trotter_gates
    assert all(gate.where == (0, 1) for gate in state.trotter_gates)


def test_gibbs_mps_applies_disconnected_onsite_terms_exactly():
    """One-site terms outside the interaction graph avoid fake edges."""
    state = GibbsMps(
        [
            (("ZZ", 0.7), (0, 1)),
            (("X", -0.2), 3),
        ],
        shape=4,
    )
    state.prepare(0.4, n_steps=2, cutoff=0.0)

    assert len(state.trotter_gates) == 1
    assert state.gates[-1][1] == (6,)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(
        np.asarray(state.gates[-1][0]),
        expm(0.2 * 0.2 * x),
        atol=1.0e-12,
    )


def test_gibbs_mps_forwards_direct_mps_controls_and_normalization():
    """Common MpsOptimizer controls are available without nested kwargs."""
    state = GibbsMps([(("ZZ", 0.7), (0, 1))], shape=2)
    state.prepare(
        0.2,
        n_steps=2,
        chi=8,
        mode="direct",
        contraction_opt="auto",
        n_iter=1,
        normalize_every=True,
        normalize_final=True,
        cutoff=0.0,
    )

    assert state.optimizer.mode == "quimb-direct"
    assert state.optimizer.get_normalizations()
    assert np.isfinite(float(np.real(state.trace())))


def test_gibbs_mps_tracks_log_partition_function_through_rescaling():
    """Log-Z and normalized readout retain scale-control bookkeeping."""
    terms = [
        (("ZZ", 0.7), (0, 1)),
        (("X", -0.2), 0),
    ]
    state = GibbsMps(terms, shape=2)
    state.prepare(
        0.4,
        n_steps=1,
        chi=32,
        cutoff=0.0,
        normalize_every=True,
    )

    z = 4.170801899580418
    assert state.trace() == pytest.approx(z / 4.0)
    assert state.partition_function() == pytest.approx(z)
    assert state.log_partition_function() == pytest.approx(np.log(z))
    np.testing.assert_allclose(
        np.asarray(state.to_mpo().to_dense()),
        np.asarray(state.raw_mpo.to_dense()) / (z / 4.0),
        atol=1.0e-12,
    )
    assert np.trace(np.asarray(state.to_mpo().to_dense())) == pytest.approx(1.0)


def test_gibbs_mps_accepts_lattice_terms_through_one_d_map():
    """Regular-lattice terms use the same OneDMap traversal as MPOBasis."""
    terms = [
        {"operator": "Z", "location": (0, 0), "coefficient": 0.2},
        {"operator": "ZZ", "location": ((0, 0), (1, 0)), "coefficient": 0.4},
    ]
    state = GibbsMps(terms, shape=(2, 2), map_mode="row-major")

    expected_basis = MPOBasis.from_terms(terms, shape=(2, 2), map_mode="row-major")
    assert state.length == 4
    assert state.mapper.mode == "row-major"
    assert state.basis.lattice_to_chain == expected_basis.lattice_to_chain
    assert state.basis.terms[1].sites == (
        expected_basis.lattice_to_chain[(0, 0)],
        expected_basis.lattice_to_chain[(1, 0)],
    )


def test_gibbs_mps_infers_coordinate_dimension_and_shape():
    """Natural 2D coordinates work without an explicit shape argument."""
    lx, ly = 3, 4
    edges = qtn.edges_2d_square(lx, ly, cyclic=True)
    sites = sorted({site for edge in edges for site in edge})
    terms = [(("zz", 0.3), (u, v)) for (u, v) in edges]
    terms += [(("x", -0.1), site) for site in sites]

    state = GibbsMps(terms, map_mode="row-major")

    assert state.shape == (lx, ly)
    assert state.length == lx * ly
    assert state.basis.lattice_shape == (lx, ly)
    assert state.mapper.mode == "row-major"
    assert state.basis.lattice_to_chain[(1, 0)] == 4


def test_gibbs_mps_infers_triangular_coordinate_graph():
    """Graph geometry comes from the supplied triangular edge list."""
    lx, ly = 2, 3
    edges = qtn.edges_2d_triangular(lx, ly, cyclic=True)
    sites = sorted({site for edge in edges for site in edge})
    terms = [(("zz", 0.3), (u, v)) for (u, v) in edges]
    terms += [(("x", -0.1), site) for site in sites]

    state = GibbsMps(terms, map_mode="snake")
    state.prepare(
        beta=0.05,
        n_steps=1,
        trotter_order=1,
        trotter_fuse_adjacent=False,
        cutoff=0.0,
    )

    assert state.shape == (lx, ly)
    assert state.length == lx * ly
    assert len(state._trotter_ham.terms) == len(set(edges))
    assert state.trotter_gates


def test_gibbs_mps_fuses_connected_onsite_terms_into_pair_terms():
    """Onsite terms are combined before the Quimb Trotter schedule."""
    state = GibbsMps(
        [
            (("ZZ", 0.7), (0, 1)),
            (("X", 0.2), 0),
            (("Z", -0.1), 0),
            (("X", 0.3), 1),
        ],
        shape=2,
    )
    state.prepare(
        beta=0.2,
        n_steps=1,
        trotter_order=1,
        trotter_fuse_adjacent=False,
        cutoff=0.0,
    )

    identity = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    expected = (
        0.7 * np.kron(z, z)
        + 0.2 * np.kron(x, identity)
        - 0.1 * np.kron(z, identity)
        + 0.3 * np.kron(identity, x)
    )
    np.testing.assert_allclose(
        np.asarray(state._trotter_ham.get_gate((0, 1))),
        expected,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(state.trotter_gates[0].U),
        expm(-0.1 * expected),
        atol=1.0e-12,
    )
    assert len(state.trotter_gates) == 1
    assert len(state.gates) == 1


def test_gibbs_mps_defaults_to_direct_replay_mode():
    """The public default uses the explicit direct-mode spelling."""
    state = GibbsMps([(("ZZ", 0.2), (0, 1))], shape=2)
    state.prepare(beta=0.1, n_steps=1, cutoff=0.0)

    assert state.optimizer.mode == "quimb-direct"


def test_gibbs_mps_infers_flat_integer_chain_shape():
    """Flat integer locations remain an inferred one-dimensional chain."""
    state = GibbsMps(
        [
            (("zz", 0.3), (0, 1)),
            (("x", -0.1), 0),
        ]
    )

    assert state.shape == (2,)
    assert state.length == 2
    assert state.basis.lattice_shape is None


def test_gibbs_mps_materializes_term_generators_once():
    """Generator-based term input remains available for backend inference."""
    terms = ((("Z", 0.2), site) for site in range(2))
    state = GibbsMps(terms, shape=2)

    assert state.basis.num_terms == 2
    assert state.mps.L == 4


def test_gibbs_mps_rejects_unsupported_multisite_and_string_gap_terms():
    """The first step reports unsupported gate arities explicitly."""
    with pytest.raises(NotImplementedError, match="only one- and two-site"):
        GibbsMps([(("XXX", 0.1), (0, 1, 2))], shape=3)

    with pytest.raises(NotImplementedError, match="string operators across"):
        GibbsMps(
            [
                {
                    "operator": "XX",
                    "location": (0, 2),
                    "coefficient": 0.1,
                    "string_operators": ("Z",),
                }
            ],
            shape=3,
        )


def test_gibbs_mps_rejects_layout_override_during_replay():
    """Ancilla positions remain meaningful only without MPS reordering."""
    state = GibbsMps([(("ZZ", 0.1), (0, 1))], shape=2)

    with pytest.raises(TypeError, match="layout"):
        state.prepare(0.1, n_steps=1, run_kwargs={"layout": "quality"})


def test_gibbs_mps_keeps_identity_snapshot_with_inplace_optimizer():
    """The identity accessor remains un evolved even for in-place replay."""
    state = GibbsMps([(("Z", 0.1), 0)], shape=1)
    state.prepare(
        0.2,
        n_steps=1,
        optimizer_kwargs={"inplace": True},
        cutoff=0.0,
    )

    np.testing.assert_allclose(
        np.asarray(
            state.identity_mps.partial_trace_to_mpo(
                keep=state.physical_sites,
                upper_ind_id="b{}",
            ).to_dense()
        ),
        np.eye(2) / 2.0,
        atol=1.0e-12,
    )


def test_gibbs_mps_keeps_explicit_backend_for_state_and_gates():
    """Explicit ``to_backend`` reaches Bell tensors, gates, and the MPO."""
    torch = pytest.importorskip("torch")
    import pepsy

    backend = pepsy.backend_torch(dtype=torch.complex128, device="cpu")
    state = GibbsMps(
        [(("ZZ", 0.4), (0, 1)), (("X", -0.1), 0)],
        shape=2,
        to_backend=backend,
    )
    state.prepare(0.2, n_steps=2, chi=16, cutoff=0.0)

    assert all(isinstance(tensor.data, torch.Tensor) for tensor in state.mps)
    assert all(isinstance(gate, torch.Tensor) for gate, _where in state.gates)
    rho = state.to_mpo(normalized=False)
    assert all(isinstance(tensor.data, torch.Tensor) for tensor in rho)


def test_gibbs_mps_preserves_autograd_gate_payloads():
    """Backend placement must not detach differentiable generated gates."""
    torch = pytest.importorskip("torch")
    import pepsy

    beta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    backend = pepsy.backend_torch(dtype=torch.complex128, device="cpu")
    state = GibbsMps(
        [(("Z", coefficient), 0)],
        shape=1,
        to_backend=backend,
    )
    state.prepare(beta, n_steps=2, chi=8, cutoff=0.0)
    loss = torch.real(state.to_mpo(normalized=False).to_dense()[0, 0])
    grad_beta, grad_coefficient = torch.autograd.grad(
        loss,
        (beta, coefficient),
    )

    assert torch.isfinite(grad_beta)
    assert torch.isfinite(grad_coefficient)


def test_gibbs_mps_jax_gate_schedule_preserves_backend_arrays():
    """Unhashable JAX exponents use the native schedule plus Autoray gates."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import pepsy

    backend = pepsy.backend_jax(dtype=jnp.complex64)
    state = GibbsMps(
        [
            (("ZZ", jnp.asarray(0.4)), (0, 1)),
            (("X", jnp.asarray(-0.1)), 0),
        ],
        shape=2,
        to_backend=backend,
    )
    state.prepare(jnp.asarray(0.2), n_steps=2, cutoff=0.0)

    assert isinstance(state.trotter_gates[0].U, jax.Array)
    assert isinstance(state.gates[0][0], jax.Array)
