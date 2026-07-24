"""Regression tests for the Symmray/backend handling in the sweep optimizer.

These cover the helper machinery that keeps finite-difference local objectives
on their original backend and preserves Torch-backed Symmray autograd paths.
"""

import numpy as np
import pytest
from types import SimpleNamespace
import quimb.tensor as qtn

import pepsy
from pepsy.optimizers.sweep.environments import QuimbMpsBoundaryStore
from pepsy.optimizers.sweep.optimizer import SweepOptimizer
from pepsy.tensors.symmetric import (
    SymPEPS,
    default_physical_sectors,
    site_charge_from_occupations,
)

torch = pytest.importorskip("torch")


def test_symmray_fd_solver_maps_autograd_solvers_to_finite_difference():
    # nlopt / torch / jax autograd solvers cannot differentiate through NumPy
    # Symmray blocks, so they must be routed to a finite-difference backend.
    assert SweepOptimizer._symmray_fd_solver("nlopt", {})[0] == "fd-nlopt"
    assert SweepOptimizer._symmray_fd_solver("scipy", {})[0] == "fd-scipy"
    assert SweepOptimizer._symmray_fd_solver("torch-adam", {})[0] == "fd-adam"
    assert SweepOptimizer._symmray_fd_solver("jax-lbfgs", {})[0] == "fd-adam"


def test_symmray_fd_solver_keeps_explicit_fd_solver():
    for name in ("fd-nlopt", "fd-scipy", "fd-adam"):
        assert SweepOptimizer._symmray_fd_solver(name, {})[0] == name


def test_symmray_fd_solver_drops_lr_option():
    # 'lr' is a first-order autograd step size and is rejected by the LBFGS-style
    # finite-difference backends; it must be stripped.
    _, opts = SweepOptimizer._symmray_fd_solver("nlopt", {"lr": 1e-2, "maxeval": 5})
    assert "lr" not in opts
    assert opts["maxeval"] == 5


def test_symmray_fd_adam_keeps_lr_option():
    _, opts = SweepOptimizer._symmray_fd_solver("torch-adam", {"lr": 1e-2})
    assert opts["lr"] == pytest.approx(1e-2)


def test_coerce_param_tree_numpy_converts_torch_leaves():
    tree = {
        "a": torch.ones(2, dtype=torch.float64),
        "b": [torch.zeros(3, dtype=torch.float64), np.arange(3.0)],
        "c": {"d": torch.tensor([1.0, 2.0])},
    }
    out = SweepOptimizer._coerce_param_tree_numpy(tree)
    assert isinstance(out["a"], np.ndarray)
    assert isinstance(out["b"][0], np.ndarray)
    assert isinstance(out["b"][1], np.ndarray)
    assert isinstance(out["c"]["d"], np.ndarray)
    np.testing.assert_allclose(out["a"], np.ones(2))
    np.testing.assert_allclose(out["c"]["d"], np.array([1.0, 2.0]))


def test_coerce_leaf_to_numpy_is_noop_for_numpy():
    arr = np.arange(4.0)
    assert SweepOptimizer._coerce_leaf_to_numpy(arr) is arr


def test_match_leaf_backend_restores_numpy_reference():
    reference = np.zeros(2)
    value = torch.ones(2, dtype=torch.float64)
    restored = SweepOptimizer._match_leaf_backend(value, reference)
    assert isinstance(restored, np.ndarray)
    np.testing.assert_allclose(restored, np.ones(2))


def test_match_leaf_backend_noop_when_backends_match():
    reference = np.zeros(2)
    value = np.ones(2)
    assert SweepOptimizer._match_leaf_backend(value, reference) is value


def test_restore_leaf_backends_maps_each_leaf():
    params_ref = {"x": np.zeros(2), "y": np.zeros(3)}
    params_opt = {
        "x": torch.ones(2, dtype=torch.float64),
        "y": torch.full((3,), 2.0, dtype=torch.float64),
    }
    restored = SweepOptimizer._restore_leaf_backends(params_opt, params_ref)
    assert isinstance(restored["x"], np.ndarray)
    assert isinstance(restored["y"], np.ndarray)
    np.testing.assert_allclose(restored["x"], np.ones(2))
    np.testing.assert_allclose(restored["y"], np.full(3, 2.0))


def test_match_param_tree_backends_restores_nested_numpy_reference():
    reference = {"site": [np.zeros(2), {"block": np.zeros(1)}]}
    trial = {
        "site": [
            torch.ones(2, dtype=torch.float64),
            {"block": torch.ones(1, dtype=torch.float64)},
        ]
    }
    restored = SweepOptimizer._match_param_tree_backends(trial, reference)
    assert isinstance(restored["site"][0], np.ndarray)
    assert isinstance(restored["site"][1]["block"], np.ndarray)


def test_params_require_finite_differences_only_for_non_torch_trees():
    assert SweepOptimizer._params_require_finite_differences({"x": np.zeros(2)})
    assert not SweepOptimizer._params_require_finite_differences(
        {"x": torch.zeros(2)}
    )


def test_sweep_rejects_mixed_symmray_and_dense_or_backend_inputs():
    """State and target must have one compatible array representation."""

    class _FakeSymmrayData:
        shape = (1,)
        dtype = "complex128"

        def __init__(self, backend):
            self.backend = backend

    _FakeSymmrayData.__module__ = "symmray.fake"

    def _symmray_state(backend):
        return SimpleNamespace(
            tensor_map={"site": SimpleNamespace(data=_FakeSymmrayData(backend))}
        )

    dense = SimpleNamespace(
        tensor_map={"site": SimpleNamespace(data=np.zeros(1))}
    )
    with pytest.raises(TypeError, match="mix Symmray and dense"):
        SweepOptimizer._validate_symmray_input_backends(_symmray_state("torch"), dense)
    with pytest.raises(TypeError, match="one common array backend"):
        SweepOptimizer._validate_symmray_input_backends(
            _symmray_state("torch"),
            _symmray_state("numpy"),
        )


def test_torch_symmray_u1u1_slice_uses_autograd_without_backend_conversion():
    """A real Torch Symmray local update must remain Torch-backed."""
    pytest.importorskip("symmray")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    charges = {
        (0, 0): (1, 0),
        (0, 1): (0, 1),
        (1, 0): (1, 0),
        (1, 1): (0, 1),
    }
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1U1",
        phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
        fermionic=True,
        site_charge=site_charge_from_occupations(charges),
        bond_dim=1,
        seed=71,
        dtype="complex128",
        to_backend=to_backend,
    ).tn
    target = state.copy()
    target.mangle_inner_()
    opt = SweepOptimizer(
        state,
        target,
        chi=4,
        boundary_engine="quimb-mps",
        renormalize_state=False,
    )

    assert opt._symmray_backends == {"torch"}
    assert opt._symmray_torch is True
    assert opt._symmray_requires_fd is False

    norm_tn, overlap_tn = opt._prepare_current_double_layers()
    opt.bdy.start_sweep(norm_tn, "y", "left")
    opt.bdy_overlap.start_sweep(overlap_tn, "y", "left")
    run = opt._optimize_axis_slice_with_current_env(
        0,
        axis="y",
        solver="nlopt",
        solver_options={"algorithm": "LD_LBFGS", "n_steps": 1, "maxeval": 1},
    )

    assert np.isfinite(run["loss_initial"])
    assert np.isfinite(run["loss_final"])
    assert all(
        isinstance(block, torch.Tensor)
        for tensor in opt.state.tensor_map.values()
        for block in tensor.data.blocks.values()
    )


def test_torch_symmray_interior_slice_attaches_both_cached_boundaries():
    """The overlap local bra must match the cached double-layer index names."""
    pytest.importorskip("symmray")
    to_backend = pepsy.backend_torch(device="cpu", dtype=torch.complex128)
    charges = {
        (x, y): (1, 0) if (x + y) % 2 == 0 else (0, 1)
        for x in range(3)
        for y in range(3)
    }
    state = SymPEPS.random(
        3,
        3,
        symmetry="U1U1",
        phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
        fermionic=True,
        site_charge=site_charge_from_occupations(charges),
        bond_dim=1,
        seed=79,
        dtype="complex128",
        to_backend=to_backend,
    ).tn
    target = state.copy()
    target.mangle_inner_()
    opt = SweepOptimizer(
        state,
        target,
        chi=4,
        boundary_engine="quimb-mps",
        renormalize_state=False,
    )

    norm_tn, overlap_tn = opt._prepare_current_double_layers()
    opt.bdy.start_sweep(norm_tn, "y", "left")
    opt.bdy_overlap.start_sweep(overlap_tn, "y", "left")
    # Seed the moving ymin boundary exactly as the forward half-sweep does
    # after processing row zero, then optimize the interior row.
    opt.bdy.advance_sweep(norm_tn, 0, axis="y", update_side="left")
    opt.bdy_overlap.advance_sweep(overlap_tn, 0, axis="y", update_side="left")
    run = opt._optimize_axis_slice_with_current_env(
        1,
        axis="y",
        solver="nlopt",
        solver_options={"algorithm": "LD_LBFGS", "n_steps": 1, "maxeval": 1},
    )

    assert np.isfinite(run["loss_initial"])
    assert np.isfinite(run["loss_final"])
    assert all(
        isinstance(block, torch.Tensor)
        for tensor in opt.state.tensor_map.values()
        for block in tensor.data.blocks.values()
    )


def test_quimb_boundary_return_move_matches_fresh_ymax_environment():
    """Backward cached moves must retain Quimb's ymax-side compressed row."""
    network = qtn.PEPS.rand(3, 3, bond_dim=2, phys_dim=2, seed=83)
    double_layer = network.H & network
    store = QuimbMpsBoundaryStore(chi=4, layer_tags=None)
    store.start_sweep(double_layer, "y", "right")
    # Seed row two, then extend the ymax boundary across row one.
    store.advance_sweep(double_layer, 2, axis="y", update_side="right")
    store.advance_sweep(double_layer, 1, axis="y", update_side="right")
    cached = store.mps_b["Y1_r"]
    fresh = double_layer.compute_ymax_environments(
        max_bond=4,
        cutoff=store.cutoff,
        canonize=store.canonize,
        mode=store.mode,
        dense=store.dense,
        equalize_norms=store.equalize_norms,
        envs={},
    )["ymax", 0]

    assert set(cached.outer_inds()) == set(fresh.outer_inds())
    assert set(cached.ind_map) == set(fresh.ind_map)


def test_nested_param_tree_flatten_roundtrip():
    tree = {
        "site": {
            "blocks": {(0, 0): np.arange(4.0), (1, 1): np.ones(2)},
            "meta": [np.zeros(1), (np.ones(1),)],
        }
    }
    assert SweepOptimizer._needs_nested_param_flatten(tree) is True
    flat, spec = SweepOptimizer._flatten_param_tree(tree)
    # All leaves are arrays (no nested containers left).
    assert all(isinstance(v, np.ndarray) for v in flat.values())
    rebuilt = SweepOptimizer._unflatten_param_tree(flat, spec)
    np.testing.assert_allclose(rebuilt["site"]["blocks"][(0, 0)], np.arange(4.0))
    np.testing.assert_allclose(rebuilt["site"]["meta"][1][0], np.ones(1))
    assert isinstance(rebuilt["site"]["meta"], list)
    assert isinstance(rebuilt["site"]["meta"][1], tuple)


def test_needs_nested_param_flatten_false_for_flat_mapping():
    flat = {"a": np.zeros(2), "b": np.ones(3)}
    assert SweepOptimizer._needs_nested_param_flatten(flat) is False
