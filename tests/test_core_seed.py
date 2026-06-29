"""Tests for optimizer/fidelity helper wiring and defaults."""

import builtins
import sys
import warnings
import numpy as np
import autoray as ar
import quimb as qu
import pepsy.tensors.core as core
import pytest
import quimb.tensor as qtn


def test_build_optimizer_loads_cotengrust_before_cotengra(monkeypatch):
    """build_optimizer should make cotengrust available to cotengra."""
    events = []

    def fake_ensure_cotengrust():
        events.append("cotengrust")
        return object()

    class DummyOpt:
        def __init__(self, **kwargs):
            _ = kwargs
            events.append("cotengra")

    monkeypatch.setattr(core, "_ensure_cotengrust", fake_ensure_cotengrust)
    monkeypatch.setattr(core.ctg, "ReusableHyperOptimizer", DummyOpt)

    out = core.build_optimizer(
        progbar=False,
        parallel=False,
        optlib="random",
        directory=None,
        max_repeats=1,
        max_time="rate:1e2",
    )

    assert isinstance(out, DummyOpt)
    assert events == ["cotengrust", "cotengra"]


def test_expec_mpo_mps_normalized_copy_does_not_divide_twice():
    """MPS expectations should be invariant to the input state's scale."""
    mps = qtn.MPS_rand_state(5, bond_dim=2, phys_dim=2, dtype="complex128", seed=26)
    mps[0].modify(data=0.2 * mps[0].data)
    mpo = core.id_to_mpo(5, dtype="complex128")

    measured = core.expec_mpo(mpo, mps, contraction_opt="auto-hq")

    assert abs(measured - 1.0) < 1e-12


def test_build_optimizer_constructs_without_seed(monkeypatch):
    """build_optimizer should configure cotengra optimizer without seed kwarg."""
    captured = {}

    class DummyOpt:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(core.ctg, "ReusableHyperOptimizer", DummyOpt)

    out = core.build_optimizer(
        progbar=False,
        parallel=False,
        optlib="random",
        directory=None,
        max_repeats=1,
        max_time="rate:1e2",
    )

    assert isinstance(out, DummyOpt)
    assert "seed" not in captured


def test_build_optimizer_forwards_slicing_related_options(monkeypatch):
    """build_optimizer should forward slicing/reconfig option dictionaries."""
    captured = {}

    class DummyOpt:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(core.ctg, "ReusableHyperOptimizer", DummyOpt)

    slicing_opts = {"target_size": 2048}
    reconf_opts = {"subtree_size": 8}
    slicing_reconf_opts = {"target_slices": 16}

    out = core.build_optimizer(
        progbar=False,
        slicing_opts=slicing_opts,
        reconf_opts=reconf_opts,
        slicing_reconf_opts=slicing_reconf_opts,
    )

    assert isinstance(out, DummyOpt)
    assert captured["slicing_opts"] is slicing_opts
    assert captured["reconf_opts"] is reconf_opts
    assert captured["slicing_reconf_opts"] is slicing_reconf_opts


def test_build_compressed_optimizer_constructs_without_seed(monkeypatch):
    """build_compressed_optimizer should not pass seed to cotengra."""
    captured = {}

    class DummyCOpt:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured.update(kwargs)

    monkeypatch.setattr(core.ctg, "ReusableHyperCompressedOptimizer", DummyCOpt)

    out = core.build_compressed_optimizer(
        progbar=False,
        chi=4,
        directory=None,
        max_repeats=1,
        max_time="rate:1e2",
    )

    assert isinstance(out, DummyCOpt)
    assert "seed" not in captured


def test_build_compressed_optimizer_loads_cotengrust_before_cotengra(monkeypatch):
    """build_compressed_optimizer should make cotengrust available to cotengra."""
    events = []

    def fake_ensure_cotengrust():
        events.append("cotengrust")
        return object()

    class DummyCOpt:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs
            events.append("cotengra")

    monkeypatch.setattr(core, "_ensure_cotengrust", fake_ensure_cotengrust)
    monkeypatch.setattr(core.ctg, "ReusableHyperCompressedOptimizer", DummyCOpt)

    out = core.build_compressed_optimizer(
        progbar=False,
        chi=4,
        directory=None,
        max_repeats=1,
        max_time="rate:1e2",
    )

    assert isinstance(out, DummyCOpt)
    assert events == ["cotengrust", "cotengra"]


def test_tn_fidelity_uses_default_optimizer_settings(monkeypatch):
    """tn_fidelity should call build_optimizer with progbar disabled."""
    captured = {}

    def fake_build_optimizer(**kwargs):
        captured.update(kwargs)
        return "auto-hq"

    monkeypatch.setattr(core, "build_optimizer", fake_build_optimizer)

    psi = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=5)
    psi_fix = psi.copy()

    fidelity = core.tn_fidelity(psi, psi_fix)

    assert abs(fidelity - 1.0) < 1e-12
    assert captured["progbar"] is False
    assert "seed" not in captured


def test_tn_fidelity_uses_supplied_optimizer_without_build(monkeypatch):
    """Passing ``contraction_opt`` should bypass default optimizer construction."""
    called = {"build": 0}

    def fake_build_optimizer(**kwargs):  # pylint: disable=unused-argument
        called["build"] += 1
        return "auto-hq"

    monkeypatch.setattr(core, "build_optimizer", fake_build_optimizer)

    psi = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=6)
    psi_fix = psi.copy()

    fidelity = core.tn_fidelity(psi, psi_fix, contraction_opt="auto-hq")

    assert abs(fidelity - 1.0) < 1e-12
    assert called["build"] == 0


def test_tn_fidelity_simplifies_closed_overlap_networks(monkeypatch):
    """tn_fidelity can simplify closed norm/overlap TNs before contraction."""

    psi = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=7)
    psi_fix = psi.copy()
    simplify_output_inds = []
    original = qtn.TensorNetwork.full_simplify_

    def spy_full_simplify(self, *args, **kwargs):
        simplify_output_inds.append(kwargs.get("output_inds"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.TensorNetwork, "full_simplify_", spy_full_simplify)

    fidelity = core.tn_fidelity(
        psi,
        psi_fix,
        contraction_opt="greedy",
        simplify=True,
    )

    assert abs(fidelity - 1.0) < 1e-12
    assert simplify_output_inds == [(), (), ()]


def test_tn_fidelity_defaults_to_no_closed_overlap_simplification(monkeypatch):
    """The default tn_fidelity path should avoid full_simplify_ calls."""

    psi = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=8)
    psi_fix = psi.copy()
    called = {"simplify": 0}

    def fail_full_simplify(self, *args, **kwargs):  # pylint: disable=unused-argument
        called["simplify"] += 1
        raise AssertionError("full_simplify_ should not be called")

    monkeypatch.setattr(qtn.TensorNetwork, "full_simplify_", fail_full_simplify)

    fidelity = core.tn_fidelity(
        psi,
        psi_fix,
        contraction_opt="greedy",
    )

    assert abs(fidelity - 1.0) < 1e-12
    assert called["simplify"] == 0


def test_tn_norm_simplifies_closed_network_when_requested(monkeypatch):
    """tn_norm can simplify the closed norm TN before contraction."""

    psi = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=9)
    simplify_output_inds = []
    original = qtn.TensorNetwork.full_simplify_

    def spy_full_simplify(self, *args, **kwargs):
        simplify_output_inds.append(kwargs.get("output_inds"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(qtn.TensorNetwork, "full_simplify_", spy_full_simplify)

    norm = core.tn_norm(psi, contraction_opt="greedy", simplify=True)

    assert norm > 0
    assert simplify_output_inds == [()]


def test_tn_norm_defaults_to_no_closed_network_simplification(monkeypatch):
    """The default tn_norm path should avoid full_simplify_ calls."""

    psi = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=10)
    called = {"simplify": 0}

    def fail_full_simplify(self, *args, **kwargs):  # pylint: disable=unused-argument
        called["simplify"] += 1
        raise AssertionError("full_simplify_ should not be called")

    monkeypatch.setattr(qtn.TensorNetwork, "full_simplify_", fail_full_simplify)

    norm = core.tn_norm(psi, contraction_opt="greedy")

    assert norm > 0
    assert called["simplify"] == 0


def test_default_backend_setters_roundtrip():
    """Default backend setters/getters should round-trip callables."""
    array_backend = lambda x: x  # noqa: E731
    grad_backend = lambda x: x  # noqa: E731

    core.reset_default_backends()
    try:
        core.set_default_array_backend(array_backend)
        core.set_default_grad_backend(grad_backend)
        assert core.get_default_array_backend() is array_backend
        assert core.get_default_grad_backend() is grad_backend
    finally:
        core.reset_default_backends()


def test_core_all_exports_backend_jax():
    """core.__all__ should include the JAX backend caster for parity."""
    assert "backend_jax" in core.__all__
    assert "reg_rel_svd_torch" in core.__all__
    assert "reg_complex_svd_torch" in core.__all__
    assert "reg_complex_svd_jax" in core.__all__


def test_backend_jax_casts_arrays_on_cpu_with_dtype():
    """backend_jax should cast inputs to JAX arrays and honor dtype."""
    jnp = pytest.importorskip("jax.numpy")

    caster = core.backend_jax(device="cpu", dtype="float32")
    out = caster(np.array([1.0, 2.0], dtype=np.float32))

    assert ar.infer_backend(out) == "jax"
    assert out.dtype == jnp.float32
    assert np.allclose(np.asarray(out), np.array([1.0, 2.0], dtype=np.float32))


def test_backend_jax_accepts_torch_tensor_input_if_available():
    """backend_jax should accept torch tensors via explicit host conversion."""
    torch = pytest.importorskip("torch")
    jnp = pytest.importorskip("jax.numpy")

    caster = core.backend_jax(device="cpu", dtype=jnp.float32)
    out = caster(torch.tensor([3.0, -1.0], dtype=torch.float64))

    assert ar.infer_backend(out) == "jax"
    assert out.dtype == jnp.float32
    assert np.allclose(np.asarray(out), np.array([3.0, -1.0], dtype=np.float32))


def test_backend_jax_invalid_device_raises():
    """Invalid device spec should produce a clear ValueError."""
    pytest.importorskip("jax")
    with pytest.raises(ValueError, match="not available"):
        core.backend_jax(device="cpu:9999")


def test_backend_jax_canonicalizes_x64_dtype_without_truncation_warning():
    """float64 requests should respect JAX x64 policy without warning spam."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    caster = core.backend_jax(device="cpu", dtype="float64")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = caster(np.array([1.0, 2.0], dtype=np.float32))

    truncation_warnings = [
        w for w in caught if "requested dtype float64" in str(w.message)
    ]
    assert not truncation_warnings
    expected_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    assert out.dtype == expected_dtype


def test_bdymps_uses_default_array_backend_when_not_provided():
    """BdyMPS should pick global default array backend for ``array_backend``."""
    import pepsy

    array_backend = lambda x: x  # noqa: E731
    core.reset_default_backends()
    try:
        core.set_default_array_backend(array_backend)
        ket = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=101, dtype="complex128")
        _, norm = pepsy.build_bra_ket(ket=ket)
        bdy = pepsy.BdyMPS(
            tn_double=norm,
            chi=4,
            single_layer=False,
        )
        assert bdy.array_backend is array_backend
    finally:
        core.reset_default_backends()


def test_contract_hypercompressed_tn_builds_copt_when_missing(monkeypatch):
    """Missing copt should be built from chi via build_compressed_optimizer."""
    captured = {}

    def fake_build_compressed_optimizer(**kwargs):
        captured.update(kwargs)
        return "dummy-copt"

    monkeypatch.setattr(core, "build_compressed_optimizer", fake_build_compressed_optimizer)

    class DummyTN:  # pylint: disable=too-few-public-methods
        def copy(self):
            return DummyTN()

        def full_simplify_(self, **kwargs):
            self.simplify_kwargs = kwargs

        def contraction_tree(self, copt):
            self.copt = copt
            return "dummy-tree"

        def contract_compressed_(self, **kwargs):
            self.contract_kwargs = kwargs

    tn = DummyTN()
    out = core.contract_hypercompressed_tn(
        tn,
        copt=None,
        max_bond=9,
        chi=7,
        progbar=True,
    )

    assert out is not tn
    assert captured["chi"] == 7
    assert captured["progbar"] is True
    assert out.copt == "dummy-copt"
    assert out.contract_kwargs["max_bond"] == 9
    assert out.simplify_kwargs == {"seq": "R", "split_method": "svd", "inplace": True}


def test_contract_hypercompressed_tn_requires_chi_if_copt_missing():
    """If copt is absent, chi must be provided so core can build one."""
    tn = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=7)
    with pytest.raises(ValueError, match="provide `chi`"):
        _ = core.contract_hypercompressed_tn(tn, copt=None, max_bond=None, chi=None)


def test_contract_hypercompressed_tn_inplace_mutates_input(monkeypatch):
    """inplace=True should operate on the original TensorNetwork object."""
    monkeypatch.setattr(core, "build_compressed_optimizer", lambda **kwargs: "dummy-copt")

    class DummyTN:  # pylint: disable=too-few-public-methods
        def copy(self):
            self.copy_called = True
            return DummyTN()

        def full_simplify_(self, **kwargs):
            self.simplify_kwargs = kwargs

        def contraction_tree(self, copt):
            _ = copt
            return "dummy-tree"

        def contract_compressed_(self, **kwargs):
            self.contract_kwargs = kwargs

    tn = DummyTN()
    out = core.contract_hypercompressed_tn(
        tn,
        copt=None,
        max_bond=5,
        chi=5,
        inplace=True,
    )

    assert out is tn
    assert not hasattr(tn, "copy_called")
    assert tn.contract_kwargs["max_bond"] == 5


def test_contract_hypercompressed_tn_can_skip_full_simplify(monkeypatch):
    """do_full_simplify=False should bypass pre-contraction simplification."""
    monkeypatch.setattr(core, "build_compressed_optimizer", lambda **kwargs: "dummy-copt")

    class DummyTN:  # pylint: disable=too-few-public-methods
        def copy(self):
            return DummyTN()

        def full_simplify_(self, **kwargs):
            self.simplify_kwargs = kwargs

        def contraction_tree(self, copt):
            _ = copt
            return "dummy-tree"

        def contract_compressed_(self, **kwargs):
            self.contract_kwargs = kwargs

    tn = DummyTN()
    out = core.contract_hypercompressed_tn(
        tn,
        copt=None,
        max_bond=5,
        chi=5,
        do_full_simplify=False,
    )

    assert out is not tn
    assert not hasattr(out, "simplify_kwargs")
    assert out.contract_kwargs["max_bond"] == 5


def test_contract_hypercompressed_tn_passes_custom_seq(monkeypatch):
    """A custom seq should be forwarded to full_simplify_."""
    monkeypatch.setattr(core, "build_compressed_optimizer", lambda **kwargs: "dummy-copt")

    class DummyTN:  # pylint: disable=too-few-public-methods
        def copy(self):
            return DummyTN()

        def full_simplify_(self, **kwargs):
            self.simplify_kwargs = kwargs

        def contraction_tree(self, copt):
            _ = copt
            return "dummy-tree"

        def contract_compressed_(self, **kwargs):
            self.contract_kwargs = kwargs

    tn = DummyTN()
    out = core.contract_hypercompressed_tn(
        tn,
        copt=None,
        max_bond=5,
        chi=5,
        seq="ADCRS",
    )

    assert out.simplify_kwargs["seq"] == "ADCRS"
    assert out.simplify_kwargs["split_method"] == "svd"
    assert out.simplify_kwargs["inplace"] is True


def test_measure_obs_normalizes_when_bra_not_provided():
    """Default branch should compute <tn|O|tn>/<tn|tn>."""
    mps = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=11)
    mps[0].modify(data=2.0 * mps[0].data)
    obs = qu.pauli("Z")
    opt = "auto-hq"

    measured = core.measure_obs(mps, obs, where=1, contraction_opt=opt)

    tn_obs = qtn.tensor_network_gate_inds(mps, obs, ["k1"], contract=False, inplace=False)
    expected = (mps.H & tn_obs).contract(all, optimize=opt) / core.tn_norm(
        mps, contraction_opt=opt
    )
    assert abs(measured - expected) < 1e-12


def test_measure_obs_uses_explicit_bra_without_normalization():
    """If bra is supplied, return raw <bra|O|tn> without tn_norm division."""
    mps = qtn.MPS_rand_state(3, bond_dim=2, phys_dim=2, dtype="complex128", seed=12)
    obs = qu.pauli("X")
    bra = mps.H.copy()
    bra[0].modify(data=3.0 * bra[0].data)
    opt = "auto-hq"

    measured = core.measure_obs(mps, obs, where=[1], bra=bra, contraction_opt=opt)

    tn_obs = qtn.tensor_network_gate_inds(mps, obs, ["k1"], contract=False, inplace=False)
    expected = (bra & tn_obs).contract(all, optimize=opt)
    assert abs(measured - expected) < 1e-12


def test_measure_obs_normalize_false_skips_tn_norm(monkeypatch):
    """normalize=False should avoid calling tn_norm and return raw expectation."""
    mps = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=18)
    obs = qu.pauli("Z")
    opt = "auto-hq"

    def _fail_tn_norm(*args, **kwargs):  # pylint: disable=unused-argument
        raise AssertionError("tn_norm should not be called when normalize=False")

    monkeypatch.setattr(core, "tn_norm", _fail_tn_norm)

    measured = core.measure_obs(mps, obs, where=1, normalize=False, contraction_opt=opt)

    tn_obs = qtn.tensor_network_gate_inds(mps, obs, ["k1"], contract=False, inplace=False)
    expected = (mps.H & tn_obs).contract(all, optimize=opt)
    assert abs(measured - expected) < 1e-12


def test_measure_obs_normalize_false_allows_zero_norm_state():
    """normalize=False should return raw value even when state norm is zero."""
    mps = qtn.MPS_computational_state("000", dtype="complex128")
    mps[0].modify(data=np.zeros_like(mps[0].data))
    obs = qu.pauli("Z")
    opt = "auto-hq"

    measured = core.measure_obs(mps, obs, where=1, normalize=False, contraction_opt=opt)

    tn_obs = qtn.tensor_network_gate_inds(mps, obs, ["k1"], contract=False, inplace=False)
    expected = (mps.H & tn_obs).contract(all, optimize=opt)
    assert abs(measured - expected) < 1e-12


def test_measure_obs_accepts_2d_coordinate_where():
    """2D coordinate tuples should map to ``k{x},{y}`` indices."""
    peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, seed=13, dtype="complex128")
    obs = qu.pauli("Z")
    opt = "auto-hq"

    measured = core.measure_obs(
        peps,
        obs,
        where=(1, 0),
        ind_id="k{},{}",
        contraction_opt=opt,
    )

    tn_obs = qtn.tensor_network_gate_inds(
        peps,
        obs,
        ["k1,0"],
        contract=False,
        inplace=False,
    )
    expected = (peps.H & tn_obs).contract(all, optimize=opt) / core.tn_norm(
        peps, contraction_opt=opt
    )
    assert abs(measured - expected) < 1e-12


def test_measure_obs_1d_tuple_where_maps_to_two_sites():
    """For 1D states, ``where=(i, j)`` should map to ``k{i}``, ``k{j}``."""
    mps = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=15)
    obs = qu.kron(qu.pauli("Z"), qu.pauli("Z"))
    opt = "auto-hq"

    measured = core.measure_obs(mps, obs, where=(1, 2), contraction_opt=opt)

    tn_obs = qtn.tensor_network_gate_inds(
        mps,
        obs,
        ["k1", "k2"],
        contract=False,
        inplace=False,
    )
    expected = (mps.H & tn_obs).contract(all, optimize=opt) / core.tn_norm(
        mps, contraction_opt=opt
    )
    assert abs(measured - expected) < 1e-12


def test_measure_obs_accepts_batched_1d_ops_and_sites():
    """Batched 1D terms like [rzz, rxx] with where=[(i,j), ...] should work."""
    mps = qtn.MPS_rand_state(6, bond_dim=2, phys_dim=2, dtype="complex128", seed=21)
    rzz = qu.kron(qu.pauli("Z"), qu.pauli("Z"))
    rxx = qu.kron(qu.pauli("X"), qu.pauli("X"))
    opt = "auto-hq"

    measured = core.measure_obs(
        mps,
        obs=[rzz, rxx],
        where=[(1, 2), (2, 3)],
        ind_id="k{}",
        contraction_opt=opt,
    )

    tn_obs = qtn.tensor_network_gate_inds(mps, rzz, ["k1", "k2"], contract=False, inplace=False)
    tn_obs = qtn.tensor_network_gate_inds(tn_obs, rxx, ["k2", "k3"], contract=False, inplace=False)
    expected = (mps.H & tn_obs).contract(all, optimize=opt) / core.tn_norm(
        mps, contraction_opt=opt
    )
    assert abs(measured - expected) < 1e-12


def test_measure_obs_accepts_batched_2d_ops_and_sites():
    """Batched 2D terms should honor ind_id='k{},{}' formatting."""
    peps = qtn.PEPS.rand(Lx=5, Ly=5, bond_dim=2, seed=22, dtype="complex128")
    rzz = qu.kron(qu.pauli("Z"), qu.pauli("Z"))
    rxx = qu.kron(qu.pauli("X"), qu.pauli("X"))
    opt = "auto-hq"

    measured = core.measure_obs(
        peps,
        obs=(rzz, rxx),
        where=[((1, 0), (2, 2)), ((2, 3), (3, 4))],
        ind_id="k{},{}",
        contraction_opt=opt,
    )

    tn_obs = qtn.tensor_network_gate_inds(
        peps, rzz, ["k1,0", "k2,2"], contract=False, inplace=False
    )
    tn_obs = qtn.tensor_network_gate_inds(
        tn_obs, rxx, ["k2,3", "k3,4"], contract=False, inplace=False
    )
    expected = (peps.H & tn_obs).contract(all, optimize=opt) / core.tn_norm(
        peps, contraction_opt=opt
    )
    assert abs(measured - expected) < 1e-12


def test_measure_obs_requires_where_for_batched_observables():
    """Batched observables should require matching ``where`` entries."""
    mps = qtn.MPS_rand_state(6, bond_dim=2, phys_dim=2, dtype="complex128", seed=25)
    rzz = qu.kron(qu.pauli("Z"), qu.pauli("Z"))
    rxx = qu.kron(qu.pauli("X"), qu.pauli("X"))
    opt = "auto-hq"

    with pytest.raises(ValueError, match="matching sequence"):
        core.measure_obs(
            mps,
            obs=[rzz, rxx],
            where=None,
            ind_id="k{}",
            contraction_opt=opt,
        )


def test_measure_obs_non_k_prefix_requires_explicit_ind_id():
    """Non-``k`` physical index prefixes should require explicit ``ind_id``."""
    mps = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=23)
    mps.reindex_({f"k{i}": f"b{i}" for i in range(4)})
    obs = qu.pauli("Z")
    opt = "auto-hq"

    with pytest.raises(ValueError, match="pass ind_id explicitly"):
        _ = core.measure_obs(mps, obs, where=1, contraction_opt=opt)

    tn_obs = qtn.tensor_network_gate_inds(
        mps,
        obs,
        ["b1"],
        contract=False,
        inplace=False,
    )
    measured = core.measure_obs(mps, obs, where=1, ind_id="b{}", contraction_opt=opt)
    expected = (mps.H & tn_obs).contract(all, optimize=opt) / core.tn_norm(
        mps, contraction_opt=opt
    )
    assert abs(measured - expected) < 1e-12


def test_measure_obs_accepts_string_index_where():
    """String-index where should be supported via gate() dispatch."""
    mps = qtn.MPS_rand_state(4, bond_dim=2, phys_dim=2, dtype="complex128", seed=24)
    obs = qu.pauli("Z")
    opt = "auto-hq"

    measured = core.measure_obs(mps, obs, where="k1", contraction_opt=opt)

    tn_obs = qtn.tensor_network_gate_inds(
        mps,
        obs,
        ["k1"],
        contract=False,
        inplace=False,
    )
    expected = (mps.H & tn_obs).contract(all, optimize=opt) / core.tn_norm(
        mps, contraction_opt=opt
    )
    assert abs(measured - expected) < 1e-12


def test_measure_obs_non_k_3d_prefix_requires_explicit_ind_id():
    """3D non-``k`` prefixes should require explicit ``ind_id``."""
    tn = qtn.TensorNetwork(
        [
            qtn.Tensor(
                data=np.array([1.0, 0.0], dtype=np.complex128),
                inds=("b2,3,4",),
            )
        ]
    )
    obs = qu.pauli("Z")
    opt = "auto-hq"

    with pytest.raises(ValueError, match="pass ind_id explicitly"):
        _ = core.measure_obs(tn, obs, where=(2, 3, 4), contraction_opt=opt)

    tn_obs = qtn.tensor_network_gate_inds(
        tn,
        obs,
        ["b2,3,4"],
        contract=False,
        inplace=False,
    )
    measured = core.measure_obs(
        tn,
        obs,
        where=(2, 3, 4),
        ind_id="b{},{},{}",
        contraction_opt=opt,
    )
    expected = (tn.H & tn_obs).contract(all, optimize=opt) / core.tn_norm(
        tn, contraction_opt=opt
    )
    assert abs(measured - expected) < 1e-12


def test_expec_tn_1d_alias_removed():
    """Legacy ``expec_TN_1D`` name should no longer exist in core."""
    with pytest.raises(AttributeError):
        _ = core.expec_TN_1D


def test_register_torch_linalg_does_not_require_jax(monkeypatch):
    """Torch linalg registration should not import or require JAX."""
    torch = pytest.importorskip("torch")
    monkeypatch.delitem(sys.modules, "pepsy.backends.linalg_torch", raising=False)

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("simulated missing jax")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    core.register_torch_linalg(mode="complex")
    A = torch.randn(4, 3, dtype=torch.complex128, requires_grad=True)
    _U, S, _Vh = ar.do("linalg.svd", A)
    loss = S.real.sum()
    loss.backward()
    assert A.grad is not None


def test_complex_svd_torch_backward_matches_native():
    """Custom complex SVD backward should match native torch off degeneracies."""
    torch = pytest.importorskip("torch")
    from pepsy.backends import linalg_torch as lrt  # pylint: disable=import-outside-toplevel

    torch.manual_seed(1234)
    dtype = torch.complex128

    def matrix_with_separated_singular_values(m, n):
        A = torch.randn(m, n, dtype=dtype)
        U, _S, Vh = torch.linalg.svd(A, full_matrices=False)
        values = torch.linspace(3.0, 1.0, min(m, n), dtype=torch.float64)
        return (U * values.unsqueeze(0)) @ Vh

    def grad_for(A, loss_fn, *, custom):
        A = A.clone().detach().requires_grad_(True)
        if custom:
            U, S, Vh = lrt.SVD.apply(A)
        else:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        loss = loss_fn(U, S, Vh)
        loss.backward()
        return A.grad

    c_left = torch.randn(4, 4, dtype=dtype)
    c_left = 0.5 * (c_left + c_left.conj().T)
    c_right = torch.randn(4, 4, dtype=dtype)
    c_right = 0.5 * (c_right + c_right.conj().T)

    cases = [
        (
            matrix_with_separated_singular_values(4, 3),
            lambda _U, S, _Vh: S.sum(),
        ),
        (
            matrix_with_separated_singular_values(4, 3),
            lambda U, S, Vh: ((U * S.unsqueeze(0)) @ Vh).real.sum(),
        ),
        (
            matrix_with_separated_singular_values(4, 3),
            lambda U, _S, _Vh: ((U @ U.conj().T) * c_left).real.sum(),
        ),
        (
            matrix_with_separated_singular_values(3, 4),
            lambda _U, _S, Vh: ((Vh.conj().T @ Vh) * c_right).real.sum(),
        ),
        (
            matrix_with_separated_singular_values(4, 4),
            lambda U, _S, Vh: (U[0, 0] * Vh.conj().T[0, 0].conj()).real,
        ),
    ]

    for A, loss_fn in cases:
        native_grad = grad_for(A, loss_fn, custom=False)
        custom_grad = grad_for(A, loss_fn, custom=True)
        torch.testing.assert_close(custom_grad, native_grad, rtol=1e-9, atol=1e-9)


def test_complex_svd_torch_backward_supports_batched_autoray():
    """Registered complex torch SVD should backpropagate through batches."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(5678)
    core.reg_complex_svd_torch()

    A = torch.randn(2, 4, 3, dtype=torch.complex128, requires_grad=True)
    U, S, Vh = ar.do("linalg.svd", A)
    loss = ((U * S.unsqueeze(-2)) @ Vh).real.sum()
    loss.backward()

    assert A.grad is not None
    assert A.grad.shape == A.shape
    assert torch.isfinite(A.grad).all()


def test_reg_rel_svd_torch_registers_stable_autoray_rule():
    """Relative SVD registration should stabilize degenerate spectra."""
    torch = pytest.importorskip("torch")

    core.reg_rel_svd_torch()

    A = torch.eye(3, dtype=torch.complex128, requires_grad=True)
    U, S, Vh = ar.do("linalg.svd", A)
    loss = S.sum() + (U[0, 0] * Vh.conj().T[0, 0].conj()).real
    loss.backward()

    assert A.grad is not None
    assert torch.isfinite(A.grad).all()


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_reg_rel_svd_torch_forward_falls_back_to_scipy_gesvd(monkeypatch, dtype):
    """Relative SVD should recover when torch's CPU SVD driver fails."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy.linalg")
    from pepsy.backends import linalg_torch as lrt  # pylint: disable=import-outside-toplevel

    torch_dtype = getattr(torch, dtype)

    def fail_svd(*_args, **_kwargs):
        raise RuntimeError("forced gesdd failure")

    monkeypatch.setattr(lrt.torch.linalg, "svd", fail_svd)

    torch.manual_seed(151617)
    A = torch.randn(2, 4, 3, dtype=torch_dtype, requires_grad=True)
    U, S, Vh = lrt.SVD.apply(A)
    recon = (U * S.unsqueeze(-2)) @ Vh

    torch.testing.assert_close(recon, A.detach(), rtol=1e-10, atol=1e-10)
    loss = recon.real.square().sum() + S.sum()
    loss.backward()

    assert A.grad is not None
    assert torch.isfinite(A.grad).all()


def test_complex_svd_torch_backward_regularizes_degenerate_spectra():
    """Custom complex SVD backward should stay finite at degenerate spectra."""
    torch = pytest.importorskip("torch")
    from pepsy.backends import linalg_torch as lrt  # pylint: disable=import-outside-toplevel

    torch.manual_seed(91011)
    dtype = torch.complex128

    def matrix_with_singular_values(values):
        A = torch.randn(len(values), len(values), dtype=dtype)
        U, _S, Vh = torch.linalg.svd(A, full_matrices=False)
        values = torch.tensor(values, dtype=torch.float64)
        return (U * values.unsqueeze(0)) @ Vh

    def assert_finite_grad(A, loss_fn):
        A = A.clone().detach().requires_grad_(True)
        U, S, Vh = lrt.SVD.apply(A)
        loss = loss_fn(U, S, Vh)
        loss.backward()
        assert A.grad is not None
        assert torch.isfinite(A.grad).all()

    cases = [
        matrix_with_singular_values([3.0, 1.0, 1.0]),
        matrix_with_singular_values([1.0, 1.0, 1.0]),
        torch.zeros(3, 3, dtype=dtype),
    ]
    losses = [
        lambda U, S, Vh: ((U * S.unsqueeze(-2)) @ Vh).real.sum(),
        lambda U, _S, Vh: (U[0, 0] * Vh.conj().T[0, 0].conj()).real,
    ]

    for A in cases:
        for loss_fn in losses:
            assert_finite_grad(A, loss_fn)


def test_complex_svd_torch_regularization_is_scale_aware():
    """SVD regularization should preserve inverse-gradient scaling."""
    torch = pytest.importorskip("torch")
    from pepsy.backends import linalg_torch as lrt  # pylint: disable=import-outside-toplevel

    torch.manual_seed(121314)
    dtype = torch.complex128
    A = torch.randn(4, 4, dtype=dtype)
    U, _S, Vh = torch.linalg.svd(A, full_matrices=False)
    values = torch.linspace(4.0, 1.0, 4, dtype=torch.float64)
    A = (U * values.unsqueeze(0)) @ Vh

    def grad_for(X):
        X = X.clone().detach().requires_grad_(True)
        U, _S, Vh = lrt.SVD.apply(X)
        loss = (U[0, 0] * Vh.conj().T[0, 0].conj()).real
        loss.backward()
        return X.grad

    grad = grad_for(A)
    for scale in (1.0e-6, 1.0e6):
        scaled_grad = grad_for(scale * A)
        torch.testing.assert_close(scale * scaled_grad, grad, rtol=1e-5, atol=1e-8)


def test_reg_complex_svd_jax_does_not_require_torch_or_scipy(monkeypatch):
    """JAX SVD registration should not import or require torch/scipy."""
    pytest.importorskip("jax")
    monkeypatch.delitem(sys.modules, "pepsy.backends.linalg_jax", raising=False)

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "torch"
            or name.startswith("torch.")
            or name == "scipy"
            or name.startswith("scipy.")
        ):
            raise ImportError(f"simulated missing dependency: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    core.reg_complex_svd_jax()


def test_safe_inverse_honors_eps_abs():
    """safe_inverse should honor caller-provided eps_abs regularization."""
    pytest.importorskip("torch")
    from pepsy.backends import linalg_torch as lrt  # pylint: disable=import-outside-toplevel

    x = lrt.torch.tensor([1.0e-8], dtype=lrt.torch.float64)
    small = lrt.safe_inverse(x, eps_abs=1.0e-12)
    large = lrt.safe_inverse(x, eps_abs=1.0e-2)
    assert not lrt.torch.allclose(small, large)
