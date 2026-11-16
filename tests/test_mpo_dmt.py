"""Protected compression in the ordinary ket/bra matrix-unit basis."""

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

from pepsy.operators import PauliMPO, MPODMTReport, compress_mpo_dmt
from pepsy.operators.pauli_mpo import _host_array


def _density():
    rng = np.random.default_rng(31)
    terms = [(1 / 64, 'IIIIII'), (.02 / 64, 'ZZ'), (.01 / 64, 'X')]
    terms += [(float(rng.uniform(-.005, .005)) / 64, ''.join(rng.choice(list('IXYZ'), 6)))
              for _ in range(60)]
    return PauliMPO.from_terms(6, terms).compress_pauli(cutoff=0.)


def _marginal(mpo, region, d=2):
    oracle = mpo.copy()
    oracle.apply_to_arrays(_host_array)
    return qu.partial_trace(oracle.to_dense(backend="numpy"), [d] * mpo.L, region)


@pytest.mark.parametrize("variant", ["density", "operator"])
@pytest.mark.parametrize("regions", [None, [(0, 3), (1, 4), (2, 5)]])
def test_compression_preserves_dense_marginals_and_matches_pauli(variant, regions):
    pauli = _density()
    mpo = pauli.to_mpo()
    before = mpo.to_dense()
    result, report = compress_mpo_dmt(mpo, max_bond=12, variant=variant,
                                      regions=regions, return_report=True)
    assert isinstance(report, MPODMTReport)
    assert result.max_bond() <= 12
    assert np.linalg.norm(result.to_dense() - before) > 1e-6
    check_regions = regions or [tuple(range(i, i + 3)) for i in range(4)]
    for region in check_regions:
        np.testing.assert_allclose(_marginal(result, region), _marginal(mpo, region), atol=2e-13)
    np.testing.assert_array_equal(mpo.to_dense(), before)


@pytest.mark.parametrize("variant", ["density", "operator"])
def test_single_truncated_bond_agrees_with_independent_pauli_basis(variant):
    rng = np.random.default_rng(51)
    a = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    density = a @ a.conj().T
    density /= np.trace(density)
    pauli = PauliMPO.from_dense(density).compress_pauli(cutoff=0.)
    normal = compress_mpo_dmt(pauli.to_mpo(), max_bond=10, variant=variant)
    reference = pauli.compress_dmt(max_bond=10, variant=variant)
    np.testing.assert_allclose(normal.to_dense(), reference.to_dense(), atol=1e-12)


def test_layout_tags_exponent_and_transactional_inplace():
    mpo = _density().to_mpo()
    mpo.upper_ind_id = "out{}"
    mpo.lower_ind_id = "in{}"
    mpo.add_tag("density")
    mpo.exponent = 2
    for tensor in mpo:
        tensor.transpose_(*reversed(tensor.inds))
    before = mpo.to_dense()
    tags = [set(mpo[i].tags) for i in range(mpo.L)]
    inds = [mpo[i].inds for i in range(mpo.L)]
    with pytest.raises(ValueError, match="protected budget"):
        compress_mpo_dmt(mpo, max_bond=2, inplace=True)
    np.testing.assert_array_equal(mpo.to_dense(), before)
    result = compress_mpo_dmt(mpo, max_bond=12, inplace=True)
    assert result is mpo
    assert result.exponent == 2
    assert [set(result[i].tags) for i in range(mpo.L)] == tags
    assert [result[i].inds for i in range(mpo.L)] == inds
    np.testing.assert_allclose(_marginal(result, (0, 1)), qu.partial_trace(before, [2] * 6, (0, 1)), atol=1e-11)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_backend_dtype_device_and_gradient(backend, dtype):
    module = pytest.importorskip(backend)
    mpo = _density().to_mpo()
    if backend == "torch":
        mpo.apply_to_arrays(lambda a: module.tensor(a, dtype=getattr(module, dtype), requires_grad=True))
    else:
        jnp = pytest.importorskip("jax.numpy")
        module.config.update("jax_enable_x64", True)
        mpo.apply_to_arrays(lambda a: jnp.asarray(a, dtype=dtype))
    # Local density preservation itself is checked without converting inputs.
    result = compress_mpo_dmt(mpo, max_bond=12)
    assert result[0].data.dtype == mpo[0].data.dtype
    assert result[0].data.device == mpo[0].data.device
    np.testing.assert_allclose(_marginal(result, (1, 2, 3)), _marginal(mpo, (1, 2, 3)),
                               atol=2e-6 if dtype == "complex64" else 1e-12)
    if backend == "torch":
        result[0].data.abs().square().sum().backward()
        assert mpo[0].data.grad is not None
        assert module.isfinite(mpo[0].data.grad).all()


def test_traceless_qutrit_and_small_chains():
    rng = np.random.default_rng(7)
    mpo = qtn.MatrixProductOperator([
        rng.normal(size=(4, 3, 3)),
        rng.normal(size=(4, 4, 3, 3)),
        rng.normal(size=(4, 3, 3)),
    ])
    out = compress_mpo_dmt(mpo, max_bond=9, variant="operator")
    np.testing.assert_allclose(out.to_dense(), mpo.to_dense(), atol=1e-12)
    single = qtn.MatrixProductOperator([np.diag([1., -1.])])
    np.testing.assert_allclose(compress_mpo_dmt(single, max_bond=1, variant="operator").to_dense(), single.to_dense())
    with pytest.raises(ValueError, match="trace"):
        compress_mpo_dmt(single, max_bond=1)


def test_rejects_unsupported_topology_and_physical_dimensions():
    with pytest.raises(NotImplementedError, match="open"):
        compress_mpo_dmt(qtn.MPO_rand(4, 3, cyclic=True), max_bond=10)
    with pytest.raises(ValueError, match="square"):
        compress_mpo_dmt(qtn.MatrixProductOperator([np.ones((2, 3))]), max_bond=1)


def test_qutrit_actual_compression_and_nonhermitian_marginals():
    mpo = qtn.MPO_rand(4, 30, phys_dim=3, dtype="complex128", seed=21)
    result = compress_mpo_dmt(mpo, max_bond=20, variant="operator")
    assert result.max_bond() <= 20
    assert np.linalg.norm(result.to_dense() - mpo.to_dense()) > 1e-6
    for region in [(0, 1, 2), (1, 2, 3)]:
        np.testing.assert_allclose(_marginal(result, region, d=3), _marginal(mpo, region, d=3), atol=1e-12)


@pytest.mark.parametrize("mode,cutoff", [
    ("rel", .25), ("abs", .0005), ("sum1", .0008),
    ("rsum1", .15), ("sum2", 1e-6), ("rsum2", .05),
])
def test_cutoff_and_diagnostics_include_external_exponent(mode, cutoff):
    mpo = _density().to_mpo()
    base, report = compress_mpo_dmt(mpo, max_bond=12, cutoff=cutoff,
                                     cutoff_mode=mode, return_report=True)
    mpo.exponent = 2
    power = 2 if mode == "sum2" else 1
    scaled_cutoff = cutoff * 100 ** power if mode in {"abs", "sum1", "sum2"} else cutoff
    scaled, scaled_report = compress_mpo_dmt(mpo, max_bond=12, cutoff=scaled_cutoff,
                                            cutoff_mode=mode, return_report=True)
    np.testing.assert_allclose(scaled.to_dense(), 100 * base.to_dense(), atol=1e-12)
    assert scaled_report.cutoff == scaled_cutoff
    np.testing.assert_allclose([r.discarded_weight for r in scaled_report.bond_reports],
                               [100 * r.discarded_weight for r in report.bond_reports], atol=1e-12)


def test_native_symmray_rejected_without_densification():
    pytest.importorskip("symmray")
    from pepsy.operators import MPOClusterProductExpansion, MPOProductTerm

    owner = MPOClusterProductExpansion(
        2, [[MPOProductTerm.from_pauli((0, 1), "ZZ")]], cluster_size=2,
        factorization="fixed", cutoff=0., symmetry="U1", physical_charges=(0, 1),
    )
    mpo = owner.exp(-.1j).to_mpo()
    with pytest.raises(TypeError, match="dense"):
        compress_mpo_dmt(mpo, max_bond=4)


def test_no_pauli_conversion_or_global_dense_in_production(monkeypatch):
    mpo = _density().to_mpo().view_as(qtn.MatrixProductOperator)

    def forbidden(*args, **kwargs):
        raise AssertionError("global dense or Pauli conversion reached")

    monkeypatch.setattr(qtn.MatrixProductOperator, "to_dense", forbidden)
    monkeypatch.setattr(PauliMPO, "from_dense", forbidden)
    monkeypatch.setattr(PauliMPO, "from_pauli_cores", forbidden)
    result = compress_mpo_dmt(mpo, max_bond=12)
    assert result.max_bond() <= 12


def test_torch_trace_gradient_through_actual_compression():
    torch = pytest.importorskip("torch")
    from pepsy.backends import TorchLinalgConfig

    mpo = qtn.MPO_rand(4, 16, dtype="float64", seed=43)
    reference = float(np.trace(mpo.to_dense()))
    theta = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    mpo.apply_to_arrays(lambda a: torch.tensor(a, dtype=torch.float64))
    mpo[0].modify(data=theta * mpo[0].data)
    with TorchLinalgConfig(stabilized=True, svd_gap_rtol=1e-9).activated():
        result = compress_mpo_dmt(mpo, max_bond=10, variant="operator")
        assert result.max_bond() == 10
        value = torch.ones(1, dtype=torch.float64)
        for i in range(result.L):
            bonds = (result.bond(i - 1, i),) if i else ()
            bonds += (result.bond(i, i + 1),) if i + 1 < result.L else ()
            tensor = result[i].transpose(*(bonds + (result.upper_ind(i), result.lower_ind(i))))
            dl = value.shape[0]
            core = tensor.data.reshape(dl, -1, 2, 2)
            value = value @ torch.diagonal(core, dim1=-2, dim2=-1).sum(-1)
        value.square().sum().backward()
    np.testing.assert_allclose(theta.grad, 2 * .7 * reference ** 2, rtol=1e-7, atol=1e-10)


def test_writeback_preparation_failure_does_not_partially_modify_input(monkeypatch):
    from pepsy.operators import mpo_dmt

    mpo = _density().to_mpo()
    before = [array.copy() for array in mpo.arrays]
    transpose = mpo_dmt._transpose
    calls = []

    def fail_second_array(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("backend writeback preparation failed")
        return transpose(*args)

    monkeypatch.setattr(mpo_dmt, "_transpose", fail_second_array)
    with pytest.raises(RuntimeError, match="writeback preparation"):
        compress_mpo_dmt(mpo, max_bond=12, inplace=True)
    for original, current in zip(before, mpo.arrays):
        np.testing.assert_array_equal(current, original)
