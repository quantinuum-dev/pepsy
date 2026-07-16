"""Tests for MERA local-energy optimization helpers."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.energy import EnergyEstimate
from pepsy.optimizers.mera import (
    LocalTerm,
    MeraEnergyOptimizer,
    build_lightcone_chunks,
    local_lightcone_expectation,
    normalize_local_terms,
)
from pepsy.optimizers.mera.optimizer import (
    MeraEnergyOptimizer as ModuleMeraEnergyOptimizer,
)


def _zz_term():
    z_op = np.diag([1.0, -1.0])
    return np.kron(z_op, z_op).reshape(2, 2, 2, 2)


def _small_mera(seed=23):
    return qtn.MERA.rand(L=8, max_bond=2, dtype="complex128", seed=seed)


def test_normalize_local_terms_accepts_mapping_iterable_and_local_term():
    """Hamiltonian input should normalize to explicit LocalTerm objects."""
    op = _zz_term()

    terms = normalize_local_terms({(0, 1): op})
    assert terms == (LocalTerm(where=(0, 1), operator=op),)

    weighted = normalize_local_terms([((1, 2), op, 0.5)])
    assert weighted[0].where == (1, 2)
    assert weighted[0].operator is op
    assert weighted[0].weight == pytest.approx(0.5)

    local = LocalTerm(where=((0, 0),), operator=op)
    assert normalize_local_terms([local])[0].where == ((0, 0),)


def test_normalize_local_terms_rejects_bad_supports():
    """Invalid local-term structure should fail before contraction."""
    op = _zz_term()

    with pytest.raises(ValueError, match="empty"):
        normalize_local_terms({(): op})
    with pytest.raises(ValueError, match="duplicate"):
        normalize_local_terms({(0, 0): op})
    with pytest.raises(ValueError, match="form"):
        normalize_local_terms([((0, 1), op, 1.0, "extra")])


def test_mera_lightcone_expectation_matches_quimb_exact_contraction():
    """The cached lightcone kernel should match quimb's full exact oracle."""
    mera = _small_mera()
    where = (0, 1)
    op = _zz_term()
    terms = normalize_local_terms({where: op})
    chunk = build_lightcone_chunks(mera, terms)[0]

    local_value = local_lightcone_expectation(
        mera,
        chunk,
        optimize="auto-hq",
        normalized=True,
        real=False,
    )
    direct = mera.compute_local_expectation_exact(
        {where: op},
        optimize="auto-hq",
        normalized=True,
    )

    assert complex(local_value) == pytest.approx(complex(direct))
    assert chunk.tags == ("I0", "I1")
    assert chunk.physical_width == 2


def test_mera_energy_loss_matches_quimb_exact_sum():
    """MeraEnergyOptimizer.loss() should sum local lightcone contractions."""
    mera = _small_mera(seed=24)
    op = _zz_term()
    terms = {(0, 1): op, (2, 3): op}
    direct = mera.compute_local_expectation_exact(
        terms,
        optimize="auto-hq",
        normalized=True,
    )
    opt = MeraEnergyOptimizer(
        mera,
        terms,
        energy_per_site=False,
        normalized=True,
        contraction_opt="auto-hq",
    )

    assert complex(opt.loss(real=False)) == pytest.approx(complex(direct))


def test_mera_energy_estimate_reports_lightcone_metadata():
    """energy() should return the shared EnergyEstimate dataclass."""
    mera = _small_mera(seed=25)
    opt = MeraEnergyOptimizer(
        mera,
        {(0, 1): _zz_term()},
        energy_per_site=True,
        normalized=True,
    )

    estimate = opt.energy()

    assert isinstance(estimate, EnergyEstimate)
    assert estimate.num_sites == 8
    assert estimate.boundary_mode == "lightcone-exact"
    assert estimate.energy_per_site == pytest.approx(estimate.energy / 8)
    assert estimate.metadata["num_terms"] == 1
    assert estimate.metadata["max_physical_width"] == 2
    assert estimate.metadata["max_lightcone_tensors"] >= 1


def test_mera_energy_make_tn_optimizer_and_optimize(monkeypatch):
    """TNOptimizer construction should receive loss constants and norm hook."""
    calls = []
    out = _small_mera(seed=27)

    class _FakeTNOptimizer:  # pylint: disable=too-few-public-methods
        def __init__(self, state, loss_fn, **kwargs):
            calls.append((state, loss_fn, kwargs))
            self.losses = [2.0, 1.0]

        def optimize(self, n=220, **kwargs):
            calls.append(("optimize", n, kwargs))
            return out

    monkeypatch.setattr(
        "pepsy.optimizers.mera.optimizer.qtn.TNOptimizer",
        _FakeTNOptimizer,
    )
    mera = _small_mera(seed=26)
    terms = {(0, 1): _zz_term()}
    opt = MeraEnergyOptimizer(mera, terms)

    tnopt = opt.make_tn_optimizer(
        optimizer="lbfgs",
        autodiff_backend="jax",
        progbar=False,
        loss_kwargs={"precompute_tags": False},
    )
    assert isinstance(tnopt, _FakeTNOptimizer)
    _, loss_fn, kwargs = calls[0]
    assert loss_fn is ModuleMeraEnergyOptimizer._tnopt_loss
    assert kwargs["loss_constants"]["terms"] == opt.terms
    assert kwargs["loss_constants"]["chunks"] is None
    assert kwargs["loss_kwargs"]["precompute_tags"] is False
    assert kwargs["optimizer"] == "L-BFGS-B"
    assert kwargs["autodiff_backend"] == "jax"
    assert callable(kwargs["norm_fn"])

    optimized, losses = opt.optimize(n=3, progbar=False, return_losses=True)
    assert optimized is out
    assert opt.state is out
    assert losses == (2.0, 1.0)
    assert calls[-1] == ("optimize", 3, {})
