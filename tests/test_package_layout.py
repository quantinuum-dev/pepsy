"""Tests for the responsibility-based package namespaces."""

import importlib

import pytest

from pepsy.boundary import (
    BdyMPS,
    contract_boundary,
    contract_flat,
    peps_fidelity,
    peps_infidelity,
    peps_norm,
    peps_normalize,
)
from pepsy.operators import gate, rx
from pepsy.optimizers import MpsOptimizer, PepsOptimizer, SweepOptimizer
from pepsy.sampling import MpsSampler, PepsBpSampler
from pepsy.solvers import FDSolver
from pepsy.tensors import (
    OneDMap,
    backend_torch,
    haar_random_state,
    ps_to_3dpeps,
    ps_to_peps,
    reg_complex_svd_torch,
)


def test_new_namespace_imports_resolve():
    """Common new namespace imports should resolve to usable objects."""
    assert BdyMPS is not None
    assert callable(contract_boundary)
    assert callable(contract_flat)
    assert callable(peps_norm)
    assert callable(peps_normalize)
    assert callable(peps_infidelity)
    assert callable(peps_fidelity)
    assert callable(gate)
    assert callable(rx)
    assert MpsOptimizer is not None
    assert PepsOptimizer is not None
    assert SweepOptimizer is not None
    assert MpsSampler is not None
    assert PepsBpSampler is not None
    assert FDSolver is not None
    assert OneDMap is not None
    assert callable(backend_torch)
    assert callable(haar_random_state)
    assert callable(ps_to_peps)
    assert callable(ps_to_3dpeps)
    assert callable(reg_complex_svd_torch)


@pytest.mark.parametrize(
    "old_module",
    [
        "pepsy.boundary_metrics",
        "pepsy.boundary_states",
        "pepsy.boundary_sweeps",
        "pepsy.core",
        "pepsy.fit",
        "pepsy.ft_solver",
        "pepsy.gates",
        "pepsy.gradient_solver",
        "pepsy.ham",
        "pepsy.optimize_energy",
        "pepsy.optimize_global",
        "pepsy.optimize_mpo",
        "pepsy.optimize_mps",
        "pepsy.optimize_sweep",
        "pepsy.sampler",
    ],
)
def test_old_layout_modules_are_removed(old_module):
    """Old flat module paths should no longer be importable."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old_module)
