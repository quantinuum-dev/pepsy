"""Tests for the responsibility-based package namespaces."""

import importlib

import pytest

from pepsy.boundary import BdyMPS, contract_boundary
from pepsy.operators import gate, rx
from pepsy.optimizers import MpsOptimizer, SweepOptimizer
from pepsy.sampling import MpsSampler, PepsBpSampler
from pepsy.solvers import FDSolver
from pepsy.tensors import OneDMap, backend_torch, ps_to_peps, reg_complex_svd_torch


def test_new_namespace_imports_resolve():
    """Common new namespace imports should resolve to usable objects."""
    assert BdyMPS is not None
    assert callable(contract_boundary)
    assert callable(gate)
    assert callable(rx)
    assert MpsOptimizer is not None
    assert SweepOptimizer is not None
    assert MpsSampler is not None
    assert PepsBpSampler is not None
    assert FDSolver is not None
    assert OneDMap is not None
    assert callable(backend_torch)
    assert callable(ps_to_peps)
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
