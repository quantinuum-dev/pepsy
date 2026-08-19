"""Dependency-light contracts for reusable contraction optimizers."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from pepsy.tensors import contractions


ROOT = Path(__file__).resolve().parents[1]


def test_cotengrust_is_best_effort(monkeypatch):
    """Cotengra's native pathfinders remain usable without Cotengrust."""
    monkeypatch.setattr(contractions, "_optional_import", lambda _name: None)

    assert contractions._ensure_cotengrust() is None


def test_missing_cmaes_uses_builtin_hyper_optimizer(monkeypatch):
    """The default search has a built-in fallback when CMA-ES is absent."""
    monkeypatch.setattr(contractions, "_module_available", lambda _name: False)

    with pytest.warns(RuntimeWarning, match="CMA-ES is unavailable"):
        optlib = contractions._resolve_optlib("cmaes")

    assert optlib == "sbplx"
    assert contractions._resolve_optlib("sbplx") == "sbplx"


def test_reusable_optimizer_builds_without_acceleration_extras(monkeypatch):
    """The fallback path constructs the same reusable optimizer surface."""
    monkeypatch.setattr(contractions, "_optional_import", lambda _name: None)
    monkeypatch.setattr(contractions, "_module_available", lambda _name: False)

    with pytest.warns(RuntimeWarning, match="CMA-ES is unavailable"):
        optimizer = contractions.build_optimizer(
            max_time=0,
            max_repeats=1,
            parallel=False,
            progbar=False,
        )

    assert optimizer is not None


def test_compressed_optimizer_builds_without_cotengrust(monkeypatch):
    """Compressed path construction also tolerates missing Rust acceleration."""
    monkeypatch.setattr(contractions, "_optional_import", lambda _name: None)

    optimizer = contractions.build_compressed_optimizer(
        progbar=False,
        chi=2,
        max_repeats=1,
        max_time=0,
    )

    assert optimizer is not None


def test_subprocess_build_without_acceleration_modules():
    """The installed core path works when both acceleration modules are absent."""
    script = """
import importlib.abc
import importlib.util
import sys


real_find_spec = importlib.util.find_spec


def find_spec(fullname, package_path=None):
    if fullname in {"cmaes", "cotengrust"}:
        return None
    return real_find_spec(fullname, package_path)


importlib.util.find_spec = find_spec


class BlockAccelerationModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname in {"cmaes", "cotengrust"}:
            raise ModuleNotFoundError(fullname)
        return None


sys.meta_path.insert(0, BlockAccelerationModules())
from pepsy.tensors.contractions import build_optimizer

optimizer = build_optimizer(
    max_time=0,
    max_repeats=1,
    parallel=False,
    progbar=False,
)
assert type(optimizer).__name__ == "ReusableHyperOptimizer"
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["NUMBA_DISABLE_JIT"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
