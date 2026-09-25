"""Import-weight contracts for the core package facade."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

_OPTIONAL_ROOTS = (
    "flax",
    "guppy",
    "jax",
    "netket",
    "nevergrad",
    "nlopt",
    "scipy",
    "stim",
    "symmray",
    "torch",
)

_CORE_MODULES = (
    "pepsy.backends",
    "pepsy.boundary",
    "pepsy.fitting",
    "pepsy.interop",
    "pepsy.operators",
    "pepsy.optimizers",
    "pepsy.sampling",
    "pepsy.solvers",
    "pepsy.tensors",
)

_LAZY_OPTIMIZER_MODULES = tuple(
    f"pepsy.optimizers.{name}"
    for name in ("mpo", "peps", "sweep", "tree", "tree_peps", "energy", "qmera")
)


def _run_clean_import(script: str) -> set[str]:
    """Run an import probe with only Pepsy's source tree added."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.split())


def test_import_pepsy_does_not_load_optional_backends():
    """The root facade must stay usable without advanced backend imports."""
    roots = repr(_OPTIONAL_ROOTS)
    loaded = _run_clean_import(
        f"""
import sys
import pepsy

roots = {roots}
print(*sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in roots)
))
"""
    )
    assert not loaded


def test_core_namespaces_do_not_load_optional_backends():
    """Core namespace discovery must remain independent of optional stacks."""
    modules = repr(_CORE_MODULES)
    roots = repr(_OPTIONAL_ROOTS)
    loaded = _run_clean_import(
        f"""
import importlib
import sys

for module_name in {modules}:
    importlib.import_module(module_name)

roots = {roots}
print(*sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in roots)
))
"""
    )
    assert not loaded


def test_lazy_namespace_discovery_does_not_resolve_exports():
    """Completion lists fresh exports without imports, warnings, or caching."""
    modules = (
        "pepsy", *_CORE_MODULES, "pepsy.experimental", "pepsy.vmc",
        *_LAZY_OPTIMIZER_MODULES,
    )
    loaded = _run_clean_import(
        f"""
import importlib
import sys
import warnings

for module_name in {modules!r}:
    module = importlib.import_module(module_name)
    before_modules = set(sys.modules)
    before_attributes = set(vars(module))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        names = dir(module)
        assert set(module.__all__) <= set(names), module_name
        assert before_attributes <= set(names), module_name
        assert names == sorted(set(names)), module_name
        assert set(sys.modules) == before_modules, module_name
        assert set(vars(module)) == before_attributes, module_name

roots = {(*_OPTIONAL_ROOTS, 'numpy', 'quimb', 'autoray', 'cotengra')!r}
print(*sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in roots)
))
"""
    )
    assert not loaded


def test_mps_namespace_and_support_modules_do_not_import_numerical_stack():
    """MPS discovery and reporting helpers must not initialize numerical code."""
    loaded = _run_clean_import(
        """
import sys
import pepsy.optimizers.mps as mps

assert set(mps.__all__) | {'gibbs'} <= set(dir(mps))
assert 'MpsOptimizer' not in vars(mps)
assert 'GibbsMps' not in vars(mps)
from pepsy.optimizers.mps import compression, diagnostics, normalization
assert compression.__all__ == diagnostics.__all__ == normalization.__all__ == []
assert diagnostics._summarize_fit_timing([])["calls"] == 0
assert diagnostics._layout_report_text({}) is None
roots = ('numpy', 'quimb', 'autoray', 'cotengra', 'torch', 'jax', 'symmray')
print(*sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in roots)
    or name in {'pepsy.optimizers.mps.optimizer', 'pepsy.optimizers.mps.gibbs'}
))
"""
    )
    assert not loaded


def test_mps_layout_import_does_not_initialize_replay_or_gibbs():
    """Layout data remains usable independently of the simulation engines."""
    loaded = _run_clean_import(
        """
import pickle
import sys
from pepsy.optimizers.mps import MpsGateStreamSchedule
from pepsy.optimizers.mps.layout import MpsGateStreamSchedule as direct

assert MpsGateStreamSchedule is direct
schedule = MpsGateStreamSchedule(stream=(), site_order=(0, 1))
assert pickle.loads(pickle.dumps(schedule)) == schedule
print(*sorted(
    name for name in sys.modules
    if name in {
        'pepsy.optimizers.mps.optimizer',
        'pepsy.optimizers.mps.gibbs',
        'pepsy.optimizers.mpo.optimizer',
        'pepsy.fitting.local',
    }
))
"""
    )
    assert not loaded


def test_tree_layout_and_stream_parsers_do_not_import_replay_engines():
    """Shared stream syntax must not make geometry load MPS replay or FIT."""
    loaded = _run_clean_import(
        """
import sys
from pepsy.optimizers.tree import TreePlan
from pepsy.optimizers._stream_events import conditional_event_parts

assert TreePlan.from_order(range(4)).n == 4
name, payload, where = conditional_event_parts(('if', -1, 1, ('x', 2)))
assert (name, where) == ('conditional', (2,))
assert payload['record'] == -1
print(*sorted(
    name for name in sys.modules
    if name in {
        'pepsy.optimizers.mps.optimizer', 'pepsy.optimizers.mpo.optimizer',
        'pepsy.optimizers.tree.optimizer', 'pepsy.fitting.local',
    }
))
"""
    )
    assert not loaded


def test_geometry_exports_work_without_numerical_dependencies():
    """Geometry objects can be constructed and serialized without engines."""
    loaded = _run_clean_import(
        """
import pickle
import sys
from pepsy.optimizers.qmera import QMeraGeometry
from pepsy.optimizers.tree_peps import TreePepsPlan

geometry = QMeraGeometry((2, 2))
assert pickle.loads(pickle.dumps(geometry)) == geometry
assert TreePepsPlan.__module__ == 'pepsy.optimizers.tree_peps.plan'
roots = ('numpy', 'quimb', 'autoray', 'cotengra', 'torch', 'jax', 'symmray')
print(*sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in roots)
))
"""
    )
    assert not loaded


def test_basic_operator_and_boundary_imports_bypass_compatibility_aggregator():
    """Direct ownership avoids loading the old core and unrelated observables."""
    for module_name in (
        "pepsy.operators.gates", "pepsy.operators.hamiltonians", "pepsy.boundary.states",
    ):
        loaded = _run_clean_import(
            f"""
import importlib
import sys
importlib.import_module({module_name!r})
print(*sorted(
    name for name in sys.modules
    if name in {{'pepsy.tensors.core', 'pepsy.tensors.observables'}}
))
"""
        )
        assert not loaded, (module_name, loaded)


def test_implementation_imports_bypass_core_compatibility_module():
    """Numerical consumers work without loading legacy patch-hook wrappers."""
    loaded = _run_clean_import(
        """
import importlib
import importlib.abc
import sys

class BlockLegacyCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pepsy.tensors.core':
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockLegacyCore())
for module_name in (
    'pepsy.boundary.sweeps', 'pepsy.fitting.local',
    'pepsy.optimizers.mps.optimizer', 'pepsy.optimizers.mpo.optimizer',
    'pepsy.optimizers.sweep.optimizer', 'pepsy.optimizers.global_opt',
    'pepsy.optimizers.stabilizer_tn.mps_stab_optimizer',
    'pepsy.sampling.samplers',
):
    importlib.import_module(module_name)

import quimb.tensor as qtn
from pepsy.optimizers.mps.optimizer import tn_fidelity
state = qtn.MPS_computational_state('01')
assert abs(tn_fidelity(state, state, contraction_opt='greedy') - 1) < 1e-12

# Exercise imports deferred until these diagnostic and sampling paths run.
from pepsy.optimizers.stabilizer_tn.mps_stab_optimizer import StabilizerMpsSimulator
simulator = object.__new__(StabilizerMpsSimulator)
simulator.contraction_opt = 'greedy'
assert simulator._fit_overlap_diagnostics_for_target(state, state)['fit_overlap_fidelity'] == 1

from pepsy.sampling.samplers import PepsBpSampler
from pepsy.tensors import contractions
contractions.build_optimizer = lambda **kwargs: 'greedy'
sampler = PepsBpSampler(qtn.PEPS.rand(2, 2, 2, seed=7))
assert sampler._get_optimizer() == 'greedy'
print(*sorted(name for name in sys.modules if name == 'pepsy.tensors.core'))
"""
    )
    assert not loaded


def test_symmetric_mapping_does_not_load_core_compatibility_module():
    """Mapping symmetric chains needs maps, not the compatibility aggregator."""
    loaded = _run_clean_import(
        """
import sys
from pepsy.tensors.maps import OneDMap
from pepsy.tensors.symmetric import _resolve_chain_mapper, _resolve_mpo_mapping

mapper = OneDMap(2, 2)
assert _resolve_chain_mapper(mapper, {'num_sites': 4}) == mapper.build()[0]
assert _resolve_mpo_mapping(mapper=mapper)[:2] == mapper.build()
print(*sorted(name for name in sys.modules if name == 'pepsy.tensors.core'))
"""
    )
    assert not loaded


def test_sampling_namespace_is_lazy():
    """Discovering samplers does not import sampler implementations eagerly."""
    loaded = _run_clean_import(
        """
import sys
import pepsy.sampling

print(*sorted(
    name for name in sys.modules
    if name in {'pepsy.sampling.samplers', 'pepsy.sampling.tree'}
))
"""
    )
    assert not loaded


def test_experimental_namespace_is_lazy():
    """Discovering advanced domains must not import their implementations."""
    loaded = _run_clean_import(
        """
import sys
import pepsy.experimental

print(*sorted(
    name for name in sys.modules
    if name == 'pepsy.vmc'
    or name.startswith('pepsy.vmc.')
    or name == 'pepsy.bp'
    or name.startswith('pepsy.bp.')
    or name == 'pepsy.optimizers.qmera'
    or name.startswith('pepsy.optimizers.qmera.')
))
"""
    )
    assert not loaded


def test_legacy_mera_namespace_is_lazy():
    """The former qMERA namespace does not load qMERA just by being imported."""
    loaded = _run_clean_import(
        """
import sys
import pepsy.optimizers.mera

print(*sorted(
    name for name in sys.modules
    if name == 'pepsy.optimizers.qmera'
    or name.startswith('pepsy.optimizers.qmera.')
))
"""
    )
    assert not loaded


def test_legacy_mera_child_namespace_is_lazy():
    """Direct imports of old qMERA child paths stay lazy as well."""
    loaded = _run_clean_import(
        """
import sys
import pepsy.optimizers.mera.builders

print(*sorted(
    name for name in sys.modules
    if name == 'pepsy.optimizers.qmera'
    or name.startswith('pepsy.optimizers.qmera.')
))
"""
    )
    assert not loaded
