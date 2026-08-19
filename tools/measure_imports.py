#!/usr/bin/env python3
"""Measure clean-process import cost for Pepsy's public namespaces.

Each sample runs in a fresh interpreter so an earlier profile cannot warm the
module cache for a later one. The probe uses only the source tree and reports
the median wall-clock import time, the number of loaded Pepsy modules, and
whether known optional roots leaked into the import boundary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_ROOTS = (
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
CORE_MODULES = (
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


def _profiles():
    """Return the import statements used by the benchmark profiles."""
    core_import = (
        "import importlib\n"
        f"for _name in {CORE_MODULES!r}:\n"
        "    importlib.import_module(_name)"
    )
    return (
        ("root", "import pepsy"),
        ("core", core_import),
        ("sampling", "import pepsy.sampling"),
        ("optimizers", "import pepsy.optimizers"),
        ("experimental", "import pepsy.experimental"),
    )


def _sample(statement):
    """Run one clean interpreter and return its JSON probe result."""
    probe = f"""
import json
import sys
import time

start = time.perf_counter()
{statement}
elapsed_ms = (time.perf_counter() - start) * 1000.0
optional_roots = {OPTIONAL_ROOTS!r}
optional = sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in optional_roots)
)
pepsy_modules = sorted(
    name for name in sys.modules
    if name == 'pepsy' or name.startswith('pepsy.')
)
print(json.dumps({{
    'elapsed_ms': elapsed_ms,
    'optional': optional,
    'pepsy_modules': pepsy_modules,
}}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.setdefault("NUMBA_DISABLE_JIT", "1")
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="clean interpreter samples per profile (default: 3)",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    print("profile       median_ms  pepsy_modules  optional_modules")
    print("------------- ---------- -------------- ----------------")
    for label, statement in _profiles():
        samples = [_sample(statement) for _ in range(args.repeat)]
        median_ms = statistics.median(sample["elapsed_ms"] for sample in samples)
        module_count = len(samples[0]["pepsy_modules"])
        optional = sorted({name for sample in samples for name in sample["optional"]})
        print(f"{label:13} {median_ms:10.2f} {module_count:14d} {', '.join(optional) or '-'}")


if __name__ == "__main__":
    main()
