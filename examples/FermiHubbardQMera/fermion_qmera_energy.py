"""Optimize a small native Symmray Fermi-Hubbard qMERA energy."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pprint

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/pepsy_numba_cache")

from pepsy import Fermion
from pepsy.backends import backend_torch
from pepsy.optimizers.mera import (
    QMeraBuilder,
    QMeraSymmrayFermionBackend,
    symmray_fermion_gate_registry,
)


def _as_float(value):
    """Convert a scalar-like NumPy or Torch value to a real float."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return float(np.real(np.asarray(value)))


def build_demo(*, steps=0):
    """Build and optionally optimize a two-site spinful qMERA."""
    if steps and importlib.util.find_spec("torch") is None:
        raise RuntimeError("--steps requires the optional Torch dependency.")

    array_backend = None
    if steps:
        import torch

        array_backend = backend_torch(dtype=torch.complex128)

    backend = QMeraSymmrayFermionBackend(to_backend=array_backend)
    registry = symmray_fermion_gate_registry(backend=backend)

    # The register order is (0, up), (1, up), (0, down), (1, down). This
    # product state has one up fermion on site 0 and one down fermion on site 1.
    occupations = {0: 1, 1: 0, 2: 0, 3: 1}

    def product_state_factory(schedule, sites, **kwargs):
        return backend.product_state(
            schedule,
            sites,
            occupations=occupations,
            **kwargs,
        )

    builder = QMeraBuilder(
        shape=2,
        site_modes=("up", "down"),
        mode_order="mode-major",
        gate_registry=registry,
        array_backend=array_backend,
        disentangler={"block_size": 2, "circuit_depth": 0},
        isometry={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        max_layers=1,
        seed=7,
        param_scale=0.01,
        product_state_factory=product_state_factory,
    )
    fermion = Fermion(
        spinful=True,
        symmetry="U1U1",
        t=0.2,
        U=0.5,
        mu=0.1,
    )

    terms = builder.fermion_terms(fermion)
    optimizer = builder.fermion_parametric_optimizer(
        fermion,
        energy_per_site=False,
    )
    initial = optimizer.loss(energy_per_site=False)
    report = {
        "num_terms": len(terms),
        "initial_energy": _as_float(initial),
    }

    if steps:
        result = optimizer.run(
            solver="torch-adam",
            n_steps=steps,
            log_every=1,
            options={"lr": 0.01},
        )
        report["energy_history"] = [float(value) for value in result.history]
        report["final_energy"] = float(result.history[-1])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Optional number of native Torch Adam steps.",
    )
    args = parser.parse_args()
    pprint.pp(build_demo(steps=args.steps))


if __name__ == "__main__":
    main()
