"""Schedule-first qMERA local-energy example."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pprint

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

from pepsy.optimizers import QMeraBuilder, build_qmera_contraction_optimizer


def zz_operator():
    """Return a two-site ZZ operator in quimb gate tensor shape."""
    z_op = np.diag([1.0, -1.0])
    return np.kron(z_op, z_op).reshape(2, 2, 2, 2)


def zz_chain_hamiltonian(length):
    """Return nearest-neighbor ZZ terms on an open chain."""
    h2 = zz_operator()
    return {(site, site + 1): h2 for site in range(length - 1)}


def as_float(value):
    """Convert NumPy/Torch/JAX scalar-like values to a display float."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return float(np.real(np.asarray(value)))


def run_demo(*, steps=0, cache_dir=False):
    """Build a small qMERA, evaluate compiled local-cone energy, optionally train."""
    builder = QMeraBuilder(
        shape=4,
        gate_family="rxx",
        isometry_gate_family="rzz",
        seed=7,
        param_scale=0.05,
    )
    schedule = builder.build_schedule()
    params = builder.initialize_parameters(schedule)
    hamiltonian = zz_chain_hamiltonian(schedule.geometry.num_sites)
    chunks = builder.parametric_lightcone_chunks(hamiltonian, schedule)
    contraction_opt = build_qmera_contraction_optimizer(
        directory=cache_dir,
        max_repeats=4,
        max_time=0.25,
        progbar=False,
    )

    energy = builder.parametric_loss(
        params,
        schedule=schedule,
        chunks=chunks,
        energy_per_site=False,
        contraction_opt=contraction_opt,
    )
    compiled = builder.compile_parametric_lightcones(
        schedule=schedule,
        chunks=chunks,
        contraction_opt=contraction_opt,
    )
    compiled_energy = builder.compiled_parametric_loss(
        params,
        schedule=schedule,
        compiled_chunks=compiled,
        energy_per_site=False,
    )

    report = {
        "num_sites": schedule.geometry.num_sites,
        "num_layers": len(schedule.layers),
        "num_gates": schedule.num_gates,
        "num_terms": len(chunks),
        "max_lightcone_gates": max(chunk.num_gates for chunk in chunks),
        "energy": as_float(energy),
        "compiled_energy": as_float(compiled_energy),
    }

    if steps:
        if importlib.util.find_spec("torch") is None:
            report["torch_optimization"] = "skipped: torch is not installed"
        else:
            opt = builder.parametric_optimizer(
                hamiltonian,
                schedule=schedule,
                chunks=chunks,
                parameters=params,
                energy_per_site=False,
                contraction_opt=contraction_opt,
            )
            result = opt.run(
                solver="torch-adam",
                n_steps=steps,
                log_every=1,
                options={"lr": 0.05},
                compiled=True,
            )
            report["torch_initial_energy"] = float(result.history[0])
            report["torch_final_energy"] = float(result.history[-1])

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Optional number of Torch Adam steps to run.",
    )
    parser.add_argument(
        "--cache-dir",
        default=False,
        help="Optional cotengra path-cache directory. Omit for memory-only paths.",
    )
    args = parser.parse_args()
    pprint.pp(run_demo(steps=args.steps, cache_dir=args.cache_dir))


if __name__ == "__main__":
    main()
