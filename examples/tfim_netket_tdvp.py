"""Bounded PEPO startup followed by native NetKet/JAX PEPS tVMC.

Run with the optional NetKet dependencies installed:
    python examples/tfim_netket_tdvp.py --reference --steps 3

The optional dense oracle is restricted to at most 16 spins. Boundary caps
are fixed independently for PEPO fitting and single-layer tVMC amplitudes.
"""

import argparse
from functools import partial
import json
import time

import jax
import netket as nk
import numpy as np
from scipy.sparse.linalg import expm_multiply

from pepsy.optimizers import PepsOptimizer
from pepsy.tensors import ps_to_peps
from pepsy.vmc import build_ising_vmc

from tfim_pepo import trotter_pepo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=int, default=2)
    parser.add_argument("--ly", type=int, default=2)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--startup-steps", type=int, default=2)
    parser.add_argument("--startup-dt", type=float, default=.01)
    parser.add_argument("--norm-chi", type=int, default=18)
    parser.add_argument("--overlap-chi", type=int, default=72)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--chi", type=int, default=9)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--chains", type=int, default=16)
    parser.add_argument("--discard", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--diag-shift", type=float, default=1e-6)
    parser.add_argument("--solver", choices=("pinv_smooth", "cholesky"), default="pinv_smooth")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--j", type=float, default=1.)
    parser.add_argument("--h", type=float, default=1.)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    if min(args.lx, args.ly, args.d, args.chi, args.chains, args.samples, args.chunk_size) < 1:
        parser.error("Geometry, bonds, sample counts and chunks must be positive")
    if args.startup_steps < 1 or args.steps < 1 or min(args.startup_dt, args.dt) <= 0:
        parser.error("Use positive time steps and at least one startup and tVMC step")
    if args.samples % args.chains:
        parser.error("Samples must divide into chains")
    if args.reference and args.lx * args.ly > 16:
        parser.error("The dense oracle supports at most 16 spins")
    jax.config.update("jax_enable_x64", True)
    # A named sample mesh keeps NetKet sharding explicit rather than relying
    # on process-global mesh state installed by an earlier optional import.
    jax.sharding.set_mesh(jax.make_mesh((len(jax.devices()),), ("S",),
                                      axis_types=(jax.sharding.AxisType.Auto,)))
    started = time.perf_counter()

    def emit(event, **data):
        print(json.dumps({"event": event, "elapsed_seconds": time.perf_counter() - started,
                          **data}), flush=True)

    emit("setup", shape=[args.lx, args.ly], D=args.d, amplitude_chi=args.chi,
         fit_chis=[args.norm_chi, args.overlap_chi], cutoff=0., samples=args.samples,
         chains=args.chains, discard=args.discard, dt=args.dt, diag_shift=args.diag_shift,
         solver=args.solver, netket=nk.__version__, jax=jax.__version__, device=str(jax.devices()[0]))
    state = ps_to_peps(args.lx, args.ly, theta=np.pi / 4)
    fit = PepsOptimizer(state, [], chi=args.d, mode="sweep", optimizer="als",
        boundary_chi=(args.norm_chi, args.overlap_chi), normalize_chi=args.norm_chi,
        evaluation_chi=(args.norm_chi, args.overlap_chi), boundary_convergence=False,
        contraction_opt="greedy", sweep_kwargs={"local_contraction_opt": "greedy", "cutoff": 0.},
        normalize_kwargs={"cutoff": 0.}, infidelity_kwargs={"cutoff": 0.},
        sweep_optimize_kwargs={"n_round_trips": 1})
    operator = trotter_pepo(args.lx, args.ly, args.startup_dt, j=args.j, h=args.h)
    for step in range(args.startup_steps):
        fit.run_pepo(operator, unitary=True, cutoff=0.)
        record = fit.get_step_records()[0]
        emit("startup_step", step=step + 1, D=fit.state.max_bond(),
             infidelity=record["final_infidelity"], reason=record["reason"])
    # NetKet Ising uses +J ZZ-h X with Pauli operators. The PEPO uses
    # -j ZZ-h X, so negate J exactly once at this integration boundary.
    setup = build_ising_vmc(fit.state, Lx=args.lx, Ly=args.ly, h=args.h, J=-args.j,
        contraction="boundary", chi=args.chi, cutoff=0., parameter_mode="real-imag",
        n_samples=args.samples, n_chains=args.chains, n_discard_per_chain=args.discard,
        chunk_size=args.chunk_size, seed=args.seed, sampler_seed=args.seed + 1, use_sr=False)
    reference = hamiltonian = positions = initial_energy = None
    if args.reference:
        configurations = np.asarray(setup.hilbert.all_states())
        bits = ((1 - configurations) // 2).astype(int)
        positions = bits @ (2 ** np.arange(args.lx * args.ly - 1, -1, -1))
        vector = np.asarray(fit.state.to_dense(optimize="greedy")).reshape(-1)
        reference = vector[positions] / np.linalg.norm(vector)
        hamiltonian = setup.hamiltonian.to_sparse()
        initial_energy = float(np.vdot(reference, hamiltonian @ reference).real)
    solver = getattr(nk.optimizer.solver, args.solver)
    if args.solver == "pinv_smooth":
        solver = partial(solver, rtol_smooth=1e-10)
    dynamics = setup.make_tdvp_driver(dt=args.dt, t0=args.startup_steps * args.startup_dt,
        diag_shift=args.diag_shift, linear_solver=solver)
    emit("tdvp_ready", tensor_parameters=setup.n_params,
         real_coordinates=sum(z.size for z in jax.tree_util.tree_leaves(setup.vstate.parameters)),
         D=setup.to_peps().max_bond())
    for step in range(args.steps):
        emit("tdvp_step_start", step=step + 1)
        dynamics.advance(args.dt)
        leaves = jax.tree_util.tree_leaves(setup.vstate.parameters)
        if not all(np.isfinite(np.asarray(z)).all() for z in leaves):
            raise RuntimeError("TDVP produced non-finite parameters")
        diagnostics = {}
        if args.reference:
            reference = expm_multiply(-1j * args.dt * hamiltonian, reference)
            vector = np.asarray(setup.to_peps().to_dense(optimize="greedy")).reshape(-1)[positions]
            vector /= np.linalg.norm(vector)
            energy = float(np.vdot(vector, hamiltonian @ vector).real)
            diagnostics = {"dense_infidelity": float(1 - abs(np.vdot(reference, vector))**2),
                           "dense_energy": energy, "energy_drift": energy - initial_energy}
        emit("tdvp_step", step=step + 1, time=float(dynamics.t), **diagnostics)
    emit("complete", time=float(dynamics.t), D=setup.to_peps().max_bond())


if __name__ == "__main__":
    main()
