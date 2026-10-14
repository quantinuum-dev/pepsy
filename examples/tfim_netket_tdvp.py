"""Bounded PEPO startup followed by native NetKet/JAX PEPS tVMC.

Run with the optional NetKet dependencies installed:
    python examples/tfim_netket_tdvp.py --reference --steps 3

The optional dense oracle is restricted to at most 16 spins. Boundary caps
are fixed independently for PEPO fitting and single-layer tVMC amplitudes.
"""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import time

import jax
import netket as nk
import numpy as np
import quimb.tensor as qtn
from scipy.sparse.linalg import expm_multiply

from pepsy.optimizers import PepsOptimizer
from pepsy.tensors import ps_to_peps
from pepsy.vmc import build_ising_vmc

from tfim_pepo import trotter_pepo


_RESUME_SETTINGS = (
    "lx", "ly", "d", "startup_steps", "startup_dt", "norm_chi", "overlap_chi",
    "dt", "chi", "samples", "chains", "discard", "chunk_size", "diag_shift",
    "solver", "cg_tol", "cg_maxiter", "solver_max_relative_residual", "seed",
    "j", "h", "reference", "reference_origin",
)


def load_checkpoint(path):
    """Read a numeric checkpoint with enough layout information to restart."""
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    metadata = json.loads(str(arrays["metadata"]))
    if metadata.get("checkpoint_version") != 2:
        raise ValueError("This snapshot lacks restart layout; use a checkpoint from the current example")
    if set(_RESUME_SETTINGS) - metadata["simulation_settings"].keys():
        raise ValueError("The checkpoint is missing simulation settings")
    if not np.isfinite(metadata["time"]) or metadata["time"] < 0 or metadata["step"] < 0:
        raise ValueError("The checkpoint has an invalid time or step")
    tensors = []
    for entry in metadata["tensor_layout"]:
        data = arrays[entry["array"]]
        if not np.isfinite(data).all():
            raise ValueError("The checkpoint contains non-finite PEPS tensors")
        tensors.append(qtn.Tensor(data, inds=entry["inds"], tags=entry["tags"]))
    network = qtn.TensorNetwork(tensors)
    network.exponent = metadata["peps_exponent"]
    peps = qtn.PEPS.from_TN(network, **metadata["peps_layout"])
    return metadata, arrays, peps


def write_checkpoint(path, arrays):
    """Replace a checkpoint only after a complete numeric archive is written."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def normalized_peps_vector(peps, positions=None):
    """Exactly contract a PEPS after stripping its harmless overall scale."""
    scaled = peps.copy()
    scaled.equalize_norms_(value=1.)
    exponent = float(scaled.exponent)
    scaled.exponent = 0.
    vector = np.asarray(scaled.to_dense(optimize="greedy")).reshape(-1)
    if positions is not None:
        vector = vector[positions]
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("The exactly contracted state has a non-finite or zero norm")
    log_squared_norm = float(2 * np.log(norm) + 2 * exponent * np.log(10.))
    return vector / norm, log_squared_norm


def dense_spin_observables(vector, configurations, edges):
    """Pauli observables in NetKet's complete spin-1/2 basis ordering."""
    n = configurations.shape[1]
    indices = np.arange(vector.size)
    probabilities = abs(vector)**2
    mx = sum(np.vdot(vector, vector[indices ^ (1 << (n - i - 1))]).real
             for i in range(n)) / n
    zz = sum(np.dot(probabilities, configurations[:, i] * configurations[:, j])
             for i, j in edges) / len(edges) if edges else 0.
    return {"mx": float(mx), "zz": float(zz),
            "parity_x": float(np.vdot(vector, vector[::-1]).real)}


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
    parser.add_argument("--final-time", type=float,
                        help="Total final time including PEPO startup; overrides --steps")
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--chi", type=int, default=9)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--chains", type=int, default=16)
    parser.add_argument("--discard", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--diag-shift", type=float, default=1e-6)
    parser.add_argument("--solver", choices=("pinv_smooth", "cholesky", "cg"), default="pinv_smooth")
    parser.add_argument("--cg-tol", type=float, default=1e-10)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument("--solver-max-relative-residual", type=float, default=1e-5,
                        help="Maximum regularized solve residual at each RK stage (default: 1e-5)")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--j", type=float, default=1.)
    parser.add_argument("--h", type=float, default=1.)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--tdvp-diagnostics", action="store_true",
                        help="Recompute solve, tangent and sampling diagnostics at each accepted endpoint")
    parser.add_argument("--reference-origin", choices=("handoff", "initial"), default="handoff")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", type=Path,
                        help="Restart a saved PEPS/time with fresh sampling; --steps adds further steps")
    args = parser.parse_args()
    resumed = None
    if args.resume is not None:
        try:
            resumed = load_checkpoint(args.resume)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"Cannot resume checkpoint: {exc}")
        settings = resumed[0]["simulation_settings"]
        parser.set_defaults(**settings)
        args = parser.parse_args()
        for name in _RESUME_SETTINGS:
            if getattr(args, name) != settings[name]:
                parser.error(f"--{name.replace('_', '-')} must match the resumed checkpoint")
    if min(args.lx, args.ly, args.d, args.chi, args.chains, args.samples, args.chunk_size) < 1:
        parser.error("Geometry, bonds, sample counts and chunks must be positive")
    if args.startup_steps < 1 or args.steps < 1 or min(args.startup_dt, args.dt) <= 0:
        parser.error("Use positive time steps and at least one startup and tVMC step")
    if args.samples % args.chains:
        parser.error("Samples must divide into chains")
    if args.reference and args.lx * args.ly > 16:
        parser.error("The dense oracle supports at most 16 spins")
    if args.checkpoint_every < 1:
        parser.error("checkpoint-every must be positive")
    startup_time = args.startup_steps * args.startup_dt
    initial_time = startup_time if resumed is None else resumed[0]["time"]
    initial_step = 0 if resumed is None else resumed[0]["step"]
    if args.final_time is not None:
        if not np.isfinite(args.final_time) or args.final_time <= initial_time:
            parser.error("final-time must be finite and after the initial/resumed time")
        args.steps = max(1, int(np.ceil((args.final_time - initial_time) / args.dt - 1e-10)))
    if args.checkpoint_dir:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
         final_time=args.final_time, reference_origin=args.reference_origin,
         solver=args.solver, netket=nk.__version__, jax=jax.__version__, device=str(jax.devices()[0]))
    if resumed is None:
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
        state = fit.state
    else:
        state = resumed[2]
        emit("resumed", path=str(args.resume), time=initial_time, step=initial_step,
             sampling="fresh", D=state.max_bond())
    # NetKet Ising uses +J ZZ-h X with Pauli operators. The PEPO uses
    # -j ZZ-h X, so negate J exactly once at this integration boundary.
    setup = build_ising_vmc(state, Lx=args.lx, Ly=args.ly, h=args.h, J=-args.j,
        contraction="boundary", chi=args.chi, cutoff=0., parameter_mode="real-imag",
        n_samples=args.samples, n_chains=args.chains, n_discard_per_chain=args.discard,
        chunk_size=args.chunk_size, seed=args.seed, sampler_seed=args.seed + 1, use_sr=False)
    reference = handoff_reference = handoff_vector = hamiltonian = positions = initial_energy = None
    configurations = edges = None
    if args.reference:
        configurations = np.asarray(setup.hilbert.all_states())
        bits = ((1 - configurations) // 2).astype(int)
        positions = bits @ (2 ** np.arange(args.lx * args.ly - 1, -1, -1))
        handoff_vector, _ = normalized_peps_vector(state, positions)
        handoff_reference = handoff_vector.copy()
        reference = handoff_vector.copy()
        hamiltonian = setup.hamiltonian.to_sparse()
        if resumed is not None:
            reference = resumed[1]["reference_vector"]
            handoff_reference = resumed[1]["handoff_reference_vector"]
            handoff_vector = resumed[1]["handoff_vector"]
            initial_energy = resumed[0]["initial_energy"]
        elif args.reference_origin == "initial":
            reference = np.full(reference.size, 1 / np.sqrt(reference.size), dtype=complex)
            initial_energy = float(np.vdot(reference, hamiltonian @ reference).real)
            reference = expm_multiply(-1j * startup_time * hamiltonian, reference)
        else:
            initial_energy = float(np.vdot(reference, hamiltonian @ reference).real)
        edges = tuple(setup.graph.edges())
        emit("reference_ready", initial_energy=initial_energy,
             handoff_energy=float(np.vdot(handoff_vector, hamiltonian @ handoff_vector).real),
             startup_infidelity=(float(1 - abs(np.vdot(reference, handoff_vector))**2)
                                 if resumed is None else None),
             observables=dense_spin_observables(handoff_vector, configurations, edges))
    solver = args.solver
    solver_kwargs = None
    if args.solver == "pinv_smooth":
        solver = partial(nk.optimizer.solver.pinv_smooth, rtol_smooth=1e-10)
    elif args.solver == "cg":
        solver_kwargs = {"tol": args.cg_tol, "maxiter": args.cg_maxiter}
    dynamics = setup.make_tdvp_driver(dt=args.dt, t0=initial_time,
        diag_shift=args.diag_shift, linear_solver=solver, linear_solver_kwargs=solver_kwargs,
        solver_max_relative_residual=args.solver_max_relative_residual)
    emit("tdvp_ready", tensor_parameters=setup.n_params,
         real_coordinates=sum(z.size for z in jax.tree_util.tree_leaves(setup.vstate.parameters)),
         D=setup.to_peps().max_bond())

    def checkpoint(step, vector=None):
        if args.checkpoint_dir is None:
            return
        arrays = {}
        for name, leaf in setup.vstate.parameters.items():
            arrays[f"{name}_real"] = np.asarray(leaf["real"])
            arrays[f"{name}_imag"] = np.asarray(leaf["imag"])
        current = setup.to_peps()
        tensor_layout = []
        for index, tensor in enumerate(current):
            name = f"peps_tensor_{index}"
            arrays[name] = np.asarray(tensor.data)
            tensor_layout.append({"array": name, "inds": list(tensor.inds), "tags": list(tensor.tags)})
        if reference is not None:
            arrays["reference_vector"] = reference
            arrays["handoff_reference_vector"] = handoff_reference
            arrays["handoff_vector"] = handoff_vector
            arrays["peps_vector"] = vector if vector is not None else handoff_vector
            arrays["basis_physical_positions"] = positions
        arrays["metadata"] = np.asarray(json.dumps({
            "checkpoint_version": 2,
            "simulation_settings": {name: getattr(args, name) for name in _RESUME_SETTINGS},
            "tensor_layout": tensor_layout, "peps_exponent": float(current.exponent),
            "peps_layout": {"Lx": args.lx, "Ly": args.ly,
                            "site_tag_id": current.site_tag_id, "x_tag_id": current.x_tag_id,
                            "y_tag_id": current.y_tag_id, "site_ind_id": current.site_ind_id},
            "initial_energy": initial_energy,
            "step": step, "time": float(dynamics.t), "lx": args.lx, "ly": args.ly,
            "d": args.d, "chi": args.chi, "dt": args.dt, "samples": args.samples,
            "seed": args.seed, "j": args.j, "h": args.h,
            "startup_steps": args.startup_steps, "startup_dt": args.startup_dt,
            "diag_shift": args.diag_shift, "solver": args.solver,
            "reference_origin": args.reference_origin,
        }))
        write_checkpoint(args.checkpoint_dir / f"step-{step:04d}.npz", arrays)

    if resumed is None:
        checkpoint(0)
    for step in range(args.steps):
        completed_step = initial_step + step + 1
        step_started = time.perf_counter()
        emit("tdvp_step_start", step=completed_step)
        step_dt = args.dt
        if args.final_time is not None:
            step_dt = min(step_dt, args.final_time - float(dynamics.t))
        dynamics.advance(step_dt)
        tdvp_seconds = time.perf_counter() - step_started
        leaves = jax.tree_util.tree_leaves(setup.vstate.parameters)
        if not all(np.isfinite(np.asarray(z)).all() for z in leaves):
            raise RuntimeError("TDVP produced non-finite parameters")
        diagnostics = {}
        vector = None
        if args.reference:
            reference = expm_multiply(-1j * step_dt * hamiltonian, reference)
            handoff_reference = expm_multiply(-1j * step_dt * hamiltonian, handoff_reference)
            vector, log_squared_norm = normalized_peps_vector(setup.to_peps(), positions)
            with np.errstate(over="ignore", under="ignore"):
                squared_norm = float(np.exp(log_squared_norm))
            energy = float(np.vdot(vector, hamiltonian @ vector).real)
            overlap = np.vdot(reference, vector)
            phase = np.conjugate(overlap) / abs(overlap) if overlap != 0 else 1.
            diagnostics = {"dense_infidelity": float(1 - abs(np.vdot(reference, vector))**2),
                           "handoff_infidelity": float(1 - abs(np.vdot(handoff_reference, vector))**2),
                           "phase_aligned_distance": float(np.linalg.norm(reference - phase * vector)),
                           "dense_energy": energy, "energy_drift": energy - initial_energy,
                           "raw_squared_norm": squared_norm if np.isfinite(squared_norm) else None,
                           "raw_log_squared_norm": log_squared_norm,
                           "observables": dense_spin_observables(vector, configurations, edges),
                           "exact_observables": dense_spin_observables(reference, configurations, edges)}
            if not all(np.isfinite(x) for x in (log_squared_norm, energy)):
                raise RuntimeError("The dense state has a non-finite scale or energy")
            # Norm is a gauge-dependent scale in TDVP; observables use the
            # normalized state, while this value records numerical range.
            expected_time = initial_time + (step + 1) * args.dt
            if args.final_time is not None:
                expected_time = min(expected_time, args.final_time)
            if abs(float(dynamics.t) - expected_time) > 1e-10:
                raise RuntimeError("The accepted endpoint does not match the requested time")
        if completed_step % args.checkpoint_every == 0 or step + 1 == args.steps:
            checkpoint(completed_step, vector)
        if args.tdvp_diagnostics:
            diagnostics["tdvp_diagnostics"] = dynamics.diagnostics()
        emit("tdvp_step", step=completed_step, time=float(dynamics.t),
             step_seconds=time.perf_counter() - step_started,
             tdvp_seconds=tdvp_seconds,
             acceptance=float(setup.vstate.sampler_state.acceptance), **diagnostics)
    emit("complete", time=float(dynamics.t), D=setup.to_peps().max_bond())


if __name__ == "__main__":
    main()
