"""Live, reproducible full-PEPO boundary-cap probe with a dense TFIM oracle.

Run: python examples/tfim_pepo_chi_probe.py --lx 4 --ly 4 --d 3
Each JSON line is flushed immediately. Only this bounded probe is launched;
states stay in memory and no persistent background job is created.
"""

import argparse
import json
import time

import numpy as np
import quimb.tensor as qtn

from pepsy.optimizers import PepsOptimizer
from pepsy.tensors import ps_to_peps


def dense_step(vector, lx, ly, dt, *, j=1., h=1.):
    """Independent Pauli Strang evolution without constructing a dense PEPO."""
    n = lx * ly
    tensor = np.array(vector, dtype=np.complex128, copy=True).reshape((2,) * n)
    half = np.cos(h * dt / 2) * np.eye(2) + 1j * np.sin(h * dt / 2) * np.array([[0, 1], [1, 0]])
    for site in range(n):
        tensor = np.tensordot(half, tensor, axes=(1, site)).transpose(
            tuple(range(1, site + 1)) + (0,) + tuple(range(site + 1, n)))
    codes = np.arange(2**n, dtype=np.uint64)
    spin = [1 - 2 * ((codes >> (n - 1 - site)) & 1).astype(np.int8) for site in range(n)]
    diagonal = np.zeros(2**n)
    for x in range(lx):
        for y in range(ly):
            i = x * ly + y
            if x + 1 < lx:
                diagonal += spin[i] * spin[i + ly]
            if y + 1 < ly:
                diagonal += spin[i] * spin[i + 1]
    tensor *= np.exp(1j * j * dt * diagonal).reshape(tensor.shape)
    for site in range(n):
        tensor = np.tensordot(half, tensor, axes=(1, site)).transpose(
            tuple(range(1, site + 1)) + (0,) + tuple(range(site + 1, n)))
    return tensor.reshape(-1)


def dense_loss(vector, target):
    a, b = np.asarray(vector).reshape(-1), np.asarray(target).reshape(-1)
    return float(1 - abs(np.vdot(a, b))**2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def main():
    from tfim_pepo import trotter_pepo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lx', type=int, default=4)
    parser.add_argument('--ly', type=int, default=4)
    parser.add_argument('--d', type=int, default=3)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--initial', choices=('random', 'plus'), default='random')
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--norm-chi', type=int, default=36)
    parser.add_argument('--overlap-chi', type=int, default=144)
    parser.add_argument('--max-norm-chi', type=int, default=144)
    parser.add_argument('--max-overlap-chi', type=int, default=576)
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--round-trips', type=int, default=1)
    parser.add_argument('--probe-chis', type=int, nargs='*', default=[18, 36, 72, 144])
    args = parser.parse_args()
    if args.lx * args.ly > 16:
        parser.error('Dense validation is restricted to at most 16 qubits')
    started = time.perf_counter()

    def emit(event, **data):
        print(json.dumps(dict(event=event, elapsed=time.perf_counter() - started, **data)), flush=True)

    emit('setup', shape=[args.lx, args.ly], D=args.d, seed=args.seed,
         caps=[args.norm_chi, args.overlap_chi], cutoff=0., initial=args.initial)
    state = (qtn.PEPS.rand(args.lx, args.ly, bond_dim=args.d, dtype='complex128', seed=args.seed)
             if args.initial == 'random' else ps_to_peps(args.lx, args.ly, theta=np.pi / 4))
    vector = np.asarray(state.to_dense(optimize='greedy')).reshape(-1)
    state.multiply_(1 / np.linalg.norm(vector))
    reference = vector / np.linalg.norm(vector)
    operator = trotter_pepo(args.lx, args.ly, args.dt)
    emit('ready', PEPS_D=state.max_bond(), PEPO_D=operator.max_bond())
    fit = PepsOptimizer(state, [], chi=args.d, mode='sweep', optimizer='als',
        boundary_chi=(args.norm_chi, args.overlap_chi), normalize_chi=args.norm_chi,
        evaluation_chi=(args.norm_chi, args.overlap_chi), contraction_opt='greedy',
        boundary_convergence=False if args.fixed else {
            'policy': 'on_failure', 'max_chi': (args.max_norm_chi, args.max_overlap_chi)},
        normalize_initial=False, sweep_kwargs={'local_contraction_opt': 'greedy', 'cutoff': 0.},
        normalize_kwargs={'cutoff': 0.}, infidelity_kwargs={'cutoff': 0.},
        sweep_optimize_kwargs={'n_round_trips': args.round_trips})
    for step in range(args.steps):
        source = fit.state.copy()
        before = np.asarray(source.to_dense(optimize='greedy')).reshape(-1)
        target_vector = dense_step(before, args.lx, args.ly, args.dt)
        reference = dense_step(reference, args.lx, args.ly, args.dt)
        warm = operator.apply(source, contract=True, compress=True, max_bond=args.d, cutoff=0.)
        warm_vector = np.asarray(warm.to_dense(optimize='greedy')).reshape(-1)
        emit('fit_start', step=step + 1, caps=fit.boundary_chi,
             warm_dense_infidelity=dense_loss(warm_vector, target_vector))
        try:
            fit.run_pepo(operator, unitary=True, infidelity_tol=0., cutoff=0., timing=True)
        except Exception as error:
            emit('fit_failed', step=step + 1, error=repr(error),
                 growth=None if fit._boundary_growth is None else fit._boundary_growth.events,
                 timing=fit.get_timing())
            raise
        record = fit.get_step_records()[0]
        final = np.asarray(fit.state.to_dense(optimize='greedy')).reshape(-1)
        emit('fit_complete', step=step + 1, PEPS_D=fit.state.max_bond(),
             caps=fit.boundary_chi, reason=record['reason'],
             pre_infidelity=record['pre_infidelity'], reported_infidelity=record['final_infidelity'],
             dense_step_infidelity=dense_loss(final, target_vector),
             dense_circuit_infidelity=dense_loss(final, reference),
             dense_squared_norm=float(np.vdot(final, final).real),
             growth=record['boundary_growth'], timing=fit.get_timing())
        target = operator.apply(source, contract=False)
        for chi in args.probe_chis:
            emit('metric_start', step=step + 1, caps=[chi, 4 * chi])
            try:
                loss = fit.estimate_infidelity(fit.state, target, evaluation_chi=(chi, 4 * chi),
                    evaluation_max_retries=0, norm_target=float(np.vdot(before, before).real), cutoff=0.)
                emit('metric_complete', step=step + 1, caps=[chi, 4 * chi], infidelity=loss,
                     dense_error=loss - dense_loss(final, target_vector))
            except Exception as error:
                emit('metric_failed', step=step + 1, caps=[chi, 4 * chi], error=repr(error))
    emit('complete')


if __name__ == '__main__':
    main()
