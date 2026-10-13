"""One exact bond-two TFIM Trotter PEPO, followed by one whole-state fit.

Run from the checkout: python examples/tfim_pepo.py --method als --steps 1
This is a small CPU example; no benchmark or checkpoint jobs are launched.
"""

import argparse
import math

from pepsy.operators import build_pepo_from_gates, rx, rzz
from pepsy.optimizers import PepsOptimizer
from pepsy.tensors import ps_to_peps


def trotter_pepo(lx, ly, dt, *, j=1., h=1., backend='numpy'):
    """Exact Strang step for H=-J sum ZZ-h sum X, with open boundaries.

    Each ZZ edge is used once and the edges commute. Onsite rotations do not
    change operator bonds, so the full step has PEPO bond dimension two.
    ``max_bond=2`` is therefore an exact algebraic cap for this model/ordering,
    rather than a generic assertion about products of arbitrary gate layers.
    """
    sites = [(x, y) for x in range(lx) for y in range(ly)]
    x_angle, zz_angle = -h * dt, -2 * j * dt
    if backend == 'torch':
        import torch
        x_angle, zz_angle = (torch.tensor(t, dtype=torch.float64) for t in (x_angle, zz_angle))
    elif backend != 'numpy':
        raise ValueError('This example supports numpy and torch')
    x_gate, zz_gate = rx(x_angle), rzz(zz_angle)
    one = [(x_gate, (site,)) for site in sites]
    edges = [(site, (site[0] + dx, site[1] + dy)) for site in sites
             for dx, dy in ((1, 0), (0, 1))
             if site[0] + dx < lx and site[1] + dy < ly]
    return build_pepo_from_gates(one + [(zz_gate, pair) for pair in edges] + one,
                                 max_bond=2, cutoff=0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('als', 'column-lbfgs', 'global'), default='als')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--lx', type=int, default=2)
    parser.add_argument('--ly', type=int, default=3)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--norm-chi', type=int, default=32)
    parser.add_argument('--overlap-chi', type=int, default=128)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--initial', choices=('zero', 'plus'), default='zero')
    parser.add_argument('--tvmc-steps', type=int, default=0)
    parser.add_argument('--tvmc-dt', type=float, default=.01)
    parser.add_argument('--tvmc-chi', type=int, default=32)
    parser.add_argument('--tvmc-samples', type=int, default=256)
    parser.add_argument('--tvmc-chains', type=int, default=16)
    args = parser.parse_args()
    if args.tvmc_steps and args.steps < 1:
        parser.error('Use at least one PEPO startup step before tVMC')
    backend = 'numpy' if args.method == 'als' else 'torch'
    state = ps_to_peps(args.lx, args.ly, theta=0. if args.initial == 'zero' else math.pi / 4)
    if backend == 'torch':
        import torch
        torch.set_num_threads(1)
        state.apply_to_arrays(lambda x: torch.as_tensor(x, dtype=torch.complex128))
    operator = trotter_pepo(args.lx, args.ly, args.dt, backend=backend)
    fit = PepsOptimizer(state, [], chi=args.d, mode='global' if args.method == 'global' else 'sweep',
        boundary_chi=(args.norm_chi, args.overlap_chi), normalize_chi=args.norm_chi,
        evaluation_chi=(args.norm_chi, args.overlap_chi), contraction_opt='greedy',
        optimizer={'global': 'L-BFGS-B', 'als': 'als', 'column-lbfgs': 'scipy'}[args.method],
        global_optimize_kwargs={'n': 20, 'autodiff_backend': 'torch'},
        sweep_kwargs={'local_contraction_opt': 'greedy'},
        sweep_optimize_kwargs={} if args.method != 'column-lbfgs' else {
            'axes': ('y',), 'optimizer_options': {'n_steps': 20, 'ftol': 1e-12, 'gtol': 1e-10}})
    for step in range(args.steps):
        fit.run_pepo(operator, unitary=True)
        record = fit.get_step_records()[0]
        print(f"step={step + 1} PEPO_D={operator.max_bond()} "
              f"PEPS_D={fit.state.max_bond()} infidelity={record['final_infidelity']:.6g} "
              f"status={record['reason']}")
    if args.tvmc_steps:
        import torch
        from pepsy.vmc import (SamplingConfig, TorchPEPSAmplitude, TorchTDVPDriver,
                               TorchVMCDriver)

        torch.set_num_threads(1)
        model = TorchPEPSAmplitude(fit.state, dtype=torch.complex128,
            contraction='boundary', chi=args.tvmc_chi, cutoff=0.,
            contraction_opts={'optimize': 'greedy'})
        sites = tuple(model.sites)
        positions = {site: i for i, site in enumerate(sites)}
        edges = [(positions[x, y], positions[other]) for x, y in sites
                 for other in ((x + 1, y), (x, y + 1)) if other in positions]
        # The native helper uses spin-1/2 operators, hence J=-4 and h=-2
        # give exactly the same H=-sum ZZ-sum X as the PEPO startup.
        vmc = TorchVMCDriver(model, edges,
            torch.zeros((args.tvmc_chains, len(sites)), dtype=torch.long),
            connection_fn='transverse_ising', connection_kwargs={'J': -4., 'h': -2.},
            proposal='local')
        dynamics = TorchTDVPDriver(vmc, dt=args.tvmc_dt, t0=args.steps * args.dt,
                                  parameter_mode='real-imag')
        sampling = SamplingConfig(n_samples=args.tvmc_samples, n_chains=args.tvmc_chains,
                                  n_discard_per_chain=8, sweep_size=2)
        for _ in range(args.tvmc_steps):
            result = dynamics.step(sampling=sampling)
            stage = result.stages[-1]
            print(f"t={result.time:.6f} PEPS_D={model.to_peps().max_bond()} "
                  f"E={stage.energy_mean.real.item():.6g} method={stage.method} "
                  f"TDVP_residual={stage.info['tdvp_relative_residual']:.3g}")


if __name__ == '__main__':
    main()
