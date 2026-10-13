"""Fixed-bond flat Symmray Z2 PEPS tVMC with optional Torch compilation.

TFIM is written in its parity-preserving basis, H=-J sum XX-h sum Z.
Try: python examples/tvmc_flat_z2.py --compile-backend aot_eager --steps 1
"""

import argparse
import json

import symmray as sr
import torch

from pepsy.backends import TorchLinalgConfig
from pepsy.vmc import SamplingConfig, TorchPEPSAmplitude, TorchTDVPDriver, TorchVMCDriver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lx', type=int, default=2)
    parser.add_argument('--ly', type=int, default=2)
    parser.add_argument('--d', type=int, default=4)
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--contraction', choices=('exact', 'boundary', 'ctmrg'), default='boundary')
    parser.add_argument('--compile-backend', choices=('none', 'eager', 'aot_eager', 'inductor'), default='aot_eager')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--optimize-steps', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=.01)
    parser.add_argument('--sr-diag-shift', type=float, default=1e-4)
    parser.add_argument('--dt', type=float, default=.001)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--chains', type=int, default=8)
    parser.add_argument('--seed', type=int, default=22)
    args = parser.parse_args()
    if args.d % 2:
        parser.error('Flat Z2 requires equal sector sizes: use an even total D')
    if args.contraction == 'ctmrg' and args.compile_backend != 'none':
        parser.error('CTMRG uses eager/vmap here; select --compile-backend none')
    if args.lx * args.ly < 2:
        parser.error('Parity pair sampling requires at least two connected sites')
    torch.set_num_threads(1)
    with TorchLinalgConfig(stabilized=True, quimb_split_drivers=True,
                           qr_rank_policy='native').activated():
        peps = sr.PEPS_abelian_rand('Z2', args.lx, args.ly, bond_dim=args.d,
                                   flat=True, dtype='complex128', seed=args.seed)
        model = TorchPEPSAmplitude(peps, contraction=args.contraction,
            chi=None if args.contraction == 'exact' else args.chi, cutoff=0.,
            dtype=torch.complex128, device=args.device, amplitude_batching='vmap')
        sites = tuple(model.sites)
        positions = {site: i for i, site in enumerate(sites)}
        edges = [(positions[x, y], positions[other]) for x, y in sites
                 for other in ((x + 1, y), (x, y + 1)) if other in positions]
        x = torch.randint(2, (args.chains, len(sites)),
                          generator=torch.Generator().manual_seed(args.seed))
        x[:, -1] = x[:, :-1].sum(dim=1) % 2
        x = x.to(args.device)
        if args.compile_backend != 'none':
            model.export_and_compile(x, backend=args.compile_backend)
        z = torch.diag(torch.tensor([1., -1.], device=args.device))
        pauli_x = torch.tensor([[0., 1.], [1., 0.]], device=args.device)
        terms = {i: -z for i in range(len(sites))}
        terms.update({edge: -torch.kron(pauli_x, pauli_x).reshape(2, 2, 2, 2) for edge in edges})
        vmc = TorchVMCDriver(model, edges, x, terms=terms, proposal='parity')
        dynamics = TorchTDVPDriver(vmc, dt=args.dt, method='minsr',
            parameter_mode='real-imag', gauge_projection=False, derivative_backend='batched')
        sampling = SamplingConfig(n_samples=args.samples, n_chains=args.chains,
                                   n_discard_per_chain=8, sweep_size=2, seed=args.seed)
        print(json.dumps({'event': 'ready', 'D': peps.max_bond(), 'chi': model.chi,
                          'cutoff': model.cutoff, 'compile_backend': args.compile_backend}), flush=True)
        for step in range(args.optimize_steps):
            samples = vmc.sample(sampling=sampling)
            update = vmc.step(samples=samples, sr=True, sr_method='minsr',
                sr_parameter_mode='real-imag', learning_rate=args.learning_rate,
                sr_diag_shift=args.sr_diag_shift, derivative_backend='batched')
            print(json.dumps({'event': 'optimization', 'step': step + 1,
                'energy_before_update': update.energy_mean.real.item(),
                'D': model.to_peps().max_bond(), 'solver': update.sr.info['solver']}), flush=True)
        for _ in range(args.steps):
            result = dynamics.step(sampling=sampling)
            stage = result.stages[-1]
            print(json.dumps({'event': 'step', 'time': result.time,
                'energy': stage.energy_mean.real.item(), 'D': model.to_peps().max_bond(),
                'tdvp_residual': stage.info['tdvp_relative_residual'],
                'unique_samples': stage.info['n_unique_configs']}), flush=True)


if __name__ == '__main__':
    main()
