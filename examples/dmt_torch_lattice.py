"""Public Torch interval/full DMT on a physical open square-lattice Ising model.

Both representations use identical exact transfer MPOs and protected physical
plaquettes. Ordinary MPOs use matrix units; PauliMPO stores I/X/Y/Z coefficients.
No global density matrix or statevector is constructed for 6x6.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
import torch

from pepsy.backends import TorchLinalgConfig
from pepsy.operators import PauliMPO
from pepsy.optimizers import MpoOptimizer, PauliDMT

try:
    from examples import dmt_2d_window as model_builder
except ModuleNotFoundError:
    import dmt_2d_window as model_builder


def matrix_unit_transfers(operators):
    """Exact coordinate change, E = B P and B^{-1}=B†/2; no core rounding."""
    sample = operators[0]
    basis = torch.as_tensor(model_builder.ref.PAULIS.transpose(1, 2, 0).reshape(4, 4),
                            dtype=torch.complex128, device=sample.device)
    inverse = basis.conj().T / 2
    return tuple(torch.einsum('oq,abqr,ri->aboi', basis, op.to(torch.complex128), inverse)
                 for op in operators)


def driver(model, chi, representation, dtype, *, check_regions=False):
    rho = PauliMPO.from_pauli_cores([torch.as_tensor(a, dtype=dtype) for a in model['cores']])
    options = dict(regions=model['regions'], region_selection='min_cover',
                   preservation_check='warn', check_regions=check_regions, cutoff=0.)
    if representation == 'pauli':
        return PauliDMT(rho, max_bond=chi, **options), options
    if dtype != torch.complex128:
        raise ValueError('Ordinary matrix-unit comparison requires complex128')
    return MpoOptimizer(rho.to_mpo(), gates=[], chi=chi, mode='dmt', contraction_opt='greedy'), options


def apply_transfer(run, options, representation, start, operators, update='interval'):
    if representation == 'pauli':
        return run.apply_transfer_mpo(start, operators, absorb='right', update=update)
    run.apply_transfer_mpo(start, operators, absorb='right', update=update, **options)
    reports = run.get_dmt_reports()
    return reports[-1] if reports else None


def readout(run, representation, model):
    mpo = run.state.to_mpo() if representation == 'pauli' else run.p
    observed = mpo.copy()
    observed.apply_to_arrays(lambda a: a.detach().cpu().numpy())
    cores = model_builder.ref.mpo_cores(observed)
    panels = model_builder.ref.Marginals(cores)
    regions = [panels.region(r) for r in model['regions']]
    one = [panels.region((j,)) for j in range(model['n'])]
    energy = sum(np.trace(panels.region(tuple(sorted(edge))) @
                           np.kron(model_builder.ref.Z, model_builder.ref.Z)) for edge in model['edges'])
    energy += sum(np.trace(p @ (model['hx']*model_builder.ref.X+model['hz']*model_builder.ref.Z)) for p in one)
    return dict(trace_error=float(abs(panels.trace-1)), energy_per_site=float(energy.real/model['n']),
                max_local_hermiticity=float(max(np.linalg.norm(p-p.conj().T) for p in regions)),
                min_local_eigenvalue=float(min(np.linalg.eigvalsh((p+p.conj().T)/2)[0] for p in regions)),
                actual_max_bond=int(mpo.max_bond())), np.stack(regions)


def trajectory(shape=(6, 6), *, chi=96, steps=20, dt=.1, representation='pauli',
               update='interval', dtype='complex128', cpu_svd='scipy_gesvd',
               cpu_qr='torch', max_seconds=600., output=None, sample_every=1,
               hx=1.05, hz=.5, pure_weight=.6):
    if representation not in {'pauli', 'mpo'} or update not in {'interval', 'full'}:
        raise ValueError('Unknown representation or update route')
    if dtype not in {'float64', 'complex128'} or steps < 1 or sample_every < 1:
        raise ValueError('Require double precision and positive step/sample counts')
    torch.set_num_threads(1)
    model = model_builder.problem(*shape, dt, hx=hx, hz=hz, weight=pure_weight)
    policy = TorchLinalgConfig(cpu_svd=cpu_svd, cpu_qr=cpu_qr,
                               quimb_split_drivers=True, qr_rank_policy='native')
    result = dict(shape=list(shape), chi=chi, dt=dt, requested_steps=steps, completed_steps=0,
                  representation=representation, dtype=dtype, backend='torch', device='cpu',
                  cpu_svd=cpu_svd, cpu_qr=cpu_qr, update=update, status='running', samples=[], step_seconds=[],
                  preservation_errors=[], physics=dict(Jzz=1., hx=model['hx'], hz=model['hz'],
                  initial_pure_weight=model['weight'], edges=len(model['edges']), plaquettes=len(model['regions'])),
                  canonical_info=[], global_positivity='unmeasured')
    saved = {}

    def checkpoint():
        if output:
            Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            np.savez_compressed(str(output)+'.npz', **saved)

    with policy.activated():
        run, options = driver(model, chi, representation, getattr(torch, dtype))
        field = torch.as_tensor(model_builder.field_transfer(dt, model['hx'], model['hz']),
                                dtype=getattr(torch, dtype)).reshape(1, 1, 4, 4)
        field = (field,) if representation == 'pauli' else matrix_unit_transfers((field,))
        edge_maps = {}
        for a, b in model['edges']:
            maps = tuple(torch.as_tensor(op, dtype=getattr(torch, dtype))
                         for op in model_builder.zz_transfer(b-a+1, dt))
            edge_maps[a, b] = maps if representation == 'pauli' else matrix_unit_transfers(maps)

        def sample(step):
            row, panels = readout(run, representation, model)
            result['samples'].append(dict(time=step*dt, step=step, **row))
            saved[str(step)] = panels

        sample(0)
        try:
            for step in range(1, steps+1):
                started = perf_counter()
                for j in range(model['n']):
                    report = apply_transfer(run, options, representation, j, field)
                    if report:
                        result['preservation_errors'].extend(report.preservation_errors)
                for edge in model['edges']:
                    report = apply_transfer(run, options, representation, edge[0], edge_maps[edge], update)
                    if report:
                        result['preservation_errors'].extend(report.preservation_errors)
                for j in range(model['n']):
                    report = apply_transfer(run, options, representation, j, field)
                    if report:
                        result['preservation_errors'].extend(report.preservation_errors)
                result['step_seconds'].append(perf_counter()-started)
                result['completed_steps'] = step
                result['canonical_info'].append(run.info_c['cur_orthog'])
                exhausted = sum(result['step_seconds']) >= max_seconds
                if step % sample_every == 0 or step == steps or exhausted:
                    sample(step)
                checkpoint()
                print(f'{representation}/{dtype} chi={chi} {update} t={step*dt:.2f} '
                      f'{result["step_seconds"][-1]:.2f}s', flush=True)
                if exhausted:
                    break
            result['status'] = 'completed' if result['completed_steps'] == steps else 'time_budget'
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                          traceback=traceback.format_exc())
        checkpoint()
    return result


def summarize(paths, output):
    """Compare saved panels only at common times and with matched settings.

    Failed runs remain in the record but are excluded from comparisons.
    Passing the stability checks does not certify trajectory accuracy or PSD.
    """
    rows, panels = [], []
    for path in paths:
        row = json.loads(Path(path).read_text())
        row.update(source=Path(path).name, cpu_qr=row.get('cpu_qr', 'torch'))
        row['comparison_eligible'] = (row['status'] in {'completed', 'time_budget'}
                                      and not row['preservation_errors'])
        rows.append(row)
        with np.load(str(path)+'.npz') as saved:
            panels.append({key: saved[key] for key in saved.files})
    comparisons = []
    controls = ('chi', 'representation', 'dtype', 'update', 'cpu_svd', 'cpu_qr')
    for i, j in combinations(range(len(rows)), 2):
        a, b = rows[i], rows[j]
        if (not a['comparison_eligible'] or not b['comparison_eligible']
                or any(a[k] != b[k] for k in ('shape', 'dt', 'physics', 'backend', 'device'))):
            continue
        changed = [key for key in controls if a[key] != b[key]]
        if len(changed) != 1:
            continue
        common = sorted(panels[i].keys() & panels[j].keys(), key=int)
        comparisons.append(dict(a=a['source'], b=b['source'], varied=changed[0], samples=[
            dict(step=int(step), time=int(step)*a['dt'], max_plaquette_difference=float(
                np.linalg.norm(panels[i][step]-panels[j][step], axis=(-2, -1)).max()))
            for step in common]))
    result = dict(runs=rows, comparisons=comparisons,
                  norm='max physical plaquette Frobenius norm; raw unnormalized marginals',
                  timing='concurrent CPU jobs; not an isolated speed benchmark',
                  global_positivity='unmeasured')
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', nargs=2, type=int, default=[6, 6])
    parser.add_argument('--chi', type=int, default=96)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--representation', choices=['pauli', 'mpo'], default='pauli')
    parser.add_argument('--update', choices=['interval', 'full'], default='interval')
    parser.add_argument('--dtype', choices=['float64', 'complex128'], default='complex128')
    parser.add_argument('--cpu-svd', choices=['torch', 'scipy_gesvd', 'scipy_gesdd'], default='scipy_gesvd')
    parser.add_argument('--cpu-qr', choices=['torch', 'scipy_pivoted'], default='torch')
    parser.add_argument('--hx', type=float, default=1.05)
    parser.add_argument('--hz', type=float, default=.5)
    parser.add_argument('--pure-weight', type=float, default=.6)
    parser.add_argument('--max-seconds', type=float, default=600.)
    parser.add_argument('--output', type=Path, default=Path('/tmp/dmt-torch-lattice.json'))
    parser.add_argument('--summarize', nargs='+', type=Path,
                        help='Compare saved JSON/NPZ runs at common times; skip evolution')
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize, args.output)
        return
    trajectory(args.shape, chi=args.chi, steps=args.steps, dt=args.dt, representation=args.representation,
               update=args.update, dtype=args.dtype, cpu_svd=args.cpu_svd,
               cpu_qr=args.cpu_qr, hx=args.hx, hz=args.hz, pure_weight=args.pure_weight,
               max_seconds=args.max_seconds, output=args.output)


if __name__ == '__main__':
    main()
