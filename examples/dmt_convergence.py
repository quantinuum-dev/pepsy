"""Matched small-2D chi/timestep study with exact, MPS and FIT references."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quimb.tensor as qtn

from pepsy.optimizers import MpsOptimizer

try:
    from examples import dmt_2d_dynamics as bench
except ModuleNotFoundError:
    import dmt_2d_dynamics as bench


def mps_trajectory(model, references, *, chi, steps, dt):
    state = qtn.MatrixProductState.from_dense(model['psi'], dims=[2]*model['n'], cutoff=0.)
    opt = MpsOptimizer(state, model['gates'], chi=chi, mode='direct', contraction_opt='greedy')
    row = dict(method='mps', chi=chi, status='completed', completed_steps=0,
               update_seconds=0., samples=[], rank_meaning='wavefunction bond')
    for step in range(steps + 1):
        if step:
            start = perf_counter()
            opt.run(cutoff=0., target_cutoff=0.)
            row['update_seconds'] += perf_counter()-start
            row['completed_steps'] = step
        if step in references:
            psi = np.asarray(opt.p.to_dense()).reshape(-1)
            norm = float(np.vdot(psi, psi).real)
            if not np.isfinite(norm) or norm <= 0:
                raise FloatingPointError('MPS norm is unresolved')
            # The physical mixture contains the normalized pure component.
            actual = bench.density(psi / np.sqrt(norm), model['pure_weight'])
            row['samples'].append(dict(time=step*dt, raw_wavefunction_norm_squared=norm,
                                       **bench.diagnostics(actual, *references[step], model)))
    return row


def study(shape=(3, 2), *, chis=(24, 32, 64), dts=(.1, .05), final_time=2.,
          methods=('dmt', 'fit', 'mps'), output=None):
    if not np.isfinite(final_time) or final_time <= 0 or not chis or min(chis) < 1:
        raise ValueError('Require positive final_time and bond caps')
    if not methods or any(m not in {'dmt', 'optimized', 'fit', 'mps'} for m in methods):
        raise ValueError('Unknown comparison method')
    result = dict(shape=list(shape), final_time=final_time, dtype='complex128', rows=[],
                  rank_comparison='Equal caps do not imply equal storage: MPO operator bonds versus MPS wavefunction bonds.')
    for dt in dts:
        if not np.isfinite(dt) or dt <= 0 or not np.isclose(final_time/dt, round(final_time/dt)):
            raise ValueError('Each positive dt must divide final_time')
        steps = round(final_time/dt)
        model = bench.problem(*shape, dt)
        references = bench.exact_references(model, dt, steps, max(1, steps//4))
        for chi in chis:
            for method in methods:
                try:
                    row = (mps_trajectory(model, references, chi=chi, steps=steps, dt=dt)
                           if method == 'mps' else bench.trajectory(
                               model, references, method=method, chi=chi, steps=steps, dt=dt))
                except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
                    row = dict(method=method, chi=chi, status='failed', error=str(exc), samples=[])
                row['dt'] = dt
                row['reached_final_time'] = (row['status'] == 'completed'
                                             and row.get('completed_steps') == steps)
                row.setdefault('rank_meaning', 'operator MPO bond')
                result['rows'].append(row)
                if output:
                    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in {'samples', 'warning_examples'}}), flush=True)
    return result


def plot_summary(data, prefix):
    """Render the saved aggregate without rerunning numerical trajectories."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    rows = data['convergence_3x3']['rows'] + data['convergence_3x3_high']['rows']
    for dt, style in [(.1, '-'), (.05, '--')]:
        for method, color, marker in [('dmt', 'C0', 'o'), ('fit', 'C1', 's')]:
            chosen = sorted([r for r in rows if r['dt'] == dt and r['method'] == method
                             and r['status'] == 'completed'], key=lambda r: r['chi'])
            axes[0].loglog([r['chi'] for r in chosen],
                           [r['samples'][-1]['max_plaquette_error'] for r in chosen],
                           style+marker, color=color, label=f'{method.upper()}, dt={dt}')
        floor = next(r['samples'][-1]['max_plaquette_error'] for r in rows
                     if r['method'] == 'mps' and r['dt'] == dt)
        axes[0].axhline(floor, color='.35', ls=style, lw=1, label=f'Exact circuit dt={dt}')
    axes[0].set(title='3×3 at t=2: rank and timestep error', xlabel='Operator MPO bond cap χ',
                ylabel='Max plaquette error vs continuous time')
    axes[0].legend(fontsize=7, loc='lower left')
    for chi, row in data['stability_6x6'].items():
        axes[1].semilogy([s['time'] for s in row['samples']],
                         [max(s['trace_error'], 1e-16) for s in row['samples']], 'o-', label=f'χ={chi}')
    axes[1].set(title='6×6: short-run stability after QR fix', xlabel='Time', ylabel='Absolute trace error')
    axes[1].legend(fontsize=8)
    axes[1].text(.04, .12, 'Inter-χ plaquette gaps ≈ 0.01\nNot converged; global PSD unmeasured',
                 transform=axes[1].transAxes, fontsize=8)
    rows = data['positive_factor_3x2']['rows']
    kappas = [r['kappa'] for r in rows]
    axes[2].plot(kappas, [r['samples'][-1]['max_plaquette_error'] for r in rows], 'o-',
                 label='Final plaquette error')
    axes[2].plot(kappas, [r['max_protected_target_defect'] for r in rows], 's--',
                 label='Max per-gate marginal change')
    axes[2].set(title='3×2 at t=1: positive-factor control', xlabel='Factor bond κ (density bond ≤ κ²)',
                ylabel='Frobenius error')
    axes[2].legend(fontsize=8)
    axes[2].text(.04, .43, 'ρ = XX† remains positive\nExact DMT marginals are not retained',
                 transform=axes[2].transAxes, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle('DMT robustness: scoped QR workaround, bounded convergence, explicit positivity tradeoff')
    for extension in ('png', 'svg'):
        fig.savefig(Path(prefix).with_suffix('.'+extension), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', nargs=2, type=int, default=[3, 2])
    parser.add_argument('--chis', nargs='+', type=int, default=[24, 32, 64])
    parser.add_argument('--dts', nargs='+', type=float, default=[.1, .05])
    parser.add_argument('--final-time', type=float, default=2.)
    parser.add_argument('--methods', nargs='+', choices=['dmt', 'optimized', 'fit', 'mps'],
                        default=['dmt', 'fit', 'mps'])
    parser.add_argument('--output', type=Path, default=Path('/tmp/dmt-convergence.json'))
    parser.add_argument('--plot-summary', type=Path, help='Plot a saved robustness aggregate; skip evolution')
    args = parser.parse_args()
    if args.plot_summary:
        plot_summary(json.loads(args.plot_summary.read_text()), args.output.with_suffix(''))
        return
    study(args.shape, chis=args.chis, dts=args.dts, final_time=args.final_time,
          methods=args.methods, output=args.output)


if __name__ == '__main__':
    main()
