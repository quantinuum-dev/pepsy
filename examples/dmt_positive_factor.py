"""Research control: truncate a locally purified factor, retaining positivity.

This is not DMT and does not preserve its exact marginal constraints. The
factor bond is kappa; the represented density MPO can have bond kappa**2.
Only unitary gates and NumPy complex128 are included in this prototype.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quimb.tensor as qtn

try:
    from examples import dmt_2d_dynamics as bench
except ModuleNotFoundError:
    import dmt_2d_dynamics as bench


class PositiveFactor:
    """MPS on interleaved system/ancilla pairs, representing rho = X X†."""

    def __init__(self, factor, kappa):
        if isinstance(kappa, bool) or not isinstance(kappa, (int, np.integer)) or kappa < 2:
            raise ValueError('This rank-two-input prototype requires integer kappa >= 2')
        self.factor = factor.copy()
        self.kappa = int(kappa)
        self.info = {'cur_orthog': None}
        self.last_norm_squared = 1.

    @classmethod
    def depolarized_product(cls, vectors, weight=.6, *, kappa=8):
        """Exact rank-two factor sqrt(a I + w |psi><psi|), with no dense state."""
        vectors = [np.asarray(v, dtype=np.complex128) for v in vectors]
        if len(vectors) < 2 or not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('Require at least two sites and weight in [0, 1]')
        if any(v.shape != (2,) or not np.isclose(np.vdot(v, v), 1., atol=1e-13, rtol=1e-13)
               for v in vectors):
            raise ValueError('Require normalized two-component site vectors')
        a = (1-weight)*2.**(-len(vectors))
        coefficients = [np.sqrt(a), np.sqrt(a+weight)-np.sqrt(a)]
        arrays = []
        for j, vector in enumerate(vectors):
            core = np.zeros((2, 2, 4), dtype=np.complex128)
            core[0, 0] = np.eye(2).reshape(-1)
            core[1, 1] = np.outer(vector, vector.conj()).reshape(-1)
            if j == 0:
                core = np.einsum('a,abp->bp', coefficients, core)
            elif j == len(vectors)-1:
                core = core.sum(axis=1)
            arrays.append(core)
        return cls(qtn.MatrixProductState(arrays, shape='lrp'), kappa)

    def apply_gate(self, unitary, where):
        where = tuple(where)
        u = np.asarray(unitary, dtype=np.complex128)
        if (len(where) not in (1, 2) or len(set(where)) != len(where)
                or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                       or not 0 <= i < self.factor.L for i in where)
                or u.shape != (2**len(where),)*2):
            raise ValueError('Require a one- or two-site gate on distinct valid sites')
        if not np.all(np.isfinite(u)) or not np.allclose(u.conj().T@u, np.eye(len(u)), atol=2e-13, rtol=2e-13):
            raise ValueError('This prototype requires a finite unitary gate')
        # Disposable update: a failed factorization cannot change live state.
        candidate, info = self.factor.copy(), dict(self.info)
        if len(where) == 1:
            candidate.gate_(np.kron(u, np.eye(2)), where[0], contract=True)
        else:
            transfer = np.einsum('ABab,Cc,Dd->ACBDacbd', u.reshape((2,)*4),
                                 np.eye(2), np.eye(2)).reshape(16, 16)
            candidate.gate_with_auto_swap(transfer, where, info=info, inplace=True,
                                          max_bond=self.kappa, cutoff=0.)
        center = info.get('cur_orthog')
        center = center[0] if isinstance(center, tuple) else center
        norm = float(np.real(candidate.normalize(insert=0 if center is None else center)))
        if not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError('Purification norm is unresolved')
        self.factor, self.info, self.last_norm_squared = candidate, info, norm
        return norm

    def to_dense(self, *, max_sites=8):
        if self.factor.L > max_sites:
            raise ValueError('Dense purification readout exceeds max_sites')
        n = self.factor.L
        vector = np.asarray(self.factor.to_dense(optimize='greedy')).reshape((2,)*(2*n))
        # Site legs are (system, ancilla); gather each family before XX†.
        x = vector.transpose(tuple(range(0, 2*n, 2)) + tuple(range(1, 2*n, 2))).reshape(2**n, 2**n)
        return x @ x.conj().T


def trajectory(model, references, *, vectors, kappa, steps, dt):
    state = PositiveFactor.depolarized_product(vectors, model['pure_weight'], kappa=kappa)
    row = dict(method='purified-factor', kappa=kappa, density_bond_ceiling=kappa**2,
               update_seconds=0., samples=[], max_protected_target_defect=0.)
    for step in range(steps+1):
        if step:
            for u, where in model['gates']:
                # Small-system oracle audits the tradeoff rather than hiding it.
                before = state.to_dense()
                target = bench.apply_density(before, u, where, model['n'])
                start = perf_counter()
                state.apply_gate(u, where)
                row['update_seconds'] += perf_counter()-start
                actual = state.to_dense()
                defect = max(np.linalg.norm(bench.ref.partial_trace(actual, r, model['n'])-
                                             bench.ref.partial_trace(target, r, model['n']))
                             for r in model['regions'])
                row['max_protected_target_defect'] = max(row['max_protected_target_defect'], float(defect))
        if step in references:
            row['samples'].append(dict(time=step*dt, factor_bond=state.factor.max_bond(),
                                       **bench.diagnostics(state.to_dense(), *references[step], model)))
    return row


def initial_vectors(shape):
    from pepsy.tensors import OneDMap

    coords, _ = OneDMap(*shape, mode='snake-row-major').build()
    result = []
    for j in range(shape[0]*shape[1]):
        x, y = coords[j]
        theta = np.pi/2 + .35*(-1)**(x+y)
        phi = .7 + .3*np.cos(np.pi*x/(shape[0]-1))
        result.append(np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', type=int, nargs=2, default=[3, 2])
    parser.add_argument('--kappas', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('/tmp/dmt-positive-factor.json'))
    args = parser.parse_args()
    if min(args.steps, *args.kappas) < 1 or np.prod(args.shape) > 8:
        parser.error('Require positive steps/kappas and at most eight sites for dense audits')
    model = bench.problem(*args.shape, args.dt)
    references = bench.exact_references(model, args.dt, args.steps, max(1, args.steps//4))
    result = dict(shape=args.shape, dt=args.dt, steps=args.steps, dtype='complex128',
                  representation='rho=XX†; factor compression, not exact-marginal DMT', rows=[])
    for kappa in args.kappas:
        row = trajectory(model, references, vectors=initial_vectors(args.shape),
                         kappa=kappa, steps=args.steps, dt=args.dt)
        result['rows'].append(row)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k != 'samples'}), flush=True)


if __name__ == '__main__':
    main()
