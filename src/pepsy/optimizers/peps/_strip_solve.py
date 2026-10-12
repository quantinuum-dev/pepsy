"""Native pseudoinverse solves on the positive support of a Hermitian norm."""

import math

import autoray as ar

from ...bp._backend import all_finite, dag


def solve_positive(matrix, rhs, *, rcond):
    """Solve N x=b on its positive numerical support, without host arrays.

    RHS columns are independent physical batches. Negative eigenvalues and
    eigenvalues below the relative cutoff are discarded, not lifted.
    """
    matrix = .5 * (matrix + dag(matrix))
    values, vectors = ar.do('linalg.eigh', matrix)
    largest = float(values[-1])
    if not math.isfinite(largest) or largest <= 0:
        raise ValueError('local norm has no finite positive support')
    positive = ar.do('clip', values, 0., None)
    report = {'solver': 'pinv', 'matrix_size': int(matrix.shape[0]),
              'negative_weight': float((-ar.do('clip', values, None, 0.)).sum() / positive.sum())}
    support = values > rcond * largest
    safe = ar.do('where', support, values, ar.do('ones_like', values))
    reciprocal = ar.do('where', support, 1. / safe, ar.do('zeros_like', values))
    projected = dag(vectors) @ rhs
    discarded = ar.do('where', support[:, None], ar.do('zeros_like', projected), projected)
    rhs_norm = float(ar.do('linalg.norm', rhs))
    answer = vectors @ (reciprocal[:, None] * projected)
    if not all_finite(answer):
        raise ValueError('local supported-subspace solve produced nonfinite tensors')
    kept = ar.do('where', support, values, ar.do('ones_like', values) * largest)
    report.update(retained_rank=int(support.sum()),
                  condition=float(values[-1] / kept.min()),
                  discarded_rhs_relative=(float(ar.do('linalg.norm', discarded)) / rhs_norm
                                          if rhs_norm else 0.))
    return answer, report
