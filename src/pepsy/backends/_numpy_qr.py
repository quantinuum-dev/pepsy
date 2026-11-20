"""Scoped QR compatibility for real Accelerate arrays on macOS 26."""

from functools import lru_cache
import platform
import warnings

import autoray as ar
import numpy as np


@lru_cache(maxsize=1)
def _affected_accelerate():
    if (platform.system() != 'Darwin' or platform.machine() != 'arm64'
            or platform.mac_ver()[0].split('.')[0] != '26'):
        return False
    try:
        config = np.show_config(mode='dicts')
        return 'accelerate' in config['Build Dependencies']['blas']['name'].lower()
    except (TypeError, KeyError, AttributeError):
        return False


@lru_cache(maxsize=1)
def _warn_workaround():
    warnings.warn(
        'Using exact pivoted SciPy QR for large real NumPy arrays on macOS 26 '
        'with Accelerate to avoid native QR heap corruption '
        '(numpy/numpy#33161). The column permutation is absorbed into the '
        'other factor; the canonical gauge can differ.', RuntimeWarning, stacklevel=4,
    )


def _numpy_qr_workaround(x):
    """Return exact reduced factors for affected arrays, otherwise ``None``."""
    if (not isinstance(x, np.ndarray) or x.ndim != 2 or x.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or max(x.shape) < 544 or not _affected_accelerate()):
        return None
    from scipy.linalg import qr

    _warn_workaround()
    # GEQRF can overwrite unrelated live allocations while returning valid
    # factors. Post-call Gram checks cannot make that route safe. GEQP3 uses
    # pivoting instead; undo its column permutation in R, with no truncation.
    # Conservatively cover large real shapes rather than hard-code only 768².
    q, r, permutation = qr(x, mode='economic', pivoting=True, overwrite_a=False)
    phase = np.sign(np.diag(r))
    phase = np.where(phase == 0, np.ones_like(phase), phase)
    return q * phase[None, :], (phase[:, None] * r)[:, np.argsort(permutation)]


def _qr_reduced(x):
    result = _numpy_qr_workaround(x)
    return result if result is not None else ar.do('linalg.qr', x, mode='reduced')
