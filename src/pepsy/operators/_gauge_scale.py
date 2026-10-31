"""Shared detached RMS scale for dense SU vectors and independent batches."""

import autoray as ar


def dense_gauge_scale(s, smudge):
    try:
        detached = ar.do('stop_gradient', s)
    except Exception:
        detached = s
    magnitude = ar.do('abs', detached)
    peak = ar.do('max', magnitude, axis=-1, keepdims=True)
    safe_peak = ar.do('where', peak > 0, peak, peak*0+1)
    rms = peak * ar.do('sqrt', ar.do('sum', (magnitude/safe_peak)**2,
                                    axis=-1, keepdims=True)/s.shape[-1])
    rms = ar.do('where', rms > 0, rms, safe_peak)
    scale = ar.do('where', rms > smudge, rms, rms*0+smudge)
    return ar.do('where', peak > 0, scale, peak*0+1)
