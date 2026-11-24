import numpy as np

def metrics(truth, predicted):
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError('Expected equally sized, nonempty one-dimensional arrays')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite labels or predictions')
    error = p - y
    with np.errstate(over='raise', invalid='raise'):
        nasa = np.expm1(np.where(error < 0, -error / 13, error / 10))
    return dict(n=len(y), rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.abs(error).mean()), nasa=float(nasa.sum()))
