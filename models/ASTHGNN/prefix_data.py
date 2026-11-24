from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
from release_paths import DATA_DIR
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data import prepare,normalize,endpoints
import prefix_core as job
ROOT=Path(__file__).resolve().parent
write=job.write
def prefix_data(fd, data):
    rows = np.loadtxt(DATA_DIR/f'train_{fd}.txt')
    xs, ys, meta = [], [], []
    for unit in np.unique(rows[:, 0]).astype(int):
        r = rows[rows[:, 0] == unit]
        assert np.array_equal(r[:, 1], np.arange(1, len(r)+1))
        z = normalize(r, data['state'])
        for length in range(1, min(60, len(r)+1)):
            x = np.concatenate([np.repeat(z[:1], 60-length, axis=0), z[:length]])
            assert np.array_equal(x[-length:], z[:length])
            assert np.array_equal(x[:60-length], np.repeat(z[:1], 60-length, axis=0))
            xs.append(x)
            ys.append(min(125., r[-1, 1]-r[length-1, 1]))
            meta.append([unit, length])
    xs, ys, meta = np.array(xs, np.float32), np.array(ys, np.float32), np.array(meta)
    ti, vi = train_test_split(np.arange(len(xs)), test_size=.1, random_state=42)
    assert not (set(map(tuple, meta[ti])) & set(map(tuple, meta[vi])))
    assert np.isfinite(xs).all()
    return dict(tx=xs[ti], ty=ys[ti], tm=meta[ti], vx=xs[vi], vy=ys[vi], vm=meta[vi])

def official(fd, model, data, out):
    xx, meta = endpoints(fd, data['state'], 60)
    pred = job.infer(model, xx).astype(np.float64)
    pd.DataFrame(dict(unit=meta[:, 0], cycle=meta[:, 1], predicted_rul=pred)).to_csv(out/'official_predictions_before_labels.csv', index=False, float_format='%.17g')
    raw = np.loadtxt(DATA_DIR/f'RUL_{fd}.txt')
    assert np.array_equal(meta[:, 0], np.arange(1, len(raw)+1))
    y = np.minimum(raw, 125.)
    error = pred-y
    table = pd.DataFrame(dict(unit=meta[:, 0], cycle=meta[:, 1], left_pad=meta[:, 2], true_rul=y, raw_true_rul=raw,
                              predicted_rul=pred, error=error, absolute_error=abs(error), squared_error=error**2,
                              nasa=np.expm1(np.where(error < 0, -error/13, error/10))))
    table.to_csv(out/'official_terminal_predictions.csv', index=False, float_format='%.17g')
    results = []
    for name, mask in [('all', np.ones(len(y), bool)), ('complete_window', meta[:, 2] == 0), ('padded_window', meta[:, 2] > 0)]:
        assert mask.any()
        results.append(dict(group=name, **job.metric(y[mask], pred[mask])))
    pd.DataFrame(results).to_csv(out/'official_group_metrics.csv', index=False)
    return results
