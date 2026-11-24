from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SENSORS = np.array([2,3,4,7,8,9,11,12,13,14,15,17,20,21]) + 4


def prepare(fd, width):
    table = np.loadtxt(DATA_DIR / f'train_{fd}.txt')
    fit, val = train_test_split(np.unique(table[:, 0]).astype(int), test_size=.1, random_state=42)
    fit, val = sorted(fit.tolist()), sorted(val.tolist())
    fitting = table[np.isin(table[:, 0], fit)][:, SENSORS]
    lo, hi = fitting.min(0), fitting.max(0)
    scale = np.maximum(hi-lo, 1e-8)
    def windows(ids):
        xx, yy = [], []
        for unit in ids:
            r = table[table[:, 0] == unit]
            z = (2*(r[:, SENSORS]-lo)/scale-1).astype('float32')
            for end in range(width-1, len(r)):
                xx.append(z[end-width+1:end+1])
                yy.append(min(125, r[-1, 1]-r[end, 1]))
        return np.asarray(xx, dtype='float32'), np.asarray(yy, dtype='float32')
    return windows(fit), windows(val), dict(minimum=lo, scale=scale), dict(fitting_engines=fit, validation_engines=val)


def terminal(fd, width, state):
    table = np.loadtxt(DATA_DIR / f'test_{fd}.txt')
    xx, ids = [], []
    for unit in np.unique(table[:, 0]).astype(int):
        r = table[table[:, 0] == unit]
        assert len(r) >= width, (fd, unit, len(r), width)
        z = (2*(r[:, SENSORS]-state['minimum'])/state['scale']-1).astype('float32')
        xx.append(z[-width:]); ids.append(unit)
    return np.asarray(xx), np.asarray(ids)
