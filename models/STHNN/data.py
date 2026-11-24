from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
import hashlib
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
WINDOWS = {'FD001': 30, 'FD002': 20, 'FD003': 30, 'FD004': 15}


def conditions(rows, state):
    if int(state['regimes']) == 1:
        return np.zeros(len(rows), dtype=int)
    z = (rows[:, 2:5] - state['settings_mean']) / state['settings_scale']
    return ((z[:, None] - state['centres'][None])**2).sum(-1).argmin(-1)


def normalize(rows, state):
    c = conditions(rows, state)
    return ((rows[:, np.array(SENSORS)+4] - state['mean'][c]) / state['std'][c]).astype(np.float32)


def fit_normalizer(rows, regimes):
    settings = StandardScaler().fit(rows[:, 2:5])
    if regimes > 1:
        km = KMeans(regimes, n_init=10, random_state=42).fit(settings.transform(rows[:, 2:5]))
        c, centres = km.labels_, km.cluster_centers_
    else:
        c, centres = np.zeros(len(rows), dtype=int), np.zeros((1, 3))
    sensors = rows[:, np.array(SENSORS)+4]
    mean = np.stack([sensors[c == i].mean(0) for i in range(regimes)])
    std = np.stack([sensors[c == i].std(0) for i in range(regimes)])
    state = dict(regimes=np.array(regimes), settings_mean=settings.mean_,
                 settings_scale=settings.scale_, centres=centres,
                 mean=mean, std=np.where(std < 1e-12, 1., std))
    assert np.array_equal(c, conditions(rows, state))
    return state


def windows(table, ids, state, width):
    xx, yy, meta = [], [], []
    for unit in ids:
        rows = table[table[:, 0] == unit]
        z = normalize(rows, state)
        for end in range(width-1, len(rows)):
            xx.append(z[end-width+1:end+1])
            yy.append(min(125, rows[-1, 1]-rows[end, 1]))
            meta.append([int(unit), int(rows[end, 1])])
    return np.asarray(xx), np.asarray(yy, dtype=np.float32), np.asarray(meta, dtype=np.int64)


def prepare(fd):
    path = DATA_DIR / f'train_{fd}.txt'
    table = np.loadtxt(path)
    fit, val = train_test_split(np.unique(table[:, 0]).astype(int), test_size=.2, random_state=42)
    fit, val = sorted(fit.tolist()), sorted(val.tolist())
    assert not (set(fit) & set(val))
    fitting = table[np.isin(table[:, 0], fit)]
    state = fit_normalizer(fitting, 6 if fd in ['FD002', 'FD004'] else 1)
    z = normalize(fitting, state).astype(np.float64)
    norms = np.linalg.norm(z, axis=0)
    cosine = (z.T @ z) / np.maximum(norms[:, None] * norms[None, :], 1e-12)
    adjacency = cosine > 0
    np.fill_diagonal(adjacency, False)
    tx, ty, tm = windows(table, fit, state, WINDOWS[fd])
    vx, vy, vm = windows(table, val, state, WINDOWS[fd])
    return dict(train_x=tx, train_y=ty, train_meta=tm, val_x=vx, val_y=vy, val_meta=vm,
                state=state, adjacency=adjacency, cosine=cosine,
                metadata=dict(fd=fd, fitting_engines=fit, validation_engines=val,
                              window=WINDOWS[fd], sensors=SENSORS, split_seed=42,
                              training_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
