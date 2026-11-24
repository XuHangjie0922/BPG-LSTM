from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


def condition(rows, state):
    z = (rows[:, 2:5] - state['settings_mean']) / state['settings_scale']
    return ((z[:, None] - state['centers'][None])**2).sum(-1).argmin(-1)


def normalize(rows, state):
    c = condition(rows, state)
    values = (rows[:, np.array(SENSORS)+4] - state['minimum'][c]) / state['span'][c]
    # Use observed age only. Never divide by the engine's future failure cycle.
    age = rows[:, 1:2] / state['cycle_scale']
    return np.concatenate([values, age], axis=-1).astype(np.float32)


def prepare(fd, width=60):
    rows = np.loadtxt(DATA_DIR/f'train_{fd}.txt')
    indices, targets, meta = [], [], []
    for unit in np.unique(rows[:, 0]).astype(int):
        ix = np.flatnonzero(rows[:, 0] == unit)
        assert np.all(np.diff(rows[ix, 1]) == 1)
        for end in range(width-1, len(ix)):
            indices.append(ix[end-width+1:end+1])
            targets.append(min(125., rows[ix[-1], 1]-rows[ix[end], 1]))
            meta.append([unit, int(rows[ix[end], 1])])
    indices = np.asarray(indices)
    target, meta = np.asarray(targets, np.float32), np.asarray(meta)
    ti, vi = train_test_split(np.arange(len(indices)), test_size=.1, random_state=42)
    fit_rows = rows[np.unique(indices[ti])]
    settings = StandardScaler().fit(fit_rows[:, 2:5])
    regimes = 6 if fd in ['FD002', 'FD004'] else 1
    if regimes == 6:
        km = KMeans(6, n_init=10, random_state=42).fit(settings.transform(fit_rows[:, 2:5]))
        centers, labels = km.cluster_centers_, km.labels_
    else:
        centers, labels = np.zeros((1, 3)), np.zeros(len(fit_rows), int)
    values = fit_rows[:, np.array(SENSORS)+4]
    minimum = np.stack([values[labels == c].min(0) for c in range(regimes)])
    span = np.stack([values[labels == c].max(0) for c in range(regimes)]) - minimum
    state = dict(settings_mean=settings.mean_, settings_scale=settings.scale_, centers=centers,
                 minimum=minimum, span=np.where(span < 1e-10, 1., span),
                 cycle_scale=np.array(fit_rows[:, 1].max()))
    assert np.array_equal(condition(fit_rows, state), labels)
    z = normalize(rows, state)
    metadata = dict(fd=fd, width=width, sensors=SENSORS, input_nodes=15,
                    train_windows=len(ti), validation_windows=len(vi),
                    shared_engines=sorted(map(int, set(meta[ti, 0]) & set(meta[vi, 0]))),
                    validation_scope='paper-like 10% window split; NOT engine-independent',
                    normalization_fitted_on='unique rows used by fitting windows only')
    return dict(tx=z[indices[ti]], ty=target[ti], vx=z[indices[vi]], vy=target[vi],
                tm=meta[ti], vm=meta[vi], state=state, metadata=metadata)


def endpoints(fd, state, width=60):
    rows = np.loadtxt(DATA_DIR/f'test_{fd}.txt')
    xx, meta = [], []
    for unit in np.unique(rows[:, 0]).astype(int):
        r = rows[rows[:, 0] == unit]
        z = normalize(r, state)
        padded = max(0, width-len(z))
        if padded:
            z = np.concatenate([np.repeat(z[:1], padded, axis=0), z])
        xx.append(z[-width:])
        meta.append([unit, int(r[-1, 1]), padded])
    return np.asarray(xx, np.float32), np.asarray(meta)
