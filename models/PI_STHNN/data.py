from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
NAMES = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi',
         'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']


def grey_graph(raw, strong, rho=.5, threshold=.7):
    # Eqs17-22: homogenize unscaled physical sensor series by their means.
    z = raw / np.maximum(np.abs(raw.mean(0)), 1e-12)
    r = np.zeros((14, 14), dtype=np.float64)
    for i in range(14):
        difference = np.abs(z - z[:, i:i+1])
        lo, hi = difference.min(), difference.max()
        r[i] = ((lo + rho * hi) / np.maximum(difference + rho * hi, 1e-12)).mean(0)
    mean = (r.sum(1) - np.diag(r)) / 13
    coefficient = np.where(strong, 1., mean[:, None])
    np.fill_diagonal(coefficient, 0)
    fused = coefficient * r
    fused[fused < threshold] = 0
    # Fig5 coefficient support is asymmetric, but section5 describes the final
    # graph as bidirectional. Retain raw intermediates and disclose this union.
    final = np.maximum(fused, fused.T)
    np.fill_diagonal(final, 0)
    return r, coefficient, fused, final


def windows(table, ids, lo, span, cap=125, w=30):
    xs, ys, metadata = [], [], []
    columns = np.array(SENSORS) + 4
    for unit in ids:
        rows = table[table[:, 0] == unit]
        values = (2 * (rows[:, columns] - lo) / span - 1).astype(np.float32)
        for end in range(w - 1, len(rows)):
            xs.append(values[end-w+1:end+1])
            ys.append(min(cap, rows[-1, 1] - rows[end, 1]))
            metadata.append([int(unit), int(rows[end, 1])])
    return np.asarray(xs), np.asarray(ys, dtype=np.float32), np.asarray(metadata)


def prepare(root, fd, cap=125, split_seed=42):
    root = Path(root)
    raw_file = DATA_DIR / f'train_{fd}.txt'
    table = np.loadtxt(raw_file)
    fit, val = train_test_split(np.unique(table[:, 0]).astype(int),
                               test_size=.2, random_state=split_seed)
    fit, val = sorted(fit.tolist()), sorted(val.tolist())
    columns = np.array(SENSORS) + 4
    raw = table[np.isin(table[:, 0], fit)][:, columns]
    lo, hi = raw.min(0), raw.max(0)
    span = np.maximum(hi - lo, 1e-12)
    strong = np.load(root / 'reference' / 'strong_support_figure5.npy')
    r, coefficient, fused, final = grey_graph(raw, strong)
    train_x, train_y, train_meta = windows(table, fit, lo, span, cap)
    val_x, val_y, val_meta = windows(table, val, lo, span, cap)
    assert not (set(fit) & set(val))
    return dict(train_x=train_x, train_y=train_y, train_meta=train_meta,
                val_x=val_x, val_y=val_y, val_meta=val_meta, minimum=lo, span=span,
                grey=r, coefficient=coefficient, fused_directed=fused, adjacency=final,
                protocol={'fit_engines': fit, 'validation_engines': val,
                          'split_seed': split_seed, 'rul_cap': cap, 'window': 30,
                          'sensor_ids': SENSORS, 'sensor_names': NAMES,
                          'raw_sha256': hashlib.sha256(raw_file.read_bytes()).hexdigest(),
                          'graph_fit': 'fitting engines only; raw physical values',
                          'scaler_fit': 'fitting engines only; global minmax [-1,1]',
                          'graph_symmetrization': 'union; max of directed fused values',
                          'selection': 'lowest engine-held-out validation-window RMSE',
                          'official_labels_used_for_selection': False})
