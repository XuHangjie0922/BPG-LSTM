from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
from release_paths import DATA_DIR
from pathlib import Path
import json,random,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from data import prepare,normalize,endpoints
import sigmoid
ROOT=Path(__file__).resolve().parent
OUT=ROOT
write=sigmoid.training.write
infer=sigmoid.training.infer
def metric(y, p):
    e = p.astype(np.float64)-y
    score = np.expm1(np.where(e < 0, -e/13, e/10))
    return dict(count=len(y), rmse=float(np.mean(e**2)**.5), mae=float(abs(e).mean()), nasa=float(score.sum()), asf=float(score.mean()))

def evaluation(model, data, prefix):
    full = infer(model, data['vx'])
    short = infer(model, prefix['vx'])
    # Fixed equal weight to each input regime; no official-test population weighting.
    score = float((.5*np.mean((full.astype(np.float64)-data['vy'])**2) + .5*np.mean((short.astype(np.float64)-prefix['vy'])**2))**.5)
    return score, full, short

def save_val(out, data, prefix, full, short):
    tables = []
    for group, d, p in [('complete_window', data, full), ('short_prefix', prefix, short)]:
        tables.append(pd.DataFrame(dict(group=group, unit=d['vm'][:, 0], cycle=d['vm'][:, 1], true_rul=d['vy'], predicted_rul=p)))
    pd.concat(tables).to_csv(out/'validation_predictions.csv', index=False, float_format='%.17g')

def official(model, data, out):
    x, meta = endpoints('FD002', data['state'], 60)
    pred = infer(model, x).astype(np.float64)
    pd.DataFrame(dict(unit=meta[:, 0], cycle=meta[:, 1], predicted_rul=pred)).to_csv(out/'official_predictions_before_labels.csv', index=False, float_format='%.17g')
    raw = np.loadtxt(DATA_DIR/'RUL_FD002.txt')
    assert np.array_equal(meta[:, 0], np.arange(1, len(raw)+1))
    y = np.minimum(raw, 125.)
    e = pred-y
    table = pd.DataFrame(dict(unit=meta[:, 0], cycle=meta[:, 1], left_pad=meta[:, 2], true_rul=y, raw_true_rul=raw,
                              predicted_rul=pred, error=e, absolute_error=abs(e), squared_error=e**2,
                              nasa=np.expm1(np.where(e < 0, -e/13, e/10))))
    table.to_csv(out/'official_terminal_predictions.csv', index=False, float_format='%.17g')
    results = []
    for name, mask in [('all', np.ones(len(y), bool)), ('complete_window', meta[:, 2] == 0), ('padded_window', meta[:, 2] > 0)]:
        results.append(dict(group=name, **metric(y[mask], pred[mask])))
    pd.DataFrame(results).to_csv(out/'official_group_metrics.csv', index=False)
    return results

def run(arm, data, prefix, protocol, manifest):
    sigmoid.training.verify(manifest)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = sigmoid.model_with_sigmoid()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    out = OUT/arm
    out.mkdir(exist_ok=False)
    loader = DataLoader(TensorDataset(torch.from_numpy(data['tx']).cuda(), torch.from_numpy(data['ty']).cuda()), batch_size=64, shuffle=True)
    px, py = torch.from_numpy(prefix['tx']).cuda(), torch.from_numpy(prefix['ty']).cuda()
    rng = np.random.default_rng(45042)
    best, best_epoch, history, steps, start = float('inf'), 0, [], 0, time.time()
    for epoch in range(1, 201):
        model.train()
        total, count, replaced = 0., 0, 0
        for x, y in loader:
            # Both arms have the same batch sizes, shuffle and step budget per epoch.
            mask = rng.random(len(y)) < .25
            choice = rng.integers(0, len(py), size=int(mask.sum()))
            if arm == 'prefix_25pct' and mask.any():
                x, y = x.clone(), y.clone()
                m = torch.as_tensor(mask, device='cuda')
                ix = torch.as_tensor(choice, device='cuda')
                x[m], y[m] = px[ix], py[ix]
                replaced += len(choice)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x)-y).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            total += loss.item()*len(y)
            count += len(y)
            steps += 1
        score, full, short = evaluation(model, data, prefix)
        row = dict(epoch=epoch, train_rmse=(total/count)**.5, selection_rmse=score,
                   full_validation_rmse=metric(data['vy'], full)['rmse'], prefix_validation_rmse=metric(prefix['vy'], short)['rmse'],
                   replaced_training_examples=replaced, optimizer_steps=steps, seconds=time.time()-start)
        if score < best:
            best, best_epoch = score, epoch
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, arm=arm, protocol=protocol), out/'best.pt')
            save_val(out, data, prefix, full, short)
        history.append(row)
        pd.DataFrame(history).to_csv(out/'history.csv', index=False)
        write(OUT/'progress.json', dict(status='running', arm=arm, fd='FD002', seed=42, best_epoch=best_epoch, best_selection_rmse=best, **row))
        print(json.dumps(dict(arm=arm, **row)), flush=True)
        if epoch >= 40 and epoch-best_epoch >= 25:
            break
    model.load_state_dict(torch.load(out/'best.pt', weights_only=False)['model'])
    score, full, short = evaluation(model, data, prefix)
    saved = pd.read_csv(out/'validation_predictions.csv')
    parity = float(abs(saved.predicted_rul.to_numpy()-np.concatenate([full, short])).max())
    assert parity < .002
    write(OUT/'progress.json', dict(status='evaluating_official', arm=arm, best_epoch=best_epoch))
    result = dict(arm=arm, epochs=epoch, best_epoch=best_epoch, best_selection_rmse=best,
                  full_validation=metric(data['vy'], full), prefix_validation=metric(prefix['vy'], short),
                  optimizer_steps=steps, official=official(model, data, out), validation_parity_max_abs=parity)
    sigmoid.training.verify(manifest)
    write(out/'COMPLETE.json', result)
    del model, optimizer, loader, px, py
    torch.cuda.empty_cache()
    return result
