from pathlib import Path
import json,random,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from model import ASTHGNN
from data import prepare
import prefix_data as four
ROOT=Path(__file__).resolve().parent
OUT=ROOT
write=four.write
infer=four.job.infer
metric=four.job.metric
SIGMOID=four.job.sigmoid
TRAINING_SEED=42
def seed():
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)

def make_model(attention):
    return SIGMOID.model_with_sigmoid() if attention == 'sigmoid' else ASTHGNN().cuda()

def evaluate(model, data, prefix):
    a, b = infer(model, data['vx']), infer(model, prefix['vx'])
    full, short = metric(data['vy'], a), metric(prefix['vy'], b)
    mixed = float((.5*full['rmse']**2+.5*short['rmse']**2)**.5)
    return dict(full=full, prefix=short, mixed_rmse=mixed), a, b

def save_predictions(path, data, prefix, a, b):
    tables = [pd.DataFrame(dict(group=name, unit=d['vm'][:, 0], cycle=d['vm'][:, 1], true_rul=d['vy'], predicted_rul=p))
              for name, d, p in [('full', data, a), ('prefix', prefix, b)]]
    pd.concat(tables).to_csv(path, index=False, float_format='%.17g')

def run(config, data, prefix, manifest, protocol):
    SIGMOID.training.verify(manifest)
    seed()
    dest = OUT/config['name']
    dest.mkdir(exist_ok=False)
    model = make_model(config['attention'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loader = DataLoader(TensorDataset(torch.from_numpy(data['tx']).cuda(), torch.from_numpy(data['ty']).cuda()), batch_size=64, shuffle=True)
    px, py = torch.from_numpy(prefix['tx']).cuda(), torch.from_numpy(prefix['ty']).cuda()
    rng = np.random.default_rng(45042)
    best_full, best_mixed, best_eligible = float('inf'), float('inf'), float('inf')
    full_epoch, history, steps, start = 0, [], 0, time.time()
    for epoch in range(1, 201):
        model.train()
        total, n, replaced = 0., 0, 0
        for x, y in loader:
            mask = rng.random(len(y)) < config['prefix_fraction']
            index = rng.integers(0, len(py), size=int(mask.sum()))
            if mask.any():
                x, y = x.clone(), y.clone()
                mask_gpu, index_gpu = torch.as_tensor(mask, device='cuda'), torch.as_tensor(index, device='cuda')
                x[mask_gpu], y[mask_gpu] = px[index_gpu], py[index_gpu]
                replaced += len(index)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x)-y).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            steps += 1
            total += float(loss.item())*len(y)
            n += len(y)
        val, a, b = evaluate(model, data, prefix)
        improvements = []
        if val['full']['rmse'] < best_full:
            best_full, full_epoch = val['full']['rmse'], epoch
            improvements.append('best_full')
        if val['mixed_rmse'] < best_mixed:
            best_mixed = val['mixed_rmse']
            improvements.append('best_mixed')
        if val['prefix']['rmse'] <= protocol['prefix_validation_rmse_limit'] and val['full']['rmse'] < best_eligible:
            best_eligible = val['full']['rmse']
            improvements.append('best_eligible')
        for name in improvements:
            torch.save(dict(model=model.state_dict(), epoch=epoch, config=config, validation=val, protocol=protocol), dest/f'{name}.pt')
            save_predictions(dest/f'{name}_validation.csv', data, prefix, a, b)
            write(dest/f'{name}.json', dict(epoch=epoch, config=config, validation=val))
        with torch.no_grad():
            _, hs = model(torch.from_numpy(data['vx'][:64]).cuda(), audit=True)
        row = dict(epoch=epoch, train_rmse=(total/n)**.5, full_validation_rmse=val['full']['rmse'], prefix_validation_rmse=val['prefix']['rmse'],
                   mixed_rmse=val['mixed_rmse'], full_validation_nasa=val['full']['nasa'], prefix_validation_nasa=val['prefix']['nasa'],
                   entropy=[float((-(h*h.clamp_min(1e-10).log()).sum(-1)).mean()) for h in hs],
                   assignment_node_std=[float(h.std(dim=-2).mean()) for h in hs],
                   optimizer_steps=steps, replaced=replaced, seconds=time.time()-start)
        history.append(row)
        pd.DataFrame(history).to_csv(dest/'history.csv', index=False)
        write(OUT/'progress.json', dict(status='running', fd='FD004', seed=42, candidate=config['name'],
              best_full_rmse=best_full, best_full_epoch=full_epoch, **row))
        print(json.dumps(dict(candidate=config['name'], **row)), flush=True)
        if epoch >= 40 and epoch-full_epoch >= 25:
            break
    result = dict(config=config, epochs=epoch, optimizer_steps=steps,
                  best_full=json.loads((dest/'best_full.json').read_text()),
                  best_mixed=json.loads((dest/'best_mixed.json').read_text()),
                  best_eligible=json.loads((dest/'best_eligible.json').read_text()) if (dest/'best_eligible.json').exists() else None)
    write(dest/'COMPLETE.json', result)
    SIGMOID.training.verify(manifest)
    del model, optimizer, loader, px, py
    torch.cuda.empty_cache()
    return result
