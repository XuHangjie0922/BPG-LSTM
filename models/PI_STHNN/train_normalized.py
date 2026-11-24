from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('LOKY_MAX_CPU_COUNT','8')
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT
ALIGNMENT=ROOT
sys.path.insert(0,str(ALIGNMENT))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from model import StudyModel
from train_study import score,infer
from normalize_data import normalized_data,transform


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fd',choices=['FD002','FD004'],required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--phase',choices=['screen','full'],required=True)
    a=ap.parse_args()
    verify_release()
    rules=json.loads((ROOT/'config.json').read_text())
    if a.phase=='screen' and a.seed!=42:
        raise ValueError('Fixed screen seed42')
    if a.phase=='full':
        import hashlib
        selected=json.loads((ROOT/'selection.json').read_text())
        if not selected['passed'] or a.seed not in rules['final_seeds']:
            raise RuntimeError('Full run not authorized by fixed selection')
        if hashlib.sha256((ROOT/'selection.json').read_bytes()).hexdigest()!=(ROOT/'selection.sha256').read_text().strip():
            raise RuntimeError('Selection changed')
    out=OUTPUT_DIR/'PI_STHNN'/a.phase/f'{a.fd}_seed{a.seed}'
    if out.exists():
        raise FileExistsError(out)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(1)
    d,state,diagnostic=normalized_data(a.fd)
    params=json.loads((ROOT/'base_parameters.json').read_text())[a.fd]
    edges=np.array(np.nonzero(d['adjacency']))
    model=StudyModel(torch.tensor(edges),torch.tensor(d['adjacency'][tuple(edges)]),'paper_window',d['protocol']['window']).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=params['lr'],weight_decay=params['weight_decay'])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,50,.1)
    loader=DataLoader(TensorDataset(torch.tensor(d['train_x']),torch.tensor(d['train_y'])),batch_size=256,shuffle=True,num_workers=0)
    out.mkdir(parents=True)
    protocol={**d['protocol'],**vars(a),'parameters':params,'epochs':50,'batch_size':256,
              'adaptation':'condition-wise minmax; raw physical graph unchanged'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (out/'normalization_diagnostic.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf-8')
    np.savez(out/'normalizer.npz',**state)
    np.savez(out/'adjacency.npz',adjacency=d['adjacency'])
    history=[]
    best=float('inf')
    best_val=None
    best_epoch=None
    started=time.time()
    for epoch in range(1,51):
        model.train()
        total,count,gradmax=0.,0,0.
        for x,y in loader:
            x,y=x.cuda(),y.cuda()
            opt.zero_grad(set_to_none=True)
            pred=model(x)
            loss=(pred-y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise FloatingPointError(f'Nonfinite gradient:{name}')
                    gradmax=max(gradmax,float(p.grad.abs().max()))
            opt.step()
            total+=float(loss.detach())*len(y)
            count+=len(y)
        pred=infer(model,d['val_x'])
        val=score(d['val_y'],pred)
        if not all(np.isfinite(v) for v in val.values()):
            raise FloatingPointError('Nonfinite validation metrics')
        row=dict(epoch=epoch,train_rmse=(total/count)**.5,validation_rmse=val['rmse'],
                 validation_nasa=val['nasa'],prediction_std=val['prediction_std'],max_abs_gradient=gradmax,
                 seconds=time.time()-started)
        history.append(row)
        if val['rmse']<best:
            best,best_val,best_epoch=val['rmse'],val,epoch
            torch.save(dict(model=model.state_dict(),epoch=epoch,protocol=protocol),out/'best.pt')
            pd.DataFrame(dict(unit=d['val_meta'][:,0],cycle=d['val_meta'][:,1],true_rul=d['val_y'].astype(float),
                              predicted_rul=pred.astype(float))).to_csv(out/'validation_predictions.csv',index=False,float_format='%.17g')
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        scheduler.step()
        print(json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),epoch=50,protocol=protocol),out/'last.pt')
    result=dict(fd=a.fd,seed=a.seed,phase=a.phase,parameters=params,epochs=50,best_validation_rmse=best,
                best_validation_metrics=best_val,best_epoch=best_epoch,all_gradients_finite=True,
                official_labels_read=False,common_validation_sha256=d['protocol']['common_validation_sha256'])
    if a.phase=='full':
        saved=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model'])
        table=np.loadtxt(DATA_DIR/f'test_{a.fd}.txt')
        units=np.unique(table[:,0]).astype(int)
        xx,ends=[],[]
        w=d['protocol']['window']
        for unit in units:
            rows=table[table[:,0]==unit]
            z=transform(rows,state)
            if len(z)<w:
                z=np.concatenate([np.repeat(z[:1],w-len(z),axis=0),z])
            xx.append(z[-w:])
            ends.append(int(rows[-1,1]))
        pred=np.maximum(0,infer(model,np.asarray(xx,dtype=np.float32))).astype(float)
        pd.DataFrame(dict(unit=units,cycle=ends,predicted_rul=pred)).to_csv(out/'official_predictions_before_labels.csv',index=False,float_format='%.17g')
        labels=np.loadtxt(DATA_DIR/f'RUL_{a.fd}.txt')
        if not np.array_equal(units,np.arange(1,len(labels)+1)):
            raise RuntimeError('Official identity mismatch')
        y=np.minimum(labels,125)
        pd.DataFrame(dict(unit=units,cycle=ends,true_rul=y,raw_true_rul=labels,predicted_rul=pred)).to_csv(
            out/'official_terminal_predictions.csv',index=False,float_format='%.17g')
        result.update(official_labels_read=True,official_terminal=score(y,pred),official_terminal_raw=score(labels,pred))
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
