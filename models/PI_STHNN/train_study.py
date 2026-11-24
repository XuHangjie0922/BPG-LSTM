from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT
sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from data import SENSORS
from study_data import common_data
from model import StudyModel


def score(y,p):
    e=np.asarray(p,dtype=float)-np.asarray(y,dtype=float)
    return dict(rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.abs(e).mean()),
                nasa=float(np.expm1(np.where(e<0,-e/13,e/10)).sum()),
                prediction_std=float(np.std(p)),n=len(e))


def infer(model,x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(t.cuda()).cpu().numpy() for t in torch.from_numpy(x).split(256)])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fd',required=True)
    ap.add_argument('--seed',required=True,type=int)
    ap.add_argument('--method',required=True)
    ap.add_argument('--phase',choices=['screen','full'],required=True)
    a=ap.parse_args()
    verify_release()
    rules=json.loads((ROOT/'study_protocol.json').read_text())
    if a.fd not in rules['subsets'] or a.method not in rules['methods']:
        raise ValueError('Outside fixed cases')
    if a.phase=='screen' and a.seed!=42:
        raise ValueError('Screen seed must be42')
    if a.phase=='full':
        choice=json.loads((ROOT/'selection.json').read_text())
        if a.method not in ['control',choice['selected']] or a.seed not in rules['final_seeds']:
            raise ValueError('Full case does not match selected study')
        if hashlib.sha256((ROOT/'selection.json').read_bytes()).hexdigest()!=(ROOT/'selection.sha256').read_text().strip():
            raise RuntimeError('Selection hash mismatch')
    folder=OUTPUT_DIR/'PI_STHNN'/a.phase/a.method/f'{a.fd}_seed{a.seed}'
    if folder.exists():
        raise FileExistsError(folder)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(1)
    params=json.loads((ROOT/'base_parameters.json').read_text())[a.fd]
    d=common_data(a.fd,a.method)
    edge=np.array(np.nonzero(d['adjacency']))
    model=StudyModel(torch.tensor(edge),torch.tensor(d['adjacency'][tuple(edge)]),a.method,d['protocol']['window']).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=params['lr'],weight_decay=params['weight_decay'])
    schedule=torch.optim.lr_scheduler.StepLR(opt,50,.1)
    loader=DataLoader(TensorDataset(torch.tensor(d['train_x']),torch.tensor(d['train_y'])),batch_size=256,shuffle=True,num_workers=0)
    protocol={**d['protocol'],**vars(a),**params,'epochs':50,'batch_size':256,
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'training_windows':len(d['train_y']),'validation_windows':len(d['val_y']),
              'selection_labels':'internal common validation only',
              'scope':'paper-based reconstruction, not confirmed original author code'}
    folder.mkdir(parents=True)
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    np.savez(folder/'preprocessing.npz',**{k:d[k] for k in ['minimum','span','adjacency']})
    history=[]
    best=float('inf')
    best_val=None
    best_epoch=None
    started=time.time()
    for epoch in range(1,51):
        model.train()
        total,count,gradmax=0.,0,0.
        for batch,(x,y) in enumerate(loader):
            x,y=x.cuda(),y.cuda()
            opt.zero_grad(set_to_none=True)
            prediction=model(x)
            loss=(prediction-y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss:epoch{epoch},batch{batch}')
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
            raise FloatingPointError('Nonfinite validation result')
        row=dict(epoch=epoch,train_rmse=(total/count)**.5,validation_rmse=val['rmse'],
                 validation_nasa=val['nasa'],prediction_std=val['prediction_std'],max_abs_gradient=gradmax,
                 seconds=time.time()-started)
        history.append(row)
        if val['rmse']<best:
            best,best_val,best_epoch=val['rmse'],val,epoch
            torch.save(dict(model=model.state_dict(),epoch=epoch,protocol=protocol),folder/'best.pt')
            pd.DataFrame(dict(unit=d['val_meta'][:,0],cycle=d['val_meta'][:,1],true_rul=d['val_y'].astype(float),
                              predicted_rul=pred.astype(float))).to_csv(folder/'validation_predictions.csv',index=False,float_format='%.17g')
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        schedule.step()
        print(json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),epoch=50,protocol=protocol),folder/'last.pt')
    result=dict(fd=a.fd,seed=a.seed,method=a.method,phase=a.phase,parameters=params,epochs=50,
                best_validation_rmse=best,best_validation_metrics=best_val,best_epoch=best_epoch,
                common_validation_sha256=d['protocol']['common_validation_sha256'],
                all_gradients_finite=True,official_labels_read=False,seconds=time.time()-started)
    if a.phase=='full':
        ckpt=torch.load(folder/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(ckpt['model'])
        table=np.loadtxt(DATA_DIR/f'test_{a.fd}.txt')
        units=np.unique(table[:,0]).astype(int)
        xs,ends=[],[]
        w=d['protocol']['window']
        for unit in units:
            rows=table[table[:,0]==unit]
            z=2*(rows[:,np.array(SENSORS)+4]-d['minimum'])/d['span']-1
            if len(z)<w:
                z=np.concatenate([np.repeat(z[:1],w-len(z),axis=0),z])
            xs.append(z[-w:])
            ends.append(int(rows[-1,1]))
        pred=np.maximum(0,infer(model,np.asarray(xs,dtype=np.float32))).astype(float)
        pd.DataFrame(dict(unit=units,cycle=ends,predicted_rul=pred)).to_csv(
            folder/'official_predictions_before_labels.csv',index=False,float_format='%.17g')
        labels=np.loadtxt(DATA_DIR/f'RUL_{a.fd}.txt')
        if not np.array_equal(units,np.arange(1,len(labels)+1)):
            raise RuntimeError('Official engine identity mismatch')
        y=np.minimum(labels,125)
        pd.DataFrame(dict(unit=units,cycle=ends,true_rul=y,raw_true_rul=labels,predicted_rul=pred)).to_csv(
            folder/'official_terminal_predictions.csv',index=False,float_format='%.17g')
        result.update(official_labels_read=True,official_terminal=score(y,pred),official_terminal_raw=score(labels,pred))
    (folder/'COMPLETE.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
