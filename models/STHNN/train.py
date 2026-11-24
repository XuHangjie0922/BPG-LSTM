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
os.environ.setdefault('LOKY_MAX_CPU_COUNT','8')
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT
sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from data import prepare,normalize,WINDOWS
from model import STHNN
from metrics import metrics


def verify():
    verify_release()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fd',required=True,choices=list(WINDOWS))
    ap.set_defaults(phase='full')
    ap.set_defaults(candidate='selected')
    ap.add_argument('--seed',required=True,type=int)
    a=ap.parse_args()
    verify()
    rules=json.loads((ROOT/'protocol.json').read_text())
    if a.phase=='screen':
        assert a.seed==42
        params=rules['candidates'][a.candidate]
    else:
        assert a.seed in rules['final_seeds']
        path=ROOT/'selection.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==(ROOT/'selection.sha256').read_text().strip()
        params=json.loads(path.read_text())[a.fd]['parameters']
    out=OUTPUT_DIR/'STHNN'/a.phase/a.candidate/f'{a.fd}_seed{a.seed}'
    if out.exists():
        raise FileExistsError(out)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    d=prepare(a.fd)
    ref=json.loads((ROOT/'validation_reference_hashes.json').read_text())[a.fd]
    digest=hashlib.sha256(d['val_meta'].astype(np.int64).tobytes()+d['val_y'].astype(np.float32).tobytes()).hexdigest()
    assert digest==ref,'Validation engine/cycle/label pairing changed'
    model=STHNN(d['adjacency'],WINDOWS[a.fd]).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=params['lr'],weight_decay=params['wd'])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=0)
    loader=DataLoader(TensorDataset(torch.from_numpy(d['train_x']),torch.from_numpy(d['train_y'])),batch_size=128,shuffle=True,num_workers=0)
    out.mkdir(parents=True)
    np.savez(out/'preprocessing.npz',**d['state'],adjacency=d['adjacency'])
    (out/'protocol.json').write_text(json.dumps(dict(**vars(a),parameters=params,data=d['metadata'],rules=rules),indent=2))
    history=[]
    best=float('inf')
    started=time.time()
    def infer(x):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(t.cuda()).cpu().numpy() for t in torch.from_numpy(x).split(128)])
    for epoch in range(1,161):
        model.train()
        total,count=0.,0
        for x,y in loader:
            x,y=x.cuda(),y.cuda()
            opt.zero_grad(set_to_none=True)
            loss=(model(x)-y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            for name,p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError(name)
            opt.step()
            total+=loss.detach().item()*len(y)
            count+=len(y)
        pred=infer(d['val_x'])
        val=metrics(d['val_y'],pred)
        row=dict(epoch=epoch,train_rmse=(total/count)**.5,val_rmse=val['rmse'],val_nasa=val['nasa'],seconds=time.time()-started)
        if val['rmse']<best:
            best,best_epoch,best_metrics=val['rmse'],epoch,val
            torch.save(dict(model=model.state_dict(),epoch=epoch,data=d['metadata']),out/'best.pt')
            pd.DataFrame(dict(unit=d['val_meta'][:,0],cycle=d['val_meta'][:,1],true_rul=d['val_y'],predicted_rul=pred)).to_csv(out/'validation_predictions.csv',index=False,float_format='%.17g')
        history.append(row)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        schedule.step()
        print(json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),epoch=160),out/'last.pt')
    result=dict(**vars(a),parameters=params,epochs=160,best_epoch=best_epoch,best_validation_metrics=best_metrics,official_labels_read=False)
    if a.phase=='full':
        model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=False)['model'])
        table=np.loadtxt(DATA_DIR/f'test_{a.fd}.txt')
        units=np.unique(table[:,0]).astype(int)
        xx,ends=[],[]
        for unit in units:
            rows=table[table[:,0]==unit]
            z=normalize(rows,d['state'])
            w=WINDOWS[a.fd]
            if len(z)<w:
                z=np.concatenate([np.repeat(z[:1],w-len(z),axis=0),z])
            xx.append(z[-w:])
            ends.append(int(rows[-1,1]))
        pred=infer(np.asarray(xx,dtype=np.float32)).astype(float)
        pd.DataFrame(dict(unit=units,cycle=ends,predicted_rul=pred)).to_csv(out/'predictions_before_labels.csv',index=False,float_format='%.17g')
        raw=np.loadtxt(DATA_DIR/f'RUL_{a.fd}.txt')
        assert np.array_equal(units,np.arange(1,len(raw)+1))
        true=np.minimum(raw,125)
        pd.DataFrame(dict(unit=units,cycle=ends,true_rul=true,raw_true_rul=raw,predicted_rul=pred)).to_csv(out/'official_terminal_predictions.csv',index=False,float_format='%.17g')
        result.update(official_labels_read=True,official_terminal=metrics(true,pred),raw_terminal=metrics(raw,pred))
    verify()
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
