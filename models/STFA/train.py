from release_paths import DATA_DIR,OUTPUT_DIR,verify_release
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,csv,json,random,time,hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from model import STFA
from data import prepare,terminal
BASE=Path(__file__).resolve().parent
OUT=OUTPUT_DIR/'STFA'
SEEDS=[3407,2024,137,123,12345]
WIDTHS=dict(FD001=30,FD002=20,FD003=30,FD004=15)
START=time.time()

def lock():
    verify_release()
    return json.loads((BASE/'config.json').read_text())


def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class SelfLoopSTFA(STFA):
    def __init__(self,width):
        super().__init__(width)
        self.adjacency.fill_diagonal_(True)

def tests():
    torch.manual_seed(42)
    old=STFA(30).eval()
    torch.manual_seed(42)
    new=SelfLoopSTFA(30).eval()
    assert all(torch.equal(p,dict(new.named_parameters())[name]) for name,p in old.named_parameters())
    assert torch.equal(old.adjacency^new.adjacency,torch.eye(14,dtype=torch.bool))
    assert isinstance(new.lstm,torch.nn.LSTM)
    assert not any('peephole' in name for name,_ in new.named_parameters())
    y,beta,a=new(torch.randn(3,30,14),diagnostic=True)
    assert torch.isfinite(y).all() and torch.allclose(beta.sum(1),torch.ones(3),atol=1e-6)
    assert torch.allclose(a.sum(-1),torch.ones_like(a.sum(-1)),atol=1e-6)
    assert (a.diagonal(dim1=-2,dim2=-1)>0).all()
    y.square().mean().backward()
    assert new.v.weight.grad.norm()>0
    save(OUT/'MODEL_TESTS.json',dict(passed=True,selfloops=14,no_peepholes=True,
        original_common_parameters_identical=True,trainable_w30=sum(p.numel() for p in new.parameters() if p.requires_grad),
        temporal_attention_has_gradient=True,spatial_normalization=True))

def predict(model,x):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(torch.from_numpy(x[start:start+128]).cuda()).cpu().numpy()
                               for start in range(0,len(x),128)])

def select_epoch(history):
    assert [r['epoch'] for r in history]==list(range(1,31)), 'Incomplete training history'
    assert all(np.isfinite(r['validation_rmse']) for r in history)
    return min(history,key=lambda r:(r['validation_rmse'],r['epoch']))

def completed():
    return [json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*/COMPLETE.json'))]

def report():
    rows=completed()
    save(OUT/'per_seed_results.json',rows)
    summary=[]
    for fd in WIDTHS:
        selected=[r for r in rows if r['fd']==fd]
        if not selected:
            continue
        item=dict(fd=fd,completed_seeds=len(selected),all_five_complete=len(selected)==5)
        for key in ['rmse','mae','nasa']:
            values=[r[key] for r in selected]
            item[key+'_mean']=float(np.mean(values))
            item[key+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(item)
    save(OUT/'summary.json',summary)
    if rows:
        with (OUT/'per_seed_results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    if summary:
        with (OUT/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]))
            writer.writeheader();writer.writerows(summary)

def train(fd,seed,config):
    target=OUT/'runs'/f'{fd}_seed{seed}'
    if (target/'COMPLETE.json').exists():
        return
    target.mkdir(parents=True,exist_ok=False)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    tr,va,state,split=prepare(fd,WIDTHS[fd])
    save(target/'split.json',split);np.savez(target/'preprocessing.npz',**state)
    model=SelfLoopSTFA(WIDTHS[fd]).cuda()
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    loader=DataLoader(TensorDataset(torch.from_numpy(tr[0]),torch.from_numpy(tr[1])),
        batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    history=[];best=float('inf');start=time.time()
    for epoch in range(1,31):
        model.train();total=0.;count=0
        for x,y in loader:
            x,y=x.cuda(),y.cuda()
            opt.zero_grad(set_to_none=True)
            loss=(model(x)-y).square().mean()
            assert torch.isfinite(loss)
            loss.backward();opt.step()
            total+=loss.detach().item()*len(y);count+=len(y)
        pred=predict(model,va[0])
        val=float(np.sqrt(np.mean((pred-va[1])**2)))
        row=dict(fd=fd,seed=seed,epoch=epoch,train_rmse=(total/count)**.5,
                 validation_rmse=val,elapsed_seconds=time.time()-start,device='cuda')
        history.append(row);save(target/'history.json',history)
        if val<best:
            best=val;torch.save(dict(model=model.state_dict(),epoch=epoch),target/'validation_best.pt')
        save(OUT/'progress.json',dict(state='training',completed=len(completed()),total=20,pid=os.getpid(),
            total_elapsed_seconds=time.time()-START,**row))
        print(json.dumps(row),flush=True)
    torch.save(dict(model=model.state_dict(),epoch=30,protocol=config),target/'final.pt')
    selected=select_epoch(history)
    ckpt=torch.load(target/'validation_best.pt',map_location='cpu',weights_only=False)
    assert ckpt['epoch']==selected['epoch']
    model.load_state_dict(ckpt['model'])
    save(target/'selection.json',dict(rule=config['checkpoint'],selected_epoch=selected['epoch'],
         validation_rmse=selected['validation_rmse'],checkpoint_sha256=sha(target/'validation_best.pt')))
    x,ids=terminal(fd,WIDTHS[fd],state)
    pred=predict(model,x)
    np.savez(target/'predictions_before_test_labels.npz',unit=ids,predicted_rul=pred)
    true=np.minimum(np.loadtxt(DATA_DIR/f'RUL_{fd}.txt'),125)
    assert np.array_equal(ids,np.arange(1,len(true)+1))
    error=pred-true
    result=dict(fd=fd,seed=seed,checkpoint='minimum validation RMSE',selected_epoch=selected['epoch'],
        validation_rmse=selected['validation_rmse'],n=len(ids),
        rmse=float(np.sqrt(np.mean(error**2))),mae=float(abs(error).mean()),
        nasa=float(np.expm1(np.where(error<0,-error/13,error/10)).sum()),seconds=time.time()-start)
    with (target/'official_terminal_predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['unit','true_rul','predicted_rul']);writer.writerows(zip(ids,true,pred))
    save(target/'COMPLETE.json',result)
    lock();report()
    print(json.dumps(dict(event='case_complete',**result)),flush=True)
    del model,opt,loader
    torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser(description='STFA: 30 epochs; minimum validation RMSE checkpoint')
    p.add_argument('--fd',required=True,choices=list(WIDTHS))
    p.add_argument('--seed',required=True,type=int,choices=SEEDS)
    a=p.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    config=lock()
    tests()
    assert torch.cuda.is_available(),'CUDA required for the recorded training protocol'
    train(a.fd,a.seed,config)

if __name__=='__main__':main()
