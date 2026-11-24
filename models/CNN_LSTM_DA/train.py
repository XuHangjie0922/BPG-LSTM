from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
import argparse
PARSER=argparse.ArgumentParser(description='Final CNN-LSTM + wcMMD PatternII reconstruction')
PARSER.add_argument('--fd',choices=['FD001','FD002','FD003','FD004'])
PARSER.add_argument('--seed',type=int,required=True,choices=[3407,2024,137,123,12345])
ARGS=PARSER.parse_args() if __name__=='__main__' else argparse.Namespace(fd=None,seed=3407)
"""Cap125 matched-label adaptation; preserves the separate cap130 reconstruction."""
from pathlib import Path
import os
import sys
import json
import hashlib
import random
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
os.environ.setdefault('OMP_NUM_THREADS','1')
MODEL_DIR=Path(__file__).resolve().parent
ROOT=OUTPUT_DIR/'CNN_LSTM_DA'/f'seed{ARGS.seed}'
RAW=DATA_DIR
# No shared baseline imports.
os.environ['RUL_PINN_OUTPUT_DIR']=str(ROOT/'runtime')
os.environ['RUL_PINN_RESULTS_DIR']=str(ROOT/'runtime'/'results')
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from model import CNNLSTM
from adaptation import condition_arrays,weighted_mmd

FDS=['FD001','FD002','FD003','FD004']
WINDOWS=dict(zip(FDS,[32,22,39,19]))
EPOCHS=dict(zip(FDS,[167,245,120,198]))
COLS=['unit','cycle']+[f'setting_{i}' for i in range(1,4)]+[f's{i}' for i in range(1,22)]
FEATURES=[f's{i}' for i in range(1,22) if i not in [1,5,10,16]]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(name):
    df=pd.read_csv(RAW/name,sep=r'\s+',header=None)
    assert df.shape[1]==26
    df.columns=COLS
    return df

def windows(df,w,labels):
    x=[];y=[];units=[]
    for unit,g in df.groupby('unit',sort=True):
        g=g.sort_values('cycle');a=g[FEATURES].to_numpy(dtype=np.float32)
        for end in range(w-1,len(g)):
            x.append(a[end-w+1:end+1]);units.append(unit)
            if labels:
                y.append(min(125,float(g['cycle'].max()-g.iloc[end]['cycle'])))
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32),np.array(units)

def mmd(a,b):
    """Biased multiscale Gaussian MMD^2; median bandwidth detached from gradients."""
    joined=torch.cat([a,b],dim=0)
    distance=torch.cdist(joined,joined,p=2).square()
    with torch.no_grad():
        positive=distance.detach()[distance.detach()>0]
        bandwidth=positive.median().clamp_min(1e-6) if positive.numel() else distance.new_tensor(1.)
    kernel=sum(torch.exp(-distance/(bandwidth*factor)) for factor in [.25,.5,1.,2.,4.])/5
    n=len(a)
    return kernel[:n,:n].mean()+kernel[n:,n:].mean()-2*kernel[:n,n:].mean()

def metrics(truth,pred):
    error=pred-truth
    nasa=np.where(error<0,np.expm1(-error/13),np.expm1(error/10))
    return {'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(abs(error).mean()),'nasa_score':float(nasa.sum())}

def main():
    verify_release()
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark=True
    torch.set_float32_matmul_precision('high')
    original=MODEL_DIR/'model.py'
    protected={str(original):sha(original)}
    protocol={'identity':'Numerically repaired CNN-LSTM DA adaptation; output parameterization is not verified author implementation.',
      'reference':'10.1109/ACCESS.2019.2943076','seed':ARGS.seed,'split_seed':42,'validation_ratio':.2,
      'known_settings':{'conv_channels':[16,32,64,64],'kernel':3,'dropout':.3,'lstm_hidden':32,
        'lstm_layers':2,'batch':128,'learning_rate':.001,'windows':WINDOWS,'epochs':EPOCHS,
        'output_scale':125.0,'CNN_negative_slope':0.0,'rul_cap':125,'loss':'sqrt(batch_MSE+1e-8)','normalization':'global sensor MinMax [0,1] fit only fitting engines',
        'input':'17 sensors only; exclude 3 operating settings','augmentation':'none'},
      'assumptions':{'sensor_ids_removed':[1,5,10,16],
        'sensor_reason':'Original says four removed but does not list identities; these four are an explicit hypothesis, not verified.',
        'pool':'kernel2 stride2 ceil_modeTrue, same convolution padding; unspecified by original',
        'dense_widths':[32,32,16,1],'recurrent_activation':'standard PyTorch sigmoid/tanh; original ReLU wording ambiguous',
        'lstm_layers_conflict':'Text/figure=2; parameter table=3. Follow text/figure, do not choose by official outcomes.',
        'sequence_layout':'Within-window pooled temporal sequence; original inter-window explanation is ambiguous',
        'checkpoint':'min internal validation RMSE; original validation split and checkpoint rule unspecified',
        'da_loss_weight':1.,'da_kernel':'mean of 5 RBF kernels, median squared distance bandwidth x .25/.5/1/2/4',
        'da_feature':'CNN flattened feature map before LSTM',
        'DA':'wcMMD PatternII Eq8 first weighted-kernel equality; condition priors from eligible domain windows'},
      'variants':['plain','DA'],
      'information_policy':'DA uses same-subset official test FEATURES only as unlabeled target, as transductive protocol. No test labels until all training choices are frozen.',
      'test_label_policy':'Report both raw NASA labels and labels capped125 because original test capping unclear. Primary capped125 fixed here before training; never select the closer score.',
      'limits':'Public RUL repository contains README only; executable author RUL implementation not found; estimated choices remain. Historical official tests have already been inspected.',
      'first_pass_reference':'The earlier common-project-protocol results stay unchanged and are not relabeled as original reproduction.',
      'runner_sha256':sha(Path(__file__)),'protected':protected}
    path=ROOT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))==protocol
    else:
        path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        (ROOT/'protocol.sha256').write_text(sha(path)+'\n',encoding='ascii')
    # Mechanism-level MMD checks before actual training.
    a=torch.randn(8,12,device='cuda',requires_grad=True)
    same=mmd(a,a);different=mmd(a,a+3)
    assert abs(same.item())<1e-6 and different.item()>same.item()
    different.backward();assert torch.isfinite(a.grad).all()
    print('MMD_TESTS_PASSED',torch.cuda.get_device_name(0),flush=True)
    collected=[]
    for fd in ([ARGS.fd] if ARGS.fd else FDS):
        raw=read(f'train_{fd}.txt')
        all_ids=np.array(sorted(raw.unit.unique()))
        np.random.default_rng(42).shuffle(all_ids)
        nv=round(len(all_ids)*.2);vid=sorted(all_ids[:nv].tolist());tid=sorted(all_ids[nv:].tolist())
        assert not set(vid)&set(tid)
        fit=raw[raw.unit.isin(tid)].copy();val=raw[raw.unit.isin(vid)].copy()
        target=read(f'test_{fd}.txt')
        scaler=MinMaxScaler().fit(fit[FEATURES])
        for df in [fit,val,target]:
            df[FEATURES]=df[FEATURES].astype(np.float32)
            df.loc[:,FEATURES]=scaler.transform(df[FEATURES]).astype(np.float32)
        xt,yt,_=windows(fit,WINDOWS[fd],True)
        xv,yv,_=windows(val,WINDOWS[fd],True)
        xu,_,_=windows(target,WINDOWS[fd],False)
        end_x=[];end_units=[]
        for unit,g in target.groupby('unit',sort=True):
            a=g.sort_values('cycle')[FEATURES].to_numpy(dtype=np.float32)
            if len(a)<WINDOWS[fd]:
                a=np.concatenate([np.repeat(a[[0]],WINDOWS[fd]-len(a),axis=0),a])
            end_x.append(a[-WINDOWS[fd]:]);end_units.append(unit)
        endpoint=np.array(end_x)
        # Conditions use unlabeled settings; no change to the CNN input.
        ca,cb,pa,pb,condition_metadata=condition_arrays(read(f'train_{fd}.txt'),read(f'test_{fd}.txt'),tid,WINDOWS[fd],1 if fd in ['FD001','FD003'] else 6)
        source_conditions,target_conditions=torch.as_tensor(ca,device='cuda'),torch.as_tensor(cb,device='cuda')
        source_prior,target_prior=torch.as_tensor(pa,device='cuda',dtype=torch.float32),torch.as_tensor(pb,device='cuda',dtype=torch.float32)
        # Preload windows to GPU to avoid repeated CPU data transfer.
        tx=torch.as_tensor(xt,device='cuda');ty=torch.as_tensor(yt,device='cuda');ux=torch.as_tensor(xu,device='cuda')
        print('DATA',fd,len(xt),len(xv),len(xu),flush=True)
        for variant in ['DA']:
            out=ROOT/fd/variant;out.mkdir(parents=True,exist_ok=True)
            if (out/'metrics.json').exists():
                collected.append(json.loads((out/'metrics.json').read_text(encoding='utf-8')));continue
            random.seed(ARGS.seed);np.random.seed(ARGS.seed);torch.manual_seed(ARGS.seed);torch.cuda.manual_seed_all(ARGS.seed)
            model=CNNLSTM(17).cuda()
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            val_tensor=torch.as_tensor(xv,device='cuda')
            best=float('inf');hist=[];begin=time.time()
            def predict(x):
                model.eval();outputs=[]
                with torch.inference_mode():
                    for start in range(0,len(x),128):
                        outputs.append(model(x[start:start+128]).clamp(min=0).cpu().numpy())
                return np.concatenate(outputs)
            print('START',fd,variant,'epochs',EPOCHS[fd],flush=True)
            for epoch in range(1,EPOCHS[fd]+1):
                model.train();order=torch.randperm(len(tx),device='cuda')
                total=0.;regtotal=0.;datotal=0.;steps=0
                for start in range(0,len(tx),128):
                    ix=order[start:start+128];x=tx[ix];y=ty[ix]
                    optimizer.zero_grad(set_to_none=True)
                    if variant=='DA':
                        target_ix=torch.randint(len(ux),(len(ix),),device='cuda')
                        latent=model.conv(torch.cat([x,ux[target_ix]]).transpose(1,2)).transpose(1,2)
                        source_latent=latent[:len(ix)]
                        h,_=model.lstm(source_latent)
                        pred=model.head(h[:,-1]).squeeze(-1)
                        adaptation=weighted_mmd(source_latent.flatten(1),latent[len(ix):].flatten(1),source_conditions[ix],target_conditions[target_ix],source_prior,target_prior)
                    else:
                        pred=model(x);adaptation=pred.new_tensor(0.)
                    regression=torch.sqrt((pred-y).square().mean()+1e-8)
                    loss=regression+adaptation
                    assert torch.isfinite(loss)
                    loss.backward();optimizer.step()
                    total+=loss.item();regtotal+=regression.item();datotal+=adaptation.item();steps+=1
                vp=predict(val_tensor);vm=metrics(yv,vp)
                hist.append({'epoch':epoch,'loss':total/steps,'regression_rmse':regtotal/steps,'mmd':datotal/steps,'validation_prediction_std':float(vp.std()),**vm})
                payload={'model_state_dict':model.state_dict(),'epoch':epoch,'val_metrics':vm,
                         'sensor_cols':FEATURES,'train_ids':tid,'val_ids':vid,
                         'scaler_min':scaler.min_.tolist(),'scaler_scale':scaler.scale_.tolist(),'output_scale':125.0,'CNN_negative_slope':0.0}
                if vm['rmse']<best:
                    best=vm['rmse'];torch.save(payload,out/'best.pt')
                if epoch==EPOCHS[fd]:
                    torch.save(payload,out/'last.pt')
                if epoch%10==0 or epoch==1 or epoch==EPOCHS[fd]:
                    print(fd,variant,f'{epoch}/{EPOCHS[fd]} val={vm["rmse"]:.3f} best={best:.3f} mmd={datotal/steps:.5f}',flush=True)
            pd.DataFrame(hist).to_csv(out/'history.csv',index=False)
            # Freeze both predeclared checkpoint predictions before opening labels.
            predictions={};epochs={}
            for key in ['best','last']:
                cp=torch.load(out/f'{key}.pt',map_location='cuda',weights_only=False)
                model.load_state_dict(cp['model_state_dict']);epochs[key]=cp['epoch']
                predictions[key]=predict(torch.as_tensor(endpoint,device='cuda'))
            pd.DataFrame({'unit':end_units,'best_prediction':predictions['best'],'last_prediction':predictions['last']}).to_csv(out/'predictions_before_labels.csv',index=False)
            label=pd.read_csv(RAW/f'RUL_{fd}.txt',sep=r'\s+',header=None).iloc[:,0].to_numpy()
            assert np.array_equal(end_units,np.arange(1,len(label)+1))
            row={'fd':fd,'variant':variant,'seed':ARGS.seed,'best_epoch':epochs['best'],
                 'epochs':EPOCHS[fd],'elapsed_seconds':time.time()-begin,
                 'best_capped125':metrics(np.minimum(label,125),predictions['best']),
                 'best_raw_labels':metrics(label,predictions['best']),
                 'last_capped125':metrics(np.minimum(label,125),predictions['last']),
                 'last_raw_labels':metrics(label,predictions['last'])}
            pd.DataFrame({'unit':end_units,'true_rul_raw':label,'true_rul_cap125':np.minimum(label,125),
                          **{key+'_prediction':value for key,value in predictions.items()}}).to_csv(out/'predictions.csv',index=False)
            (out/'metrics.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
            collected.append(row)
            (ROOT/'progress.json').write_text(json.dumps(collected,indent=2),encoding='utf-8')
            print('DONE',json.dumps(row),flush=True)
        del tx,ty,ux
        torch.cuda.empty_cache()
    assert all(sha(Path(p))==h for p,h in protected.items())
    (ROOT/'COMPLETE.json').write_text(json.dumps({'results':collected,'earlier_runner_unchanged':True},indent=2),encoding='utf-8')
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':
    main()
