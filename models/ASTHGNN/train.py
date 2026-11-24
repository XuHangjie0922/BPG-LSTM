from release_paths import DATA_DIR,OUTPUT_DIR,verify_release
from pathlib import Path
import os,sys,json,inspect,textwrap,ast,argparse
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch
import repair_core as r
ROOT=Path(__file__).resolve().parent
OUT=OUTPUT_DIR/'ASTHGNN'
j=r.four.job
write=r.write
SEEDS=[3407,2024,137,123,12345]
FDS=['FD001','FD002','FD003','FD004']

def verify():
    return verify_release()


def seeded_function(module, seed):
    # Structured AST specialization changes only the fixed training seed and prefix sampler seed.
    tree=ast.parse(textwrap.dedent(inspect.getsource(module.run)))
    class Seeds(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value in (42,45042):
                return ast.copy_location(ast.Constant(seed if node.value==42 else 45000+seed),node)
            return node
    tree=ast.fix_missing_locations(Seeds().visit(tree))
    scope={}
    exec(compile(tree,'<seed-specialized-locked-training>','exec'),module.__dict__,scope)
    return scope['run']

def worker(fd,seed):
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    r.TRAINING_SEED=seed
    manifest=verify()
    dest=OUT/f'{fd}_seed{seed}'
    dest.mkdir(exist_ok=False)
    data=r.prepare(fd,60)
    prefix=r.four.prefix_data(fd,data)
    np.savez(dest/'preprocessing.npz',**data['state'])
    np.savez(dest/'split_indices.npz',full_fitting=data['tm'],full_validation=data['vm'],prefix_fitting=prefix['tm'],prefix_validation=prefix['vm'])
    def writer(path,value):
        write(path,dict(value,fd=fd,seed=seed) if isinstance(value,dict) else value)
    protocol=dict(json.loads((ROOT/'config.json').read_text()),fd=fd,seed=seed)
    r.OUT=dest
    r.write=writer
    j.OUT=dest
    j.write=writer
    train_r=seeded_function(r,seed)
    train_j=seeded_function(j,seed)
    if fd in ['FD002','FD003']:
        j.official=lambda *args: []
        result=train_j('prefix_25pct',data,prefix,protocol,manifest)
        checkpoint=dest/'prefix_25pct'/'best.pt'
        model=r.make_model('sigmoid')
        model.load_state_dict(torch.load(checkpoint,weights_only=False,map_location='cpu')['model'])
        selected=dict(rule='minimum mixed validation RMSE',checkpoint=str(checkpoint),epoch=result['best_epoch'])
    else:
        # No historical seed42 checkpoint is reused for a new seed.
        config=dict(name='leaky_prefix0',attention='leaky',prefix_fraction=0.)
        base=train_r(config,data,prefix,manifest,dict(protocol,prefix_validation_rmse_limit=1e12))
        checkpoint=dest/'leaky_prefix0'/'best_full.pt'
        model=r.make_model('leaky')
        model.load_state_dict(torch.load(checkpoint,weights_only=False,map_location='cpu')['model'])
        selected=dict(rule='minimum full validation RMSE',checkpoint=str(checkpoint),epoch=base['best_full']['epoch'])
        if fd=='FD004':
            source,_,_=r.evaluate(model,data,prefix)
            limit=source['prefix']['rmse']*1.05
            config=dict(name='leaky_prefix5',attention='leaky',prefix_fraction=.05)
            adapted=train_r(config,data,prefix,manifest,dict(protocol,prefix_validation_rmse_limit=limit))
            eligible=adapted['best_eligible']
            alpha=0.
            adapted_model=None
            screen=[]
            if eligible:
                adapted_path=dest/'leaky_prefix5'/'best_eligible.pt'
                adapted_model=r.make_model('leaky')
                adapted_model.load_state_dict(torch.load(adapted_path,weights_only=False,map_location='cpu')['model'])
                p0=r.infer(model,prefix['vx']).astype(np.float64)
                p1=r.infer(adapted_model,prefix['vx']).astype(np.float64)
                original_score=r.metric(prefix['vy'],p0)
                for a in [0.,.25,.5,.75,1.]:
                    score=r.metric(prefix['vy'],(1-a)*p0+a*p1)
                    screen.append(dict(alpha=a,eligible=score['nasa']<=original_score['nasa']+1e-8,**score))
                alpha=min((q for q in screen if q['eligible']),key=lambda q:(q['rmse'],q['alpha']))['alpha']
                selected.update(adapted_checkpoint=str(adapted_path),adapted_epoch=eligible['epoch'])
            selected.update(alpha=alpha,source_alpha_screen=screen,rule='original full windows; source-selected blend only for short windows')
            write(dest/'SELECTION.json',selected)
            # The original official evaluator is reused with a strictly length-based inference wrapper.
            original_infer=j.infer
            def routed(m,x):
                p=original_infer(m,x).astype(np.float64)
                # Prefix padding repeats the age feature; actual length is taken from raw endpoint metadata.
                _,meta=r.four.endpoints(fd,data['state'],60)
                short=meta[:,2]>0
                assert len(p)==len(meta)
                result=p.copy()
                if alpha>0:
                    result[short]=(1-alpha)*p[short]+alpha*original_infer(adapted_model,x[short])
                assert np.array_equal(result[~short],p[~short])
                return result
            j.infer=routed
    write(dest/'SELECTION.json',selected)
    official_dir=dest/'official'
    official_dir.mkdir(exist_ok=False)
    official=r.four.official(fd,model,data,official_dir)
    verify()
    result=dict(fd=fd,seed=seed,selected=selected,official=official)
    write(dest/'COMPLETE.json',result)
    write(dest/'progress.json',dict(status='complete',**result))

def main():
    p=argparse.ArgumentParser(description='Frozen final ASTHGNN per-subset adaptation')
    p.add_argument('--fd',required=True,choices=FDS)
    p.add_argument('--seed',required=True,type=int,choices=SEEDS)
    a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    worker(a.fd,a.seed)

if __name__=='__main__':main()
