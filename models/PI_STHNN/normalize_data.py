from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT
ALIGNMENT=ROOT
sys.path.insert(0,str(ALIGNMENT))
sys.path.insert(0,str(PROJECT))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data import SENSORS
from study_data import common_data


def assign(rows,state):
    if int(state['clusters'])==1:
        return np.zeros(len(rows),dtype=int)
    settings=(rows[:,2:5]-state['settings_mean'])/state['settings_scale']
    distances=((settings[:,None,:]-state['centres'][None,:,:])**2).sum(-1)
    return distances.argmin(-1)


def transform(rows,state):
    values=rows[:,np.array(SENSORS)+4]
    if int(state['clusters'])==1:
        return 2*(values-state['minimum'][0])/state['span'][0]-1
    labels=assign(rows,state)
    return 2*(values-state['minimum'][labels])/state['span'][labels]-1


def make_windows(table,ids,state,w,cutoff=None):
    xx,yy,meta=[],[],[]
    for unit in ids:
        rows=table[table[:,0]==unit]
        z=transform(rows,state).astype(np.float32)
        for end in range(w-1,len(rows)):
            if cutoff is not None and rows[end,1]<cutoff:
                continue
            xx.append(z[end-w+1:end+1])
            yy.append(min(125,rows[-1,1]-rows[end,1]))
            meta.append([int(unit),int(rows[end,1])])
    return np.asarray(xx),np.asarray(yy,dtype=np.float32),np.asarray(meta,dtype=np.int64)


def variance_fraction(x,labels,k):
    centre=x.mean(0)
    between=sum((labels==c).mean()*(x[labels==c].mean(0)-centre)**2 for c in range(k))
    return between/np.maximum(x.var(0),1e-12)


def normalized_data(fd):
    d=common_data(fd,'paper_window')
    k=json.loads((ROOT/'config.json').read_text())['regimes'][fd]
    if k==1:
        state=dict(clusters=np.array(1),minimum=d['minimum'][None],span=d['span'][None],
                   settings_mean=np.zeros(3),settings_scale=np.ones(3),centres=np.zeros((1,3)))
        return d,state,dict(fd=fd,clusters=1,exact_global_delegation=True)
    table=np.loadtxt(DATA_DIR/f'train_{fd}.txt')
    fitting=table[np.isin(table[:,0],d['protocol']['fit_engines'])]
    settings=StandardScaler().fit(fitting[:,2:5])
    z=settings.transform(fitting[:,2:5])
    km=KMeans(n_clusters=k,random_state=42,n_init=10).fit(z)
    labels=km.labels_
    x=fitting[:,np.array(SENSORS)+4]
    minimum=np.stack([x[labels==c].min(0) for c in range(k)])
    span=np.stack([np.ptp(x[labels==c],axis=0) for c in range(k)])
    span=np.where(span<=1e-12,1.,span)
    state=dict(clusters=np.array(k),settings_mean=settings.mean_,settings_scale=settings.scale_,
               centres=km.cluster_centers_,minimum=minimum,span=span)
    assert np.array_equal(assign(fitting,state),labels)
    before=variance_fraction(x,labels,k)
    after=variance_fraction(transform(fitting,state),labels,k)
    for prefix,ids,cutoff in [('train',d['protocol']['fit_engines'],None),
                              ('val',d['protocol']['validation_engines'],d['protocol']['common_validation_cutoff'])]:
        xx,yy,meta=make_windows(table,ids,state,d['protocol']['window'],cutoff)
        if prefix=='val':
            assert np.array_equal(meta,d['val_meta']) and np.array_equal(yy,d['val_y'])
        d[prefix+'_x'],d[prefix+'_y'],d[prefix+'_meta']=xx,yy,meta
    digest=hashlib.sha256(d['val_meta'].tobytes()+d['val_y'].tobytes()).hexdigest()
    assert digest==d['protocol']['common_validation_sha256']
    d['protocol'].update(normalization='fitting-only condition-wise minmax[-1,1]',regime_count=k,
                         graph_normalization='raw fitting-data graph unchanged')
    diagnostic=dict(fd=fd,clusters=k,regime_samples=np.bincount(labels,minlength=k).tolist(),
                    between_variance_fraction_before=before.tolist(),between_variance_fraction_after=after.tolist(),
                    between_variance_median_before=float(np.median(before)),between_variance_median_after=float(np.median(after)),
                    interpretation='Descriptive fitting-only variance decomposition; not explanation of prediction error')
    return d,state,diagnostic
