"""Final wcMMD PatternII; exact kernel and condition implementation used for adopted results."""
import torch
import numpy as np

def kernel(a,b):
    distance=torch.cdist(torch.cat([a,b]),torch.cat([a,b]),p=2).square()
    with torch.no_grad():
        positive=distance.detach()[distance.detach()>0]
        bandwidth=positive.median().clamp_min(1e-6) if positive.numel() else distance.new_tensor(1.)
    return sum(torch.exp(-distance/(bandwidth*f)) for f in [.25,.5,1.,2.,4.])/5

def weighted_mmd(a,b,source_conditions,target_conditions,source_prior,target_prior):
    k=kernel(a,b);n=len(a)
    ws=source_prior[source_conditions];wt=target_prior[target_conditions]
    ws=ws/ws.sum();wt=wt/wt.sum()
    return ws@k[:n,:n]@ws+wt@k[n:,n:]@wt-2*(ws@k[:n,n:]@wt)

def condition_arrays(raw_source,raw_target,fit_ids,window,clusters):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    fitting=raw_source[raw_source.unit.isin(fit_ids)]
    cols=['setting_1','setting_2','setting_3']
    scaler=StandardScaler().fit(fitting[cols])
    km=KMeans(n_clusters=clusters,random_state=42,n_init=10).fit(scaler.transform(fitting[cols])) if clusters>1 else None
    def assign(frame):
        labels=[]
        for _,g in frame.groupby('unit',sort=True):
            settings=g.sort_values('cycle').iloc[window-1:][cols]
            # The original window generator emits no samples for short engines.
            # sklearn transforms reject empty arrays, so skip the same engines.
            if len(settings)==0:continue
            z=km.predict(scaler.transform(settings)) if km is not None else np.zeros(len(settings),dtype=np.int64)
            labels.extend(z.tolist())
        return np.asarray(labels,dtype=np.int64)
    a,b=assign(fitting),assign(raw_target)
    pa=np.bincount(a,minlength=clusters)/len(a);pb=np.bincount(b,minlength=clusters)/len(b)
    return a,b,pa,pb,dict(clusters=clusters,settings_mean=scaler.mean_.tolist(),settings_scale=scaler.scale_.tolist(),
        centres=km.cluster_centers_.tolist() if km is not None else None,source_prior=pa.tolist(),target_prior=pb.tolist())
