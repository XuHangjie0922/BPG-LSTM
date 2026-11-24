"""Score published engine endpoints; do not use these scores for checkpoint selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def metrics(true,pred):
    true,pred=np.asarray(true,dtype=float),np.asarray(pred,dtype=float)
    if true.shape!=pred.shape or true.ndim!=1 or not len(true):raise ValueError('Expected paired nonempty vectors')
    if not np.isfinite(true).all() or not np.isfinite(pred).all():raise ValueError('Nonfinite values')
    e=pred-true
    return dict(n=len(e),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.abs(e).mean()),
                nasa=float(np.expm1(np.where(e<0,-e/13,e/10)).sum()))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('csv',type=Path)
    a=p.parse_args()
    groups={};identities=set()
    with a.csv.open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            key=(r.get('Model','model'),r.get('FD','dataset'),r.get('Seed','seed'))
            identity=key+(r['unit'],)
            if identity in identities:raise ValueError('Repeated engine endpoint')
            identities.add(identity)
            groups.setdefault(key,[]).append((float(r['true_rul']),float(r['predicted_rul'])))
    rows=[]
    for key,pairs in sorted(groups.items()):
        y,pred=np.array(pairs).T
        rows.append(dict(model=key[0],fd=key[1],seed=key[2],**metrics(y,pred)))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
