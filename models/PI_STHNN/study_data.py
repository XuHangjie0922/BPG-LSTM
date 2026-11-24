from release_paths import DATA_DIR, OUTPUT_DIR, verify_release
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT
sys.path.insert(0,str(PROJECT))
import numpy as np
from data import prepare,windows


def common_data(fd,method):
    rules=json.loads((ROOT/'study_protocol.json').read_text())
    cutoff=rules['paper_count_implied_windows'][fd]
    w=cutoff if method=='paper_window' else 30
    d=prepare(PROJECT,fd)
    if w!=30:
        table=np.loadtxt(DATA_DIR/f'train_{fd}.txt')
        for prefix,ids in [('train',d['protocol']['fit_engines']),('val',d['protocol']['validation_engines'])]:
            xx,yy,meta=windows(table,ids,d['minimum'],d['span'],cap=125,w=w)
            d[prefix+'_x'],d[prefix+'_y'],d[prefix+'_meta']=xx,yy,meta
    mask=d['val_meta'][:,1]>=cutoff
    for key in ['val_x','val_y','val_meta']:
        d[key]=d[key][mask]
    digest=hashlib.sha256(d['val_meta'].tobytes()+d['val_y'].tobytes()).hexdigest()
    d['protocol'].update(window=w,common_validation_cutoff=cutoff,common_validation_sha256=digest,
                         checkpoint='minimum common-endpoint internal validation RMSE')
    expected=json.loads((ROOT/'PRECHECK.json').read_text())['common_validation_hashes'][fd]
    if digest!=expected:
        raise RuntimeError('Validation engine/cycle/label pairing differs across methods')
    return d
