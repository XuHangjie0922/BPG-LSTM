"""Run the frozen final recipe for one case or all cases of one model."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

MODELS=['CNN_LSTM_DA','PI_STHNN','STHNN','ASTHGNN','STFA']
FDS=['FD001','FD002','FD003','FD004']
SEEDS_BY_MODEL={'CNN_LSTM_DA': [3407, 2024, 137, 123, 12345], 'PI_STHNN': [3407, 2024, 137, 123, 12345], 'STHNN': [3407, 2024, 137, 123, 12345], 'ASTHGNN': [3407, 2024, 137, 123, 12345], 'STFA': [3407, 2024, 137, 123, 12345]}
SEEDS=sorted(set(sum(SEEDS_BY_MODEL.values(),[])))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=MODELS)
    p.add_argument('--fd',choices=FDS)
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--dry-run',action='store_true',help='Print the selected commands without training')
    p.add_argument('--all',action='store_true',help='Run all20 cases for the specified model')
    p.add_argument('--data-dir',type=Path,default=Path('data/CMAPSS'))
    p.add_argument('--output-dir',type=Path,default=Path('runs'))
    a=p.parse_args()
    if a.all and (a.fd or a.seed):p.error('--all cannot be combined with --fd or --seed')
    if not a.all and (a.fd is None or a.seed is None):p.error('Provide both --fd and --seed, or --all')
    if a.seed is not None and a.seed not in SEEDS_BY_MODEL[a.model]:p.error('Seed is not in the final five-seed set for this model')
    root=Path(__file__).resolve().parent
    env=dict(os.environ,CMAPSS_DATA_DIR=str(a.data_dir.resolve()),RUL_OUTPUT_DIR=str(a.output_dir.resolve()),
             PYTHONDONTWRITEBYTECODE='1',KMP_DUPLICATE_LIB_OK='TRUE')
    jobs=[(fd,seed) for fd in FDS for seed in SEEDS_BY_MODEL[a.model]] if a.all else [(a.fd,a.seed)]
    for fd,seed in jobs:
        if a.dry_run:
            print(a.model,fd,seed,flush=True);continue
        subprocess.run([sys.executable,'-B',str(root/'models'/a.model/'train.py'),
                        '--fd',fd,'--seed',str(seed)],cwd=root,env=env,check=True)

if __name__=='__main__':main()
