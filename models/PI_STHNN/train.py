import argparse
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(description='Frozen final PI-STHNN reconstruction')
    p.add_argument('--fd',required=True,choices=['FD001','FD002','FD003','FD004'])
    p.add_argument('--seed',required=True,type=int,choices=[3407,2024,137,123,12345])
    a=p.parse_args()
    normalized=a.fd in ['FD002','FD004']
    script='train_normalized.py' if normalized else 'train_study.py'
    cmd=[sys.executable,str(Path(__file__).with_name(script)),'--fd',a.fd,'--seed',str(a.seed),'--phase','full']
    if not normalized:cmd+=['--method','paper_window']
    subprocess.run(cmd,check=True)

if __name__=='__main__':main()
