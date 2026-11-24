from release_paths import verify_release
from pathlib import Path
import json
import numpy as np
import torch
def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)

def infer(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(b.cuda()).cpu().numpy() for b in torch.from_numpy(x).split(64)])

def verify(manifest):
    verify_release()
