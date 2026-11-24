from pathlib import Path
import hashlib
import json
import os

MODEL_DIR = Path(__file__).resolve().parent
REPO_DIR = MODEL_DIR.parents[1]
DATA_DIR = Path(os.environ.get('CMAPSS_DATA_DIR', REPO_DIR/'data'/'CMAPSS'))
OUTPUT_DIR = Path(os.environ.get('RUL_OUTPUT_DIR', REPO_DIR/'runs'))

def verify_release(require_data=True):
    manifest=json.loads((MODEL_DIR/'source_manifest.json').read_text())
    for name,expected in manifest['source'].items():
        actual=hashlib.sha256((MODEL_DIR/name).read_bytes()).hexdigest()
        if actual!=expected:
            raise RuntimeError(f'Source differs from release manifest: {name}')
    if require_data:
        for name,expected in manifest['data'].items():
            p=DATA_DIR/name
            if not p.is_file():
                raise FileNotFoundError(f'Place the original C-MAPSS files in {DATA_DIR}; missing {name}')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'Dataset differs from recorded experiment: {name}')
    return manifest
