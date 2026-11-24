"""Check the source-only release and all six model manifests."""
import ast
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
models = {p.name for p in (root / "models").iterdir() if p.is_dir()}
assert models == {"ASTHGNN", "BPG_LSTM", "CNN_LSTM_DA", "PI_STHNN", "STFA", "STHNN"}
for model in sorted(models):
    folder = root / "models" / model
    manifest = json.loads((folder / "source_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["source"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, (model, name)
for path in root.rglob("*"):
    if not path.is_file() or {".git", "__pycache__", "runs", "outputs"} & set(path.relative_to(root).parts):
        continue
    assert path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".log", ".pdf"}, path
    if path.suffix in {".py", ".json", ".md", ".txt"}:
        content = path.read_text(encoding="utf-8")
        assert not re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", content), path
        if path.suffix == ".py":
            ast.parse(content)
print("Six model source manifests and release contents verified.")
