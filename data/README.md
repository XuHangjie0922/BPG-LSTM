# C-MAPSS data

Obtain the original C-MAPSS benchmark separately and place these twelve files in `data/CMAPSS/`:

- `train_FD001.txt` through `train_FD004.txt`
- `test_FD001.txt` through `test_FD004.txt`
- `RUL_FD001.txt` through `RUL_FD004.txt`

Dataset files are excluded from this repository. Each model's `source_manifest.json`
records the twelve dataset SHA256 values used by the retained experiments.
The trainers verify those files before training. A custom directory can be supplied
with the top-level `--data-dir` option.
