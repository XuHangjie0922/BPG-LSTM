# BPG-LSTM and five C-MAPSS baselines

This release contains the five locally reconstructed models used in the adopted results: CNN-LSTM (wcMMD Pattern II), PI-STHNN with final softmax fusion, STHNN, seed-corrected ASTHGNN, and validation-selected STFA. Source, active settings, and result tables are synchronized. These are paper-based reconstructions, not confirmed author code.

The proposed [BPG-LSTM](models/BPG_LSTM/) has its own runner and historical run provenance.

## Run

Install the pinned dependencies in `requirements.txt` with a compatible CUDA PyTorch build. Place the twelve original C-MAPSS files in `data/CMAPSS/`; the source manifests validate their hashes. Raw datasets and vendored dependencies are not bundled.

```bash
python train.py --model CNN_LSTM_DA --fd FD001 --seed 3407
python train.py --model PI_STHNN --all
python train.py --model STFA --all --dry-run
python tools/check_release.py
python tools/score_predictions.py results/official_terminal_predictions.csv
```

Use `--data-dir` and `--output-dir` to set data and new training output directories. Each baseline has 20 final cases, across four subsets and five seeds. Existing outputs are not silently overwritten.

## Exact seed sets

| Models | Seeds |
|---|---|
| CNN-LSTM (wcMMD), PI_STHNN | 3407,2024,137,123,12345 |
| STHNN, ASTHGNN, STFA | 3407,2024,137,123,12345 |

All five baselines use the seed values listed above. Training splits and preprocessing vary by model.

## Results and contents

The root results below cover the five baselines. The proposed model’s historical
settings, reported metrics and inference checks are in
[its provenance record](docs/BPG_LSTM_provenance.json).

`results/official_terminal_predictions.csv` contains 17,675 engine endpoints from 100 final cases. `results/per_seed_metrics.json` and `results/five_seed_summary.json` can be recomputed from these predictions. Standard deviations use ddof=1. Baseline checkpoints are selected using internal source validation. CNN adaptation also uses unlabeled official target features; target RUL labels are not used for checkpoint selection.

`models/` contains training code, active configuration, required graph support, and hash manifests. `docs/MODEL_NOTES.md` documents implementation assumptions. Checkpoints, obsolete trials, tuning grids, logs, caches and dependency copies are excluded from this GitHub source package. The separate local model folders retain the 100 selected cases, selected checkpoints and preprocessing/selection evidence.

Source migration is checked against the retained training implementations and checkpoints. See `docs/verification.json` for the checks actually performed. Full training of all100 cases has not been repeated after packaging, and a fresh-machine dependency install has not been tested. CUDA/library differences can affect floating-point training results.
