# BPG-LSTM (proposed model)

This directory contains the implementation of the model proposed in the accompanying
manuscript: bounded physics-inspired sensor gating combined with stage-adaptive
degradation constraints. It is **not** one of the five paper-based reconstructions in
the sibling directories, and it is not a reconstruction of an external paper.

## Data and training

Place the original C-MAPSS files under the repository's `data/CMAPSS/` directory.

```bash
python models/BPG_LSTM/run.py --fd FD003 --seed 3407 --dry-run
python models/BPG_LSTM/run.py --fd FD003 --seed 3407
```

The paper configuration uses 50 epochs with AMP disabled and selects the minimum
validation RMSE checkpoint. FD003 uses three LSTM layers of width 256.
See [config.json](config.json) and [run provenance](../../docs/BPG_LSTM_provenance.json).
The five historical seeds were selected retrospectively from previously evaluated runs.
This release checks the archived settings and scores without replaying the full training runs.
