"""Run the BPG-LSTM protocol for one C-MAPSS subset and one seed.

The wrapper supplies the per-subset configuration recorded in config.json to
src/train_condition_brdr_physics_gate_rul.py. Place the twelve original
C-MAPSS files under data/CMAPSS/ before running.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "src" / "train_condition_brdr_physics_gate_rul.py"


def build_command(fd: str, seed: int, epochs: int, prefix: str) -> list[str]:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    subset = config["subsets"][fd]
    command = [
        sys.executable, str(TRAIN),
        "--fd", fd,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--output-prefix", prefix,
    ]
    command += config["common_args"]
    command += [
        "--window-size", str(subset["window"]),
        "--hidden-dim", str(subset["hidden"]),
        "--num-layers", str(subset["layers"]),
        "--scaler-mode", subset["scaler"],
        "--stage-prior-range", str(subset["stage_prior"]),
        "--mono-weight", str(subset["mono"]),
        "--smooth-weight", str(subset["smooth"]),
        "--failure-weight", str(subset["failure"]),
        "--augment-noise-std", str(subset["noise"]),
        "--augment-feature-dropout", str(subset["dropout"]),
    ]
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prefix = args.output_prefix or f"bpg_{args.fd.lower()}_s{args.seed}"
    command = build_command(args.fd, args.seed, args.epochs, prefix)
    if args.dry_run:
        print(" ".join(command))
        return 0
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["RUL_PINN_OUTPUT_DIR"] = str(ROOT / "outputs")
    env["RUL_PINN_RESULTS_DIR"] = str(ROOT / "results")
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
