from pathlib import Path
import argparse
import csv
import json
import os
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from losses import StageConditionAwareBRDRPINNLoss
from metrics import regression_metrics
from train_lstm_baseline import (
    DROP_COLS,
    OUTPUT_DIR,
    RESULTS_DIR,
    configure_device,
    ensure_fd_files,
    load_test,
    load_train,
    save_torch_checkpoint,
    set_seed,
)
from train_stage_condition_brdr_pinn_rul import (
    StageConditionRULSequenceDataset,
    add_condition_labels,
    append_summary,
    assign_stage_ids,
    make_censored_validation_windows,
    make_test_condition_windows,
    make_train_condition_windows,
    predict as stage_predict,
    save_loss_history,
    save_metrics,
    save_predictions,
    split_train_validation,
)


FD = "FD001"
WINDOW_SIZE = 30
MAX_RUL = 125
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3


def gradient_alignment(
    data_grads: tuple[torch.Tensor | None, ...],
    physics_grads: tuple[torch.Tensor | None, ...],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return gradient dot product, data-gradient norm squared, and cosine."""
    pairs = [
        (data_grad, physics_grad)
        for data_grad, physics_grad in zip(data_grads, physics_grads)
        if data_grad is not None and physics_grad is not None
    ]
    if not pairs:
        zero = torch.tensor(0.0)
        return zero, zero, 0.0
    dot = sum((data_grad * physics_grad).sum() for data_grad, physics_grad in pairs)
    data_norm_sq = sum(data_grad.square().sum() for data_grad, _ in pairs)
    physics_norm_sq = sum(physics_grad.square().sum() for _, physics_grad in pairs)
    cosine = dot / (data_norm_sq.sqrt() * physics_norm_sq.sqrt()).clamp_min(1e-12)
    return dot, data_norm_sq, float(cosine.detach().cpu())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the stage-condition-aware BRDR-PINN-RUL model on C-MAPSS."
    )
    parser.add_argument("--fd", default=FD)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--max-rul", type=int, default=MAX_RUL)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--selection-metric", choices=["rmse", "nasa_score"], default="rmse")
    parser.add_argument(
        "--validation-protocol", choices=["all_windows", "censored"], default="all_windows"
    )
    parser.add_argument(
        "--validation-rul-grid", type=float, nargs="+", default=[20.0, 50.0, 80.0, 110.0, 125.0]
    )
    parser.add_argument("--num-conditions", type=int, default=6)
    parser.add_argument("--auto-num-conditions", action="store_true")
    parser.add_argument("--scaler-mode", choices=["global", "condition"], default="condition")
    parser.add_argument("--middle-rul", type=float, default=100.0)
    parser.add_argument("--late-rul", type=float, default=40.0)
    parser.add_argument("--late-stage-weight", type=float, default=1.5)
    parser.add_argument("--learnable-stage-prior", action="store_true")
    parser.add_argument("--stage-prior-reg-weight", type=float, default=0.001)
    parser.add_argument("--stage-prior-range", type=float, default=0.25)
    parser.add_argument("--mono-weight", type=float, default=0.1)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--rul-dynamics-weight", type=float, default=0.0)
    parser.add_argument("--rul-dynamics-threshold", type=float, default=80.0)
    parser.add_argument("--sensor-trend-weight", type=float, default=0.0)
    parser.add_argument("--sensor-trend-min-corr", type=float, default=0.15)
    parser.add_argument("--sensor-trend-top-k", type=int, default=8)
    parser.add_argument("--sensor-trend-min-stage", type=int, default=1)
    parser.add_argument("--sensor-trend-huber-delta", type=float, default=1.0)
    parser.add_argument("--health-weight", type=float, default=0.0)
    parser.add_argument("--health-mono-weight", type=float, default=0.0)
    parser.add_argument("--health-index-pretrain-epochs", type=int, default=0)
    parser.add_argument("--weak-health-index", action="store_true")
    parser.add_argument("--health-margin", type=float, default=0.01)
    parser.add_argument("--hi-min-std", type=float, default=0.05)
    parser.add_argument("--hi-endpoint-gap", type=float, default=0.2)
    parser.add_argument("--hi-range-eps", type=float, default=0.02)
    parser.add_argument("--unified-health-state", action="store_true")
    parser.add_argument(
        "--health-state-mode",
        choices=["off", "aux", "residual", "condition_base", "condition_residual"],
        default="off",
    )
    parser.add_argument("--condition-residual", action="store_true")
    parser.add_argument("--condition-residual-baseline-rul", type=float, default=100.0)
    parser.add_argument("--failure-weight", type=float, default=0.1)
    parser.add_argument("--failure-threshold", type=float, default=5.0)
    parser.add_argument("--asym-weight", type=float, default=0.0)
    parser.add_argument("--asym-over-alpha", type=float, default=10.0)
    parser.add_argument("--asym-under-beta", type=float, default=13.0)
    parser.add_argument("--asym-clip", type=float, default=50.0)
    parser.add_argument("--over-margin-weight", type=float, default=0.0)
    parser.add_argument("--over-margin", type=float, default=0.0)
    parser.add_argument("--beta-c", type=float, default=0.99)
    parser.add_argument("--beta-w", type=float, default=0.95)
    parser.add_argument("--brdr-min-weight", type=float, default=0.25)
    parser.add_argument("--brdr-max-weight", type=float, default=4.0)
    parser.add_argument("--brdr-weight-power", type=float, default=0.75)
    parser.add_argument("--brdr-warmup-steps", type=int, default=20)
    parser.add_argument("--physics-gradient-diagnostics", action="store_true")
    parser.add_argument("--physics-gradient-conflict-control", action="store_true")
    parser.add_argument("--physics-gradient-diagnostic-every", type=int, default=10)
    parser.add_argument("--physics-gradient-projection-strength", type=float, default=1.0)
    parser.add_argument(
        "--sensor-gate-control", choices=["physics", "learned", "shuffled"], default="physics"
    )
    parser.add_argument("--trajectory-align-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-align-temperature", type=float, default=0.08)
    parser.add_argument("--stage-trajectory-align-weight", type=float, default=0.0)
    parser.add_argument("--stage-trajectory-align-temperature", type=float, default=0.12)
    parser.add_argument("--stage-trajectory-align-window", type=int, default=5)
    parser.add_argument(
        "--stage-trajectory-align-source",
        choices=["target", "prediction"],
        default="target",
        help="Use labels or the model's current RUL prediction to form soft-stage affinities.",
    )
    parser.add_argument(
        "--stage-trajectory-align-condition-mode",
        choices=["all", "same", "soft"],
        default="all",
        help="Pair samples across all conditions, only within the same cluster, or by soft setting similarity.",
    )
    parser.add_argument("--stage-trajectory-align-condition-temperature", type=float, default=0.5)
    parser.add_argument("--stage-trajectory-align-fault-aware", action="store_true")
    parser.add_argument("--stage-trajectory-align-fault-temperature", type=float, default=0.5)
    parser.add_argument("--fault-prototypes", type=int, default=2)
    parser.add_argument("--fault-prototype-weight", type=float, default=0.0)
    parser.add_argument("--fault-prototype-temperature", type=float, default=0.2)
    parser.add_argument("--fault-mixture-experts", action="store_true")
    parser.add_argument("--fault-mixture-expert-count", type=int, default=2)
    parser.add_argument("--fault-mixture-temperature", type=float, default=0.5)
    parser.add_argument("--fault-mixture-balance-weight", type=float, default=0.01)
    parser.add_argument(
        "--stage-trajectory-align-condition-source",
        choices=["window", "raw_settings"],
        default="window",
        help="Build soft condition similarity from scaled windows or raw setting cluster profiles.",
    )
    parser.add_argument("--condition-lstm-init", action="store_true")
    parser.add_argument("--condition-lstm-bounded-residual", action="store_true")
    parser.add_argument("--condition-lstm-residual-max", type=float, default=0.10)
    parser.add_argument("--condition-output-residual", action="store_true")
    parser.add_argument("--condition-output-residual-max", type=float, default=0.10)
    parser.add_argument("--condition-dropout", type=float, default=0.20)
    parser.add_argument("--condition-output-center-weight", type=float, default=0.01)
    parser.add_argument("--condition-balanced-sampling", action="store_true")
    parser.add_argument("--rul-kernel-reweighting", action="store_true")
    parser.add_argument("--rul-kernel-bandwidth", type=float, default=8.0)
    parser.add_argument("--rul-kernel-power", type=float, default=0.5)
    parser.add_argument("--rul-kernel-min-weight", type=float, default=0.25)
    parser.add_argument("--rul-kernel-max-weight", type=float, default=4.0)
    parser.add_argument("--multi-scale-temporal", action="store_true")
    parser.add_argument("--multi-scale-residual-max", type=float, default=0.20)
    parser.add_argument("--stage-calibration", action="store_true")
    parser.add_argument("--stage-calibration-shrinkage", type=float, default=10.0)
    parser.add_argument("--stage-calibration-min-samples", type=int, default=20)
    parser.add_argument("--condition-stage-calibration", action="store_true")
    parser.add_argument("--soft-condition-stage-calibration", action="store_true")
    parser.add_argument("--safe-soft-condition-stage-calibration", action="store_true")
    parser.add_argument("--soft-stage-calibration-temperature", type=float, default=15.0)
    parser.add_argument("--safe-condition-calibration-temperature", type=float, default=1.0)
    parser.add_argument("--safe-calibration-variance-scale", type=float, default=100.0)
    parser.add_argument("--safe-calibration-positive-scale", type=float, default=0.5)
    parser.add_argument("--stage-calibration-correction-max", type=float, default=10.0)
    parser.add_argument("--condition-rul-bias", action="store_true")
    parser.add_argument("--condition-rul-bias-max", type=float, default=0.05)
    parser.add_argument("--condition-rul-bias-center-weight", type=float, default=0.01)
    parser.add_argument("--condition-consistency-weight", type=float, default=0.0)
    parser.add_argument("--condition-consistency-temperature", type=float, default=0.15)
    parser.add_argument("--condition-consistency-equalization-weight", type=float, default=0.5)
    parser.add_argument("--condition-consistency-fault-aware", action="store_true")
    parser.add_argument("--condition-consistency-fault-temperature", type=float, default=0.5)
    parser.add_argument("--condition-consistency-change-point-aware", action="store_true")
    parser.add_argument("--condition-consistency-change-point-temperature", type=float, default=0.20)
    parser.add_argument("--condition-consistency-fault-mode-aware", action="store_true")
    parser.add_argument("--condition-consistency-fault-modes", type=int, default=2)
    parser.add_argument("--condition-consistency-fault-mode-threshold", type=float, default=0.60)
    parser.add_argument(
        "--within-engine-ranking-weight",
        type=float,
        default=0.0,
        help="Weight for a same-engine temporal RUL ranking constraint.",
    )
    parser.add_argument("--within-engine-ranking-margin", type=float, default=0.5)
    parser.add_argument("--condition-normalized-ranking-weight", type=float, default=0.0)
    parser.add_argument("--condition-normalized-ranking-margin", type=float, default=0.2)
    parser.add_argument("--condition-normalized-ranking-min-rul-gap", type=float, default=10.0)
    parser.add_argument("--condition-normalized-ranking-warmup-epochs", type=int, default=5)
    parser.add_argument("--condition-normalized-ranking-over-weight", type=float, default=0.0)
    parser.add_argument("--condition-normalized-ranking-over-margin", type=float, default=0.0)
    parser.add_argument("--condition-normalized-ranking-gated", action="store_true")
    parser.add_argument("--condition-normalized-ranking-low-rul-max", type=float, default=40.0)
    parser.add_argument("--condition-normalized-ranking-high-rul-min", type=float, default=100.0)
    parser.add_argument("--condition-normalized-ranking-mid-weight", type=float, default=0.0)
    parser.add_argument("--physics-sensor-gate", action="store_true")
    parser.add_argument("--physics-sensor-gate-lstm-input", action="store_true")
    parser.add_argument("--sensor-gate-strength", type=float, default=0.10)
    parser.add_argument("--sensor-gate-condition-adaptive", action="store_true")
    parser.add_argument("--sensor-gate-max-strength", type=float, default=0.20)
    parser.add_argument("--sensor-gate-warmup-epochs", type=int, default=5)
    parser.add_argument("--physics-graph-branch", action="store_true")
    parser.add_argument("--physics-graph-residual-max", type=float, default=0.10)
    parser.add_argument("--thermodynamic-state-residual", action="store_true")
    parser.add_argument("--thermo-state-dim", type=int, default=16)
    parser.add_argument("--thermo-residual-max", type=float, default=0.10)
    parser.add_argument("--thermo-consistency-weight", type=float, default=0.01)
    parser.add_argument("--thermo-raw-state-input", action="store_true")
    parser.add_argument("--thermo-healthy-reference-residual", action="store_true")
    parser.add_argument("--thermo-condition-confidence-adaptive", action="store_true")
    parser.add_argument("--thermo-confidence-floor", type=float, default=0.25)
    parser.add_argument("--thermo-fusion-mode", choices=["input", "latent"], default="input")
    parser.add_argument("--thermo-consistency-gate", action="store_true")
    parser.add_argument("--thermo-transition-weight", type=float, default=0.01)
    parser.add_argument("--thermo-gate-state-dim", type=int, default=8)
    parser.add_argument("--thermo-gate-max-attenuation", type=float, default=0.50)
    parser.add_argument("--disable-adaptive-brdr-stage-prior", action="store_true")
    parser.add_argument(
        "--disable-brdr-adaptation",
        action="store_true",
        help="Freeze BRDR multipliers at one while leaving stage-prior weighting untouched.",
    )
    parser.add_argument("--shared-private-latent", action="store_true")
    parser.add_argument("--shared-private-orth-weight", type=float, default=0.0)
    parser.add_argument("--condition-invariant-latent", action="store_true")
    parser.add_argument("--condition-adversarial-weight", type=float, default=0.0)
    parser.add_argument("--private-condition-weight", type=float, default=0.0)
    parser.add_argument("--future-trajectory-mode", action="store_true")
    parser.add_argument("--future-trajectory-mode-weight", type=float, default=0.01)
    parser.add_argument("--future-trajectory-mode-residual-max", type=float, default=0.05)
    parser.add_argument("--future-trajectory-mode-fraction", type=float, default=0.20)
    parser.add_argument("--condition-observation-residual", action="store_true")
    parser.add_argument("--condition-observation-residual-max", type=float, default=0.10)
    parser.add_argument("--dynamic-condition-modulation", action="store_true")
    parser.add_argument("--dynamic-condition-modulation-max", type=float, default=0.10)
    parser.add_argument("--fault-mode-suppressed-modulation", action="store_true")
    parser.add_argument("--fault-mode-suppression-weight", type=float, default=0.01)
    parser.add_argument(
        "--fault-mode-conditional-head",
        action="store_true",
        help="Route RUL prediction through soft fault-mode experts inferred from the private latent.",
    )
    parser.add_argument("--fault-mode-conditional-weight", type=float, default=0.01)
    parser.add_argument("--stage-trajectory-align-warmup-epochs", type=int, default=0)
    parser.add_argument("--augment-noise-std", type=float, default=0.0)
    parser.add_argument("--augment-feature-dropout", type=float, default=0.0)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--model-name", default="Stage-Condition-BRDR-PhysicsGate-PINN-RUL")
    parser.add_argument("--summary-name", default="stage_condition_brdr_pinn_rul_results.csv")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Stop after validation-based checkpoint selection without evaluating the test set.",
    )
    parser.add_argument(
        "--attnpinn-protocol",
        action="store_true",
        help="Use the public AttnPINN FD004 preprocessing and row split while keeping this model.",
    )
    parser.add_argument("--attnpinn-raw-split", type=int, default=49072)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument(
        "--condition-seed",
        type=int,
        default=None,
        help="Keep operating-condition clustering fixed while changing engine splits.",
    )
    return parser.parse_args()


def output_prefix(args: argparse.Namespace) -> str:
    if args.output_prefix:
        return args.output_prefix
    return f"stage_condition_brdr_pinn_rul_{args.fd.lower()}"


def augment_windows(x: torch.Tensor, noise_std: float, feature_dropout: float) -> torch.Tensor:
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    if feature_dropout > 0:
        keep = torch.rand(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype) >= feature_dropout
        x = x * keep
    return x


def make_train_windows_with_units(
    df: pd.DataFrame, feature_cols: list[str], window_size: int, middle_rul: float, late_rul: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, condition_ids, stage_ids, unit_ids = [], [], [], [], []
    for unit, sub in df.groupby("unit"):
        sub = sub.sort_values("cycle")
        x = sub[feature_cols].to_numpy(dtype=np.float32)
        y = sub["RUL"].to_numpy(dtype=np.float32)
        c = sub["condition_id"].to_numpy(dtype=np.int64)
        s = assign_stage_ids(y, middle_rul, late_rul)
        if len(sub) < window_size:
            continue
        for i in range(window_size - 1, len(sub)):
            xs.append(x[i - window_size + 1 : i + 1])
            ys.append(y[i - window_size + 1 : i + 1])
            condition_ids.append(c[i])
            stage_ids.append(s[i])
            unit_ids.append(int(unit))
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(condition_ids, dtype=np.int64),
        np.asarray(stage_ids, dtype=np.int64),
        np.asarray(unit_ids, dtype=np.int64),
    )


def estimate_long_term_change_points(
    df: pd.DataFrame, feature_cols: list[str]
) -> dict[int, int]:
    """Estimate an unsupervised degradation onset for each engine from long trends."""
    sensor_cols = [c for c in feature_cols if c not in {"setting_1", "setting_2", "setting_3"}]
    change_points: dict[int, int] = {}
    for unit, sub in df.groupby("unit"):
        sub = sub.sort_values("cycle")
        values = sub[sensor_cols].to_numpy(dtype=np.float64)
        n = len(values)
        if n < 20 or not sensor_cols:
            change_points[int(unit)] = max(0, int(0.30 * max(n - 1, 1)))
            continue
        smooth_window = max(5, min(15, n // 10))
        smoothed = pd.DataFrame(values).rolling(smooth_window, min_periods=1, center=True).mean().to_numpy()
        baseline_end = max(smooth_window, int(0.20 * n))
        baseline = smoothed[:baseline_end]
        center = np.nanmean(baseline, axis=0)
        scale = np.nanstd(baseline, axis=0)
        scale[scale < 1e-6] = 1.0
        score = np.mean(np.abs((smoothed - center) / scale), axis=1)
        score = pd.Series(score).rolling(max(3, smooth_window // 2), min_periods=1).mean().to_numpy()
        normal_score = score[:baseline_end]
        threshold = float(np.nanmean(normal_score) + 3.0 * np.nanstd(normal_score))
        threshold = max(threshold, float(np.nanpercentile(score, 60)))
        min_start = max(baseline_end, int(0.15 * n))
        cp = None
        sustained = max(3, smooth_window // 3)
        for idx in range(min_start, max(min_start, n - sustained + 1)):
            if np.all(score[idx : idx + sustained] > threshold):
                cp = idx
                break
        change_points[int(unit)] = int(cp if cp is not None else max(min_start, int(0.30 * n)))
    return change_points


def build_window_relative_ages(
    df: pd.DataFrame,
    unit_ids: np.ndarray,
    window_size: int,
    change_points: dict[int, int],
) -> np.ndarray:
    """Map ordered windows to relative age since each unit's estimated change point."""
    unit_lengths = {
        int(unit): len(sub.sort_values("cycle"))
        for unit, sub in df.groupby("unit")
    }
    seen: dict[int, int] = {}
    ages = np.zeros(len(unit_ids), dtype=np.float32)
    for idx, unit_value in enumerate(unit_ids):
        unit = int(unit_value)
        endpoint = window_size - 1 + seen.get(unit, 0)
        seen[unit] = seen.get(unit, 0) + 1
        cp = int(change_points.get(unit, max(0, int(0.30 * (unit_lengths.get(unit, window_size) - 1)))))
        denominator = max(unit_lengths.get(unit, window_size) - 1 - cp, 1)
        ages[idx] = float(np.clip((endpoint - cp) / denominator, 0.0, 1.0))
    return ages


def estimate_long_term_fault_mode_probs(
    df: pd.DataFrame, feature_cols: list[str], num_modes: int, seed: int
) -> torch.Tensor:
    """Estimate soft fault-mode probabilities from long-term engine trends."""
    units = sorted(int(unit) for unit in df["unit"].unique())
    sensor_cols = [c for c in feature_cols if c not in {"setting_1", "setting_2", "setting_3"}]
    signatures = []
    for unit in units:
        sub = df[df["unit"] == unit].sort_values("cycle")
        values = sub[sensor_cols].to_numpy(dtype=np.float64)
        n = len(values)
        if n < 8 or not sensor_cols:
            signatures.append(np.zeros(max(len(sensor_cols), 1), dtype=np.float32))
            continue
        smooth_window = max(5, min(15, n // 10))
        smoothed = pd.DataFrame(values).rolling(smooth_window, min_periods=1, center=True).mean().to_numpy()
        half = max(3, n // 2)
        slopes = []
        for col in range(smoothed.shape[1]):
            early = np.polyfit(np.arange(half), smoothed[:half, col], 1)[0]
            late = np.polyfit(np.arange(n - half), smoothed[-(n - half):, col], 1)[0]
            slopes.append(late - early)
        signatures.append(np.asarray(slopes, dtype=np.float32))
    signature_matrix = np.nan_to_num(np.asarray(signatures, dtype=np.float32))
    if len(units) < max(2, num_modes):
        probs = np.full((len(units), 1), 1.0, dtype=np.float32)
    else:
        signature_matrix = StandardScaler().fit_transform(signature_matrix)
        clusterer = KMeans(n_clusters=num_modes, random_state=seed, n_init=10)
        clusterer.fit(signature_matrix)
        distances = clusterer.transform(signature_matrix)
        scale = max(float(np.median(distances)), 1e-3)
        logits = -distances / scale
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True).clip(min=1e-8)
    lookup = np.zeros((max(units) + 1, probs.shape[1]), dtype=np.float32)
    for row, unit in enumerate(units):
        lookup[unit] = probs[row]
    return torch.tensor(lookup, dtype=torch.float32)


def estimate_future_trajectory_mode_probs(
    df: pd.DataFrame, feature_cols: list[str], num_modes: int, seed: int, future_fraction: float
) -> torch.Tensor:
    """Build train-engine pseudo modes from future trajectory signatures only."""
    units = sorted(int(unit) for unit in df["unit"].unique())
    sensor_cols = [c for c in feature_cols if c not in {"setting_1", "setting_2", "setting_3"}]
    signatures = []
    for unit in units:
        sub = df[df["unit"] == unit].sort_values("cycle")
        values = sub[sensor_cols].to_numpy(dtype=np.float64)
        start = max(0, int(len(values) * (1.0 - future_fraction)))
        future = values[start:]
        if len(future) < 4:
            future = values
        q = max(2, len(future) // 3)
        signature = []
        for col in range(future.shape[1]):
            series = future[:, col]
            t = np.linspace(0.0, 1.0, len(series))
            signature.extend([
                float(np.mean(series)),
                float(np.mean(series[-q:]) - np.mean(series[:q])),
                float(np.polyfit(t, series, 1)[0]) if len(series) > 1 else 0.0,
            ])
        signatures.append(signature)
    signature_matrix = StandardScaler().fit_transform(np.nan_to_num(np.asarray(signatures, dtype=np.float32)))
    clusterer = KMeans(n_clusters=max(2, num_modes), random_state=seed, n_init=50)
    clusterer.fit(signature_matrix)
    distances = clusterer.transform(signature_matrix)
    scale = max(float(np.median(distances)), 1e-3)
    logits = -distances / scale
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True).clip(min=1e-8)
    lookup = np.zeros((max(units) + 1, probs.shape[1]), dtype=np.float32)
    for row, unit in enumerate(units):
        lookup[unit] = probs[row]
    return torch.tensor(lookup, dtype=torch.float32)


def make_test_windows_with_units(
    df: pd.DataFrame, feature_cols: list[str], window_size: int, middle_rul: float, late_rul: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, condition_ids, stage_ids, unit_ids = [], [], [], [], []
    for unit, sub in df.groupby("unit"):
        sub = sub.sort_values("cycle")
        x = sub[feature_cols].to_numpy(dtype=np.float32)
        y = sub["RUL"].to_numpy(dtype=np.float32)
        c = sub["condition_id"].to_numpy(dtype=np.int64)
        s = assign_stage_ids(y, middle_rul, late_rul)
        if len(sub) >= window_size:
            x_window, y_window = x[-window_size:], y[-window_size:]
        else:
            pad = window_size - len(sub)
            x_window = np.concatenate([np.repeat(x[[0]], pad, axis=0), x], axis=0)
            y_window = np.concatenate([np.repeat(y[[0]], pad), y], axis=0)
        xs.append(x_window)
        ys.append(y_window)
        condition_ids.append(c[-1])
        stage_ids.append(s[-1])
        unit_ids.append(int(unit))
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(condition_ids, dtype=np.int64),
        np.asarray(stage_ids, dtype=np.int64),
        np.asarray(unit_ids, dtype=np.int64),
    )


class TrajectoryRULSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, x, y_seq, condition_ids, stage_ids, unit_ids, relative_ages=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)
        self.condition_ids = torch.tensor(condition_ids, dtype=torch.long)
        self.stage_ids = torch.tensor(stage_ids, dtype=torch.long)
        self.unit_ids = torch.tensor(unit_ids, dtype=torch.long)
        if relative_ages is None:
            relative_ages = np.zeros(len(unit_ids), dtype=np.float32)
        self.relative_ages = torch.tensor(relative_ages, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx], self.y_seq[idx], self.condition_ids[idx], self.stage_ids[idx],
            self.unit_ids[idx], self.relative_ages[idx]
        )


def cross_engine_trajectory_alignment_loss(
    latent_seq: torch.Tensor,
    target_seq: torch.Tensor,
    unit_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Align latent degradation positions across different engines at similar RUL."""
    endpoint = latent_seq[:, -1]
    endpoint = endpoint / endpoint.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    unit_ids = unit_ids.to(endpoint.device)
    rul = (target_seq[:, -1] / 125.0).clamp(0.0, 1.0)
    pair_rul = (rul[:, None] - rul[None, :]).abs()
    pair_units = unit_ids[:, None] != unit_ids[None, :]
    weights = torch.exp(-pair_rul / max(temperature, 1e-4)) * pair_units.float()
    distances = (endpoint[:, None, :] - endpoint[None, :, :]).pow(2).mean(dim=-1)
    diagonal = ~torch.eye(len(endpoint), dtype=torch.bool, device=endpoint.device)
    weights = weights * diagonal
    return (weights * distances).sum() / weights.sum().clamp_min(1.0)


def soft_stage_trajectory_alignment_loss(
    latent_seq: torch.Tensor,
    target_seq: torch.Tensor,
    unit_ids: torch.Tensor,
    temperature: float,
    trajectory_window: int,
    max_rul: float = MAX_RUL,
    stage_rul: torch.Tensor | None = None,
    condition_ids: torch.Tensor | None = None,
    condition_mode: str = "all",
    condition_features: torch.Tensor | None = None,
    condition_temperature: float = 0.5,
    fault_features: torch.Tensor | None = None,
    fault_temperature: float = 0.5,
    relative_ages: torch.Tensor | None = None,
    change_point_temperature: float = 0.20,
    fault_mode_probs: torch.Tensor | None = None,
    fault_mode_threshold: float = 0.60,
) -> torch.Tensor:
    """Align cross-engine states and local degradation directions by soft RUL stage.

    The affinity is continuous in normalized RUL and also requires similar soft
    stage memberships, so windows from different degradation phases are not
    treated as interchangeable. The second term compares local latent movement,
    which makes this a trajectory constraint rather than endpoint matching only.
    """
    batch_size = latent_seq.size(0)
    if batch_size < 2:
        return latent_seq.new_zeros(())
    endpoint = latent_seq[:, -1]
    endpoint = endpoint / endpoint.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if stage_rul is None:
        stage_rul = target_seq[:, -1]
    rul = (stage_rul / max(float(max_rul), 1.0)).clamp(0.0, 1.0)

    centers = latent_seq.new_tensor([0.15, 0.55, 0.85])
    stage_logits = -((rul[:, None] - centers[None, :]) / max(temperature, 1e-4)).pow(2)
    stage_prob = torch.softmax(stage_logits, dim=-1)
    stage_affinity = torch.mm(stage_prob, stage_prob.transpose(0, 1))
    rul_affinity = torch.exp(-torch.abs(rul[:, None] - rul[None, :]) / max(temperature, 1e-4))
    unit_mask = (unit_ids[:, None] != unit_ids[None, :]).float()
    if condition_ids is not None and condition_mode == "same":
        condition_mask = (condition_ids[:, None] == condition_ids[None, :]).float()
    elif condition_features is not None and condition_mode == "soft":
        profile = condition_features if condition_features.dim() == 2 else condition_features.mean(dim=1)
        profile_distance = (profile[:, None, :] - profile[None, :, :]).pow(2).mean(dim=-1)
        scale = max(float(condition_temperature), 1e-4)
        condition_mask = torch.exp(-profile_distance / (2.0 * scale * scale))
    else:
        condition_mask = torch.ones_like(unit_mask)
    if fault_features is not None:
        fault_profile = fault_features if fault_features.dim() == 2 else fault_features.mean(dim=1)
        fault_profile = fault_profile / fault_profile.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        fault_distance = (fault_profile[:, None, :] - fault_profile[None, :, :]).pow(2).mean(dim=-1)
        fault_scale = max(float(fault_temperature), 1e-4)
        fault_mask = torch.exp(-fault_distance / (2.0 * fault_scale * fault_scale))
    else:
        fault_mask = torch.ones_like(unit_mask)
    diagonal = 1.0 - torch.eye(batch_size, device=latent_seq.device, dtype=latent_seq.dtype)
    weights = stage_affinity * rul_affinity * unit_mask * condition_mask * fault_mask * diagonal

    endpoint_distance = (endpoint[:, None, :] - endpoint[None, :, :]).pow(2).mean(dim=-1)
    steps = max(1, min(int(trajectory_window), latent_seq.size(1) - 1))
    direction = latent_seq[:, -1] - latent_seq[:, -1 - steps]
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction_distance = (direction[:, None, :] - direction[None, :, :]).pow(2).mean(dim=-1)
    value = endpoint_distance + 0.5 * direction_distance
    return (weights * value).sum() / weights.sum().clamp_min(1.0)


def condition_aware_degradation_consistency_loss(
    latent_seq: torch.Tensor,
    target_seq: torch.Tensor,
    condition_ids: torch.Tensor,
    stage_ids: torch.Tensor,
    unit_ids: torch.Tensor,
    temperature: float = 0.15,
    equalization_weight: float = 0.5,
    fault_features: torch.Tensor | None = None,
    fault_temperature: float = 0.5,
    relative_ages: torch.Tensor | None = None,
    change_point_temperature: float = 0.20,
    fault_mode_probs: torch.Tensor | None = None,
    fault_mode_threshold: float = 0.60,
) -> torch.Tensor:
    """Align same-stage latent states across conditions without changing RUL outputs."""
    endpoint = latent_seq[:, -1]
    endpoint = endpoint / endpoint.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    condition_ids = condition_ids.to(endpoint.device)
    stage_ids = stage_ids.to(endpoint.device)
    unit_ids = unit_ids.to(endpoint.device)
    batch_size = endpoint.size(0)
    if batch_size < 2 or condition_ids.unique().numel() < 2:
        return endpoint.new_zeros(())

    diagonal = torch.eye(batch_size, device=endpoint.device, dtype=torch.bool)
    cross_condition = condition_ids[:, None] != condition_ids[None, :]
    cross_unit = unit_ids[:, None] != unit_ids[None, :]
    same_stage = stage_ids[:, None] == stage_ids[None, :]
    positive = same_stage & cross_condition & cross_unit & ~diagonal
    if relative_ages is not None:
        age = relative_ages.to(endpoint.device).float()
        age_similarity = torch.exp(
            -torch.abs(age[:, None] - age[None, :])
            / max(float(change_point_temperature), 1e-4)
        )
        positive = positive & (age_similarity >= 0.5)
    if fault_mode_probs is not None:
        mode_probs = fault_mode_probs.to(endpoint.device)[unit_ids.long()]
        mode_similarity = mode_probs.matmul(mode_probs.transpose(0, 1))
        positive = positive & (mode_similarity >= float(fault_mode_threshold))
    if fault_features is not None:
        fault_profile = fault_features.to(endpoint.device)
        fault_profile = fault_profile / fault_profile.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        fault_distance = (fault_profile[:, None, :] - fault_profile[None, :, :]).pow(2).mean(dim=-1)
        fault_similarity = torch.exp(
            -fault_distance / (2.0 * max(float(fault_temperature), 1e-4) ** 2)
        )
        positive = positive & (fault_similarity >= 0.5)
    logits = endpoint.matmul(endpoint.transpose(0, 1)) / max(float(temperature), 1e-4)
    logits = logits.masked_fill(diagonal, float("-inf"))
    positive_count = positive.sum()
    if positive_count > 0:
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        contrastive = -(log_prob.masked_fill(~positive, 0.0).sum() / positive_count)
    else:
        contrastive = endpoint.new_zeros(())

    condition_centers = []
    for condition in condition_ids.unique():
        members = endpoint[condition_ids == condition]
        if members.size(0) > 0:
            condition_centers.append(members.mean(dim=0))
    centers = torch.stack(condition_centers)
    centers = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    equalization = (centers - centers.mean(dim=0, keepdim=True)).pow(2).mean()
    return contrastive + float(equalization_weight) * equalization


def within_engine_rul_ranking_loss(
    pred_seq: torch.Tensor,
    target_seq: torch.Tensor,
    unit_ids: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Enforce temporal RUL ordering only within the same engine.

    This avoids assuming that different fault modes share a common degradation
    origin. Pairs are formed from windows already present in the mini-batch.
    """
    pred = pred_seq[:, -1]
    target = target_seq[:, -1]
    unit_ids = unit_ids.to(pred.device)
    same_unit = unit_ids[:, None] == unit_ids[None, :]
    earlier = target[:, None] > target[None, :]
    pair_mask = same_unit & earlier
    if not pair_mask.any():
        return pred.new_zeros(())
    margin_value = pred.new_tensor(float(margin))
    pair_gap = pred[:, None] - pred[None, :]
    violation = torch.relu(margin_value - pair_gap)
    # Larger RUL separations carry more reliable temporal ordering information.
    reliability = (target[:, None] - target[None, :]).abs().clamp_min(1.0).sqrt()
    return (violation * reliability * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1.0)


def condition_normalized_degradation_ranking_loss(
    degradation_score: torch.Tensor,
    target_seq: torch.Tensor,
    condition_ids: torch.Tensor,
    margin: float = 0.2,
    min_rul_gap: float = 10.0,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rank shared degradation scores after removing condition-level offsets.

    Higher normalized score should indicate a healthier, larger-RUL sample.
    Unlike adversarial condition removal, this keeps condition information
    available to the predictor and only constrains the degradation ordering.
    """
    score = degradation_score.reshape(-1)
    target = target_seq[:, -1].reshape(-1)
    condition_ids = condition_ids.to(score.device).long()
    normalized = score.clone()
    for condition in condition_ids.unique():
        mask = condition_ids == condition
        if int(mask.sum()) >= 2:
            values = score[mask]
            normalized[mask] = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-3)
        else:
            normalized[mask] = score[mask] - score.mean()
    rul_gap = target[:, None] - target[None, :]
    pair_mask = rul_gap >= float(min_rul_gap)
    pair_weight = pair_mask.float()
    if sample_weight is not None:
        weight = sample_weight.to(score.device).reshape(-1).float().clamp(min=0.0)
        pair_weight = pair_weight * weight[:, None] * weight[None, :]
        pair_mask = pair_mask & (pair_weight > 0)
    if not pair_mask.any():
        return score.new_zeros(())
    score_gap = normalized[:, None] - normalized[None, :]
    violation = torch.relu(score.new_tensor(float(margin)) - score_gap)
    reliability = (rul_gap.clamp_min(0.0) / max(float(min_rul_gap), 1.0)).sqrt().clamp(max=4.0)
    return (violation * reliability * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)


def over_prediction_safe_loss(
    pred_seq: torch.Tensor,
    target_seq: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    pred = pred_seq[:, -1]
    target = target_seq[:, -1]
    over = torch.relu(pred - target - float(margin))
    if not torch.any(over > 0):
        return pred.new_zeros(())
    squared = over.pow(2)
    nasa_like = torch.exp((over / 10.0).clamp(max=10.0)) - 1.0
    return 0.5 * squared.mean() + 0.5 * nasa_like.mean()


def shared_private_orthogonality_loss(
    shared_seq: torch.Tensor, private_seq: torch.Tensor
) -> torch.Tensor:
    """Discourage shared and private latent channels from learning the same signal."""
    shared = shared_seq.reshape(-1, shared_seq.size(-1))
    private = private_seq.reshape(-1, private_seq.size(-1))
    shared = shared - shared.mean(dim=0, keepdim=True)
    private = private - private.mean(dim=0, keepdim=True)
    shared = shared / shared.norm(dim=0, keepdim=True).clamp_min(1e-6)
    private = private / private.norm(dim=0, keepdim=True).clamp_min(1e-6)
    cross_covariance = shared.transpose(0, 1).matmul(private)
    return cross_covariance.pow(2).mean()


def fault_prototype_balance_loss(fault_probs: torch.Tensor) -> torch.Tensor:
    """Prevent all samples from selecting one learnable fault prototype."""
    mean_probs = fault_probs.mean(dim=0)
    target = torch.full_like(mean_probs, 1.0 / max(fault_probs.size(-1), 1))
    return (mean_probs - target).pow(2).mean()


def fault_mixture_balance_loss(router_probs: torch.Tensor) -> torch.Tensor:
    """Keep the soft fault router from collapsing onto one expert."""
    mean_probs = router_probs.mean(dim=0)
    target = torch.full_like(mean_probs, 1.0 / max(router_probs.size(-1), 1))
    return (mean_probs - target).pow(2).mean()


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return GradientReverse.apply(x, float(scale))


def build_physics_adjacency(feature_cols: list[str]) -> torch.Tensor:
    names = {name: idx for idx, name in enumerate(feature_cols)}
    adjacency = torch.eye(len(feature_cols), dtype=torch.float32)

    edges = [
        ("s2", "s3"),
        ("s3", "s4"),
        ("s7", "s11"),
        ("s11", "s15"),
        ("s8", "s13"),
        ("s9", "s14"),
        ("s12", "s3"),
        ("s12", "s4"),
        ("s12", "s11"),
        ("s20", "s21"),
        ("setting_1", "setting_2"),
        ("setting_1", "setting_3"),
        ("setting_2", "setting_3"),
    ]
    groups = [
        ["s2", "s7", "s8"],
        ["s3", "s11", "s12"],
        ["s4", "s15", "s17"],
        ["s8", "s13"],
        ["s9", "s14"],
        ["s20", "s21"],
    ]
    for left, right in edges:
        if left in names and right in names:
            adjacency[names[left], names[right]] = 1.0
            adjacency[names[right], names[left]] = 1.0
    for group in groups:
        present = [names[name] for name in group if name in names]
        for i in present:
            for j in present:
                adjacency[i, j] = 1.0
    return adjacency


def estimate_sensor_degradation_signs(
    train_df: pd.DataFrame, feature_cols: list[str], rul_col: str = "RUL"
) -> torch.Tensor:
    """Estimate sensor directions toward failure from training-only correlations."""
    signs = []
    target = train_df[rul_col].to_numpy(dtype=np.float64)
    failure_axis = -target
    for feature in feature_cols:
        values = train_df[feature].to_numpy(dtype=np.float64)
        if np.std(values) < 1e-8 or np.std(failure_axis) < 1e-8:
            signs.append(0.0)
            continue
        correlation = float(np.corrcoef(values, failure_axis)[0, 1])
        signs.append(float(np.clip(correlation, -1.0, 1.0)))
    return torch.tensor(signs, dtype=torch.float32)


class PhysicsAwareSensorReliabilityGate(nn.Module):
    """Reliability from degradation direction, physical neighbors, and smoothness."""

    def __init__(
        self,
        feature_cols: list[str],
        direction_scores: torch.Tensor,
        strength: float,
        warmup_epochs: int,
        condition_adaptive: bool = False,
        max_strength: float = 0.20,
        num_conditions: int = 6,
        control: str = "physics",
    ) -> None:
        super().__init__()
        self.input_dim = len(feature_cols)
        self.strength = float(strength)
        self.condition_adaptive = condition_adaptive
        self.max_strength = max(float(max_strength), self.strength, 1e-6)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.current_epoch = 0
        self.register_buffer("direction_scores", direction_scores.float())
        adjacency = build_physics_adjacency(feature_cols)
        if control == "shuffled":
            generator = torch.Generator().manual_seed(20261007)
            sensor_indices = torch.arange(3, len(feature_cols))
            permutation = torch.arange(len(feature_cols))
            permutation[3:] = sensor_indices[torch.randperm(len(sensor_indices), generator=generator)]
            self.direction_scores = self.direction_scores[permutation]
            adjacency = adjacency[permutation][:, permutation]
        self.register_buffer("sensor_adjacency", adjacency)
        if condition_adaptive:
            ratio = min(max(self.strength / self.max_strength, 1e-4), 1.0 - 1e-4)
            init_logit = float(np.log(ratio / (1.0 - ratio)))
            self.condition_strength_logit = nn.Parameter(
                torch.full((max(int(num_conditions), 1),), init_logit)
            )

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def forward(self, x: torch.Tensor, condition_ids: torch.Tensor | None = None) -> torch.Tensor:
        slope = x[:, -1] - x[:, 0]
        direction_scores = self.direction_scores.to(device=x.device, dtype=x.dtype)
        if direction_scores.dim() == 2 and condition_ids is not None:
            direction = direction_scores[condition_ids.long().clamp(0, direction_scores.size(0) - 1)]
        else:
            direction = direction_scores[-1] if direction_scores.dim() == 2 else direction_scores
        direction_match = torch.tanh(slope * direction * 4.0)
        adjacency = self.sensor_adjacency.to(device=x.device, dtype=x.dtype)
        degree = adjacency.sum(dim=-1).clamp_min(1.0)
        signed_trend = torch.tanh(slope * 4.0)
        neighbor_support = adjacency.matmul(signed_trend.transpose(0, 1)).transpose(0, 1)
        neighbor_support = neighbor_support / degree.unsqueeze(0)
        first_diff = x[:, 1:] - x[:, :-1]
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        smoothness = 1.0 / (1.0 + second_diff.abs().mean(dim=1) /
                             first_diff.abs().mean(dim=1).clamp_min(1e-4))
        reliability = 0.45 * direction_match + 0.35 * neighbor_support + 0.20 * smoothness
        centered = reliability - reliability.mean(dim=-1, keepdim=True)
        spread = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        normalized = torch.tanh(centered / (2.0 * spread))
        ramp = (
            min(1.0, max(0.0, (self.current_epoch - 1) / self.warmup_epochs))
            if self.warmup_epochs > 0 else 1.0
        )
        if self.condition_adaptive:
            if condition_ids is None:
                condition_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            condition_ids = condition_ids.long().clamp(0, self.condition_strength_logit.numel() - 1)
            strength = self.max_strength * torch.sigmoid(self.condition_strength_logit[condition_ids])
            strength = strength.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        else:
            strength = torch.full(
                (x.size(0), 1), self.strength, device=x.device, dtype=x.dtype
            )
        gate = (1.0 + strength * ramp * normalized).clamp(1.0 - strength, 1.0 + strength)
        if self.input_dim >= 3:
            gate = torch.cat([torch.ones_like(gate[:, :3]), gate[:, 3:]], dim=-1)
        return x * gate.unsqueeze(1)


class LearnedSensorReliabilityGate(PhysicsAwareSensorReliabilityGate):
    """Small data-driven gate with the same bounded condition-wise fusion."""

    def __init__(self, feature_cols, direction_scores, strength, warmup_epochs,
                 condition_adaptive=False, max_strength=0.20, num_conditions=6):
        super().__init__(feature_cols, direction_scores, strength, warmup_epochs,
                         condition_adaptive, max_strength, num_conditions)
        self.sensor_score = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor, condition_ids: torch.Tensor | None = None) -> torch.Tensor:
        first_diff = x[:, 1:] - x[:, :-1]
        features = torch.stack((x.mean(dim=1), x.std(dim=1, unbiased=False),
                                first_diff.mean(dim=1)), dim=-1)
        reliability = self.sensor_score(features).squeeze(-1)
        centered = reliability - reliability.mean(dim=-1, keepdim=True)
        spread = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        normalized = torch.tanh(centered / (2.0 * spread))
        ramp = (
            min(1.0, max(0.0, (self.current_epoch - 1) / self.warmup_epochs))
            if self.warmup_epochs > 0 else 1.0
        )
        if self.condition_adaptive:
            if condition_ids is None:
                condition_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            condition_ids = condition_ids.long().clamp(0, self.condition_strength_logit.numel() - 1)
            strength = self.max_strength * torch.sigmoid(self.condition_strength_logit[condition_ids])
            strength = strength.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        else:
            strength = torch.full((x.size(0), 1), self.strength, device=x.device, dtype=x.dtype)
        gate = (1.0 + strength * ramp * normalized).clamp(1.0 - strength, 1.0 + strength)
        if self.input_dim >= 3:
            gate = torch.cat([torch.ones_like(gate[:, :3]), gate[:, 3:]], dim=-1)
        return x * gate.unsqueeze(1)


class ThermodynamicConsistencyGate(nn.Module):
    """Attenuate the existing physics gate when thermal transitions are unreliable."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_conditions: int,
        max_attenuation: float,
    ) -> None:
        super().__init__()
        state_dim = max(int(state_dim), 1)
        hidden_dim = max(int(hidden_dim), 4)
        self.state_dim = state_dim
        self.max_attenuation = min(max(float(max_attenuation), 0.0), 1.0)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.transition_mean = nn.Linear(hidden_dim, state_dim)
        self.transition_logvar = nn.Linear(hidden_dim, state_dim)
        self.condition_attenuation_logit = nn.Parameter(
            torch.full((max(int(num_conditions), 1),), -4.0)
        )
        self.last_transition_loss: torch.Tensor | None = None
        self.last_confidence: torch.Tensor | None = None

    def forward(
        self,
        sensor_x: torch.Tensor,
        gated_x: torch.Tensor,
        thermo_states: torch.Tensor,
        condition_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if thermo_states.size(-1) != self.state_dim:
            raise ValueError("Thermodynamic consistency gate received the wrong state dimension.")
        encoded = self.encoder(thermo_states[:, :-1])
        mean = self.transition_mean(encoded)
        logvar = self.transition_logvar(encoded).clamp(-4.0, 2.0)
        target = thermo_states[:, 1:]
        standardized_error = (target - mean).pow(2) * torch.exp(-logvar)
        self.last_transition_loss = 0.5 * (standardized_error + logvar).mean()
        confidence_next = torch.exp(-0.5 * standardized_error.mean(dim=-1)).clamp(0.05, 1.0)
        confidence = torch.cat(
            [torch.ones_like(confidence_next[:, :1]), confidence_next], dim=1
        )
        self.last_confidence = confidence.detach()

        if condition_ids is None:
            condition_ids = torch.zeros(sensor_x.size(0), dtype=torch.long, device=sensor_x.device)
        condition_ids = condition_ids.long().clamp(0, self.condition_attenuation_logit.numel() - 1)
        attenuation = self.max_attenuation * torch.sigmoid(
            self.condition_attenuation_logit[condition_ids]
        )
        blend = 1.0 - attenuation[:, None, None] * (1.0 - confidence.detach().unsqueeze(-1))
        return sensor_x + blend * (gated_x - sensor_x)

    def condition_attenuations(self) -> torch.Tensor:
        return self.max_attenuation * torch.sigmoid(self.condition_attenuation_logit)


class MinimalPhysicsGraphBranch(nn.Module):
    """One-hop fixed-physics message passing with a zero-start residual."""

    def __init__(
        self, feature_cols: list[str], hidden_dim: int, residual_max: float = 0.10
    ) -> None:
        super().__init__()
        adjacency = build_physics_adjacency(feature_cols)
        degree = adjacency.sum(dim=-1).clamp_min(1.0)
        degree_inv_sqrt = degree.rsqrt()
        normalized = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
        self.register_buffer("normalized_adjacency", normalized)
        self.projection = nn.Sequential(
            nn.Linear(len(feature_cols) * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.residual_logit = nn.Parameter(torch.zeros(()))
        self.residual_max = max(float(residual_max), 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self.normalized_adjacency.to(device=x.device, dtype=x.dtype)
        neighbor_features = torch.matmul(x, adjacency)
        return self.projection(torch.cat([x, neighbor_features], dim=-1))

    def residual_scale(self) -> torch.Tensor:
        return self.residual_max * torch.tanh(self.residual_logit)


class ThermodynamicStateResidual(nn.Module):
    """Encode gas-path state relations into a bounded sensor-space residual."""

    def __init__(
        self,
        feature_cols: list[str],
        state_dim: int = 16,
        residual_max: float = 0.10,
        raw_state_dim: int = 0,
        condition_confidence_adaptive: bool = False,
        num_conditions: int = 6,
        confidence_floor: float = 0.25,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        names = {name: idx for idx, name in enumerate(feature_cols)}
        relations = [
            ("s3", "s2"),   # Compressor temperature rise: T30 - T24.
            ("s3", "s4"),   # Turbine temperature drop: T30 - T50.
            ("s7", "s11"),  # Total/static pressure relationship near HPC exit.
            ("s9", "s8"),   # Core/fan shaft-speed relationship.
            ("s14", "s13"), # Corrected core/fan speed relationship.
            ("s20", "s21"), # Cooling-flow balance.
        ]
        active = [(names[left], names[right]) for left, right in relations if left in names and right in names]
        self.raw_state_dim = max(int(raw_state_dim), 0)
        if self.raw_state_dim == 0 and not active:
            raise ValueError("No supported thermodynamic sensor relations are present in feature_cols.")
        transform = torch.zeros(len(feature_cols), len(active), dtype=torch.float32)
        for column, (left, right) in enumerate(active):
            transform[left, column] = 1.0
            transform[right, column] = -1.0
        residual_output_dim = len(feature_cols) if output_dim is None else max(int(output_dim), 1)
        sensor_mask = torch.ones(residual_output_dim, dtype=torch.float32)
        if residual_output_dim == len(feature_cols):
            sensor_mask[: min(3, len(feature_cols))] = 0.0
        self.register_buffer("proxy_transform", transform)
        self.register_buffer("sensor_mask", sensor_mask)
        state_dim = max(int(state_dim), 4)
        encoder_input_dim = self.raw_state_dim if self.raw_state_dim > 0 else len(active)
        self.state_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.Tanh(),
        )
        self.residual_projection = nn.Linear(state_dim, residual_output_dim)
        self.health_head = nn.Sequential(nn.Linear(state_dim, 1), nn.Sigmoid())
        self.residual_logit = nn.Parameter(torch.zeros(()))
        self.residual_max = max(float(residual_max), 0.0)
        self.condition_confidence_adaptive = bool(condition_confidence_adaptive)
        self.confidence_floor = min(max(float(confidence_floor), 0.0), 1.0)
        if self.condition_confidence_adaptive:
            self.condition_residual_logit = nn.Embedding(max(int(num_conditions), 1), 1)
            nn.init.zeros_(self.condition_residual_logit.weight)
            self.confidence_head = nn.Linear(state_dim, 1)
            nn.init.zeros_(self.confidence_head.weight)
            nn.init.constant_(self.confidence_head.bias, 1.3862944)
        self.last_health: torch.Tensor | None = None
        self.last_confidence: torch.Tensor | None = None
        self.last_scale: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        raw_states: torch.Tensor | None = None,
        condition_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.raw_state_dim > 0:
            if raw_states is None or raw_states.size(-1) != self.raw_state_dim:
                raise ValueError("TSRM raw-state input is missing or has the wrong dimension.")
            proxies = raw_states
        else:
            transform = self.proxy_transform.to(device=x.device, dtype=x.dtype)
            proxies = torch.matmul(x, transform)
        state = self.state_encoder(proxies)
        self.last_health = self.health_head(state).squeeze(-1)
        residual = torch.tanh(self.residual_projection(state))
        residual = residual * self.sensor_mask.to(device=x.device, dtype=x.dtype)
        if self.condition_confidence_adaptive:
            if condition_ids is None:
                condition_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            condition_ids = condition_ids.long().clamp(0, self.condition_residual_logit.num_embeddings - 1)
            condition_logit = self.condition_residual_logit(condition_ids).view(-1, 1, 1)
            confidence = self.confidence_floor + (1.0 - self.confidence_floor) * torch.sigmoid(
                self.confidence_head(state)
            )
            scale = self.residual_max * torch.tanh(self.residual_logit + condition_logit)
            scale = scale * confidence
            self.last_confidence = confidence.squeeze(-1)
            self.last_scale = scale.squeeze(-1)
        else:
            scale = self.residual_max * torch.tanh(self.residual_logit)
            self.last_confidence = None
            self.last_scale = scale.reshape(1)
        return scale * residual


class ConditionBRDRLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_cols: list[str],
        hidden_dim: int = 64,
        num_layers: int = 1,
        condition_lstm_init: bool = False,
        condition_lstm_bounded_residual: bool = False,
        condition_lstm_residual_max: float = 0.10,
        condition_output_residual: bool = False,
        condition_output_residual_max: float = 0.10,
        condition_dropout: float = 0.20,
        condition_rul_bias: bool = False,
        condition_rul_bias_max: float = 0.05,
        num_conditions: int = 6,
        physics_sensor_gate: bool = False,
        physics_sensor_gate_lstm_input: bool = False,
        sensor_gate_direction_scores: torch.Tensor | None = None,
        sensor_gate_strength: float = 0.10,
        sensor_gate_warmup_epochs: int = 5,
        sensor_gate_condition_adaptive: bool = False,
        sensor_gate_max_strength: float = 0.20,
        sensor_gate_control: str = "physics",
        physics_graph_branch: bool = False,
        physics_graph_residual_max: float = 0.10,
        thermodynamic_state_residual: bool = False,
        thermo_state_dim: int = 16,
        thermo_residual_max: float = 0.10,
        thermo_raw_state_dim: int = 0,
        thermo_condition_confidence_adaptive: bool = False,
        thermo_confidence_floor: float = 0.25,
        thermo_fusion_mode: str = "input",
        thermo_consistency_gate: bool = False,
        thermo_gate_state_dim: int = 8,
        thermo_gate_max_attenuation: float = 0.50,
        shared_private_latent: bool = False,
        condition_invariant_latent: bool = False,
        condition_normalized_ranking: bool = False,
        fault_prototypes: int = 2,
        fault_prototype_temperature: float = 0.2,
        fault_mixture_experts: bool = False,
        fault_mixture_expert_count: int = 2,
        fault_mixture_temperature: float = 0.5,
        fault_mode_conditional_head: bool = False,
        fault_mode_conditional_weight: float = 0.01,
        unified_health_state: bool = False,
        health_state_mode: str = "off",
        multi_scale_temporal: bool = False,
        multi_scale_residual_max: float = 0.20,
        future_trajectory_mode: bool = False,
        future_trajectory_mode_residual_max: float = 0.05,
        condition_observation_residual: bool = False,
        condition_observation_residual_max: float = 0.10,
        dynamic_condition_modulation: bool = False,
        dynamic_condition_modulation_max: float = 0.10,
        fault_mode_suppressed_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.thermo_raw_state_dim = max(int(thermo_raw_state_dim), 0)
        self.thermo_fusion_mode = thermo_fusion_mode
        rng_state = torch.random.get_rng_state()
        self.thermo_consistency_gate = (
            ThermodynamicConsistencyGate(
                self.thermo_raw_state_dim,
                thermo_gate_state_dim,
                num_conditions,
                thermo_gate_max_attenuation,
            )
            if thermo_consistency_gate else None
        )
        # Optional modules must not change the frozen baseline's initialization.
        torch.random.set_rng_state(rng_state)
        self.condition_lstm_init = condition_lstm_init
        self.condition_lstm_bounded_residual = condition_lstm_bounded_residual
        self.condition_lstm_residual_max = max(float(condition_lstm_residual_max), 0.0)
        self.condition_output_residual = condition_output_residual
        self.condition_output_residual_max = max(float(condition_output_residual_max), 0.0)
        self.condition_dropout = min(max(float(condition_dropout), 0.0), 1.0)
        self.num_conditions = max(int(num_conditions), 1)
        self.fault_mode_conditional_head = bool(fault_mode_conditional_head)
        self.fault_mode_conditional_weight = max(float(fault_mode_conditional_weight), 0.0)
        self.fault_mixture_experts = bool(fault_mixture_experts or fault_mode_conditional_head)
        self.fault_mixture_expert_count = max(int(fault_mixture_expert_count), 2)
        self.fault_mixture_temperature = max(float(fault_mixture_temperature), 1e-4)
        self.last_mixture_probs: torch.Tensor | None = None
        self.multi_scale_temporal = bool(multi_scale_temporal)
        self.multi_scale_residual_max = max(float(multi_scale_residual_max), 0.0)
        self.future_trajectory_mode = bool(future_trajectory_mode)
        self.future_trajectory_mode_residual_max = max(float(future_trajectory_mode_residual_max), 0.0)
        self.condition_observation_residual = bool(condition_observation_residual)
        self.condition_observation_residual_max = max(float(condition_observation_residual_max), 0.0)
        self.dynamic_condition_modulation = bool(dynamic_condition_modulation)
        self.dynamic_condition_modulation_max = max(float(dynamic_condition_modulation_max), 0.0)
        self.fault_mode_suppressed_modulation = bool(fault_mode_suppressed_modulation)
        self.condition_rul_bias = condition_rul_bias
        self.condition_rul_bias_max = max(float(condition_rul_bias_max), 0.0)
        self.condition_invariant_latent = bool(condition_invariant_latent)
        self.last_condition_logits_shared: torch.Tensor | None = None
        self.last_condition_logits_private: torch.Tensor | None = None
        self.condition_normalized_ranking = bool(condition_normalized_ranking)
        self.last_degradation_scores: torch.Tensor | None = None
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        if self.dynamic_condition_modulation:
            modulation_hidden = max(16, hidden_dim // 2)
            self.dynamic_modulation = nn.Sequential(
                nn.Linear(input_dim, modulation_hidden), nn.Tanh(), nn.Linear(modulation_hidden, hidden_dim * 2)
            )
            nn.init.zeros_(self.dynamic_modulation[-1].weight)
            nn.init.zeros_(self.dynamic_modulation[-1].bias)
            if self.fault_mode_suppressed_modulation:
                self.modulation_mode_head = nn.Linear(hidden_dim, 2)
        if self.condition_observation_residual:
            sensor_dim = max(input_dim - 3, 1)
            self.condition_observation_embedding = nn.Embedding(self.num_conditions, 8)
            self.condition_observation_projection = nn.Linear(8, sensor_dim)
            nn.init.zeros_(self.condition_observation_projection.weight)
            nn.init.zeros_(self.condition_observation_projection.bias)
        if self.multi_scale_temporal:
            branch_dim = max(8, hidden_dim // 2)
            self.multi_scale_conv_short = nn.Conv1d(
                hidden_dim, branch_dim, kernel_size=3, padding=1
            )
            self.multi_scale_conv_medium = nn.Conv1d(
                hidden_dim, branch_dim, kernel_size=7, padding=3
            )
            self.multi_scale_projection = nn.Sequential(
                nn.Linear(branch_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
            )
        if condition_output_residual:
            self.output_condition_embedding = nn.Embedding(self.num_conditions, 8)
            self.output_condition_projection = nn.Linear(8, hidden_dim)
            nn.init.zeros_(self.output_condition_projection.weight)
            nn.init.zeros_(self.output_condition_projection.bias)
        if condition_rul_bias:
            self.rul_bias_condition_embedding = nn.Embedding(self.num_conditions, 8)
            self.rul_bias_projection = nn.Linear(8, 1)
            nn.init.zeros_(self.rul_bias_projection.weight)
            nn.init.zeros_(self.rul_bias_projection.bias)
        if condition_lstm_init:
            self.lstm_condition_embedding = nn.Embedding(max(int(num_conditions), 1), 8)
            self.lstm_h0 = nn.Linear(8, num_layers * hidden_dim)
            self.lstm_c0 = nn.Linear(8, num_layers * hidden_dim)
            nn.init.zeros_(self.lstm_h0.weight)
            nn.init.zeros_(self.lstm_h0.bias)
            nn.init.zeros_(self.lstm_c0.weight)
            nn.init.zeros_(self.lstm_c0.bias)
        self.physics_sensor_gate_lstm_input = physics_sensor_gate_lstm_input
        gate_rng_state = torch.random.get_rng_state()
        gate_type = LearnedSensorReliabilityGate if sensor_gate_control == "learned" else PhysicsAwareSensorReliabilityGate
        self.sensor_reliability_gate = (
            gate_type(
                feature_cols,
                sensor_gate_direction_scores
                if sensor_gate_direction_scores is not None
                else torch.zeros(input_dim),
                sensor_gate_strength,
                sensor_gate_warmup_epochs,
                sensor_gate_condition_adaptive,
                sensor_gate_max_strength,
                num_conditions,
                **({"control": sensor_gate_control} if sensor_gate_control == "shuffled" else {}),
            )
            if physics_sensor_gate else None
        )
        torch.random.set_rng_state(gate_rng_state)
        self.physics_graph_branch = (
            MinimalPhysicsGraphBranch(feature_cols, hidden_dim, physics_graph_residual_max)
            if physics_graph_branch else None
        )
        rng_state = torch.random.get_rng_state()
        self.thermodynamic_state_residual = (
            ThermodynamicStateResidual(
                feature_cols,
                thermo_state_dim,
                thermo_residual_max,
                self.thermo_raw_state_dim,
                thermo_condition_confidence_adaptive,
                num_conditions,
                thermo_confidence_floor,
                hidden_dim if thermo_fusion_mode == "latent" else input_dim,
            )
            if thermodynamic_state_residual else None
        )
        # Keep the downstream prediction head identical to the frozen baseline.
        torch.random.set_rng_state(rng_state)
        self.last_thermo_health: torch.Tensor | None = None
        self.shared_private_latent = shared_private_latent
        fused_dim = hidden_dim
        rank_dim = fused_dim
        if self.shared_private_latent:
            shared_dim = max(8, fused_dim // 2)
            rank_dim = shared_dim
            self.shared_projection = nn.Sequential(nn.Linear(fused_dim, shared_dim), nn.LayerNorm(shared_dim), nn.Tanh())
            private_dim = fused_dim - shared_dim
            self.private_projection = nn.Sequential(nn.Linear(fused_dim, private_dim), nn.LayerNorm(private_dim), nn.Tanh())
            self.fault_prototypes = nn.Parameter(torch.randn(fault_prototypes, private_dim) * 0.02)
            self.fault_prototype_temperature = fault_prototype_temperature
            if self.condition_invariant_latent:
                self.shared_condition_classifier = nn.Sequential(
                    nn.Linear(shared_dim, max(8, shared_dim // 2)), nn.ReLU(), nn.Linear(max(8, shared_dim // 2), self.num_conditions)
                )
                self.private_condition_classifier = nn.Sequential(
                    nn.Linear(private_dim, max(8, private_dim // 2)), nn.ReLU(), nn.Linear(max(8, private_dim // 2), self.num_conditions)
                )
            if self.future_trajectory_mode:
                adapter_hidden = max(8, fused_dim // 4)
                self.future_mode_adapters = nn.ModuleList(
                    [nn.Sequential(nn.Linear(fused_dim, adapter_hidden), nn.Tanh(), nn.Linear(adapter_hidden, fused_dim))
                     for _ in range(max(2, fault_prototypes))]
                )
        if self.condition_normalized_ranking:
            self.degradation_rank_head = nn.Sequential(
                nn.Linear(rank_dim, max(8, rank_dim // 2)), nn.ReLU(), nn.Linear(max(8, rank_dim // 2), 1)
            )
        if self.fault_mixture_experts:
            router_hidden = max(16, fused_dim // 2)
            self.fault_router = nn.Sequential(
                nn.Linear(fused_dim, router_hidden), nn.Tanh(),
                nn.Linear(router_hidden, self.fault_mixture_expert_count)
            )
            self.fault_expert_heads = nn.ModuleList(
                [nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, 1))
                 for _ in range(self.fault_mixture_expert_count)]
            )
        self.health_state_mode = health_state_mode if health_state_mode != "off" else ("aux" if unified_health_state else "off")
        self.unified_health_state = self.health_state_mode in {"aux", "residual", "condition_residual"}
        condition_aware = self.health_state_mode in {"condition_base", "condition_residual"}
        if self.health_state_mode == "condition_base":
            self.condition_embedding = nn.Embedding(6, 8)
            self.head = nn.Sequential(nn.Linear(fused_dim + 8, 64), nn.ReLU(), nn.Linear(64, 1))
        if self.unified_health_state:
            state_input_dim = fused_dim
            if condition_aware:
                self.condition_embedding = nn.Embedding(6, 8)
                # Stage labels are derived from true RUL and are unavailable at inference.
                state_input_dim += 8
            self.state_encoder = nn.Sequential(
                nn.Linear(state_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
            )
            self.health_head = nn.Sequential(
                nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
            )
            self.residual_head = nn.Sequential(nn.Linear(hidden_dim + 1, 32), nn.ReLU(), nn.Linear(32, 1))
            self.residual_gate = nn.Parameter(torch.tensor(-4.0))
            self.base_head = nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        elif self.health_state_mode == "off":
            self.head = nn.Sequential(
                nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, 1)
            )

    def forward(
        self,
        x: torch.Tensor,
        condition_ids: torch.Tensor | None = None,
        stage_ids: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[object, torch.Tensor]:
        sensor_x = x[:, :, : self.input_dim]
        raw_thermo_states = (
            x[:, :, self.input_dim : self.input_dim + self.thermo_raw_state_dim]
            if self.thermo_raw_state_dim > 0 else None
        )
        lstm_input = sensor_x
        if self.condition_observation_residual:
            if condition_ids is None:
                condition_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            condition_context = self.condition_observation_embedding(
                condition_ids.long().clamp(0, self.num_conditions - 1)
            )
            sensor_shift = self.condition_observation_residual_max * torch.tanh(
                self.condition_observation_projection(condition_context)
            )
            lstm_input = sensor_x.clone()
            lstm_input[:, :, 3:] = lstm_input[:, :, 3:] - sensor_shift.unsqueeze(1)
        if self.physics_sensor_gate_lstm_input and self.sensor_reliability_gate is not None:
            lstm_input = self.sensor_reliability_gate(sensor_x, condition_ids)
            if self.thermo_consistency_gate is not None:
                if raw_thermo_states is None:
                    raise ValueError("Thermodynamic consistency gating requires raw thermal states.")
                lstm_input = self.thermo_consistency_gate(
                    sensor_x, lstm_input, raw_thermo_states, condition_ids
                )
        if self.thermodynamic_state_residual is not None and self.thermo_fusion_mode == "input":
            lstm_input = lstm_input + self.thermodynamic_state_residual(
                sensor_x, raw_thermo_states, condition_ids
            )
            self.last_thermo_health = self.thermodynamic_state_residual.last_health
        elif self.thermodynamic_state_residual is None:
            self.last_thermo_health = None
        if self.condition_lstm_init:
            if condition_ids is None:
                condition_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            condition_ids = condition_ids.long().clamp(0, self.lstm_condition_embedding.num_embeddings - 1)
            condition_context = self.lstm_condition_embedding(condition_ids)
            h0 = self.lstm_h0(condition_context).view(
                self.num_layers, x.size(0), self.hidden_dim
            )
            c0 = self.lstm_c0(condition_context).view(
                self.num_layers, x.size(0), self.hidden_dim
            )
            if self.condition_lstm_bounded_residual:
                h0 = self.condition_lstm_residual_max * torch.tanh(h0)
                c0 = self.condition_lstm_residual_max * torch.tanh(c0)
            sequence_features, _ = self.temporal_encoder(lstm_input, (h0, c0))
        else:
            sequence_features, _ = self.temporal_encoder(lstm_input)
        if self.thermodynamic_state_residual is not None and self.thermo_fusion_mode == "latent":
            sequence_features = sequence_features + self.thermodynamic_state_residual(
                sensor_x, raw_thermo_states, condition_ids
            )
            self.last_thermo_health = self.thermodynamic_state_residual.last_health
        modulation_confidence = None
        self.last_modulation_mode_probs = None
        if self.dynamic_condition_modulation and self.fault_mode_suppressed_modulation:
            mode_probs = torch.softmax(self.modulation_mode_head(sequence_features), dim=-1)
            entropy = -(mode_probs * torch.log(mode_probs.clamp_min(1e-6))).sum(dim=-1)
            modulation_confidence = (1.0 - entropy / np.log(2.0)).clamp(0.0, 1.0)
            self.last_modulation_mode_probs = mode_probs[:, -1]
        if self.dynamic_condition_modulation:
            modulation = self.dynamic_modulation(sensor_x)
            gamma, beta = modulation.chunk(2, dim=-1)
            scale = self.dynamic_condition_modulation_max
            if modulation_confidence is not None:
                gamma = gamma * modulation_confidence.unsqueeze(-1)
                beta = beta * modulation_confidence.unsqueeze(-1)
            sequence_features = sequence_features * (1.0 + scale * torch.tanh(gamma)) \
                + scale * torch.tanh(beta)
        fused = sequence_features
        if self.physics_graph_branch is not None:
            graph_features = self.physics_graph_branch(sensor_x)
            fused = fused + self.physics_graph_branch.residual_scale() * graph_features
        if self.multi_scale_temporal:
            conv_input = sequence_features.transpose(1, 2)
            short_features = torch.relu(self.multi_scale_conv_short(conv_input)).transpose(1, 2)
            medium_features = torch.relu(self.multi_scale_conv_medium(conv_input)).transpose(1, 2)
            multi_scale_residual = self.multi_scale_projection(
                torch.cat([short_features, medium_features], dim=-1)
            )
            fused = fused + self.multi_scale_residual_max * multi_scale_residual
        if self.condition_output_residual:
            if condition_ids is None:
                condition_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            condition_ids = condition_ids.long().clamp(0, self.output_condition_embedding.num_embeddings - 1)
            condition_context = self.output_condition_embedding(condition_ids)
            if self.training and self.condition_dropout > 0:
                keep = (torch.rand(condition_context.size(0), 1, device=x.device) >= self.condition_dropout).to(
                    condition_context.dtype
                )
                condition_context = condition_context * keep
            output_residual = self.condition_output_residual_max * torch.tanh(
                self.output_condition_projection(condition_context)
            )
            fused = fused + output_residual.unsqueeze(1)
        alignment_latent = fused
        prediction_features = fused
        private_latent = None
        fault_probs = None
        self.last_condition_logits_shared = None
        self.last_condition_logits_private = None
        self.last_degradation_scores = None
        if self.shared_private_latent:
            shared_latent = self.shared_projection(fused)
            private_latent = self.private_projection(fused)
            alignment_latent = shared_latent
            prediction_features = torch.cat([shared_latent, private_latent], dim=-1)
            if self.condition_invariant_latent:
                shared_endpoint = shared_latent[:, -1]
                private_endpoint = private_latent[:, -1]
                self.last_condition_logits_shared = self.shared_condition_classifier(
                    gradient_reverse(shared_endpoint, 1.0)
                )
                self.last_condition_logits_private = self.private_condition_classifier(private_endpoint)
            private_sequence = private_latent / private_latent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            prototypes = self.fault_prototypes / self.fault_prototypes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            logits = private_sequence.matmul(prototypes.transpose(0, 1)) / max(self.fault_prototype_temperature, 1e-4)
            mode_probs_sequence = torch.softmax(logits, dim=-1)
            fault_probs = mode_probs_sequence[:, -1]
            if self.future_trajectory_mode:
                adapter_outputs = torch.stack(
                    [adapter(fused) for adapter in self.future_mode_adapters], dim=-1
                )
                mode_residual = (adapter_outputs * mode_probs_sequence.unsqueeze(2)).sum(dim=-1)
                fused = fused + self.future_trajectory_mode_residual_max * torch.tanh(mode_residual)
                prediction_features = torch.cat(
                    [self.shared_projection(fused), self.private_projection(fused)], dim=-1
                )
        if self.condition_normalized_ranking:
            self.last_degradation_scores = self.degradation_rank_head(alignment_latent[:, -1]).squeeze(-1)

        if self.health_state_mode == "condition_base":
            if condition_ids is None:
                condition_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            condition_context = self.condition_embedding(condition_ids.long()).unsqueeze(1).expand(-1, x.size(1), -1)
            conditioned_features = torch.cat([prediction_features, condition_context], dim=-1)
            output = self.head(conditioned_features).squeeze(-1)
            latent_output = (alignment_latent, private_latent, fault_probs) if self.shared_private_latent else alignment_latent
            return (output, latent_output) if return_latent else output
        if self.unified_health_state:
            state_input = prediction_features
            if self.health_state_mode == "condition_residual":
                if condition_ids is None:
                    condition_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                condition_context = self.condition_embedding(condition_ids.long()).unsqueeze(1).expand(-1, x.size(1), -1)
                state_input = torch.cat([fused, condition_context], dim=-1)
            state = self.state_encoder(state_input)
            health_seq = self.health_head(state).squeeze(-1)
            base_pred = self.base_head(prediction_features).squeeze(-1)
            if self.health_state_mode == "aux":
                pred_seq = base_pred
            else:
                residual = self.residual_head(torch.cat([state, health_seq.unsqueeze(-1)], dim=-1)).squeeze(-1)
                pred_seq = base_pred + torch.sigmoid(self.residual_gate) * residual
            output = (pred_seq, health_seq)
            latent_output = (alignment_latent, private_latent, fault_probs) if self.shared_private_latent else alignment_latent
            return (output, latent_output) if return_latent else output
        if self.fault_mixture_experts:
            if (
                self.fault_mode_conditional_head
                and fault_probs is not None
                and fault_probs.size(-1) == self.fault_mixture_expert_count
            ):
                # The private latent identifies the mode; the same soft mode
                # posterior directly selects the corresponding RUL expert.
                router_probs = torch.softmax(
                    torch.log(fault_probs.clamp_min(1e-6)) / self.fault_mixture_temperature,
                    dim=-1,
                )
            else:
                router_input = prediction_features[:, -1]
                router_probs = torch.softmax(
                    self.fault_router(router_input) / self.fault_mixture_temperature, dim=-1
                )
            expert_outputs = torch.stack(
                [head(prediction_features).squeeze(-1) for head in self.fault_expert_heads], dim=-1
            )
            output = (expert_outputs * router_probs.unsqueeze(1)).sum(dim=-1)
            self.last_mixture_probs = router_probs
        else:
            self.last_mixture_probs = None
            output = self.head(prediction_features).squeeze(-1)
        if self.condition_rul_bias:
            if condition_ids is None:
                condition_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            condition_ids = condition_ids.long().clamp(0, self.rul_bias_condition_embedding.num_embeddings - 1)
            condition_context = self.rul_bias_condition_embedding(condition_ids)
            rul_bias = self.condition_rul_bias_max * torch.tanh(self.rul_bias_projection(condition_context)).squeeze(-1)
            output = output.clone()
            output[:, -1] = output[:, -1] + rul_bias
        latent_output = (alignment_latent, private_latent, fault_probs) if self.shared_private_latent else alignment_latent
        return (output, latent_output) if return_latent else output

    def condition_output_center_loss(self) -> torch.Tensor:
        if not self.condition_output_residual:
            return next(self.parameters()).new_zeros(())
        condition_ids = torch.arange(
            self.output_condition_embedding.num_embeddings,
            device=self.output_condition_embedding.weight.device,
        )
        condition_context = self.output_condition_embedding(condition_ids)
        residuals = self.condition_output_residual_max * torch.tanh(
            self.output_condition_projection(condition_context)
        )
        return residuals.mean(dim=0).pow(2).mean()

    def condition_rul_bias_center_loss(self) -> torch.Tensor:
        if not self.condition_rul_bias:
            return next(self.parameters()).new_zeros(())
        condition_ids = torch.arange(
            self.rul_bias_condition_embedding.num_embeddings,
            device=self.rul_bias_condition_embedding.weight.device,
        )
        condition_context = self.rul_bias_condition_embedding(condition_ids)
        biases = self.condition_rul_bias_max * torch.tanh(self.rul_bias_projection(condition_context)).squeeze(-1)
        return biases.mean().pow(2)


def predict(model, loader, device, pin_memory):
    model.eval()
    preds, trues, conditions, stages = [], [], [], []
    with torch.no_grad():
        for xb, yb, cb, sb, _, _ in loader:
            xb = xb.to(device, non_blocking=pin_memory)
            cb_device = cb.to(device, non_blocking=pin_memory)
            sb_device = sb.to(device, non_blocking=pin_memory)
            output = model(xb, cb_device, sb_device)
            pred_seq = output[0] if isinstance(output, tuple) else output
            preds.extend(pred_seq[:, -1].cpu().numpy())
            trues.extend(yb.numpy()[:, -1])
            conditions.extend(cb.numpy())
            stages.extend(sb.numpy())
    return (
        np.asarray(trues, dtype=np.float32),
        np.maximum(np.asarray(preds), 0.0).astype(np.float32),
        np.asarray(conditions, dtype=np.int64),
        np.asarray(stages, dtype=np.int64),
    )


def predicted_stage_ids(
    predictions: np.ndarray, middle_rul: float, late_rul: float
) -> np.ndarray:
    return assign_stage_ids(np.asarray(predictions, dtype=np.float32), middle_rul, late_rul)


def fit_stage_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    middle_rul: float,
    late_rul: float,
    shrinkage: float,
    min_samples: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit validation-only affine calibration using stages inferred from predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    stages = predicted_stage_ids(y_pred, middle_rul, late_rul)
    global_x = np.column_stack([y_pred, np.ones_like(y_pred)])
    identity_prior = np.asarray([1.0, 0.0], dtype=np.float64)
    coeffs = np.zeros((3, 2), dtype=np.float64)
    counts = []
    for stage in range(3):
        mask = stages == stage
        counts.append(int(mask.sum()))
        x = global_x[mask] if mask.sum() >= min_samples else global_x
        y = y_true[mask] if mask.sum() >= min_samples else y_true
        normal = x.T @ x + float(shrinkage) * np.eye(2)
        rhs = x.T @ y + float(shrinkage) * identity_prior
        coeffs[stage] = np.linalg.solve(normal, rhs)
    metadata = {"coefficients": coeffs.tolist(), "validation_predicted_stage_counts": counts}
    return coeffs, metadata


def apply_stage_calibrator(
    predictions: np.ndarray,
    coeffs: np.ndarray,
    middle_rul: float,
    late_rul: float,
    max_rul: float,
) -> tuple[np.ndarray, np.ndarray]:
    stages = predicted_stage_ids(predictions, middle_rul, late_rul)
    calibrated = np.asarray(predictions, dtype=np.float64).copy()
    for stage in range(3):
        mask = stages == stage
        calibrated[mask] = coeffs[stage, 0] * calibrated[mask] + coeffs[stage, 1]
    return np.clip(calibrated, 0.0, float(max_rul)).astype(np.float32), stages


def _condition_stage_design(
    predictions: np.ndarray,
    condition_ids: np.ndarray,
    middle_rul: float,
    late_rul: float,
    num_conditions: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float64)
    conditions = np.asarray(condition_ids, dtype=np.int64)
    stages = predicted_stage_ids(predictions, middle_rul, late_rul)
    columns = [predictions, np.ones_like(predictions)]
    # Reference condition/stage levels keep the design identifiable.
    for condition in range(1, num_conditions):
        columns.append((conditions == condition).astype(np.float64))
    for stage in range(1, 3):
        columns.append((stages == stage).astype(np.float64))
    for condition in range(1, num_conditions):
        for stage in range(1, 3):
            columns.append(((conditions == condition) & (stages == stage)).astype(np.float64))
    return np.column_stack(columns), stages


def fit_condition_stage_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    condition_ids: np.ndarray,
    middle_rul: float,
    late_rul: float,
    num_conditions: int,
    shrinkage: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit validation-only hierarchical condition-stage affine calibration."""
    x, stages = _condition_stage_design(
        y_pred, condition_ids, middle_rul, late_rul, num_conditions
    )
    y_true = np.asarray(y_true, dtype=np.float64)
    prior = np.zeros(x.shape[1], dtype=np.float64)
    prior[0] = 1.0
    regularizer = np.eye(x.shape[1], dtype=np.float64) * float(shrinkage)
    # Preserve the global identity mapping while shrinking all bias terms.
    normal = x.T @ x + regularizer
    rhs = x.T @ y_true + regularizer @ prior
    coefficients = np.linalg.solve(normal, rhs)
    counts = np.zeros((num_conditions, 3), dtype=int)
    for condition in range(num_conditions):
        for stage in range(3):
            counts[condition, stage] = int(((np.asarray(condition_ids) == condition) & (stages == stage)).sum())
    return coefficients, {
        "coefficients": coefficients.tolist(),
        "validation_predicted_condition_stage_counts": counts.tolist(),
    }


def apply_condition_stage_calibrator(
    predictions: np.ndarray,
    condition_ids: np.ndarray,
    coefficients: np.ndarray,
    middle_rul: float,
    late_rul: float,
    num_conditions: int,
    max_rul: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, stages = _condition_stage_design(
        predictions, condition_ids, middle_rul, late_rul, num_conditions
    )
    calibrated = x @ np.asarray(coefficients, dtype=np.float64)
    return np.clip(calibrated, 0.0, float(max_rul)).astype(np.float32), stages


def _soft_condition_stage_design(
    predictions: np.ndarray,
    condition_ids: np.ndarray,
    middle_rul: float,
    late_rul: float,
    max_rul: float,
    num_conditions: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float64)
    conditions = np.asarray(condition_ids, dtype=np.int64)
    hard_stages = predicted_stage_ids(predictions, middle_rul, late_rul)
    centers = np.asarray(
        [float(max_rul), (float(middle_rul) + float(late_rul)) / 2.0, float(late_rul) / 2.0],
        dtype=np.float64,
    )
    logits = -((predictions[:, None] - centers[None, :]) / max(float(temperature), 1e-3)) ** 2
    stage_probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    stage_probs /= stage_probs.sum(axis=1, keepdims=True).clip(min=1e-8)
    columns = [np.ones_like(predictions)]
    for condition in range(1, num_conditions):
        columns.append((conditions == condition).astype(np.float64))
    columns.extend([stage_probs[:, 1], stage_probs[:, 2]])
    for condition in range(1, num_conditions):
        for stage in (1, 2):
            columns.append((conditions == condition).astype(np.float64) * stage_probs[:, stage])
    return np.column_stack(columns), hard_stages


def fit_soft_condition_stage_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    condition_ids: np.ndarray,
    middle_rul: float,
    late_rul: float,
    max_rul: float,
    num_conditions: int,
    temperature: float,
    shrinkage: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a smooth, bounded residual calibrator on validation predictions."""
    x, stages = _soft_condition_stage_design(
        y_pred, condition_ids, middle_rul, late_rul, max_rul, num_conditions, temperature
    )
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    normal = x.T @ x + float(shrinkage) * np.eye(x.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(normal, x.T @ residual)
    counts = np.zeros((num_conditions, 3), dtype=int)
    for condition in range(num_conditions):
        for stage in range(3):
            counts[condition, stage] = int(
                ((np.asarray(condition_ids) == condition) & (stages == stage)).sum()
            )
    return coefficients, {
        "coefficients": coefficients.tolist(),
        "temperature": float(temperature),
        "validation_predicted_condition_stage_counts": counts.tolist(),
    }


def apply_soft_condition_stage_calibrator(
    predictions: np.ndarray,
    condition_ids: np.ndarray,
    coefficients: np.ndarray,
    middle_rul: float,
    late_rul: float,
    max_rul: float,
    num_conditions: int,
    temperature: float,
    correction_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, stages = _soft_condition_stage_design(
        predictions, condition_ids, middle_rul, late_rul, max_rul, num_conditions, temperature
    )
    correction = np.clip(x @ np.asarray(coefficients, dtype=np.float64), -float(correction_max), float(correction_max))
    calibrated = np.asarray(predictions, dtype=np.float64) + correction
    return np.clip(calibrated, 0.0, float(max_rul)).astype(np.float32), stages


def _soft_stage_probabilities(
    predictions: np.ndarray,
    middle_rul: float,
    late_rul: float,
    max_rul: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float64)
    hard_stages = predicted_stage_ids(predictions, middle_rul, late_rul)
    centers = np.asarray(
        [float(max_rul), (float(middle_rul) + float(late_rul)) / 2.0, float(late_rul) / 2.0],
        dtype=np.float64,
    )
    logits = -((predictions[:, None] - centers[None, :]) / max(float(temperature), 1e-3)) ** 2
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-8)
    return probabilities, hard_stages


def fit_safe_soft_condition_stage_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    condition_ids: np.ndarray,
    raw_condition_profiles: np.ndarray,
    middle_rul: float,
    late_rul: float,
    max_rul: float,
    stage_temperature: float,
) -> dict[str, object]:
    """Store validation residuals for conservative local condition-stage calibration."""
    stage_probs, stages = _soft_stage_probabilities(
        y_pred, middle_rul, late_rul, max_rul, stage_temperature
    )
    conditions = np.asarray(condition_ids, dtype=np.int64)
    profiles = np.asarray(raw_condition_profiles, dtype=np.float64)
    sample_profiles = profiles[np.clip(conditions, 0, profiles.shape[0] - 1)]
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    counts = np.zeros((profiles.shape[0], 3), dtype=int)
    for condition in range(profiles.shape[0]):
        for stage in range(3):
            counts[condition, stage] = int(((conditions == condition) & (stages == stage)).sum())
    return {
        "val_residual": residual,
        "val_stage_probs": stage_probs,
        "val_condition_profiles": sample_profiles,
        "validation_predicted_condition_stage_counts": counts.tolist(),
        "validation_residual_mean": float(residual.mean()),
        "validation_residual_std": float(residual.std()),
    }


def apply_safe_soft_condition_stage_calibrator(
    predictions: np.ndarray,
    condition_ids: np.ndarray,
    calibrator: dict[str, object],
    raw_condition_profiles: np.ndarray,
    middle_rul: float,
    late_rul: float,
    max_rul: float,
    stage_temperature: float,
    condition_temperature: float,
    shrinkage: float,
    correction_max: float,
    variance_scale: float,
    positive_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    stage_probs, stages = _soft_stage_probabilities(
        predictions, middle_rul, late_rul, max_rul, stage_temperature
    )
    profiles = np.asarray(raw_condition_profiles, dtype=np.float64)
    conditions = np.asarray(condition_ids, dtype=np.int64)
    test_profiles = profiles[np.clip(conditions, 0, profiles.shape[0] - 1)]
    val_profiles = np.asarray(calibrator["val_condition_profiles"], dtype=np.float64)
    val_stage_probs = np.asarray(calibrator["val_stage_probs"], dtype=np.float64)
    residual = np.asarray(calibrator["val_residual"], dtype=np.float64)

    condition_scale = max(float(condition_temperature), 1e-3)
    shrink = max(float(shrinkage), 0.0)
    max_correction = max(float(correction_max), 0.0)
    var_scale = max(float(variance_scale), 1e-3)
    positive_scale = min(max(float(positive_scale), 0.0), 1.0)

    corrections = np.zeros(len(predictions), dtype=np.float64)
    effective_counts = np.zeros(len(predictions), dtype=np.float64)
    reliabilities = np.zeros(len(predictions), dtype=np.float64)
    for idx, profile in enumerate(test_profiles):
        dist2 = ((val_profiles - profile[None, :]) ** 2).sum(axis=1)
        condition_weight = np.exp(-dist2 / (2.0 * condition_scale * condition_scale))
        stage_weight = val_stage_probs @ stage_probs[idx]
        weight = condition_weight * stage_weight
        weight_sum = float(weight.sum())
        if weight_sum <= 1e-8:
            continue
        local_mean = float((weight * residual).sum() / weight_sum)
        local_var = float((weight * (residual - local_mean) ** 2).sum() / weight_sum)
        effective_count = float((weight_sum * weight_sum) / np.square(weight).sum().clip(min=1e-8))
        reliability = effective_count / (effective_count + shrink)
        reliability *= 1.0 / (1.0 + local_var / var_scale)
        if local_mean > 0.0:
            negative_residual_weight = float(weight[residual < 0.0].sum() / weight_sum)
            reliability *= (1.0 - negative_residual_weight) + positive_scale * negative_residual_weight
        corrections[idx] = np.clip(local_mean * reliability, -max_correction, max_correction)
        effective_counts[idx] = effective_count
        reliabilities[idx] = reliability

    calibrated = np.asarray(predictions, dtype=np.float64) + corrections
    stats = {
        "mean_correction": float(corrections.mean()),
        "mean_abs_correction": float(np.abs(corrections).mean()),
        "max_abs_correction": float(np.abs(corrections).max()) if corrections.size else 0.0,
        "mean_effective_count": float(effective_counts.mean()),
        "mean_reliability": float(reliabilities.mean()),
    }
    return np.clip(calibrated, 0.0, float(max_rul)).astype(np.float32), stages, stats


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    mode: str,
    num_conditions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = train_df[feature_cols].astype(np.float32)
    test_df[feature_cols] = test_df[feature_cols].astype(np.float32)
    scaler_info: dict[str, dict[str, float]] = {}

    if mode == "global":
        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        scaler_info["global"] = {"train_rows": float(len(train_df)), "test_rows": float(len(test_df))}
        return train_df, test_df, scaler_info

    global_scaler = StandardScaler()
    global_scaler.fit(train_df[feature_cols])
    for condition_idx in range(num_conditions):
        train_mask = train_df["condition_id"] == condition_idx
        test_mask = test_df["condition_id"] == condition_idx
        scaler = StandardScaler()
        if train_mask.any():
            train_df.loc[train_mask, feature_cols] = scaler.fit_transform(train_df.loc[train_mask, feature_cols])
            if test_mask.any():
                test_df.loc[test_mask, feature_cols] = scaler.transform(test_df.loc[test_mask, feature_cols])
            scaler_info[f"condition_{condition_idx}"] = {
                "train_rows": float(train_mask.sum()),
                "test_rows": float(test_mask.sum()),
            }
        elif test_mask.any():
            test_df.loc[test_mask, feature_cols] = global_scaler.transform(test_df.loc[test_mask, feature_cols])
            scaler_info[f"condition_{condition_idx}"] = {
                "train_rows": 0.0,
                "test_rows": float(test_mask.sum()),
            }
    return train_df, test_df, scaler_info


THERMO_STATE_COLS = [
    "_thermo_compressor_temp_rise",
    "_thermo_turbine_temp_drop",
    "_thermo_hpc_pressure_ratio",
    "_thermo_shaft_speed_mismatch",
    "_thermo_corrected_speed_mismatch",
    "_thermo_cooling_flow_balance",
]


def build_raw_thermodynamic_states(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    healthy_reference_residual: bool = False,
    max_rul: float = 125.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build dimensionless gas-path relations before condition normalization."""
    required = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s13", "s14", "s20", "s21"]
    missing = [column for column in required if column not in train_df.columns]
    if missing:
        raise ValueError(f"Raw thermodynamic states require missing sensors: {missing}")

    def derive(frame: pd.DataFrame) -> np.ndarray:
        values = frame[required].astype(np.float64)

        def relative_difference(left: str, right: str) -> np.ndarray:
            denominator = np.maximum(np.abs(values[right].to_numpy()), 1e-6)
            return (values[left].to_numpy() - values[right].to_numpy()) / denominator

        states = np.column_stack(
            [
                relative_difference("s3", "s2"),
                relative_difference("s3", "s4"),
                values["s7"].to_numpy() / np.maximum(np.abs(values["s11"].to_numpy()), 1e-6),
                relative_difference("s9", "s8"),
                relative_difference("s14", "s13"),
                (values["s20"].to_numpy() - values["s21"].to_numpy())
                / np.maximum(np.abs(values["s20"].to_numpy()) + np.abs(values["s21"].to_numpy()), 1e-6),
            ]
        )
        return np.nan_to_num(states, nan=0.0, posinf=10.0, neginf=-10.0)

    raw_train = derive(train_df)
    raw_evaluation = derive(evaluation_df)
    reference_metadata: dict[str, object] | None = None
    if healthy_reference_residual:
        if "condition_id" not in train_df.columns or "condition_id" not in evaluation_df.columns:
            raise ValueError("Healthy-reference TSRM requires operating-condition labels.")
        healthy_mask = train_df["RUL"].to_numpy(dtype=np.float64) >= float(max_rul) - 1e-6
        if not healthy_mask.any():
            raise ValueError("No capped healthy-stage samples are available for TSRM references.")
        train_conditions = train_df["condition_id"].to_numpy(dtype=np.int64)
        evaluation_conditions = evaluation_df["condition_id"].to_numpy(dtype=np.int64)
        condition_count = int(max(train_conditions.max(), evaluation_conditions.max()) + 1)
        global_reference = raw_train[healthy_mask].mean(axis=0)
        references = np.repeat(global_reference[None, :], condition_count, axis=0)
        healthy_counts: list[int] = []
        for condition_id in range(condition_count):
            condition_mask = healthy_mask & (train_conditions == condition_id)
            healthy_counts.append(int(condition_mask.sum()))
            if condition_mask.any():
                references[condition_id] = raw_train[condition_mask].mean(axis=0)
        raw_train = raw_train - references[train_conditions]
        raw_evaluation = raw_evaluation - references[evaluation_conditions]
        reference_metadata = {
            "mode": "condition_healthy_reference_residual",
            "healthy_rul_threshold": float(max_rul),
            "healthy_counts": healthy_counts,
            "global_reference": global_reference.tolist(),
            "condition_references": references.tolist(),
        }
    scaler = StandardScaler()
    train_states = np.clip(scaler.fit_transform(raw_train), -8.0, 8.0).astype(np.float32)
    evaluation_states = np.clip(scaler.transform(raw_evaluation), -8.0, 8.0).astype(np.float32)
    metadata = {
        "columns": THERMO_STATE_COLS,
        "raw_train_mean": raw_train.mean(axis=0).tolist(),
        "raw_train_std": raw_train.std(axis=0).tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "healthy_reference": reference_metadata,
    }
    return train_states, evaluation_states, metadata


def build_raw_condition_profiles(
    train_df: pd.DataFrame,
    condition_ids: pd.Series,
    num_conditions: int,
) -> torch.Tensor:
    """Create comparable raw operating-condition profiles before condition scaling."""
    setting_cols = [column for column in ["setting_1", "setting_2", "setting_3"] if column in train_df.columns]
    if not setting_cols:
        return torch.zeros(num_conditions, 1, dtype=torch.float32)
    raw = train_df[setting_cols].astype(np.float32)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0).replace(0.0, 1.0)
    normalized = (raw - mean) / std
    profiles = []
    for condition_idx in range(num_conditions):
        mask = condition_ids.to_numpy() == condition_idx
        if mask.any():
            profiles.append(normalized.loc[mask].mean(axis=0).to_numpy(dtype=np.float32))
        else:
            profiles.append(np.zeros(len(setting_cols), dtype=np.float32))
    return torch.tensor(np.asarray(profiles, dtype=np.float32))


def build_rul_kernel_sample_weights(
    target_rul: np.ndarray,
    max_rul: float,
    bandwidth: float,
    power: float,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    """Assign smooth inverse-density weights over the continuous RUL axis."""
    values = np.asarray(target_rul, dtype=np.float64).reshape(-1)
    grid = np.linspace(0.0, float(max_rul), 128, dtype=np.float64)
    scale = max(float(bandwidth), 1e-3)
    # A fixed grid keeps the KDE cost small while retaining a continuous RUL prior.
    density_grid = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / scale) ** 2).mean(axis=1)
    sample_density = np.interp(values, grid, density_grid)
    weights = (sample_density + 1e-8) ** (-max(float(power), 0.0))
    weights /= max(weights.mean(), 1e-8)
    weights = np.clip(weights, float(min_weight), float(max_weight))
    weights /= max(weights.mean(), 1e-8)
    return weights.astype(np.float64)


def apply_attnpinn_protocol_scaling(
    full_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Match the public AttnPINN FD004 preprocessing for our own model.

    Settings use independent train/test MinMax scaling to [-1, 1]. Sensors use
    train-set statistics separately for the six operating-condition clusters.
    The returned frames retain the condition labels for the existing model.
    """
    setting_cols = ["setting_1", "setting_2", "setting_3"]
    feature_cols = [c for c in full_train_df.columns if c not in DROP_COLS + ["RUL"]]
    train_out = full_train_df.copy()
    test_out = test_df.copy()

    # Build the exact six clusters after fitting independent MinMax transforms.
    from sklearn.preprocessing import MinMaxScaler

    train_mm = MinMaxScaler(feature_range=(-1, 1))
    test_mm = MinMaxScaler(feature_range=(-1, 1))
    train_out[setting_cols] = train_mm.fit_transform(train_out[setting_cols])
    test_out[setting_cols] = test_mm.fit_transform(test_out[setting_cols])
    centers = np.asarray(
        [[1.0, 1.0, 1.0], [-0.047, 0.66, 1.0], [0.19, 0.47, -1.0],
         [-1.0, -1.0, 1.0], [0.66, 1.0, 1.0], [-0.52, -0.4, 1.0]],
        dtype=np.float32,
    )

    def assign(frame: pd.DataFrame) -> np.ndarray:
        values = frame[setting_cols].to_numpy(dtype=np.float32)
        return ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)

    train_conditions = assign(train_out)
    test_conditions = assign(test_out)
    train_out["condition_id"] = train_conditions
    test_out["condition_id"] = test_conditions
    sensor_cols = [c for c in feature_cols if c not in setting_cols]
    train_out[sensor_cols] = train_out[sensor_cols].astype(np.float32)
    test_out[sensor_cols] = test_out[sensor_cols].astype(np.float32)
    for condition_idx in range(6):
        train_mask = train_conditions == condition_idx
        test_mask = test_conditions == condition_idx
        if not train_mask.any():
            continue
        mean = train_out.loc[train_mask, sensor_cols].mean()
        std = train_out.loc[train_mask, sensor_cols].std().replace(0.0, 1.0)
        train_out.loc[train_mask, sensor_cols] = (train_out.loc[train_mask, sensor_cols] - mean) / std
        if test_mask.any():
            test_out.loc[test_mask, sensor_cols] = (test_out.loc[test_mask, sensor_cols] - mean) / std
    return train_out, test_out, feature_cols


def main() -> None:
    args = parse_args()
    if args.thermo_consistency_gate and not args.thermo_raw_state_input:
        raise ValueError("--thermo-consistency-gate requires --thermo-raw-state-input.")
    if args.thermo_healthy_reference_residual and not args.thermo_raw_state_input:
        raise ValueError("--thermo-healthy-reference-residual requires --thermo-raw-state-input.")
    if args.thermo_consistency_gate and not (
        args.physics_sensor_gate and args.physics_sensor_gate_lstm_input
    ):
        raise ValueError("Thermodynamic consistency gating requires the LSTM-input physics sensor gate.")
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.condition_seed is None:
        args.condition_seed = args.split_seed
    prefix = output_prefix(args)
    set_seed(args.seed)
    ensure_fd_files(args.fd)

    device = configure_device(args.no_cuda)
    pin_memory = device == "cuda"
    use_amp = bool(args.amp and device == "cuda")
    print(f"Device: {device}")

    full_train_df = load_train(args.fd, args.max_rul)
    test_df = load_test(args.fd, args.max_rul)
    thermo_scaler_info: dict[str, object] | None = None
    if args.attnpinn_protocol:
        if args.thermo_raw_state_input:
            raise ValueError("Raw TSRM states are supported only by the project's native protocol.")
        if args.fd.upper() != "FD004":
            raise ValueError("--attnpinn-protocol is defined for FD004 only.")
        protocol_train, protocol_test, feature_cols = apply_attnpinn_protocol_scaling(full_train_df, test_df)
        train_df = protocol_train.iloc[: args.attnpinn_raw_split].copy()
        val_df = protocol_train.iloc[args.attnpinn_raw_split :].copy()
        test_df = protocol_test.copy()
        train_units = np.sort(train_df["unit"].unique()).tolist()
        val_units = np.sort(val_df["unit"].unique()).tolist()
        effective_conditions = 6
        raw_condition_profiles = build_raw_condition_profiles(
            train_df, train_df["condition_id"], effective_conditions
        )
        scaler_info = {"protocol": "attnpinn", "raw_split": args.attnpinn_raw_split}
        window_feature_cols = feature_cols
    else:
        train_df, val_df, train_units, val_units = split_train_validation(
            full_train_df, args.val_ratio, args.split_seed
        )
        evaluation_df = pd.concat(
            [val_df.assign(_evaluation_split="validation"), test_df.assign(_evaluation_split="test")],
            ignore_index=True,
        )
        train_df, evaluation_df, effective_conditions = add_condition_labels(
            train_df, evaluation_df, args.num_conditions, args.condition_seed
        )
        raw_condition_profiles = build_raw_condition_profiles(
            train_df, train_df["condition_id"], effective_conditions
        )
        feature_cols = [c for c in train_df.columns if c not in DROP_COLS + ["RUL", "condition_id"]]
        if args.thermo_raw_state_input:
            thermo_train_states, thermo_evaluation_states, thermo_scaler_info = (
                build_raw_thermodynamic_states(
                    train_df,
                    evaluation_df,
                    healthy_reference_residual=args.thermo_healthy_reference_residual,
                    max_rul=args.max_rul,
                )
            )
        train_df, evaluation_df, scaler_info = scale_features(
            train_df, evaluation_df, feature_cols, args.scaler_mode, effective_conditions
        )
        if args.thermo_raw_state_input:
            train_df.loc[:, THERMO_STATE_COLS] = thermo_train_states
            evaluation_df.loc[:, THERMO_STATE_COLS] = thermo_evaluation_states
            window_feature_cols = feature_cols + THERMO_STATE_COLS
        else:
            window_feature_cols = feature_cols
        val_df = evaluation_df[evaluation_df["_evaluation_split"] == "validation"].drop(columns=["_evaluation_split"])
        test_df = evaluation_df[evaluation_df["_evaluation_split"] == "test"].drop(columns=["_evaluation_split"])

    x_train, y_train_seq, c_train, s_train, u_train = make_train_windows_with_units(
        train_df, window_feature_cols, args.window_size, args.middle_rul, args.late_rul
    )
    if args.validation_protocol == "censored":
        x_val, y_val_seq, c_val, s_val = make_censored_validation_windows(
            val_df, window_feature_cols, args.window_size, args.middle_rul, args.late_rul, args.validation_rul_grid
        )
        u_val = np.arange(len(x_val), dtype=np.int64)
    else:
        x_val, y_val_seq, c_val, s_val, u_val = make_train_windows_with_units(
            val_df, window_feature_cols, args.window_size, args.middle_rul, args.late_rul
        )
    x_test, y_test_seq, c_test, s_test, u_test = make_test_windows_with_units(
        test_df, window_feature_cols, args.window_size, args.middle_rul, args.late_rul
    )
    if args.condition_consistency_change_point_aware:
        change_points = estimate_long_term_change_points(train_df, feature_cols)
        relative_age_train = build_window_relative_ages(
            train_df, u_train, args.window_size, change_points
        )
        print(
            f"Estimated long-term change points for {len(change_points)} training engines; "
            f"relative-age mean={relative_age_train.mean():.3f}",
            flush=True,
        )
    else:
        relative_age_train = np.zeros(len(u_train), dtype=np.float32)
    if args.condition_consistency_fault_mode_aware or args.fault_mode_conditional_head:
        fault_mode_lookup = estimate_long_term_fault_mode_probs(
            train_df, feature_cols, max(2, args.condition_consistency_fault_modes), args.seed
        )
        print(
            f"Estimated soft fault-mode probabilities for {int((fault_mode_lookup.sum(dim=1) > 0).sum())} training engines",
            flush=True,
        )
    else:
        fault_mode_lookup = None
    if args.future_trajectory_mode or args.fault_mode_suppressed_modulation:
        future_mode_lookup = estimate_future_trajectory_mode_probs(
            train_df, feature_cols, max(2, args.condition_consistency_fault_modes), args.seed,
            args.future_trajectory_mode_fraction,
        )
        print(
            f"Estimated future-trajectory pseudo modes for {int((future_mode_lookup.sum(dim=1) > 0).sum())} training engines",
            flush=True,
        )
    else:
        future_mode_lookup = None
    relative_age_val = np.zeros(len(u_val), dtype=np.float32)
    relative_age_test = np.zeros(len(u_test), dtype=np.float32)

    train_counts = np.bincount(c_train, minlength=effective_conditions).tolist()
    val_counts = np.bincount(c_val, minlength=effective_conditions).tolist()
    test_counts = np.bincount(c_test, minlength=effective_conditions).tolist()
    train_stage_counts = np.bincount(s_train, minlength=3).tolist()
    val_stage_counts = np.bincount(s_val, minlength=3).tolist()
    test_stage_counts = np.bincount(s_test, minlength=3).tolist()
    print(f"Effective operating-condition clusters: {effective_conditions}")
    print(f"Train windows: {x_train.shape}, Validation windows: {x_val.shape}, Test engines: {x_test.shape}")
    print(f"Features: {len(feature_cols)} -> {feature_cols}")
    if args.thermo_raw_state_input:
        print(f"Raw thermodynamic side-channel: {len(THERMO_STATE_COLS)} -> {THERMO_STATE_COLS}")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset = TrajectoryRULSequenceDataset(
        x_train, y_train_seq, c_train, s_train, u_train, relative_age_train
    )
    train_sampler = None
    if args.condition_balanced_sampling:
        condition_counts = np.bincount(c_train.astype(np.int64), minlength=effective_conditions).astype(np.float64)
        condition_weights = np.zeros_like(condition_counts)
        nonzero = condition_counts > 0
        condition_weights[nonzero] = 1.0 / condition_counts[nonzero]
        sample_weights = torch.as_tensor(condition_weights[c_train.astype(np.int64)], dtype=torch.double)
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=sampler_generator,
        )
    elif args.rul_kernel_reweighting:
        sample_weights = torch.as_tensor(
            build_rul_kernel_sample_weights(
                y_train_seq[:, -1],
                args.max_rul,
                args.rul_kernel_bandwidth,
                args.rul_kernel_power,
                args.rul_kernel_min_weight,
                args.rul_kernel_max_weight,
            ),
            dtype=torch.double,
        )
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=sampler_generator,
        )
        print(
            "RUL kernel reweighting enabled: "
            f"bandwidth={args.rul_kernel_bandwidth}, power={args.rul_kernel_power}, "
            f"weight range=[{sample_weights.min().item():.3f}, {sample_weights.max().item():.3f}]",
            flush=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        TrajectoryRULSequenceDataset(x_val, y_val_seq, c_val, s_val, u_val, relative_age_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TrajectoryRULSequenceDataset(x_test, y_test_seq, c_test, s_test, u_test, relative_age_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    sensor_gate_direction_scores = estimate_sensor_degradation_signs(train_df, feature_cols)
    model = ConditionBRDRLSTM(
        input_dim=len(feature_cols),
        feature_cols=feature_cols,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        condition_lstm_init=args.condition_lstm_init,
        condition_lstm_bounded_residual=args.condition_lstm_bounded_residual,
        condition_lstm_residual_max=args.condition_lstm_residual_max,
        condition_output_residual=args.condition_output_residual,
        condition_output_residual_max=args.condition_output_residual_max,
        condition_dropout=args.condition_dropout,
        condition_rul_bias=args.condition_rul_bias,
        condition_rul_bias_max=args.condition_rul_bias_max,
        num_conditions=effective_conditions,
        physics_sensor_gate=args.physics_sensor_gate,
        physics_sensor_gate_lstm_input=args.physics_sensor_gate_lstm_input,
        sensor_gate_direction_scores=sensor_gate_direction_scores,
        sensor_gate_strength=args.sensor_gate_strength,
        sensor_gate_warmup_epochs=args.sensor_gate_warmup_epochs,
        sensor_gate_condition_adaptive=args.sensor_gate_condition_adaptive,
        sensor_gate_max_strength=args.sensor_gate_max_strength,
        sensor_gate_control=args.sensor_gate_control,
        physics_graph_branch=args.physics_graph_branch,
        physics_graph_residual_max=args.physics_graph_residual_max,
        thermodynamic_state_residual=args.thermodynamic_state_residual,
        thermo_state_dim=args.thermo_state_dim,
        thermo_residual_max=args.thermo_residual_max,
        thermo_raw_state_dim=len(THERMO_STATE_COLS) if args.thermo_raw_state_input else 0,
        thermo_condition_confidence_adaptive=args.thermo_condition_confidence_adaptive,
        thermo_confidence_floor=args.thermo_confidence_floor,
        thermo_fusion_mode=args.thermo_fusion_mode,
        thermo_consistency_gate=args.thermo_consistency_gate,
        thermo_gate_state_dim=args.thermo_gate_state_dim,
        thermo_gate_max_attenuation=args.thermo_gate_max_attenuation,
        shared_private_latent=(
            args.shared_private_latent
            or args.fault_mode_conditional_head
            or args.condition_invariant_latent
        ),
        condition_invariant_latent=args.condition_invariant_latent,
        condition_normalized_ranking=args.condition_normalized_ranking_weight > 0,
        fault_prototypes=args.fault_prototypes,
        fault_prototype_temperature=args.fault_prototype_temperature,
        fault_mixture_experts=args.fault_mixture_experts,
        fault_mixture_expert_count=args.fault_mixture_expert_count,
        fault_mixture_temperature=args.fault_mixture_temperature,
        fault_mode_conditional_head=args.fault_mode_conditional_head,
        fault_mode_conditional_weight=args.fault_mode_conditional_weight,
        unified_health_state=args.unified_health_state,
        health_state_mode=args.health_state_mode,
        multi_scale_temporal=args.multi_scale_temporal,
        multi_scale_residual_max=args.multi_scale_residual_max,
        future_trajectory_mode=args.future_trajectory_mode,
        future_trajectory_mode_residual_max=args.future_trajectory_mode_residual_max,
        condition_observation_residual=args.condition_observation_residual,
        condition_observation_residual_max=args.condition_observation_residual_max,
        dynamic_condition_modulation=args.dynamic_condition_modulation,
        dynamic_condition_modulation_max=args.dynamic_condition_modulation_max,
        fault_mode_suppressed_modulation=args.fault_mode_suppressed_modulation,
    ).to(device)
    loss_fn = StageConditionAwareBRDRPINNLoss(
        num_conditions=effective_conditions,
        mono_weight=args.mono_weight,
        smooth_weight=args.smooth_weight,
        failure_weight=args.failure_weight,
        asym_weight=args.asym_weight,
        asym_over_alpha=args.asym_over_alpha,
        asym_under_beta=args.asym_under_beta,
        asym_clip=args.asym_clip,
        over_margin_weight=args.over_margin_weight,
        over_margin=args.over_margin,
        late_stage_weight=1.0 if args.disable_adaptive_brdr_stage_prior else args.late_stage_weight,
        health_weight=args.health_weight,
        health_mono_weight=args.health_mono_weight,
        weak_health_index=args.weak_health_index,
        health_margin=args.health_margin,
        hi_min_std=args.hi_min_std,
        hi_endpoint_gap=args.hi_endpoint_gap,
        hi_range_eps=args.hi_range_eps,
        learnable_stage_prior=args.learnable_stage_prior,
        stage_prior_reg_weight=args.stage_prior_reg_weight,
        stage_prior_range=args.stage_prior_range,
        failure_threshold=args.failure_threshold,
        beta_c=args.beta_c,
        beta_w=args.beta_w,
        brdr_min_weight=(
            1.0 if args.disable_adaptive_brdr_stage_prior or args.disable_brdr_adaptation
            else args.brdr_min_weight
        ),
        brdr_max_weight=(
            1.0 if args.disable_adaptive_brdr_stage_prior or args.disable_brdr_adaptation
            else args.brdr_max_weight
        ),
        brdr_weight_power=(1.0 if args.disable_brdr_adaptation else args.brdr_weight_power),
        brdr_warmup_steps=(0 if args.disable_brdr_adaptation else args.brdr_warmup_steps),
    ).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=args.lr)
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.health_index_pretrain_epochs > 0:
        hi_params = list(model.temporal_encoder.parameters())
        if hasattr(model, "state_encoder"):
            hi_params += list(model.state_encoder.parameters()) + list(model.health_head.parameters())
        if hasattr(model, "condition_embedding") and args.health_state_mode == "condition_residual":
            hi_params += list(model.condition_embedding.parameters())
        hi_optimizer = torch.optim.Adam(hi_params, lr=args.lr)
        print(f"HI pretraining for {args.health_index_pretrain_epochs} epochs", flush=True)
        for hi_epoch in range(1, args.health_index_pretrain_epochs + 1):
            model.train()
            hi_total = 0.0
            hi_seen = 0
            for xb, yb, cb, sb, _, _ in train_loader:
                xb = xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                cb = cb.to(device, non_blocking=pin_memory)
                sb = sb.to(device, non_blocking=pin_memory)
                hi_optimizer.zero_grad()
                output = model(xb, cb, sb)
                if not isinstance(output, tuple):
                    raise ValueError("Health-index pretraining requires a health-state model mode.")
                _, health_seq = output
                hi_mono = torch.relu(health_seq[:, 1:] - health_seq[:, :-1]).mean()
                hi_drop = health_seq[:, :-1] - health_seq[:, 1:]
                hi_pairwise = torch.relu(args.health_margin - hi_drop).mean()
                hi_smooth = (health_seq[:, 1:] - health_seq[:, :-1]).pow(2).mean()
                healthy_mask = yb[:, -1] >= args.middle_rul
                failure_mask = yb[:, -1] <= args.failure_threshold
                healthy_anchor = (torch.relu(0.8 - health_seq[healthy_mask, 0]).mean()
                                  if healthy_mask.any() else health_seq.new_tensor(0.0))
                failure_anchor = (health_seq[failure_mask, -1].pow(2).mean()
                                  if failure_mask.any() else health_seq.new_tensor(0.0))
                if args.weak_health_index:
                    hi_std = health_seq.std(dim=1, unbiased=False)
                    variance_penalty = torch.relu(args.hi_min_std - hi_std).pow(2).mean()
                    if healthy_mask.any() and failure_mask.any():
                        endpoint_gap = torch.relu(
                            args.hi_endpoint_gap
                            - (health_seq[healthy_mask, 0].mean() - health_seq[failure_mask, -1].mean())
                        ).pow(2)
                    else:
                        endpoint_gap = health_seq.new_tensor(0.0)
                    range_penalty = (
                        torch.relu(args.hi_range_eps - health_seq).pow(2)
                        + torch.relu(health_seq - (1.0 - args.hi_range_eps)).pow(2)
                    ).mean()
                    hi_loss = (hi_pairwise + 0.1 * hi_mono + 0.01 * hi_smooth
                               + 0.5 * healthy_anchor + 0.5 * failure_anchor
                               + variance_penalty + endpoint_gap + 0.1 * range_penalty)
                else:
                    hi_target = (yb / args.max_rul).clamp(0.0, 1.0)
                    hi_fit = (health_seq - hi_target).pow(2).mean()
                    hi_loss = hi_fit + 0.1 * hi_mono + 0.01 * hi_smooth
                hi_loss.backward()
                nn.utils.clip_grad_norm_(hi_params, max_norm=1.0)
                hi_optimizer.step()
                hi_total += float(hi_loss.detach().cpu()) * len(xb)
                hi_seen += len(xb)
            print(f"HI epoch {hi_epoch:03d} | Loss {hi_total / max(hi_seen, 1):.5f}", flush=True)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_selection_value = float("inf")
    best_validation_metrics: dict[str, float] | None = None
    best_epoch = 0
    stale = 0
    best_path = OUTPUT_DIR / f"best_{prefix}.pt"
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        if model.sensor_reliability_gate is not None:
            model.sensor_reliability_gate.set_epoch(epoch)
        sums = {
            "total_loss": 0.0,
            "rul_loss": 0.0,
            "raw_rul_loss": 0.0,
            "mono_loss": 0.0,
            "smooth_loss": 0.0,
            "failure_loss": 0.0,
            "asym_loss": 0.0,
            "over_margin_loss": 0.0,
            "stage_prior_reg": 0.0,
            "health_loss": 0.0,
            "health_mono_loss": 0.0,
            "w_rul_loss": 0.0,
            "w_mono_loss": 0.0,
            "w_smooth_loss": 0.0,
            "w_failure_loss": 0.0,
            "condition_output_center_loss": 0.0,
            "condition_rul_bias_center_loss": 0.0,
            "condition_consistency_loss": 0.0,
            "future_trajectory_mode_loss": 0.0,
            "fault_mode_suppression_loss": 0.0,
            "fault_mode_conditional_loss": 0.0,
            "condition_adversarial_loss": 0.0,
            "private_condition_loss": 0.0,
            "condition_normalized_ranking_loss": 0.0,
            "condition_normalized_over_loss": 0.0,
            "within_engine_ranking_loss": 0.0,
            "thermo_consistency_loss": 0.0,
            "thermo_transition_loss": 0.0,
        }
        for condition_idx in range(effective_conditions):
            sums[f"w_condition_{condition_idx}"] = 0.0
        for stage_idx in range(3):
            sums[f"w_stage_{stage_idx}"] = 0.0
            sums[f"prior_stage_{stage_idx}"] = 0.0
        for condition_idx in range(effective_conditions):
            for stage_idx in range(3):
                sums[f"w_condition_{condition_idx}_stage_{stage_idx}"] = 0.0
        seen = 0
        gradient_diagnostics = {
            name: {"cosine_sum": 0.0, "conflicts": 0, "count": 0}
            for name in ("physics", "mono", "smooth", "failure")
        }

        for batch_index, (xb, yb, cb, sb, ub, ab) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)
            cb = cb.to(device, non_blocking=pin_memory)
            sb = sb.to(device, non_blocking=pin_memory)
            if args.thermo_raw_state_input:
                augmented_sensors = augment_windows(
                    xb[:, :, : len(feature_cols)],
                    args.augment_noise_std,
                    args.augment_feature_dropout,
                )
                xb = torch.cat([augmented_sensors, xb[:, :, len(feature_cols) :]], dim=-1)
            else:
                xb = augment_windows(xb, args.augment_noise_std, args.augment_feature_dropout)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                model_output, latent_output = model(xb, cb, sb, return_latent=True)
                if isinstance(latent_output, tuple):
                    latent_seq, private_latent, fault_probs = latent_output
                else:
                    latent_seq, private_latent, fault_probs = latent_output, None, None
                if isinstance(model_output, tuple):
                    pred_seq, health_seq = model_output
                else:
                    pred_seq, health_seq = model_output, None
                loss, parts = loss_fn(
                    pred_seq,
                    yb,
                    cb,
                    sb,
                    x_seq=xb[:, :, : len(feature_cols)],
                    health_seq=health_seq,
                )
                if (
                    args.thermo_consistency_gate
                    and args.thermo_transition_weight > 0
                    and model.thermo_consistency_gate is not None
                    and model.thermo_consistency_gate.last_transition_loss is not None
                ):
                    transition_loss = model.thermo_consistency_gate.last_transition_loss
                    loss = loss + args.thermo_transition_weight * transition_loss
                    parts["thermo_transition_loss"] = float(transition_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if (
                    args.thermodynamic_state_residual
                    and args.thermo_consistency_weight > 0
                    and model.last_thermo_health is not None
                ):
                    thermo_target = (yb / max(float(args.max_rul), 1.0)).clamp(0.0, 1.0)
                    thermo_health = model.last_thermo_health
                    thermo_fit = nn.functional.mse_loss(thermo_health, thermo_target)
                    thermo_mono = torch.relu(
                        thermo_health[:, 1:] - thermo_health[:, :-1]
                    ).mean()
                    thermo_consistency = thermo_fit + 0.1 * thermo_mono
                    loss = loss + args.thermo_consistency_weight * thermo_consistency
                    parts["thermo_consistency_loss"] = float(
                        thermo_consistency.detach().cpu()
                    )
                    parts["total_loss"] = float(loss.detach().cpu())
                if args.condition_output_residual and args.condition_output_center_weight > 0:
                    center_loss = model.condition_output_center_loss()
                    loss = loss + args.condition_output_center_weight * center_loss
                    parts["condition_output_center_loss"] = float(center_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if args.condition_rul_bias and args.condition_rul_bias_center_weight > 0:
                    bias_center_loss = model.condition_rul_bias_center_loss()
                    loss = loss + args.condition_rul_bias_center_weight * bias_center_loss
                    parts["condition_rul_bias_center_loss"] = float(bias_center_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if args.condition_consistency_weight > 0:
                    consistency_loss = condition_aware_degradation_consistency_loss(
                        latent_seq,
                        yb,
                        cb,
                        sb,
                        ub,
                        args.condition_consistency_temperature,
                        args.condition_consistency_equalization_weight,
                        (xb[:, -1, 3:] - xb[:, 0, 3:])
                        if args.condition_consistency_fault_aware
                        else None,
                        args.condition_consistency_fault_temperature,
                        ab if args.condition_consistency_change_point_aware else None,
                        args.condition_consistency_change_point_temperature,
                        fault_mode_lookup if args.condition_consistency_fault_mode_aware else None,
                        args.condition_consistency_fault_mode_threshold,
                    )
                    loss = loss + args.condition_consistency_weight * consistency_loss
                    parts["condition_consistency_loss"] = float(consistency_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if args.within_engine_ranking_weight > 0:
                    ranking_loss = within_engine_rul_ranking_loss(
                        pred_seq, yb, ub, args.within_engine_ranking_margin
                    )
                    loss = loss + args.within_engine_ranking_weight * ranking_loss
                    parts["within_engine_ranking_loss"] = float(ranking_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if args.condition_normalized_ranking_weight > 0 and model.last_degradation_scores is not None:
                    rank_warmup = min(
                        1.0,
                        epoch / max(int(args.condition_normalized_ranking_warmup_epochs), 1),
                    )
                    rank_sample_weight = None
                    if args.condition_normalized_ranking_gated:
                        final_rul = yb[:, -1]
                        low_or_high = (final_rul <= args.condition_normalized_ranking_low_rul_max) | (
                            final_rul >= args.condition_normalized_ranking_high_rul_min
                        )
                        rank_sample_weight = torch.full_like(
                            final_rul,
                            float(args.condition_normalized_ranking_mid_weight),
                        )
                        rank_sample_weight = torch.where(
                            low_or_high,
                            torch.ones_like(rank_sample_weight),
                            rank_sample_weight,
                        )
                    condition_rank_loss = condition_normalized_degradation_ranking_loss(
                        model.last_degradation_scores,
                        yb,
                        cb,
                        args.condition_normalized_ranking_margin,
                        args.condition_normalized_ranking_min_rul_gap,
                        rank_sample_weight,
                    )
                    loss = loss + args.condition_normalized_ranking_weight * rank_warmup * condition_rank_loss
                    parts["condition_normalized_ranking_loss"] = float(condition_rank_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                    if args.condition_normalized_ranking_over_weight > 0:
                        over_safe_loss = over_prediction_safe_loss(
                            pred_seq,
                            yb,
                            args.condition_normalized_ranking_over_margin,
                        )
                        loss = loss + args.condition_normalized_ranking_over_weight * rank_warmup * over_safe_loss
                        parts["condition_normalized_over_loss"] = float(over_safe_loss.detach().cpu())
                        parts["total_loss"] = float(loss.detach().cpu())
                if args.trajectory_align_weight > 0:
                    trajectory_loss = cross_engine_trajectory_alignment_loss(
                        latent_seq, yb, ub, args.trajectory_align_temperature
                    )
                    align_scale = min(1.0, epoch / max(args.stage_trajectory_align_warmup_epochs, 1)) \
                        if args.stage_trajectory_align_warmup_epochs > 0 else 1.0
                    loss = loss + args.trajectory_align_weight * align_scale * trajectory_loss
                    parts["trajectory_align_loss"] = float(trajectory_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if args.stage_trajectory_align_weight > 0:
                    stage_trajectory_loss = soft_stage_trajectory_alignment_loss(
                        latent_seq,
                        yb,
                        ub,
                        args.stage_trajectory_align_temperature,
                        args.stage_trajectory_align_window,
                        args.max_rul,
                        pred_seq[:, -1] if args.stage_trajectory_align_source == "prediction" else None,
                        cb,
                        args.stage_trajectory_align_condition_mode,
                        raw_condition_profiles.to(device)[cb]
                        if args.stage_trajectory_align_condition_source == "raw_settings"
                        else xb[:, :, :3],
                        args.stage_trajectory_align_condition_temperature,
                        fault_probs
                        if args.fault_prototype_weight > 0 and fault_probs is not None
                        else ((xb[:, -1, 3:] - xb[:, 0, 3:]) if args.stage_trajectory_align_fault_aware else None),
                        args.stage_trajectory_align_fault_temperature,
                    )
                    align_scale = min(1.0, epoch / max(args.stage_trajectory_align_warmup_epochs, 1)) \
                        if args.stage_trajectory_align_warmup_epochs > 0 else 1.0
                    loss = loss + args.stage_trajectory_align_weight * align_scale * stage_trajectory_loss
                    parts["stage_trajectory_align_loss"] = float(stage_trajectory_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if private_latent is not None and args.shared_private_orth_weight > 0:
                    orth_loss = shared_private_orthogonality_loss(latent_seq, private_latent)
                    loss = loss + args.shared_private_orth_weight * orth_loss
                    parts["shared_private_orth_loss"] = float(orth_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if args.condition_invariant_latent and args.condition_adversarial_weight > 0:
                    shared_logits = model.last_condition_logits_shared
                    if shared_logits is not None:
                        adv_loss = nn.functional.cross_entropy(shared_logits, cb.long())
                        loss = loss + args.condition_adversarial_weight * adv_loss
                        parts["condition_adversarial_loss"] = float(adv_loss.detach().cpu())
                        parts["total_loss"] = float(loss.detach().cpu())
                if args.condition_invariant_latent and args.private_condition_weight > 0:
                    private_logits = model.last_condition_logits_private
                    if private_logits is not None:
                        private_condition_loss = nn.functional.cross_entropy(private_logits, cb.long())
                        loss = loss + args.private_condition_weight * private_condition_loss
                        parts["private_condition_loss"] = float(private_condition_loss.detach().cpu())
                        parts["total_loss"] = float(loss.detach().cpu())
                if fault_probs is not None and args.fault_prototype_weight > 0:
                    prototype_balance = fault_prototype_balance_loss(fault_probs)
                    loss = loss + args.fault_prototype_weight * prototype_balance
                    parts["fault_prototype_balance_loss"] = float(prototype_balance.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if (
                    args.fault_mode_conditional_head
                    and args.fault_mode_conditional_weight > 0
                    and fault_probs is not None
                    and fault_mode_lookup is not None
                ):
                    mode_targets = torch.argmax(fault_mode_lookup.to(device)[ub.long()], dim=-1)
                    mode_loss = nn.functional.nll_loss(
                        torch.log(fault_probs.clamp_min(1e-6)), mode_targets
                    )
                    loss = loss + args.fault_mode_conditional_weight * mode_loss
                    parts["fault_mode_conditional_loss"] = float(mode_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if args.future_trajectory_mode and fault_probs is not None and future_mode_lookup is not None:
                    mode_targets = torch.argmax(future_mode_lookup.to(device)[ub.long()], dim=-1)
                    mode_loss = nn.functional.nll_loss(
                        torch.log(fault_probs.clamp_min(1e-6)), mode_targets
                    )
                    loss = loss + args.future_trajectory_mode_weight * mode_loss
                    parts["future_trajectory_mode_loss"] = float(mode_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if args.fault_mode_suppressed_modulation and model.last_modulation_mode_probs is not None and future_mode_lookup is not None:
                    mode_targets = torch.argmax(future_mode_lookup.to(device)[ub.long()], dim=-1)
                    suppression_loss = nn.functional.nll_loss(
                        torch.log(model.last_modulation_mode_probs.clamp_min(1e-6)), mode_targets
                    )
                    loss = loss + args.fault_mode_suppression_weight * suppression_loss
                    parts["fault_mode_suppression_loss"] = float(suppression_loss.detach().cpu())
                    parts["total_loss"] = float(loss.detach().cpu())
                if args.fault_mixture_experts and args.fault_mixture_balance_weight > 0:
                    router_probs = model.last_mixture_probs
                    if router_probs is not None:
                        mixture_balance = fault_mixture_balance_loss(router_probs)
                        loss = loss + args.fault_mixture_balance_weight * mixture_balance
                        parts["fault_mixture_balance_loss"] = float(mixture_balance.detach().cpu())
                        parts["total_loss"] = float(loss.detach().cpu())
            control_gradients = bool(args.physics_gradient_conflict_control)
            sample_diagnostics = bool(
                (args.physics_gradient_diagnostics or control_gradients)
                and batch_index % max(int(args.physics_gradient_diagnostic_every), 1) == 0
            )
            analyze_aggregate = control_gradients or sample_diagnostics
            data_grads: tuple[torch.Tensor | None, ...] | None = None
            physics_grads: tuple[torch.Tensor | None, ...] | None = None
            physics_dot: torch.Tensor | None = None
            data_norm_sq: torch.Tensor | None = None
            if analyze_aggregate:
                if loss_fn.last_data_objective is None or loss_fn.last_physics_objective is None:
                    raise RuntimeError("Physics-gradient analysis requires decomposed loss objectives.")
                data_grads = torch.autograd.grad(
                    loss_fn.last_data_objective,
                    model_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                physics_grads = torch.autograd.grad(
                    loss_fn.last_physics_objective,
                    model_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                physics_dot, data_norm_sq, cosine = gradient_alignment(data_grads, physics_grads)
                diagnostics = gradient_diagnostics["physics"]
                diagnostics["cosine_sum"] += cosine
                diagnostics["conflicts"] += int(float(physics_dot.detach().cpu()) < 0.0)
                diagnostics["count"] += 1
            if sample_diagnostics and data_grads is not None:
                for short_name, loss_name in (
                    ("mono", "mono_loss"),
                    ("smooth", "smooth_loss"),
                    ("failure", "failure_loss"),
                ):
                    weighted_loss = loss_fn.last_weighted_physics_objectives.get(loss_name)
                    if weighted_loss is None or not weighted_loss.requires_grad:
                        continue
                    term_grads = torch.autograd.grad(
                        weighted_loss,
                        model_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    term_dot, _, term_cosine = gradient_alignment(data_grads, term_grads)
                    diagnostics = gradient_diagnostics[short_name]
                    diagnostics["cosine_sum"] += term_cosine
                    diagnostics["conflicts"] += int(float(term_dot.detach().cpu()) < 0.0)
                    diagnostics["count"] += 1

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            if (
                control_gradients
                and data_grads is not None
                and physics_dot is not None
                and data_norm_sq is not None
                and float(physics_dot.detach().cpu()) < 0.0
            ):
                strength = min(max(float(args.physics_gradient_projection_strength), 0.0), 1.0)
                coefficient = strength * (-physics_dot / data_norm_sq.clamp_min(1e-12))
                for parameter, data_grad in zip(model_parameters, data_grads):
                    if parameter.grad is not None and data_grad is not None:
                        parameter.grad.add_(data_grad, alpha=float(coefficient.detach().cpu()))
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(loss_fn.parameters()), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            batch_n = len(xb)
            seen += batch_n
            for key in sums:
                sums[key] += parts.get(key, 0.0) * batch_n

        epoch_parts = {key: value / seen for key, value in sums.items()}
        for name, diagnostics in gradient_diagnostics.items():
            count = max(int(diagnostics["count"]), 1)
            epoch_parts[f"{name}_grad_cosine"] = float(diagnostics["cosine_sum"]) / count
            epoch_parts[f"{name}_grad_conflict_rate"] = float(diagnostics["conflicts"]) / count
        if epoch == 1 or epoch % args.eval_every == 0:
            y_true, y_pred, _, _ = predict(model, val_loader, device, pin_memory)
            validation_metrics = regression_metrics(y_true, y_pred)
            loss_history.append(
                {
                    "epoch": epoch,
                    **epoch_parts,
                    "val_rmse": validation_metrics["rmse"],
                    "val_mae": validation_metrics["mae"],
                    "val_nasa_score": validation_metrics["nasa_score"],
                }
            )
            print(
                f"Epoch {epoch:03d} | Total {epoch_parts['total_loss']:.4f} | "
                f"RUL {epoch_parts['rul_loss']:.4f} | Val RMSE {validation_metrics['rmse']:.3f} | "
                f"Val MAE {validation_metrics['mae']:.3f} | Val NASA {validation_metrics['nasa_score']:.2f} | "
                f"GradConflict {epoch_parts['physics_grad_conflict_rate']:.2f}"
            )
            selection_value = validation_metrics[args.selection_metric]
            if selection_value < best_selection_value:
                best_selection_value = selection_value
                best_validation_metrics = validation_metrics
                best_epoch = epoch
                stale = 0
                save_torch_checkpoint(
                    {
                        "model_state_dict": model.state_dict(),
                        "loss_state_dict": loss_fn.state_dict(),
                        "feature_cols": feature_cols,
                        "window_size": args.window_size,
                        "max_rul": args.max_rul,
                        "validation_metrics": validation_metrics,
                        "best_epoch": epoch,
                        "selection_metric": args.selection_metric,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "unified_health_state": args.unified_health_state,
                        "health_state_mode": args.health_state_mode,
                        "effective_conditions": effective_conditions,
                        "physics_brdr_weights": dict(loss_fn.physics_balancer.weights),
                        "condition_stage_brdr_weights": dict(loss_fn.condition_stage_balancer.weights),
                        "physics_brdr_state": loss_fn.physics_balancer.state_dict(),
                        "condition_stage_brdr_state": loss_fn.condition_stage_balancer.state_dict(),
                        "final_stage_prior": [
                            float(value)
                            for value in loss_fn.stage_prior(torch.device(device)).detach().cpu().tolist()
                        ],
                    },
                    best_path,
                )
            else:
                stale += 1
            if args.early_stopping_patience > 0 and stale >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}.")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn.load_state_dict(checkpoint["loss_state_dict"])
    if "physics_brdr_state" in checkpoint:
        loss_fn.physics_balancer.load_state_dict(checkpoint["physics_brdr_state"])
    if "condition_stage_brdr_state" in checkpoint:
        loss_fn.condition_stage_balancer.load_state_dict(checkpoint["condition_stage_brdr_state"])
    if args.validation_only:
        validation_metrics = best_validation_metrics or checkpoint["validation_metrics"]
        metrics_path = save_metrics(
            prefix,
            args.fd,
            validation_metrics,
            args,
            {
                "device": device,
                "amp": use_amp,
                "effective_conditions": effective_conditions,
                "train_units": train_units,
                "validation_units": val_units,
                "train_condition_counts": train_counts,
                "validation_condition_counts": val_counts,
                "train_stage_counts": train_stage_counts,
                "validation_stage_counts": val_stage_counts,
                "scaler_info": scaler_info,
                "thermo_raw_state_scaler": thermo_scaler_info,
                "augment_noise_std": args.augment_noise_std,
                "augment_feature_dropout": args.augment_feature_dropout,
                "train_windows": int(len(x_train)),
                "validation_windows": int(len(x_val)),
                "validation_units_count": int(len(val_units)),
                "best_epoch": best_epoch,
                "best_validation_metrics": validation_metrics,
                "best_epoch_metrics": validation_metrics,
                "test_evaluated": False,
                "final_test_metrics": None,
                "final_physics_brdr_weights": dict(loss_fn.physics_balancer.weights),
                "final_condition_stage_brdr_weights": dict(loss_fn.condition_stage_balancer.weights),
                "final_stage_prior": [
                    float(value)
                    for value in loss_fn.stage_prior(torch.device(device)).detach().cpu().tolist()
                ],
                "thermo_condition_residual_scales": (
                    (
                        model.thermodynamic_state_residual.residual_max
                        * torch.tanh(
                            model.thermodynamic_state_residual.residual_logit
                            + model.thermodynamic_state_residual.condition_residual_logit.weight.squeeze(-1)
                        )
                    ).detach().cpu().tolist()
                    if model.thermodynamic_state_residual is not None
                    and model.thermodynamic_state_residual.condition_confidence_adaptive
                    else None
                ),
                "thermo_gate_condition_attenuations": (
                    model.thermo_consistency_gate.condition_attenuations().detach().cpu().tolist()
                    if model.thermo_consistency_gate is not None else None
                ),
            },
        )
        history_path = save_loss_history(prefix, loss_history)
        print(
            f"Validation only | RMSE {validation_metrics['rmse']:.3f} | "
            f"MAE {validation_metrics['mae']:.3f} | "
            f"NASA {validation_metrics['nasa_score']:.2f}",
            flush=True,
        )
        print("Test evaluation skipped.", flush=True)
        print(f"Saved model: {best_path}", flush=True)
        print(f"Saved metrics: {metrics_path}", flush=True)
        print(f"Saved loss history: {history_path}", flush=True)
        return
    if model.sensor_reliability_gate is not None:
        model.sensor_reliability_gate.set_epoch(args.epochs)
    calibration_info: dict[str, object] = {"enabled": False}
    y_true, y_pred, condition_ids, stage_ids = predict(model, test_loader, device, pin_memory)
    pre_calibration_metrics = regression_metrics(y_true, y_pred)
    if args.safe_soft_condition_stage_calibration:
        val_true_cal, val_pred_cal, val_condition_cal, _ = predict(
            model, val_loader, device, pin_memory
        )
        calibration_info = fit_safe_soft_condition_stage_calibrator(
            val_true_cal,
            val_pred_cal,
            val_condition_cal,
            raw_condition_profiles.cpu().numpy(),
            args.middle_rul,
            args.late_rul,
            args.max_rul,
            args.soft_stage_calibration_temperature,
        )
        y_pred, stage_ids, safe_stats = apply_safe_soft_condition_stage_calibrator(
            y_pred,
            condition_ids,
            calibration_info,
            raw_condition_profiles.cpu().numpy(),
            args.middle_rul,
            args.late_rul,
            args.max_rul,
            args.soft_stage_calibration_temperature,
            args.safe_condition_calibration_temperature,
            args.stage_calibration_shrinkage,
            args.stage_calibration_correction_max,
            args.safe_calibration_variance_scale,
            args.safe_calibration_positive_scale,
        )
        calibration_info["enabled"] = True
        calibration_info["type"] = "safe_soft_condition_stage_residual"
        calibration_info["stage_temperature"] = float(args.soft_stage_calibration_temperature)
        calibration_info["condition_temperature"] = float(args.safe_condition_calibration_temperature)
        calibration_info["shrinkage"] = float(args.stage_calibration_shrinkage)
        calibration_info["correction_max"] = float(args.stage_calibration_correction_max)
        calibration_info["variance_scale"] = float(args.safe_calibration_variance_scale)
        calibration_info["positive_scale"] = float(args.safe_calibration_positive_scale)
        calibration_info["test_correction_stats"] = safe_stats
        for cache_key in ("val_residual", "val_stage_probs", "val_condition_profiles"):
            calibration_info.pop(cache_key, None)
        print(
            "Validation-fitted safe soft condition-stage residual calibration: "
            f"stage_temperature={args.soft_stage_calibration_temperature}, "
            f"condition_temperature={args.safe_condition_calibration_temperature}, "
            f"correction_max={args.stage_calibration_correction_max}",
            flush=True,
        )
    elif args.soft_condition_stage_calibration:
        val_true_cal, val_pred_cal, val_condition_cal, _ = predict(
            model, val_loader, device, pin_memory
        )
        calibration_coeffs, calibration_info = fit_soft_condition_stage_calibrator(
            val_true_cal,
            val_pred_cal,
            val_condition_cal,
            args.middle_rul,
            args.late_rul,
            args.max_rul,
            effective_conditions,
            args.soft_stage_calibration_temperature,
            args.stage_calibration_shrinkage,
        )
        y_pred, stage_ids = apply_soft_condition_stage_calibrator(
            y_pred,
            condition_ids,
            calibration_coeffs,
            args.middle_rul,
            args.late_rul,
            args.max_rul,
            effective_conditions,
            args.soft_stage_calibration_temperature,
            args.stage_calibration_correction_max,
        )
        calibration_info["enabled"] = True
        calibration_info["type"] = "soft_condition_stage_residual"
        calibration_info["correction_max"] = float(args.stage_calibration_correction_max)
        print(
            "Validation-fitted soft condition-stage residual calibration: "
            f"temperature={args.soft_stage_calibration_temperature}, "
            f"correction_max={args.stage_calibration_correction_max}",
            flush=True,
        )
    elif args.condition_stage_calibration:
        val_true_cal, val_pred_cal, val_condition_cal, _ = predict(
            model, val_loader, device, pin_memory
        )
        calibration_coeffs, calibration_info = fit_condition_stage_calibrator(
            val_true_cal,
            val_pred_cal,
            val_condition_cal,
            args.middle_rul,
            args.late_rul,
            effective_conditions,
            args.stage_calibration_shrinkage,
        )
        y_pred, stage_ids = apply_condition_stage_calibrator(
            y_pred,
            condition_ids,
            calibration_coeffs,
            args.middle_rul,
            args.late_rul,
            effective_conditions,
            args.max_rul,
        )
        calibration_info["enabled"] = True
        calibration_info["type"] = "condition_stage_hierarchical"
        print(
            "Validation-fitted condition-stage calibration: "
            f"validation_condition_stage_counts={calibration_info['validation_predicted_condition_stage_counts']}",
            flush=True,
        )
    elif args.stage_calibration:
        val_true_cal, val_pred_cal, _, _ = predict(model, val_loader, device, pin_memory)
        calibration_coeffs, calibration_info = fit_stage_calibrator(
            val_true_cal,
            val_pred_cal,
            args.middle_rul,
            args.late_rul,
            args.stage_calibration_shrinkage,
            args.stage_calibration_min_samples,
        )
        y_pred, stage_ids = apply_stage_calibrator(
            y_pred,
            calibration_coeffs,
            args.middle_rul,
            args.late_rul,
            args.max_rul,
        )
        calibration_info["enabled"] = True
        print(
            "Validation-fitted stage calibration: "
            f"coefficients={calibration_info['coefficients']}, "
            f"validation_predicted_stage_counts={calibration_info['validation_predicted_stage_counts']}",
            flush=True,
        )
    final_metrics = regression_metrics(y_true, y_pred)
    predictions_path = save_predictions(prefix, args.fd, y_true, y_pred, condition_ids, stage_ids)
    metrics_path = save_metrics(
        prefix,
        args.fd,
        final_metrics,
        args,
        {
            "device": device,
            "amp": use_amp,
            "effective_conditions": effective_conditions,
            "train_units": train_units,
            "validation_units": val_units,
            "train_condition_counts": train_counts,
            "validation_condition_counts": val_counts,
            "test_condition_counts": test_counts,
            "train_stage_counts": train_stage_counts,
            "validation_stage_counts": val_stage_counts,
            "test_stage_counts": test_stage_counts,
            "scaler_info": scaler_info,
            "thermo_raw_state_scaler": thermo_scaler_info,
            "augment_noise_std": args.augment_noise_std,
            "augment_feature_dropout": args.augment_feature_dropout,
            "train_windows": int(len(x_train)),
            "validation_windows": int(len(x_val)),
            "validation_units_count": int(len(val_units)),
            "test_units": int(len(x_test)),
            "best_epoch": best_epoch,
            "best_validation_metrics": best_validation_metrics,
            "best_epoch_metrics": best_validation_metrics,
            "final_test_metrics": final_metrics,
            "pre_calibration_test_metrics": pre_calibration_metrics,
            "stage_calibration": calibration_info,
            "final_physics_brdr_weights": dict(loss_fn.physics_balancer.weights),
            "final_condition_stage_brdr_weights": dict(loss_fn.condition_stage_balancer.weights),
            "final_stage_prior": [
                float(value)
                for value in loss_fn.stage_prior(torch.device(device)).detach().cpu().tolist()
            ],
            "physics_graph_residual_scale": (
                float(model.physics_graph_branch.residual_scale().detach().cpu())
                if model.physics_graph_branch is not None else 0.0
            ),
            "thermo_residual_scale": (
                float(
                    model.thermodynamic_state_residual.residual_max
                    * torch.tanh(model.thermodynamic_state_residual.residual_logit).detach().cpu()
                )
                if model.thermodynamic_state_residual is not None else 0.0
            ),
            "thermo_condition_residual_scales": (
                (
                    model.thermodynamic_state_residual.residual_max
                    * torch.tanh(
                        model.thermodynamic_state_residual.residual_logit
                        + model.thermodynamic_state_residual.condition_residual_logit.weight.squeeze(-1)
                    )
                ).detach().cpu().tolist()
                if model.thermodynamic_state_residual is not None
                and model.thermodynamic_state_residual.condition_confidence_adaptive
                else None
            ),
            "thermo_last_batch_mean_confidence": (
                float(model.thermodynamic_state_residual.last_confidence.mean().detach().cpu())
                if model.thermodynamic_state_residual is not None
                and model.thermodynamic_state_residual.last_confidence is not None
                else None
            ),
            "thermo_gate_condition_attenuations": (
                model.thermo_consistency_gate.condition_attenuations().detach().cpu().tolist()
                if model.thermo_consistency_gate is not None else None
            ),
            "thermo_gate_last_batch_mean_confidence": (
                float(model.thermo_consistency_gate.last_confidence.mean().detach().cpu())
                if model.thermo_consistency_gate is not None
                and model.thermo_consistency_gate.last_confidence is not None
                else None
            ),
        },
    )
    history_path = save_loss_history(prefix, loss_history)
    summary_path = None
    if not args.skip_summary:
        summary_path = append_summary(
            final_metrics,
            best_validation_metrics or final_metrics,
            best_epoch,
            args,
            effective_conditions,
        )

    print(
        f"Final test | RMSE {final_metrics['rmse']:.3f} | MAE {final_metrics['mae']:.3f} | "
        f"NASA {final_metrics['nasa_score']:.2f}"
    )
    print(f"Saved model: {best_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved loss history: {history_path}")
    if summary_path is not None:
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
