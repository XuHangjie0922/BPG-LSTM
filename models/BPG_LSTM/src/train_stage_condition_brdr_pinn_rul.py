from pathlib import Path
import argparse
import csv
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
from train_pinn_rul import PINNRUL


FD = "FD001"
WINDOW_SIZE = 30
MAX_RUL = 125
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3


def infer_sensor_trend_signs(
    x_train: np.ndarray,
    y_train_seq: np.ndarray,
    condition_ids: np.ndarray,
    num_conditions: int,
    min_abs_corr: float = 0.15,
    top_k: int = 8,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    x_flat = x_train.reshape(-1, x_train.shape[-1])
    y_flat = y_train_seq.reshape(-1)
    condition_flat = np.repeat(np.asarray(condition_ids).reshape(-1), x_train.shape[1])
    signs = np.zeros((num_conditions, x_train.shape[-1]), dtype=np.float32)
    selected = []
    for condition_idx in range(num_conditions):
        condition_mask = condition_flat == condition_idx
        correlations = []
        for feature_idx in range(x_flat.shape[-1]):
            sensor = x_flat[condition_mask, feature_idx]
            target = y_flat[condition_mask]
            corr = float(np.corrcoef(sensor, target)[0, 1]) if len(sensor) > 2 and np.std(sensor) > 1e-8 else 0.0
            correlations.append(corr if np.isfinite(corr) else 0.0)
        candidates = [idx for idx, corr in enumerate(correlations) if abs(corr) >= min_abs_corr]
        candidates = sorted(candidates, key=lambda idx: abs(correlations[idx]), reverse=True)[:top_k]
        for idx in candidates:
            signs[condition_idx, idx] = -np.sign(correlations[idx])
            selected.append(
                {
                    "condition_idx": int(condition_idx),
                    "feature_idx": int(idx),
                    "rul_corr": float(correlations[idx]),
                    "degradation_sign": float(signs[condition_idx, idx]),
                }
            )
    return signs, selected


def build_sensor_health_bounds(
    x_train: np.ndarray,
    condition_ids: np.ndarray,
    trend_signs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    condition_flat = np.repeat(np.asarray(condition_ids).reshape(-1), x_train.shape[1])
    x_flat = x_train.reshape(-1, x_train.shape[-1])
    signed_scores = np.zeros(len(x_flat), dtype=np.float32)
    for condition_idx in range(trend_signs.shape[0]):
        mask = condition_flat == condition_idx
        if mask.any() and np.abs(trend_signs[condition_idx]).sum() > 0:
            signed_scores[mask] = (
                x_flat[mask] * trend_signs[condition_idx]
            ).sum(axis=1) / max(float(np.abs(trend_signs[condition_idx]).sum()), 1.0)
    low = np.zeros(trend_signs.shape[0], dtype=np.float32)
    high = np.ones(trend_signs.shape[0], dtype=np.float32)
    for condition_idx in range(trend_signs.shape[0]):
        mask = condition_flat == condition_idx
        if mask.any():
            values = signed_scores[mask]
            low[condition_idx], high[condition_idx] = np.percentile(values, [5.0, 95.0])
            if high[condition_idx] - low[condition_idx] < 1e-6:
                high[condition_idx] = low[condition_idx] + 1.0
    return low, high


def build_condition_sensor_baseline(
    x_train: np.ndarray, condition_ids: np.ndarray, num_conditions: int
) -> np.ndarray:
    condition_flat = np.repeat(np.asarray(condition_ids).reshape(-1), x_train.shape[1])
    x_flat = x_train.reshape(-1, x_train.shape[-1])
    baseline = np.zeros((num_conditions, x_train.shape[-1]), dtype=np.float32)
    for condition_idx in range(num_conditions):
        mask = condition_flat == condition_idx
        if mask.any():
            baseline[condition_idx] = np.mean(x_flat[mask], axis=0)
    return baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage-condition-aware BRDR-PINN-RUL on C-MAPSS.")
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
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Number of validation evaluations without improvement; 0 disables early stopping.",
    )
    parser.add_argument("--selection-metric", choices=["rmse", "nasa_score"], default="rmse")
    parser.add_argument(
        "--validation-protocol", choices=["all_windows", "censored"], default="all_windows"
    )
    parser.add_argument(
        "--validation-rul-grid", type=float, nargs="+", default=[20.0, 50.0, 80.0, 110.0, 125.0]
    )
    parser.add_argument("--num-conditions", type=int, default=6)
    parser.add_argument("--auto-num-conditions", action="store_true")
    parser.add_argument("--scaler-mode", choices=["global", "condition"], default="global")
    parser.add_argument("--middle-rul", type=float, default=100.0)
    parser.add_argument("--late-rul", type=float, default=40.0)
    parser.add_argument("--late-stage-weight", type=float, default=1.5)
    parser.add_argument("--learnable-stage-prior", action="store_true")
    parser.add_argument("--stage-prior-reg-weight", type=float, default=0.001)
    parser.add_argument("--stage-prior-range", type=float, default=0.75)
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
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--model-name", default="Stage-Condition-BRDR-PINN-RUL")
    parser.add_argument("--summary-name", default="stage_condition_brdr_pinn_rul_results.csv")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    return parser.parse_args()


def output_prefix(args: argparse.Namespace) -> str:
    if args.output_prefix:
        return args.output_prefix
    return f"stage_condition_brdr_pinn_rul_{args.fd.lower()}"


def assign_stage_ids(rul_values: np.ndarray, middle_rul: float, late_rul: float) -> np.ndarray:
    stage_ids = np.zeros_like(rul_values, dtype=np.int64)
    stage_ids[(rul_values <= middle_rul) & (rul_values > late_rul)] = 1
    stage_ids[rul_values <= late_rul] = 2
    return stage_ids


def split_train_validation(
    df: pd.DataFrame, val_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    units = np.asarray(sorted(df["unit"].unique()), dtype=np.int64)
    if len(units) < 2:
        raise ValueError("At least two training engines are required for a train/validation split.")

    rng = np.random.default_rng(seed)
    shuffled = units.copy()
    rng.shuffle(shuffled)
    val_count = min(len(units) - 1, max(1, int(round(len(units) * val_ratio))))
    val_units = sorted(int(unit) for unit in shuffled[:val_count])
    train_units = sorted(int(unit) for unit in shuffled[val_count:])
    train_df = df[df["unit"].isin(train_units)].copy()
    val_df = df[df["unit"].isin(val_units)].copy()
    return train_df, val_df, train_units, val_units


def add_condition_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_conditions: int,
    seed: int,
    auto_num_conditions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    setting_cols = ["setting_1", "setting_2", "setting_3"]
    condition_scaler = StandardScaler()
    train_settings = condition_scaler.fit_transform(train_df[setting_cols])
    test_settings = condition_scaler.transform(test_df[setting_cols])

    rounded = np.round(train_settings, decimals=3)
    unique_count = len(np.unique(rounded, axis=0))
    max_conditions = max(1, min(num_conditions, unique_count, len(train_df)))
    effective_conditions = max_conditions
    if auto_num_conditions and max_conditions > 1:
        sample_size = min(len(train_settings), 10000)
        sample_idx = np.linspace(0, len(train_settings) - 1, sample_size, dtype=int)
        sampled_settings = train_settings[sample_idx]
        scores: dict[int, float] = {}
        for candidate in range(2, max_conditions + 1):
            labels = KMeans(n_clusters=candidate, random_state=seed, n_init=10).fit_predict(
                sampled_settings
            )
            if len(np.unique(labels)) > 1:
                scores[candidate] = float(silhouette_score(sampled_settings, labels))
        if scores:
            effective_conditions = max(scores, key=scores.get)

    if effective_conditions == 1:
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["condition_id"] = 0
        test_df["condition_id"] = 0
        return train_df, test_df, effective_conditions

    kmeans = KMeans(n_clusters=effective_conditions, random_state=seed, n_init=10)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["condition_id"] = kmeans.fit_predict(train_settings)
    test_df["condition_id"] = kmeans.predict(test_settings)
    return train_df, test_df, effective_conditions


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
        scaler_info["global"] = {
            "train_rows": float(len(train_df)),
            "test_rows": float(len(test_df)),
        }
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


def remove_condition_sensor_baseline(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    sensor_cols: list[str],
    baseline_rul: float,
    num_conditions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Subtract an early-life sensor baseline learned only from training engines."""
    train_df = train_df.copy()
    evaluation_df = evaluation_df.copy()
    # Residualization can produce non-integer values even when raw sensors are integer-typed.
    train_df[sensor_cols] = train_df[sensor_cols].astype(np.float32)
    evaluation_df[sensor_cols] = evaluation_df[sensor_cols].astype(np.float32)
    baseline_info: dict[str, dict[str, float]] = {}
    for condition_idx in range(num_conditions):
        condition_mask = train_df["condition_id"] == condition_idx
        healthy_mask = condition_mask & (train_df["RUL"] >= baseline_rul)
        reference = train_df.loc[healthy_mask, sensor_cols]
        if reference.empty:
            reference = train_df.loc[condition_mask, sensor_cols]
        baseline = (
            reference.mean(axis=0).to_numpy(dtype=np.float32)
            if not reference.empty
            else np.zeros(len(sensor_cols), dtype=np.float32)
        )
        train_mask = train_df["condition_id"] == condition_idx
        eval_mask = evaluation_df["condition_id"] == condition_idx
        if train_mask.any():
            train_df.loc[train_mask, sensor_cols] = train_df.loc[train_mask, sensor_cols] - baseline
        if eval_mask.any():
            evaluation_df.loc[eval_mask, sensor_cols] = evaluation_df.loc[eval_mask, sensor_cols] - baseline
        baseline_info[f"condition_{condition_idx}"] = {
            "reference_rows": float(len(reference)),
            "baseline_rul": float(baseline_rul),
        }
    return train_df, evaluation_df, baseline_info


def make_train_condition_windows(
    df: pd.DataFrame, feature_cols: list[str], window_size: int, middle_rul: float, late_rul: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, condition_ids, stage_ids = [], [], [], []
    for _, sub in df.groupby("unit"):
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
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(condition_ids, dtype=np.int64),
        np.asarray(stage_ids, dtype=np.int64),
    )


def make_test_condition_windows(
    df: pd.DataFrame, feature_cols: list[str], window_size: int, middle_rul: float, late_rul: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, condition_ids, stage_ids = [], [], [], []
    for _, sub in df.groupby("unit"):
        sub = sub.sort_values("cycle")
        x = sub[feature_cols].to_numpy(dtype=np.float32)
        y = sub["RUL"].to_numpy(dtype=np.float32)
        c = sub["condition_id"].to_numpy(dtype=np.int64)
        s = assign_stage_ids(y, middle_rul, late_rul)
        if len(sub) >= window_size:
            xs.append(x[-window_size:])
            ys.append(y[-window_size:])
        else:
            pad_x = np.repeat(x[[0]], window_size - len(sub), axis=0)
            pad_y = np.repeat(y[[0]], window_size - len(sub), axis=0)
            xs.append(np.concatenate([pad_x, x], axis=0))
            ys.append(np.concatenate([pad_y, y], axis=0))
        condition_ids.append(c[-1])
        stage_ids.append(s[-1])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(condition_ids, dtype=np.int64),
        np.asarray(stage_ids, dtype=np.int64),
    )


def make_censored_validation_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    middle_rul: float,
    late_rul: float,
    rul_grid: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, condition_ids, stage_ids = [], [], [], []
    for _, sub in df.groupby("unit"):
        sub = sub.sort_values("cycle")
        x = sub[feature_cols].to_numpy(dtype=np.float32)
        y = sub["RUL"].to_numpy(dtype=np.float32)
        c = sub["condition_id"].to_numpy(dtype=np.int64)
        selected_indices = []
        for target_rul in rul_grid:
            distance = np.abs(y - target_rul)
            candidates = np.flatnonzero(distance == distance.min())
            index = int(candidates[-1])
            if index not in selected_indices:
                selected_indices.append(index)
        for index in sorted(selected_indices):
            start = max(0, index - window_size + 1)
            x_window = x[start : index + 1]
            y_window = y[start : index + 1]
            if len(x_window) < window_size:
                pad = window_size - len(x_window)
                x_window = np.concatenate([np.repeat(x_window[[0]], pad, axis=0), x_window], axis=0)
                y_window = np.concatenate([np.repeat(y_window[[0]], pad), y_window], axis=0)
            xs.append(x_window)
            ys.append(y_window)
            condition_ids.append(c[index])
            stage_ids.append(assign_stage_ids(np.asarray([y[index]]), middle_rul, late_rul)[0])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(condition_ids, dtype=np.int64),
        np.asarray(stage_ids, dtype=np.int64),
    )


class StageConditionRULSequenceDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_seq: np.ndarray,
        condition_ids: np.ndarray,
        stage_ids: np.ndarray,
    ) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)
        self.condition_ids = torch.tensor(condition_ids, dtype=torch.long)
        self.stage_ids = torch.tensor(stage_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y_seq[idx], self.condition_ids[idx], self.stage_ids[idx]


class HealthAwarePINNRUL(PINNRUL):
    """LSTM RUL model with an explicit normalized health-state head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.health_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.lstm(x)
        return self.head(hidden).squeeze(-1), self.health_head(hidden).squeeze(-1)


def predict(
    model: nn.Module, loader: DataLoader, device: str, pin_memory: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, trues, conditions, stages = [], [], [], []
    with torch.no_grad():
        for xb, yb, cb, sb in loader:
            xb = xb.to(device, non_blocking=pin_memory)
            model_output = model(xb)
            pred_seq = model_output[0] if isinstance(model_output, tuple) else model_output
            pred_seq = pred_seq.cpu().numpy()
            preds.extend(pred_seq[:, -1])
            trues.extend(yb.numpy()[:, -1])
            conditions.extend(cb.numpy())
            stages.extend(sb.numpy())
    preds = np.maximum(np.asarray(preds), 0.0)
    return (
        np.asarray(trues, dtype=np.float32),
        preds.astype(np.float32),
        np.asarray(conditions, dtype=np.int64),
        np.asarray(stages, dtype=np.int64),
    )


def save_predictions(
    prefix: str,
    fd: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    condition_ids: np.ndarray,
    stage_ids: np.ndarray,
) -> Path:
    result_df = pd.DataFrame(
        {
            "unit": np.arange(1, len(y_true) + 1),
            "condition_id": condition_ids,
            "stage_id": stage_ids,
            "true_rul": y_true,
            "pred_rul": y_pred,
            "error": y_pred - y_true,
            "abs_error": np.abs(y_pred - y_true),
        }
    )
    path = OUTPUT_DIR / f"{prefix}_predictions.csv"
    result_df.to_csv(path, index=False)
    return path


def save_loss_history(prefix: str, rows: list[dict[str, float]]) -> Path:
    path = OUTPUT_DIR / f"{prefix}_loss_history.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_metrics(prefix: str, fd: str, metrics: dict[str, float], args: argparse.Namespace, extra: dict) -> Path:
    payload = {
        "model": args.model_name,
        "fd": fd,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "nasa_score": metrics["nasa_score"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "max_rul": args.max_rul,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "lr": args.lr,
        "val_ratio": args.val_ratio,
        "eval_every": args.eval_every,
        "early_stopping_patience": args.early_stopping_patience,
        "selection_metric": args.selection_metric,
        "num_conditions": args.num_conditions,
        "auto_num_conditions": args.auto_num_conditions,
        "scaler_mode": args.scaler_mode,
        "middle_rul": args.middle_rul,
        "late_rul": args.late_rul,
        "late_stage_weight": args.late_stage_weight,
        "learnable_stage_prior": args.learnable_stage_prior,
        "stage_prior_reg_weight": args.stage_prior_reg_weight,
        "stage_prior_range": args.stage_prior_range,
        "mono_weight": args.mono_weight,
        "smooth_weight": args.smooth_weight,
        "rul_dynamics_weight": args.rul_dynamics_weight,
        "rul_dynamics_threshold": args.rul_dynamics_threshold,
        "sensor_trend_weight": args.sensor_trend_weight,
        "sensor_trend_min_corr": args.sensor_trend_min_corr,
        "sensor_trend_top_k": args.sensor_trend_top_k,
        "sensor_trend_min_stage": args.sensor_trend_min_stage,
        "sensor_trend_huber_delta": args.sensor_trend_huber_delta,
        "condition_residual": args.condition_residual,
        "condition_residual_baseline_rul": args.condition_residual_baseline_rul,
        "health_weight": args.health_weight,
        "health_mono_weight": args.health_mono_weight,
        "failure_weight": args.failure_weight,
        "failure_threshold": args.failure_threshold,
        "asym_weight": args.asym_weight,
        "asym_over_alpha": args.asym_over_alpha,
        "asym_under_beta": args.asym_under_beta,
        "asym_clip": args.asym_clip,
        "over_margin_weight": args.over_margin_weight,
        "over_margin": args.over_margin,
        "beta_c": args.beta_c,
        "beta_w": args.beta_w,
        "brdr_min_weight": args.brdr_min_weight,
        "brdr_max_weight": args.brdr_max_weight,
        "brdr_weight_power": args.brdr_weight_power,
        "brdr_warmup_steps": args.brdr_warmup_steps,
        "seed": args.seed,
        "split_seed": args.split_seed,
        **extra,
    }
    path = OUTPUT_DIR / f"{prefix}_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def append_summary(
    metrics: dict[str, float],
    validation_metrics: dict[str, float],
    best_epoch: int,
    args: argparse.Namespace,
    effective_conditions: int,
) -> Path:
    path = RESULTS_DIR / args.summary_name
    row = {
        "model": args.model_name,
        "fd": args.fd,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "nasa_score": metrics["nasa_score"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_nasa_score": validation_metrics["nasa_score"],
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "max_rul": args.max_rul,
        "val_ratio": args.val_ratio,
        "selection_metric": args.selection_metric,
        "num_conditions": effective_conditions,
        "scaler_mode": args.scaler_mode,
        "middle_rul": args.middle_rul,
        "late_rul": args.late_rul,
        "late_stage_weight": args.late_stage_weight,
        "learnable_stage_prior": args.learnable_stage_prior,
        "stage_prior_reg_weight": args.stage_prior_reg_weight,
        "stage_prior_range": args.stage_prior_range,
        "mono_weight": args.mono_weight,
        "smooth_weight": args.smooth_weight,
        "rul_dynamics_weight": args.rul_dynamics_weight,
        "rul_dynamics_threshold": args.rul_dynamics_threshold,
        "sensor_trend_weight": args.sensor_trend_weight,
        "sensor_trend_min_stage": args.sensor_trend_min_stage,
        "sensor_trend_huber_delta": args.sensor_trend_huber_delta,
        "failure_weight": args.failure_weight,
        "asym_weight": args.asym_weight,
        "over_margin_weight": args.over_margin_weight,
        "over_margin": args.over_margin,
        "beta_c": args.beta_c,
        "beta_w": args.beta_w,
        "brdr_min_weight": args.brdr_min_weight,
        "brdr_max_weight": args.brdr_max_weight,
        "brdr_weight_power": args.brdr_weight_power,
        "brdr_warmup_steps": args.brdr_warmup_steps,
        "seed": args.seed,
        "split_seed": args.split_seed,
    }
    rows = []
    if path.exists():
        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
        rows = [old for old in rows if not (old["model"] == row["model"] and old["fd"] == row["fd"])]
    rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    args = parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.eval_every < 1:
        raise ValueError(f"eval_every must be at least 1, got {args.eval_every}")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience cannot be negative")
    prefix = output_prefix(args)
    set_seed(args.seed)
    ensure_fd_files(args.fd)

    device = configure_device(args.no_cuda)
    pin_memory = device == "cuda"
    use_amp = bool(args.amp and device == "cuda")
    print(f"Device: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.1f} GB)")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"AMP: {'on' if use_amp else 'off'}")

    full_train_df = load_train(args.fd, args.max_rul)
    test_df = load_test(args.fd, args.max_rul)
    train_df, val_df, train_units, val_units = split_train_validation(
        full_train_df, args.val_ratio, args.split_seed
    )
    evaluation_df = pd.concat(
        [
            val_df.assign(_evaluation_split="validation"),
            test_df.assign(_evaluation_split="test"),
        ],
        ignore_index=True,
    )
    train_df, evaluation_df, effective_conditions = add_condition_labels(
        train_df, evaluation_df, args.num_conditions, args.split_seed, args.auto_num_conditions
    )
    print(f"Effective operating-condition clusters: {effective_conditions}")
    print(
        f"Engine split: train={len(train_units)}, validation={len(val_units)}, "
        f"test={test_df['unit'].nunique()} (split_seed={args.split_seed}, model_seed={args.seed})"
    )

    feature_cols = [c for c in train_df.columns if c not in DROP_COLS + ["RUL", "condition_id"]]
    residual_info = {}
    if args.condition_residual:
        sensor_cols = [c for c in feature_cols if c.startswith("s")]
        train_df, evaluation_df, residual_info = remove_condition_sensor_baseline(
            train_df,
            evaluation_df,
            sensor_cols,
            args.condition_residual_baseline_rul,
            effective_conditions,
        )
        print(f"Condition residual baseline enabled on {len(sensor_cols)} sensors")
    train_df, evaluation_df, scaler_info = scale_features(
        train_df,
        evaluation_df,
        feature_cols,
        args.scaler_mode,
        effective_conditions,
    )
    val_df = evaluation_df[evaluation_df["_evaluation_split"] == "validation"].drop(
        columns=["_evaluation_split"]
    )
    test_df = evaluation_df[evaluation_df["_evaluation_split"] == "test"].drop(
        columns=["_evaluation_split"]
    )
    print(f"Feature scaler mode: {args.scaler_mode}")

    x_train, y_train_seq, c_train, s_train = make_train_condition_windows(
        train_df, feature_cols, args.window_size, args.middle_rul, args.late_rul
    )
    if args.validation_protocol == "censored":
        x_val, y_val_seq, c_val, s_val = make_censored_validation_windows(
            val_df,
            feature_cols,
            args.window_size,
            args.middle_rul,
            args.late_rul,
            args.validation_rul_grid,
        )
    else:
        x_val, y_val_seq, c_val, s_val = make_train_condition_windows(
            val_df, feature_cols, args.window_size, args.middle_rul, args.late_rul
        )
    x_test, y_test_seq, c_test, s_test = make_test_condition_windows(
        test_df, feature_cols, args.window_size, args.middle_rul, args.late_rul
    )
    sensor_trend_signs, selected_sensor_trends = infer_sensor_trend_signs(
        x_train,
        y_train_seq,
        c_train,
        effective_conditions,
        min_abs_corr=args.sensor_trend_min_corr,
        top_k=args.sensor_trend_top_k,
    )
    print(f"Selected sensor trend features: {selected_sensor_trends}")
    condition_sensor_baseline = build_condition_sensor_baseline(
        x_train, c_train, effective_conditions
    )
    x_train_residual = x_train - condition_sensor_baseline[c_train][:, None, :]
    health_low, health_high = build_sensor_health_bounds(
        x_train_residual, c_train, sensor_trend_signs
    )
    train_counts = np.bincount(c_train, minlength=effective_conditions).tolist()
    val_counts = np.bincount(c_val, minlength=effective_conditions).tolist()
    test_counts = np.bincount(c_test, minlength=effective_conditions).tolist()
    train_stage_counts = np.bincount(s_train, minlength=3).tolist()
    val_stage_counts = np.bincount(s_val, minlength=3).tolist()
    test_stage_counts = np.bincount(s_test, minlength=3).tolist()
    print(f"Train windows: {x_train.shape}, Validation windows: {x_val.shape}, Test engines: {x_test.shape}")
    print(f"Condition train counts: {train_counts}")
    print(f"Condition validation counts: {val_counts}")
    print(f"Condition test counts: {test_counts}")
    print(f"Stage train counts: {train_stage_counts} (early/middle/late)")
    print(f"Stage validation counts: {val_stage_counts} (early/middle/late)")
    print(f"Stage test counts: {test_stage_counts} (early/middle/late)")
    print(f"Features: {len(feature_cols)} -> {feature_cols}")

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        StageConditionRULSequenceDataset(x_train, y_train_seq, c_train, s_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
        generator=train_generator,
    )
    val_loader = DataLoader(
        StageConditionRULSequenceDataset(x_val, y_val_seq, c_val, s_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        StageConditionRULSequenceDataset(x_test, y_test_seq, c_test, s_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = HealthAwarePINNRUL(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    loss_fn = StageConditionAwareBRDRPINNLoss(
        num_conditions=effective_conditions,
        max_rul=args.max_rul,
        mono_weight=args.mono_weight,
        smooth_weight=args.smooth_weight,
        rul_dynamics_weight=args.rul_dynamics_weight,
        rul_dynamics_threshold=args.rul_dynamics_threshold,
        sensor_trend_weight=args.sensor_trend_weight,
        sensor_trend_signs=torch.tensor(sensor_trend_signs, dtype=torch.float32),
        sensor_trend_min_stage=args.sensor_trend_min_stage,
        sensor_trend_huber_delta=args.sensor_trend_huber_delta,
        health_weight=args.health_weight,
        health_mono_weight=args.health_mono_weight,
        health_low=torch.tensor(health_low, dtype=torch.float32),
        health_high=torch.tensor(health_high, dtype=torch.float32),
        condition_sensor_baseline=torch.tensor(condition_sensor_baseline, dtype=torch.float32),
        middle_rul=args.middle_rul,
        late_rul=args.late_rul,
        failure_weight=args.failure_weight,
        asym_weight=args.asym_weight,
        asym_over_alpha=args.asym_over_alpha,
        asym_under_beta=args.asym_under_beta,
        asym_clip=args.asym_clip,
        over_margin_weight=args.over_margin_weight,
        over_margin=args.over_margin,
        late_stage_weight=args.late_stage_weight,
        learnable_stage_prior=args.learnable_stage_prior,
        stage_prior_reg_weight=args.stage_prior_reg_weight,
        stage_prior_range=args.stage_prior_range,
        failure_threshold=args.failure_threshold,
        beta_c=args.beta_c,
        beta_w=args.beta_w,
        brdr_min_weight=args.brdr_min_weight,
        brdr_max_weight=args.brdr_max_weight,
        brdr_weight_power=args.brdr_weight_power,
        brdr_warmup_steps=args.brdr_warmup_steps,
    ).to(device)
    trainable_params = list(model.parameters()) + [
        param for param in loss_fn.parameters() if param.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_selection_value = float("inf")
    best_validation_metrics: dict[str, float] | None = None
    best_epoch = 0
    evaluations_without_improvement = 0
    best_path = OUTPUT_DIR / f"best_{prefix}.pt"
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {
            "total_loss": 0.0,
            "rul_loss": 0.0,
            "raw_rul_loss": 0.0,
            "mono_loss": 0.0,
            "smooth_loss": 0.0,
            "rul_dynamics_loss": 0.0,
            "sensor_trend_loss": 0.0,
            "health_loss": 0.0,
            "health_mono_loss": 0.0,
            "failure_loss": 0.0,
            "asym_loss": 0.0,
            "over_margin_loss": 0.0,
            "stage_prior_reg": 0.0,
            "w_rul_loss": 0.0,
            "w_mono_loss": 0.0,
            "w_smooth_loss": 0.0,
            "w_rul_dynamics_loss": 0.0,
            "w_sensor_trend_loss": 0.0,
            "w_health_loss": 0.0,
            "w_health_mono_loss": 0.0,
            "w_failure_loss": 0.0,
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

        for xb, yb, cb, sb in train_loader:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)
            cb = cb.to(device, non_blocking=pin_memory)
            sb = sb.to(device, non_blocking=pin_memory)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_seq, health_seq = model(xb)
                loss, parts = loss_fn(pred_seq, yb, cb, sb, xb, health_seq)

            optimizer.zero_grad()
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            batch_n = len(xb)
            seen += batch_n
            for key in sums:
                sums[key] += parts[key] * batch_n

        epoch_parts = {key: value / seen for key, value in sums.items()}

        if epoch == 1 or epoch % args.eval_every == 0:
            y_true, y_pred, condition_ids, stage_ids = predict(model, val_loader, device, pin_memory)
            validation_metrics = regression_metrics(y_true, y_pred)
            row = {
                "epoch": epoch,
                **epoch_parts,
                "val_rmse": validation_metrics["rmse"],
                "val_mae": validation_metrics["mae"],
                "val_nasa_score": validation_metrics["nasa_score"],
            }
            loss_history.append(row)
            condition_weight_text = "/".join(
                f"{epoch_parts[f'w_condition_{idx}']:.2f}" for idx in range(effective_conditions)
            )
            stage_weight_text = "/".join(
                f"{epoch_parts[f'w_stage_{idx}']:.2f}" for idx in range(3)
            )
            stage_prior_text = "/".join(
                f"{epoch_parts[f'prior_stage_{idx}']:.2f}" for idx in range(3)
            )
            print(
                f"Epoch {epoch:03d} | "
                f"Total {epoch_parts['total_loss']:.4f} | "
                f"RUL {epoch_parts['rul_loss']:.4f} | "
                f"RawRUL {epoch_parts['raw_rul_loss']:.4f} | "
                f"Mono {epoch_parts['mono_loss']:.4f} | "
                f"Smooth {epoch_parts['smooth_loss']:.4f} | "
                f"Dyn {epoch_parts['rul_dynamics_loss']:.4f} | "
                f"Failure {epoch_parts['failure_loss']:.4f} | "
                f"Asym {epoch_parts['asym_loss']:.4f} | "
                f"Over {epoch_parts['over_margin_loss']:.4f} | "
                f"Wphys={epoch_parts['w_mono_loss']:.2f}/"
                f"{epoch_parts['w_smooth_loss']:.2f}/"
                f"{epoch_parts['w_failure_loss']:.2f} | "
                f"Wcond={condition_weight_text} | "
                f"Wstage={stage_weight_text} | "
                f"Prior={stage_prior_text} | "
                f"Val RMSE {validation_metrics['rmse']:.3f} | "
                f"Val MAE {validation_metrics['mae']:.3f} | "
                f"Val NASA {validation_metrics['nasa_score']:.2f}"
            )

            selection_value = validation_metrics[args.selection_metric]
            if selection_value < best_selection_value:
                best_selection_value = selection_value
                best_validation_metrics = validation_metrics
                best_epoch = epoch
                evaluations_without_improvement = 0
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
                        "effective_conditions": effective_conditions,
                        "physics_brdr_weights": dict(loss_fn.physics_balancer.weights),
                        "condition_stage_brdr_weights": dict(loss_fn.condition_stage_balancer.weights),
                        "physics_brdr_state": loss_fn.physics_balancer.state_dict(),
                        "condition_stage_brdr_state": loss_fn.condition_stage_balancer.state_dict(),
                        "stage_prior": [
                            float(value)
                            for value in loss_fn.stage_prior(torch.device(device)).detach().cpu().tolist()
                        ],
                        "device": device,
                        "amp": use_amp,
                    },
                    best_path,
                )
            else:
                evaluations_without_improvement += 1

            if (
                args.early_stopping_patience > 0
                and evaluations_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}; best validation "
                    f"{args.selection_metric}={best_selection_value:.3f} at epoch {best_epoch}."
                )
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn.load_state_dict(checkpoint["loss_state_dict"])
    if "physics_brdr_state" in checkpoint:
        loss_fn.physics_balancer.load_state_dict(checkpoint["physics_brdr_state"])
    if "condition_stage_brdr_state" in checkpoint:
        loss_fn.condition_stage_balancer.load_state_dict(checkpoint["condition_stage_brdr_state"])
    y_true, y_pred, condition_ids, stage_ids = predict(model, test_loader, device, pin_memory)
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
            "selected_sensor_trends": selected_sensor_trends,
            "scaler_info": scaler_info,
            "condition_residual_info": residual_info,
            "train_windows": int(len(x_train)),
            "validation_windows": int(len(x_val)),
            "validation_units_count": int(len(val_units)),
            "test_units": int(len(x_test)),
            "best_epoch": best_epoch,
            "best_validation_metrics": best_validation_metrics,
            "best_epoch_metrics": best_validation_metrics,
            "final_test_metrics": final_metrics,
            "final_physics_brdr_weights": dict(loss_fn.physics_balancer.weights),
            "final_condition_stage_brdr_weights": dict(loss_fn.condition_stage_balancer.weights),
            "final_stage_prior": [
                float(value)
                for value in loss_fn.stage_prior(torch.device(device)).detach().cpu().tolist()
            ],
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
        f"Best validation {args.selection_metric}: {best_selection_value:.3f} "
        f"(epoch {best_epoch})"
    )
    print(
        f"Final test | RMSE {final_metrics['rmse']:.3f} | "
        f"MAE {final_metrics['mae']:.3f} | NASA {final_metrics['nasa_score']:.2f}"
    )
    print(f"Saved model: {best_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved loss history: {history_path}")
    if summary_path is not None:
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
