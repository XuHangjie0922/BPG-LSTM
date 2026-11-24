from pathlib import Path
import argparse
import csv
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from losses import FixedPINNLoss
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


FD = "FD001"
WINDOW_SIZE = 30
MAX_RUL = 125
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fixed-weight PINN-RUL baseline on C-MAPSS.")
    parser.add_argument("--fd", default=FD)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--max-rul", type=int, default=MAX_RUL)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mono-weight", type=float, default=0.1)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--failure-weight", type=float, default=0.1)
    parser.add_argument("--failure-threshold", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def make_train_sequence_windows(
    df: pd.DataFrame, feature_cols: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, sub in df.groupby("unit"):
        sub = sub.sort_values("cycle")
        x = sub[feature_cols].to_numpy(dtype=np.float32)
        y = sub["RUL"].to_numpy(dtype=np.float32)
        if len(sub) < window_size:
            continue
        for i in range(window_size - 1, len(sub)):
            xs.append(x[i - window_size + 1 : i + 1])
            ys.append(y[i - window_size + 1 : i + 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def make_test_last_sequence_windows(
    df: pd.DataFrame, feature_cols: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, sub in df.groupby("unit"):
        sub = sub.sort_values("cycle")
        x = sub[feature_cols].to_numpy(dtype=np.float32)
        y = sub["RUL"].to_numpy(dtype=np.float32)
        if len(sub) >= window_size:
            xs.append(x[-window_size:])
            ys.append(y[-window_size:])
        else:
            pad_x = np.repeat(x[[0]], window_size - len(sub), axis=0)
            pad_y = np.repeat(y[[0]], window_size - len(sub), axis=0)
            xs.append(np.concatenate([pad_x, x], axis=0))
            ys.append(np.concatenate([pad_y, y], axis=0))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class RULSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y_seq: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y_seq[idx]


class PINNRUL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


def predict(model: nn.Module, loader: DataLoader, device: str, pin_memory: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=pin_memory)
            pred_seq = model(xb).cpu().numpy()
            preds.extend(pred_seq[:, -1])
            trues.extend(yb.numpy()[:, -1])
    preds = np.maximum(np.asarray(preds), 0.0)
    return np.asarray(trues, dtype=np.float32), preds.astype(np.float32)


def save_predictions(fd: str, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    result_df = pd.DataFrame(
        {
            "unit": np.arange(1, len(y_true) + 1),
            "true_rul": y_true,
            "pred_rul": y_pred,
            "error": y_pred - y_true,
            "abs_error": np.abs(y_pred - y_true),
        }
    )
    path = OUTPUT_DIR / f"pinn_rul_{fd.lower()}_predictions.csv"
    result_df.to_csv(path, index=False)
    return path


def save_loss_history(fd: str, rows: list[dict[str, float]]) -> Path:
    path = OUTPUT_DIR / f"pinn_rul_{fd.lower()}_loss_history.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_metrics(fd: str, metrics: dict[str, float], args: argparse.Namespace, extra: dict) -> Path:
    payload = {
        "model": "PINN-RUL",
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
        "mono_weight": args.mono_weight,
        "smooth_weight": args.smooth_weight,
        "failure_weight": args.failure_weight,
        "failure_threshold": args.failure_threshold,
        **extra,
    }
    path = OUTPUT_DIR / f"pinn_rul_{fd.lower()}_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def append_summary(metrics: dict[str, float], args: argparse.Namespace) -> Path:
    path = RESULTS_DIR / "pinn_rul_results.csv"
    row = {
        "model": "PINN-RUL",
        "fd": args.fd,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "nasa_score": metrics["nasa_score"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "max_rul": args.max_rul,
        "mono_weight": args.mono_weight,
        "smooth_weight": args.smooth_weight,
        "failure_weight": args.failure_weight,
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
    set_seed(42)
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

    train_df = load_train(args.fd, args.max_rul)
    test_df = load_test(args.fd, args.max_rul)
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS + ["RUL"]]

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    x_train, y_train_seq = make_train_sequence_windows(train_df, feature_cols, args.window_size)
    x_test, y_test_seq = make_test_last_sequence_windows(test_df, feature_cols, args.window_size)
    print(f"Train windows: {x_train.shape}, Test windows: {x_test.shape}")
    print(f"Features: {len(feature_cols)} -> {feature_cols}")

    train_loader = DataLoader(
        RULSequenceDataset(x_train, y_train_seq),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        RULSequenceDataset(x_test, y_test_seq),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = PINNRUL(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = FixedPINNLoss(
        mono_weight=args.mono_weight,
        smooth_weight=args.smooth_weight,
        failure_weight=args.failure_weight,
        failure_threshold=args.failure_threshold,
    )
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_rmse = float("inf")
    best_metrics: dict[str, float] | None = None
    best_path = OUTPUT_DIR / f"best_pinn_rul_{args.fd.lower()}.pt"
    loss_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"total_loss": 0.0, "rul_loss": 0.0, "mono_loss": 0.0, "smooth_loss": 0.0, "failure_loss": 0.0}
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_seq = model(xb)
                loss, parts = loss_fn(pred_seq, yb)

            optimizer.zero_grad()
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            batch_n = len(xb)
            seen += batch_n
            for key in sums:
                sums[key] += parts[key] * batch_n

        epoch_parts = {key: value / seen for key, value in sums.items()}

        if epoch == 1 or epoch % 5 == 0:
            y_true, y_pred = predict(model, test_loader, device, pin_memory)
            metrics = regression_metrics(y_true, y_pred)
            row = {"epoch": epoch, **epoch_parts, **metrics}
            loss_history.append(row)
            print(
                f"Epoch {epoch:03d} | "
                f"Total {epoch_parts['total_loss']:.4f} | "
                f"RUL {epoch_parts['rul_loss']:.4f} | "
                f"Mono {epoch_parts['mono_loss']:.4f} | "
                f"Smooth {epoch_parts['smooth_loss']:.4f} | "
                f"Failure {epoch_parts['failure_loss']:.4f} | "
                f"RMSE {metrics['rmse']:.3f} | "
                f"MAE {metrics['mae']:.3f} | "
                f"NASA {metrics['nasa_score']:.2f}"
            )

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_metrics = metrics
                save_torch_checkpoint(
                    {
                        "model_state_dict": model.state_dict(),
                        "feature_cols": feature_cols,
                        "window_size": args.window_size,
                        "max_rul": args.max_rul,
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "nasa_score": metrics["nasa_score"],
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "loss_weights": {
                            "mono": args.mono_weight,
                            "smooth": args.smooth_weight,
                            "failure": args.failure_weight,
                        },
                        "device": device,
                        "amp": use_amp,
                    },
                    best_path,
                )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = predict(model, test_loader, device, pin_memory)
    final_metrics = regression_metrics(y_true, y_pred)
    predictions_path = save_predictions(args.fd, y_true, y_pred)
    metrics_path = save_metrics(
        args.fd,
        final_metrics,
        args,
        {
            "device": device,
            "amp": use_amp,
            "train_windows": int(len(x_train)),
            "test_units": int(len(x_test)),
            "best_epoch_metrics": best_metrics or final_metrics,
        },
    )
    history_path = save_loss_history(args.fd, loss_history)
    summary_path = append_summary(final_metrics, args)

    print(f"Best RMSE: {best_rmse:.3f}")
    print(f"Saved model: {best_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved loss history: {history_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
