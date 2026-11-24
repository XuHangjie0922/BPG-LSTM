from pathlib import Path
import argparse
import io
import json
import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from metrics import regression_metrics


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("CMAPSS_DATA_DIR", ROOT.parents[1] / "data" / "CMAPSS"))
OUTPUT_DIR = Path(os.environ.get("RUL_PINN_OUTPUT_DIR", str(ROOT / "outputs")))
RESULTS_DIR = Path(os.environ.get("RUL_PINN_RESULTS_DIR", str(ROOT / "results")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_torch_checkpoint(payload: dict, path: Path) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())

FD = "FD001"
WINDOW_SIZE = 30
MAX_RUL = 125
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = DEVICE == "cuda"

COLS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Preserve the promoted profile by default; the terminal pure-LSTM arm can opt
# out of the hand-selected sensor pruning without editing the frozen source.
MANUAL_SENSOR_DROPS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
DROP_COLS = ["unit", "cycle"] + (
    [] if os.environ.get("RUL_NEUTRAL_PREPROCESS") == "1" else MANUAL_SENSOR_DROPS
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM RUL baseline on C-MAPSS.")
    parser.add_argument("--fd", default=FD, help="Dataset subset, e.g. FD001, FD002, FD003, FD004.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--max-rul", type=int, default=MAX_RUL)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def configure_device(no_cuda: bool) -> str:
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return device


def read_cmapss_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Please put train_{FD}.txt, test_{FD}.txt, and RUL_{FD}.txt in data/raw/."
        )
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df.columns = COLS
    return df


def ensure_fd_files(fd: str) -> None:
    missing = [
        RAW_DIR / f"train_{fd}.txt",
        RAW_DIR / f"test_{fd}.txt",
        RAW_DIR / f"RUL_{fd}.txt",
    ]
    missing = [path for path in missing if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required C-MAPSS files:\n{missing_text}")


def load_train(fd: str, max_rul: int) -> pd.DataFrame:
    df = read_cmapss_table(RAW_DIR / f"train_{fd}.txt")
    max_cycle = df.groupby("unit", as_index=False)["cycle"].max()
    max_cycle.columns = ["unit", "max_cycle"]
    df = df.merge(max_cycle, on="unit", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df.drop(columns=["max_cycle"])


def load_test(fd: str, max_rul: int) -> pd.DataFrame:
    df = read_cmapss_table(RAW_DIR / f"test_{fd}.txt")
    rul_path = RAW_DIR / f"RUL_{fd}.txt"
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing file: {rul_path}")

    rul = pd.read_csv(rul_path, sep=r"\s+", header=None)
    rul.columns = ["final_RUL"]
    rul["unit"] = np.arange(1, len(rul) + 1)

    max_cycle = df.groupby("unit", as_index=False)["cycle"].max()
    max_cycle.columns = ["unit", "max_cycle"]

    df = df.merge(max_cycle, on="unit", how="left")
    df = df.merge(rul, on="unit", how="left")
    df["RUL"] = df["final_RUL"] + df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df.drop(columns=["max_cycle", "final_RUL"])


def make_train_windows(
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
            ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def make_test_last_windows(
    df: pd.DataFrame, feature_cols: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, sub in df.groupby("unit"):
        sub = sub.sort_values("cycle")
        x = sub[feature_cols].to_numpy(dtype=np.float32)
        y = sub["RUL"].to_numpy(dtype=np.float32)
        if len(sub) >= window_size:
            xs.append(x[-window_size:])
        else:
            pad = np.repeat(x[[0]], window_size - len(sub), axis=0)
            xs.append(np.concatenate([pad, x], axis=0))
        ys.append(y[-1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class RULDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class LSTMRUL(nn.Module):
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
        return self.head(out[:, -1, :])


def predict(model: nn.Module, loader: DataLoader, device: str, pin_memory: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=pin_memory)
            pred = model(xb).cpu().numpy().reshape(-1)
            preds.extend(pred)
            trues.extend(yb.numpy().reshape(-1))
    preds = np.maximum(np.asarray(preds), 0.0)
    return np.asarray(trues, dtype=np.float32), preds.astype(np.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: str, pin_memory: bool) -> dict[str, float]:
    y_true, y_pred = predict(model, loader, device, pin_memory)
    return regression_metrics(y_true, y_pred)


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
    path = OUTPUT_DIR / f"lstm_{fd.lower()}_predictions.csv"
    result_df.to_csv(path, index=False)
    return path


def save_metrics(fd: str, metrics: dict[str, float], args: argparse.Namespace, extra: dict) -> Path:
    payload = {
        "model": "LSTM",
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
        **extra,
    }
    path = OUTPUT_DIR / f"lstm_{fd.lower()}_metrics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    device = configure_device(args.no_cuda)
    pin_memory = device == "cuda"
    use_amp = bool(args.amp and device == "cuda")

    set_seed(SEED)
    ensure_fd_files(args.fd)
    print(f"Device: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB)")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"AMP: {'on' if use_amp else 'off'}")

    train_df = load_train(args.fd, args.max_rul)
    test_df = load_test(args.fd, args.max_rul)
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS + ["RUL"]]

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    x_train, y_train = make_train_windows(train_df, feature_cols, args.window_size)
    x_test, y_test = make_test_last_windows(test_df, feature_cols, args.window_size)

    print(f"Train windows: {x_train.shape}, Test windows: {x_test.shape}")
    print(f"Features: {len(feature_cols)} -> {feature_cols}")

    train_loader = DataLoader(
        RULDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        RULDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = LSTMRUL(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_rmse = float("inf")
    best_path = OUTPUT_DIR / f"best_lstm_{args.fd.lower()}.pt"

    best_metrics: dict[str, float] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(xb)
                loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(xb)

        if epoch == 1 or epoch % 5 == 0:
            metrics = evaluate(model, test_loader, device, pin_memory)
            train_loss = total_loss / len(x_train)
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_loss:.4f} | "
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

    print(f"Best RMSE: {best_rmse:.3f}")
    print(f"Saved model: {best_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
