import numpy as np


def rmse_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true, y_pred) -> float:
    """NASA C-MAPSS asymmetric score. Late predictions are penalized more."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true
    score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(score))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse_score(y_true, y_pred),
        "mae": mae_score(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }
