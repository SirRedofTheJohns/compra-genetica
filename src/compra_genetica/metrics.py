"""Error metrics to compare a solver against the best known solutions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def error_columns(pred: np.ndarray, true: np.ndarray) -> pd.DataFrame:
    """Per-problem errors: E (signed), AE (absolute), RE (relative)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    E = pred - true
    return pd.DataFrame({"E": E, "AE": np.abs(E), "RE": E / true})


def summary_metrics(pred: np.ndarray, true: np.ndarray, tol: float = 1e-6) -> dict:
    """Aggregate metrics: ME, MAE, MSE, RMSE, MAPE, max AE, R2 and hit rate.

    hit rate = share of problems where the solver matches or beats the reference.
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ok = ~(np.isnan(pred) | np.isnan(true))
    pred, true = pred[ok], true[ok]
    E = pred - true
    ss_res, ss_tot = (E ** 2).sum(), ((true - true.mean()) ** 2).sum()
    return {
        "n": int(ok.sum()),
        "ME": E.mean(),
        "MAE": np.abs(E).mean(),
        "MSE": (E ** 2).mean(),
        "RMSE": np.sqrt((E ** 2).mean()),
        "MAPE_%": 100 * np.abs(E / true).mean(),
        "max_AE": np.abs(E).max(),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "hit_rate_%": 100 * (E <= tol).mean(),
    }
