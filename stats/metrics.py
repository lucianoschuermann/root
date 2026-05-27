import pandas as pd
import numpy as np
from typing import Optional


def _safe_rmse(x: pd.Series) -> float:
    """
    Root mean squared error with NaN-safe handling.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return np.nan
    return float(np.sqrt(np.mean(x ** 2)))


def _safe_quantile(x: pd.Series, q: float) -> float:
    """
    Quantile with NaN-safe handling.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return np.nan
    return float(np.quantile(x, q))


def summary_metrics(
    df: pd.DataFrame,
    by: list[str],
) -> pd.DataFrame:
    """
    Compute core accuracy and bias metrics.

    Expected columns in df:
        - Error
        - AbsError

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level evaluation dataframe.
    by : list[str]
        Grouping columns, e.g.:
            ["Model", "CalibrationScope"]
            ["Model", "CalibrationScope", "SizeBucket"]

    Returns
    -------
    pd.DataFrame
        Summary metrics by requested grouping.
    """
    required_cols = set(by) | {"Error", "AbsError"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"summary_metrics: missing required columns: {missing}")

    grouped = df.groupby(by, dropna=False)

    out = grouped.agg(
        N=("Error", "count"),
        MeanError=("Error", "mean"),
        MedianError=("Error", "median"),
        MAE=("AbsError", "mean"),
        MedAE=("AbsError", "median"),
        RMSE=("Error", _safe_rmse),
        P95AbsError=("AbsError", lambda x: _safe_quantile(x, 0.95)),
    ).reset_index()

    return out


def coverage_metrics(
    df: pd.DataFrame,
    by: Optional[list[str]] = None,
    rel_thresholds: list[float] = [0.1, 0.2],
) -> pd.DataFrame:
    """
    Compute coverage metrics:
        fraction of trades within relative error thresholds.

    Coverage is defined as:
        mean(abs(RelError) <= threshold)

    Parameters
    ----------
    df : pd.DataFrame
        Trade-level evaluation dataframe.
    by : list[str], optional
        Grouping columns. If None, defaults to ["Model"] for backward compatibility.
        Recommended new usage:
            ["Model", "CalibrationScope"]
    rel_thresholds : list[float]
        Relative error thresholds.

    Returns
    -------
    pd.DataFrame
        Coverage table with grouping columns + Threshold + Coverage.
    """
    if by is None:
        by = ["Model"]

    required_cols = set(by) | {"RelError"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"coverage_metrics: missing required columns: {missing}")

    rows = []

    for group_keys, sub in df.groupby(by, dropna=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        group_dict = dict(zip(by, group_keys))

        rel_err = pd.to_numeric(sub["RelError"], errors="coerce")
        valid = rel_err.notna()

        for thr in rel_thresholds:
            if valid.sum() == 0:
                coverage = np.nan
            else:
                coverage = float((rel_err.loc[valid].abs() <= thr).mean())

            rows.append({
                **group_dict,
                "Threshold": float(thr),
                "Coverage": coverage,
                "N": int(valid.sum()),
            })

    return pd.DataFrame(rows)