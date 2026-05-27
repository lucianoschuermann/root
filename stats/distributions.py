from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _group_cols(by: Sequence[str] | None) -> list[str]:
    return list(by) if by is not None else []


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").dropna()


def _safe_quantile(s: pd.Series, q: float) -> float:
    x = _coerce_numeric(s)
    if len(x) == 0:
        return float("nan")
    return float(x.quantile(q))


def _safe_mean(s: pd.Series) -> float:
    x = _coerce_numeric(s)
    if len(x) == 0:
        return float("nan")
    return float(x.mean())


def _safe_std(s: pd.Series) -> float:
    x = _coerce_numeric(s)
    if len(x) <= 1:
        return float("nan")
    return float(x.std(ddof=1))


def _safe_skew(s: pd.Series) -> float:
    x = _coerce_numeric(s)
    if len(x) <= 2:
        return float("nan")
    return float(x.skew())


def _safe_kurt(s: pd.Series) -> float:
    x = _coerce_numeric(s)
    if len(x) <= 3:
        return float("nan")
    # pandas kurt() returns Fisher excess kurtosis
    return float(x.kurt())


def _summarise_series(
    s: pd.Series,
    prefix: str = "",
) -> dict[str, float | int]:
    x = _coerce_numeric(s)

    if len(x) == 0:
        return {
            f"{prefix}N": 0,
            f"{prefix}Mean": float("nan"),
            f"{prefix}Std": float("nan"),
            f"{prefix}Min": float("nan"),
            f"{prefix}P01": float("nan"),
            f"{prefix}P05": float("nan"),
            f"{prefix}P25": float("nan"),
            f"{prefix}Median": float("nan"),
            f"{prefix}P75": float("nan"),
            f"{prefix}P95": float("nan"),
            f"{prefix}P99": float("nan"),
            f"{prefix}Max": float("nan"),
            f"{prefix}Skew": float("nan"),
            f"{prefix}Kurtosis": float("nan"),
        }

    return {
        f"{prefix}N": int(len(x)),
        f"{prefix}Mean": float(x.mean()),
        f"{prefix}Std": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
        f"{prefix}Min": float(x.min()),
        f"{prefix}P01": float(x.quantile(0.01)),
        f"{prefix}P05": float(x.quantile(0.05)),
        f"{prefix}P25": float(x.quantile(0.25)),
        f"{prefix}Median": float(x.quantile(0.50)),
        f"{prefix}P75": float(x.quantile(0.75)),
        f"{prefix}P95": float(x.quantile(0.95)),
        f"{prefix}P99": float(x.quantile(0.99)),
        f"{prefix}Max": float(x.max()),
        f"{prefix}Skew": float(x.skew()) if len(x) > 2 else float("nan"),
        f"{prefix}Kurtosis": float(x.kurt()) if len(x) > 3 else float("nan"),
    }


def distribution_moments(
    df: pd.DataFrame,
    by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Summary moments for the error distribution.

    Expected columns:
        - Error
        - optional grouping columns in `by`
    """
    group_cols = _group_cols(by)
    if "Error" not in df.columns:
        raise KeyError("distribution_moments: missing required column 'Error'")

    rows: list[dict[str, object]] = []

    if not group_cols:
        row: dict[str, object] = {}
        row.update(_summarise_series(df["Error"]))
        rows.append(row)
    else:
        for keys, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)

            row = {col: key for col, key in zip(group_cols, keys)}
            row.update(_summarise_series(sub["Error"]))
            rows.append(row)

    out = pd.DataFrame(rows)

    if group_cols and not out.empty:
        out = out.sort_values(group_cols).reset_index(drop=True)

    return out


def realised_moments(
    df: pd.DataFrame,
    by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Summary moments for the realised impact distribution.

    Expected columns:
        - ImpactRealised
        - optional grouping columns in `by`

    IMPORTANT:
    This function does NOT deduplicate across CalibrationScope.
    Deduplication should be handled upstream in reporting.py
    if the same realised trades appear once per scope.
    """
    group_cols = _group_cols(by)
    if "ImpactRealised" not in df.columns:
        raise KeyError("realised_moments: missing required column 'ImpactRealised'")

    rows: list[dict[str, object]] = []

    if not group_cols:
        row: dict[str, object] = {}
        row.update(_summarise_series(df["ImpactRealised"]))
        rows.append(row)
    else:
        for keys, sub in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)

            row = {col: key for col, key in zip(group_cols, keys)}
            row.update(_summarise_series(sub["ImpactRealised"]))
            rows.append(row)

    out = pd.DataFrame(rows)

    if group_cols and not out.empty:
        out = out.sort_values(group_cols).reset_index(drop=True)

    return out


def ks_tests(
    df: pd.DataFrame,
    by: Sequence[str] | None = None,
    min_n: int = 10,
) -> pd.DataFrame:
    """
    Two-sample KS tests comparing, for each model (and optional extra grouping),
    the predicted impact distribution vs the realised impact distribution.

    IMPORTANT:
    This is NOT model-vs-model.
    It is:
        ModelOutput  vs  ImpactRealised
    within each model / segment.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least:
            - Model
            - ModelOutput
            - ImpactRealised
    by : sequence of str or None
        Optional additional grouping columns, e.g.
            ["CalibrationScope", "Geography", "SizeBucket", "WindowType"]
        The function always groups by Model as primary dimension.
    min_n : int
        Minimum sample size required on both sides.

    Returns
    -------
    pd.DataFrame
        Columns:
            Model, [optional group cols], KS_D, KS_pvalue, N_pred, N_real,
            Mean_pred, Mean_real, Median_pred, Median_real, KS_valid, KS_note
    """
    required_cols = ["Model", "ModelOutput", "ImpactRealised"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"ks_tests: missing required columns: {missing}")

    extra_by = list(by) if by is not None else []
    group_cols: list[str] = ["Model"] + [c for c in extra_by if c != "Model"]

    rows: list[dict[str, object]] = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: key for col, key in zip(group_cols, keys)}

        pred = _coerce_numeric(sub["ModelOutput"])
        real = _coerce_numeric(sub["ImpactRealised"])

        n_pred = int(len(pred))
        n_real = int(len(real))

        row["N_pred"] = n_pred
        row["N_real"] = n_real
        row["Mean_pred"] = _safe_mean(pred)
        row["Mean_real"] = _safe_mean(real)
        row["Median_pred"] = _safe_quantile(pred, 0.50)
        row["Median_real"] = _safe_quantile(real, 0.50)
        row["P05_pred"] = _safe_quantile(pred, 0.05)
        row["P05_real"] = _safe_quantile(real, 0.05)
        row["P95_pred"] = _safe_quantile(pred, 0.95)
        row["P95_real"] = _safe_quantile(real, 0.95)
        row["Std_pred"] = _safe_std(pred)
        row["Std_real"] = _safe_std(real)
        row["Skew_pred"] = _safe_skew(pred)
        row["Skew_real"] = _safe_skew(real)
        row["Kurtosis_pred"] = _safe_kurt(pred)
        row["Kurtosis_real"] = _safe_kurt(real)

        if n_pred < min_n or n_real < min_n:
            row["KS_D"] = np.nan
            row["KS_pvalue"] = np.nan
            row["KS_valid"] = False
            row["KS_note"] = f"min_n not met (min_n={min_n})"
            rows.append(row)
            continue

        res = ks_2samp(
            pred.to_numpy(dtype=float),
            real.to_numpy(dtype=float),
            alternative="two-sided",
            mode="auto",
        )

        row["KS_D"] = float(res.statistic)
        row["KS_pvalue"] = float(res.pvalue)
        row["KS_valid"] = True
        row["KS_note"] = ""
        rows.append(row)

    out = pd.DataFrame(rows)

    preferred_sort = [
        "Model",
        "CalibrationScope",
        "Geography",
        "SizeBucket",
        "WindowType",
    ]
    sort_cols = [c for c in preferred_sort if c in out.columns]
    if sort_cols and not out.empty:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def impact_moments_comparison(
    df: pd.DataFrame,
    by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compare distribution moments between:
        - ImpactRealised
        - ModelOutput per model

    Returns comparable stats in one table.

    IMPORTANT:
    If the same realised trades are duplicated once per CalibrationScope,
    then this function should either be:
      - called scope-wise separately, OR
      - fed a dataframe already restricted to one scope.
    """
    required_cols = ["ImpactRealised", "ModelOutput", "Model"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"impact_moments_comparison: missing required columns: {missing}")

    group_cols = _group_cols(by)
    rows: list[dict[str, object]] = []

    # ----------------------------------------
    # grouping
    # ----------------------------------------
    if not group_cols:
        groups = [(None, df)]
    else:
        groups = df.groupby(group_cols, dropna=False)

    for keys, sub in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = {col: key for col, key in zip(group_cols, keys)} if group_cols else {}

        # ----------------------------------------
        # REALISED
        # ----------------------------------------
        realised_stats = _summarise_series(sub["ImpactRealised"])
        rows.append({
            **base,
            "Type": "Realised",
            "Model": "Realised",
            **realised_stats,
        })

        # ----------------------------------------
        # MODEL OUTPUTS
        # ----------------------------------------
        for model_name, sub_model in sub.groupby("Model", dropna=False):
            pred_stats = _summarise_series(sub_model["ModelOutput"])

            rows.append({
                **base,
                "Type": "Model",
                "Model": model_name,
                **pred_stats,
            })

    out = pd.DataFrame(rows)

    if group_cols and not out.empty:
        out = out.sort_values(group_cols + ["Type", "Model"]).reset_index(drop=True)

    return out