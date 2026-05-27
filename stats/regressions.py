import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections.abc import Sequence


def _group_cols(by: Sequence[str] | None) -> list[str]:
    return list(by) if by is not None else ["Model"]


def predicted_vs_realised_regression(
    df: pd.DataFrame,
    by: Sequence[str] | None = None,
    min_n: int = 10,
) -> pd.DataFrame:
    """
    Run realised vs predicted regression per group:

        ImpactRealised = alpha + beta * ModelOutput + error

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            - ModelOutput
            - ImpactRealised
            - grouping columns in `by`
    by : sequence[str] or None
        Grouping columns.
        If None, defaults to ["Model"] for backward compatibility.
        Recommended new usage:
            ["Model", "CalibrationScope"]
    min_n : int
        Minimum number of valid rows required.

    Returns
    -------
    pd.DataFrame
        One row per group with:
            - Alpha
            - Beta
            - R2
            - N
            - PredMean
            - RealMean
            - PredStd
            - RealStd
    """
    group_cols = _group_cols(by)

    required_cols = set(group_cols) | {"ModelOutput", "ImpactRealised"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"predicted_vs_realised_regression: missing required columns: {missing}")

    rows: list[dict[str, object]] = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        group_dict = dict(zip(group_cols, keys))

        sub = sub.copy()
        sub["ModelOutput"] = pd.to_numeric(sub["ModelOutput"], errors="coerce")
        sub["ImpactRealised"] = pd.to_numeric(sub["ImpactRealised"], errors="coerce")
        sub = sub.dropna(subset=["ModelOutput", "ImpactRealised"])

        if len(sub) < min_n:
            rows.append(
                {
                    **group_dict,
                    "Alpha": np.nan,
                    "Beta": np.nan,
                    "R2": np.nan,
                    "N": int(len(sub)),
                    "PredMean": float(sub["ModelOutput"].mean()) if len(sub) > 0 else np.nan,
                    "RealMean": float(sub["ImpactRealised"].mean()) if len(sub) > 0 else np.nan,
                    "PredStd": float(sub["ModelOutput"].std(ddof=1)) if len(sub) > 1 else np.nan,
                    "RealStd": float(sub["ImpactRealised"].std(ddof=1)) if len(sub) > 1 else np.nan,
                    "RegressionValid": False,
                    "RegressionNote": f"min_n not met (min_n={min_n})",
                }
            )
            continue

        X = sm.add_constant(sub["ModelOutput"])
        y = sub["ImpactRealised"]

        try:
            res = sm.OLS(y, X).fit()

            rows.append(
                {
                    **group_dict,
                    "Alpha": float(res.params.get("const", float("nan"))),
                    "Beta": float(res.params.get("ModelOutput", float("nan"))),
                    "R2": float(res.rsquared),
                    "N": int(res.nobs),
                    "PredMean": float(sub["ModelOutput"].mean()),
                    "RealMean": float(sub["ImpactRealised"].mean()),
                    "PredStd": float(sub["ModelOutput"].std(ddof=1)) if len(sub) > 1 else np.nan,
                    "RealStd": float(sub["ImpactRealised"].std(ddof=1)) if len(sub) > 1 else np.nan,
                    "RegressionValid": True,
                    "RegressionNote": "",
                }
            )

        except Exception as e:
            rows.append(
                {
                    **group_dict,
                    "Alpha": np.nan,
                    "Beta": np.nan,
                    "R2": np.nan,
                    "N": int(len(sub)),
                    "PredMean": float(sub["ModelOutput"].mean()) if len(sub) > 0 else np.nan,
                    "RealMean": float(sub["ImpactRealised"].mean()) if len(sub) > 0 else np.nan,
                    "PredStd": float(sub["ModelOutput"].std(ddof=1)) if len(sub) > 1 else np.nan,
                    "RealStd": float(sub["ImpactRealised"].std(ddof=1)) if len(sub) > 1 else np.nan,
                    "RegressionValid": False,
                    "RegressionNote": f"{type(e).__name__}: {e}",
                }
            )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(group_cols).reset_index(drop=True)

    return out