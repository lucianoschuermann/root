import numpy as np
import pandas as pd
from typing import Any

from time_windows import prepare_time_index
from segmentation import prepare_segments
from models import MODEL_LOOKUP


# ---------------------------------------------------------
# Columns that are metadata and must NOT be passed as params
# ---------------------------------------------------------
META_COLS = {
    "Model",
    "WindowType",
    "Geography",
    "SizeBucket",
    "CalibrationStart",
    "CalibrationEnd",
    "OutSampleMonth",
    "N_Calibration",
    "N_OutSample",

    # new / future scope columns
    "CalibrationScope",
    "RequestedScope",
    "EffectiveScope",
    "FallbackUsed",
    "FallbackFromScope",

    # evaluation segment columns
    "EvalGeography",
    "EvalSizeBucket",

    # optional calibration segment descriptors
    "CalibrationGeo",
    "CalibrationSize",
}


def _series_from_values(values: Any, out_df: pd.DataFrame) -> pd.Series:
    """
    Normalize model.calculate(...) output to a Series aligned to out_df.index.
    Supports:
      - pd.Series
      - np.ndarray
      - list-like
      - scalar (broadcast)
    """
    if isinstance(values, pd.Series):
        if values.index.equals(out_df.index):
            return values
        if len(values) == len(out_df):
            return pd.Series(values.to_numpy(), index=out_df.index)
        raise ValueError(
            f"model.calculate returned Series of length {len(values)}, "
            f"expected {len(out_df)}"
        )

    if isinstance(values, (np.ndarray, list, tuple)):
        if len(values) != len(out_df):
            raise ValueError(
                f"model.calculate returned array-like of length {len(values)}, "
                f"expected {len(out_df)}"
            )
        return pd.Series(values, index=out_df.index)

    # scalar fallback
    return pd.Series([values] * len(out_df), index=out_df.index)


def _extract_eval_segment(row: pd.Series) -> tuple[Any, Any]:
    """
    For backward compatibility:
      - prefer EvalGeography / EvalSizeBucket if available
      - otherwise fall back to Geography / SizeBucket
    """
    eval_geo = row["EvalGeography"] if "EvalGeography" in row.index else row["Geography"]
    eval_size = row["EvalSizeBucket"] if "EvalSizeBucket" in row.index else row["SizeBucket"]
    return eval_geo, eval_size


def _extract_params(row: pd.Series) -> dict[str, Any]:
    """
    Extract model params / diagnostics from calibration row.
    Everything not listed in META_COLS is passed through.
    """
    return {
        k: row[k]
        for k in row.index
        if k not in META_COLS
    }


def run_outsample_evaluation(
    trades: pd.DataFrame,
    calibration_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Out-of-sample evaluation.

    Supports both:
      1) old calibration_results schema (single implicit scope)
      2) new multi-scope schema with CalibrationScope / EvalGeography / EvalSizeBucket

    Output is trade-level and therefore suitable for:
      - MAE / RMSE / KS aggregation
      - paired scope comparisons
      - later regression analysis by scope
    """
    trades = prepare_time_index(trades)
    trades = prepare_segments(trades)

    rows: list[dict[str, Any]] = []

    if calibration_results.empty:
        return pd.DataFrame(rows)

    for _, r in calibration_results.iterrows():
        model_name = str(r["Model"])
        model = MODEL_LOOKUP[model_name]

        eval_geo, eval_size = _extract_eval_segment(r)

        out_df = trades[
            (trades["GeographyBucket"] == eval_geo) &
            (trades["SizeBucket"] == eval_size) &
            (trades["Month"] == r["OutSampleMonth"])
        ]

        if out_df.empty:
            continue

        params = _extract_params(r)
        window_type = r["WindowType"]

        values = model.calculate(
            out_df=out_df,
            window_type=window_type,
            params=params,
        )
        values_s = _series_from_values(values, out_df)

        calibration_scope = r["CalibrationScope"] if "CalibrationScope" in r.index else "by_geo_size"
        requested_scope = r["RequestedScope"] if "RequestedScope" in r.index else calibration_scope
        effective_scope = r["EffectiveScope"] if "EffectiveScope" in r.index else calibration_scope
        fallback_used = r["FallbackUsed"] if "FallbackUsed" in r.index else False
        fallback_from_scope = r["FallbackFromScope"] if "FallbackFromScope" in r.index else None

        for idx, trade in out_df.iterrows():
            model_output = values_s.loc[idx]
            impact_realised = trade["ImpactRealised"]
            error = model_output - impact_realised
            abs_error = abs(error)
            sq_error = error ** 2

            rows.append(
                {
                    "TSId": trade.get("TSId"),
                    "Arrival Time": trade["Arrival Time"],

                    # model identity
                    "Model": model_name,
                    "WindowType": window_type,

                    # scope identity
                    "CalibrationScope": calibration_scope,
                    "RequestedScope": requested_scope,
                    "EffectiveScope": effective_scope,
                    "FallbackUsed": fallback_used,
                    "FallbackFromScope": fallback_from_scope,

                    # evaluation segment (important!)
                    "EvalGeography": eval_geo,
                    "EvalSizeBucket": eval_size,
                    "OutSampleMonth": r["OutSampleMonth"],

                    # optional original calibration segment info (if available)
                    "CalibrationGeo": r["CalibrationGeo"] if "CalibrationGeo" in r.index else r.get("Geography"),
                    "CalibrationSize": r["CalibrationSize"] if "CalibrationSize" in r.index else r.get("SizeBucket"),

                    # values
                    "ModelOutput": model_output,
                    "ImpactRealised": impact_realised,
                    "Error": error,
                    "AbsError": abs_error,
                    "SquaredError": sq_error,

                    # keep selected calibration metadata if useful
                    "N_Calibration": r.get("N_Calibration"),
                    "N_OutSample": r.get("N_OutSample"),

                    # keep all params / diagnostics for debugging / reporting
                    **params,
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    sort_cols = [
        "Model",
        "CalibrationScope",
        "WindowType",
        "EvalGeography",
        "EvalSizeBucket",
        "OutSampleMonth",
        "Arrival Time",
        "TSId",
    ]
    sort_cols = [c for c in sort_cols if c in out.columns]

    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out