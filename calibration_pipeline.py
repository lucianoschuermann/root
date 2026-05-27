import numpy as np
import pandas as pd

from typing import Dict, Optional, Hashable, Any, Tuple

from config import (
    WINDOW_TYPES,
    GEOGRAPHY_BUCKETS,
    SIZE_BUCKETS,
    MIN_CALIBRATION_N,
    MIN_OMEGA_N,
    OMEGA_POOL_WINDOWTYPE,
    CALIBRATION_SCOPES,
    CALIBRATION_SCOPE_FALLBACKS,
    MIN_SCOPE_CALIBRATION_N,
    JPM_OMEGA_POOL_SCOPE,
)

from time_windows import prepare_time_index, generate_rolling_windows
from segmentation import prepare_segments
from models import ALL_MODELS


# -----------------------------
# JPMorgan model names (must match model.name exactly)
# -----------------------------
JPM_SPREAD_NAME = "JPMorgan_Spread"
JPM_NOSPREAD_NAME = "JPMorgan_NoSpread"
JPM_VARIANTS = {JPM_SPREAD_NAME, JPM_NOSPREAD_NAME}


# Cache keys
OmegaKey = Tuple[Hashable, ...]
CalibKey = Tuple[Hashable, ...]


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and np.isfinite(x)


def _subset_for_scope(
    trades: pd.DataFrame,
    scope: str,
    geo: Optional[str],
    size: Optional[str],
) -> pd.DataFrame:
    """
    Return the universe corresponding to a calibration scope.
    """
    if scope == "global":
        return trades

    if scope == "by_size":
        if size is None:
            raise ValueError("size required for scope='by_size'")
        return trades[trades["SizeBucket"] == size]

    if scope == "by_geo":
        if geo is None:
            raise ValueError("geo required for scope='by_geo'")
        return trades[trades["GeographyBucket"] == geo]

    if scope == "by_geo_size":
        if geo is None or size is None:
            raise ValueError("geo and size required for scope='by_geo_size'")
        return trades[
            (trades["GeographyBucket"] == geo) &
            (trades["SizeBucket"] == size)
        ]

    raise ValueError(f"Unknown calibration scope: {scope}")


def _scope_descriptor(
    scope: str,
    geo: Optional[str],
    size: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Return human-readable calibration segment descriptors
    written to output columns CalibrationGeo / CalibrationSize.
    """
    if scope == "global":
        return None, None
    if scope == "by_size":
        return None, size
    if scope == "by_geo":
        return geo, None
    if scope == "by_geo_size":
        return geo, size
    raise ValueError(f"Unknown calibration scope: {scope}")


def _resolve_scope_with_fallback(
    trades: pd.DataFrame,
    requested_scope: str,
    geo: Optional[str],
    size: Optional[str],
    cs: Hashable,
    ce: Hashable,
) -> tuple[str, pd.DataFrame, bool, Optional[str]]:
    """
    Resolve calibration scope using fallback hierarchy.

    Returns:
      effective_scope
      calibration_df (time-filtered)
      fallback_used
      fallback_from_scope
    """
    candidate_scopes = [requested_scope] + CALIBRATION_SCOPE_FALLBACKS.get(requested_scope, [])

    for scope in candidate_scopes:
        scope_trades = _subset_for_scope(trades, scope=scope, geo=geo, size=size)
        calib_df = scope_trades[(scope_trades["Month"] >= cs) & (scope_trades["Month"] <= ce)]

        min_n = MIN_SCOPE_CALIBRATION_N.get(scope, MIN_CALIBRATION_N)

        if len(calib_df) >= min_n:
            return scope, calib_df, (scope != requested_scope), (requested_scope if scope != requested_scope else None)

    return requested_scope, pd.DataFrame(), False, None


def _omega_pool_scope_key(
    model_name: str,
    pool_scope: str,
    geo: Optional[str],
    size: Optional[str],
    cs: Hashable,
    ce: Hashable,
) -> OmegaKey:
    """
    Cache key for pooled JPM omega calibration.
    """
    if pool_scope == "global":
        return (model_name, "global", cs, ce)

    if pool_scope == "geo":
        return (model_name, "geo", geo, cs, ce)

    if pool_scope == "size":
        return (model_name, "size", size, cs, ce)

    if pool_scope == "geo_size":
        return (model_name, "geo_size", geo, size, cs, ce)

    raise ValueError(f"Unknown omega pool scope: {pool_scope}")


def _subset_for_omega_pool_scope(
    trades: pd.DataFrame,
    pool_scope: str,
    geo: Optional[str],
    size: Optional[str],
) -> pd.DataFrame:
    """
    Return the universe used to estimate pooled omega for JPM models.
    Pool scopes are slightly different names than calibration scopes:
      global, geo, size, geo_size
    """
    if pool_scope == "global":
        return trades

    if pool_scope == "geo":
        if geo is None:
            raise ValueError("geo required for omega pool scope='geo'")
        return trades[trades["GeographyBucket"] == geo]

    if pool_scope == "size":
        if size is None:
            raise ValueError("size required for omega pool scope='size'")
        return trades[trades["SizeBucket"] == size]

    if pool_scope == "geo_size":
        if geo is None or size is None:
            raise ValueError("geo and size required for omega pool scope='geo_size'")
        return trades[
            (trades["GeographyBucket"] == geo) &
            (trades["SizeBucket"] == size)
        ]

    raise ValueError(f"Unknown omega pool scope: {pool_scope}")


def _default_calibration_diagnostics(model_name: str) -> dict[str, Any]:
    """
    Ensure that the calibration output always contains the same diagnostic fields,
    even if a specific model version does not return all of them.
    """
    is_jpm = model_name in JPM_VARIANTS

    return {
        # core params
        "alpha": None,
        "beta": None,
        "gamma": None,
        "omega": None,

        # omega diagnostics
        "omega_unclipped": None,
        "omega_clipped": None,
        "omega_best_loss": None,
        "omega_delta": None,
        "omega_loss_left": None,
        "omega_loss_right": None,
        "omega_flatness_rel": None,
        "omega_best_source": None,
        "omega_n_used": None,

        # calibration diagnostics
        "n_raw": None,
        "n_after_dropna": None,
        "n_after_basic_valid": None,
        "n_positive_target": None,
        "share_positive_target": None,
        "calibration_note": None,

        # derived / pipeline-level diagnostics
        "HasFiniteAlpha": None,
        "HasFiniteBeta": None,
        "HasFiniteGamma": None,
        "HasFiniteOmega": None,
        "OmegaHitDiagClipBand": None if is_jpm else None,
    }


def _enrich_params_with_diagnostics(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Add derived diagnostic flags on top of model-returned params.
    Also ensures the result always contains standard diagnostic keys.
    """
    out = _default_calibration_diagnostics(model_name)
    out.update(params)

    alpha = out.get("alpha")
    beta = out.get("beta")
    gamma = out.get("gamma")
    omega = out.get("omega")

    out["HasFiniteAlpha"] = _is_finite_number(alpha)
    out["HasFiniteBeta"] = _is_finite_number(beta)
    out["HasFiniteGamma"] = _is_finite_number(gamma)
    out["HasFiniteOmega"] = _is_finite_number(omega)

    if model_name in JPM_VARIANTS:
        omega_unclipped = out.get("omega_unclipped")
        omega_clipped = out.get("omega_clipped")

        if _is_finite_number(omega_unclipped) and _is_finite_number(omega_clipped):
            out["OmegaHitDiagClipBand"] = bool(abs(float(omega_unclipped) - float(omega_clipped)) > 1e-12)
        else:
            out["OmegaHitDiagClipBand"] = None
    else:
        out["OmegaHitDiagClipBand"] = None

    return out


def _calibration_cache_key(
    model_name: str,
    window_type: str,
    effective_scope: str,
    calib_geo: Optional[str],
    calib_size: Optional[str],
    cs: Hashable,
    ce: Hashable,
    omega_fixed: Optional[float],
) -> CalibKey:
    """
    Cache key for model calibration parameters.
    Rounded omega_fixed for stable hash key if present.
    """
    omega_key = None if omega_fixed is None else round(float(omega_fixed), 12)
    return (
        model_name,
        window_type,
        effective_scope,
        calib_geo,
        calib_size,
        cs,
        ce,
        omega_key,
    )


def _iter_eval_segments():
    """
    Evaluation segments remain the same as your current reporting structure:
    all Geography × Size combinations.
    """
    for geo in GEOGRAPHY_BUCKETS:
        for size in SIZE_BUCKETS:
            yield geo, size


def run_calibration_pipeline(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling calibration pipeline with explicit calibration scopes.

    Key idea:
    - Evaluation segment is always fixed at (eval_geo, eval_size, out-month)
    - Calibration scope determines where parameters are estimated from:
        global / by_size / by_geo / by_geo_size
    - This allows clean OOS comparison between global and segmented calibrations

    JPMorgan variants:
    - pooled omega is estimated according to JPM_OMEGA_POOL_SCOPE[effective_scope]
    - omega_fixed is injected into per-scope calibration if available

    Output contains:
    - requested + effective calibration scope
    - evaluation segment
    - calibration segment descriptors
    - fallback diagnostics
    - model params + diagnostics
    """
    trades = prepare_time_index(trades)
    trades = prepare_segments(trades)

    windows = generate_rolling_windows(trades["Month"].unique())
    results: list[dict[str, Any]] = []

    # Locate JPMorgan variants once
    jpm_models: Dict[str, Any] = {
        m.name: m for m in ALL_MODELS if getattr(m, "name", "") in JPM_VARIANTS
    }

    # Cache pooled omega and full calibrations
    omega_cache: Dict[OmegaKey, float] = {}
    calibration_cache: Dict[CalibKey, dict[str, Any]] = {}

    # Stable canonical window type for pooled omega
    pool_wtype = OMEGA_POOL_WINDOWTYPE if OMEGA_POOL_WINDOWTYPE in WINDOW_TYPES else WINDOW_TYPES[0]

    for window_type in WINDOW_TYPES:
        for cs, ce, om in windows:
            for eval_geo, eval_size in _iter_eval_segments():
                eval_seg = trades[
                    (trades["GeographyBucket"] == eval_geo) &
                    (trades["SizeBucket"] == eval_size)
                ]
                out_df = eval_seg[eval_seg["Month"] == om]

                if out_df.empty:
                    continue

                for requested_scope in CALIBRATION_SCOPES:
                    effective_scope, calib_df, fallback_used, fallback_from_scope = _resolve_scope_with_fallback(
                        trades=trades,
                        requested_scope=requested_scope,
                        geo=eval_geo,
                        size=eval_size,
                        cs=cs,
                        ce=ce,
                    )

                    if calib_df.empty:
                        continue

                    calib_geo, calib_size = _scope_descriptor(
                        scope=effective_scope,
                        geo=eval_geo,
                        size=eval_size,
                    )

                    # ------------------------------------------
                    # Stage 1: pooled omega for JPM variants
                    # ------------------------------------------
                    omega_pooled_by_variant: Dict[str, Optional[float]] = {name: None for name in JPM_VARIANTS}
                    omega_pool_scope = JPM_OMEGA_POOL_SCOPE.get(effective_scope, "geo")

                    omega_pool_universe = _subset_for_omega_pool_scope(
                        trades=trades,
                        pool_scope=omega_pool_scope,
                        geo=eval_geo,
                        size=eval_size,
                    )
                    omega_pool_df = omega_pool_universe[
                        (omega_pool_universe["Month"] >= cs) &
                        (omega_pool_universe["Month"] <= ce)
                    ]

                    if len(omega_pool_df) >= MIN_OMEGA_N and jpm_models:
                        for variant_name, model in jpm_models.items():
                            key = _omega_pool_scope_key(
                                model_name=variant_name,
                                pool_scope=omega_pool_scope,
                                geo=eval_geo,
                                size=eval_size,
                                cs=cs,
                                ce=ce,
                            )

                            if key in omega_cache:
                                omega_pooled_by_variant[variant_name] = omega_cache[key]
                            else:
                                pooled_params = model.calibrate(omega_pool_df, pool_wtype)
                                op = pooled_params.get("omega")

                                if _is_finite_number(op):
                                    omega_val = float(op)
                                    omega_cache[key] = omega_val
                                    omega_pooled_by_variant[variant_name] = omega_val

                    # ------------------------------------------
                    # Stage 2: calibrate each model for this scope
                    # ------------------------------------------
                    for model in ALL_MODELS:
                        model_name = str(getattr(model, "name", ""))

                        omega_pooled_used: Optional[float] = None
                        omega_source = "segment"

                        if model_name in JPM_VARIANTS and _is_finite_number(omega_pooled_by_variant.get(model_name)):
                            omega_pooled_used = float(omega_pooled_by_variant[model_name])
                            omega_source = f"pooled_{omega_pool_scope}"

                        cache_key = _calibration_cache_key(
                            model_name=model_name,
                            window_type=window_type,
                            effective_scope=effective_scope,
                            calib_geo=calib_geo,
                            calib_size=calib_size,
                            cs=cs,
                            ce=ce,
                            omega_fixed=omega_pooled_used,
                        )

                        if cache_key in calibration_cache:
                            params_enriched = calibration_cache[cache_key]
                        else:
                            if model_name in JPM_VARIANTS and _is_finite_number(omega_pooled_used):
                                try:
                                    params = model.calibrate(
                                        calib_df,
                                        window_type,
                                        omega_fixed=omega_pooled_used,
                                    )
                                except TypeError:
                                    # Fallback if model signature is not harmonized
                                    params = model.calibrate(calib_df, window_type)
                                    omega_source = "segment"
                                    omega_pooled_used = None
                            else:
                                params = model.calibrate(calib_df, window_type)

                            params_enriched = _enrich_params_with_diagnostics(model_name, params)
                            calibration_cache[cache_key] = params_enriched

                        results.append(
                            {
                                # model identity
                                "Model": model_name,
                                "WindowType": window_type,

                                # scope identity
                                "CalibrationScope": requested_scope,
                                "RequestedScope": requested_scope,
                                "EffectiveScope": effective_scope,
                                "FallbackUsed": bool(fallback_used),
                                "FallbackFromScope": fallback_from_scope,

                                # evaluation segment (used later in evaluation code)
                                "EvalGeography": eval_geo,
                                "EvalSizeBucket": eval_size,

                                # backward compatibility:
                                # keep old column names aligned to evaluation segment
                                "Geography": eval_geo,
                                "SizeBucket": eval_size,

                                # calibration segment descriptors
                                "CalibrationGeo": calib_geo,
                                "CalibrationSize": calib_size,

                                # rolling window metadata
                                "CalibrationStart": cs,
                                "CalibrationEnd": ce,
                                "OutSampleMonth": om,
                                "N_Calibration": int(len(calib_df)),
                                "N_OutSample": int(len(out_df)),

                                # JPM omega metadata
                                "OmegaSource": omega_source,
                                "OmegaPooled": omega_pooled_used,
                                "OmegaPoolScope": omega_pool_scope if model_name in JPM_VARIANTS else None,
                                "OmegaPoolWindowType": pool_wtype if model_name in JPM_VARIANTS else None,

                                # params + diagnostics
                                **params_enriched,
                            }
                        )

    out_df = pd.DataFrame(results)

    if out_df.empty:
        return out_df

    sort_cols = [
        "Model",
        "CalibrationScope",
        "EffectiveScope",
        "WindowType",
        "EvalGeography",
        "EvalSizeBucket",
        "CalibrationStart",
        "OutSampleMonth",
    ]
    sort_cols = [c for c in sort_cols if c in out_df.columns]

    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    return out_df