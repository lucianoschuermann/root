
import numpy as np
import pandas as pd
from config import MIN_CALIBRATION_N, DAY_FRAC
from .base import MarketImpactModel

class SquareRootExtendedModel(MarketImpactModel):
    """
    Gatheral (2016) square-root law compatible version:
        Impact ≈ spread/2 + alpha * Vol * sqrt(POV)
    with beta fixed at 0.5 (NOT calibrated).
    """
    name = "Sqrt_Extended"

    BETA_FIXED = 0.5

    def calibrate(self, calib_df, window_type):
        if len(calib_df) < MIN_CALIBRATION_N:
            return {"alpha": None, "beta": self.BETA_FIXED}

        vol_col = "Tages Vola"
        adv_col = "5d ADV Non Composite" if window_type == "5d" else "25d ADV Non Composite"
        abos_col = "5d ABOS" if window_type == "5d" else "25d ABOS"

        # NOTE: abos_col must be present since spread is part of the model
        df = calib_df.dropna(subset=["Order Quantity", adv_col, vol_col, abos_col, "ImpactRealised"]).copy()

        adv_eff = df[adv_col] * DAY_FRAC
        pov = df["Order Quantity"] / adv_eff

        valid = (
            (pov > 0)
            & (df[vol_col] > 0)
            & np.isfinite(pov)
            & np.isfinite(df[vol_col])
            & np.isfinite(df[abos_col])
            & np.isfinite(df["ImpactRealised"])
        )
        df = df.loc[valid]
        pov = pov.loc[valid]

        if len(df) < MIN_CALIBRATION_N:
            return {"alpha": None, "beta": self.BETA_FIXED}

        EPS = 1e-12

        # --- Spread half-term in decimal units (bps -> decimal, then /2) ---
        half_spread = (df[abos_col].astype(float).to_numpy() / 10000.0) / 2.0

        impact = df["ImpactRealised"].astype(float).to_numpy()
        vol = df[vol_col].astype(float).to_numpy()

        # --- Net impact to be explained by alpha*Vol*sqrt(POV) ---
        impact_net = impact - half_spread

        # Keep "non-negative model impact" behavior for calibration target
        impact_net = np.maximum(impact_net, EPS)

        # y = (Impact - spread/2) / vol, x = sqrt(pov)
        y = impact_net / vol
        x = np.sqrt(pov.astype(float).to_numpy())  # beta fixed at 0.5

        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]

        if len(x) < MIN_CALIBRATION_N:
            return {"alpha": None, "beta": self.BETA_FIXED}

        denom = float(np.dot(x, x))
        if denom <= 0.0 or not np.isfinite(denom):
            return {"alpha": None, "beta": self.BETA_FIXED}

        alpha_unclipped = float(np.dot(x, y) / denom)

        # Economic sanity: alpha should be non-negative
        alpha = float(max(alpha_unclipped, 0.0))

        return {"alpha": alpha, "beta": self.BETA_FIXED}

    def calculate(self, out_df, window_type, params):
        alpha = params.get("alpha")
        if alpha is None:
            return pd.Series(np.nan, index=out_df.index)

        vol_col = "Tages Vola"
        adv_col = "5d ADV Non Composite" if window_type == "5d" else "25d ADV Non Composite"
        abos_col = "5d ABOS" if window_type == "5d" else "25d ABOS"

        adv_eff = out_df[adv_col] * DAY_FRAC
        pov = out_df["Order Quantity"] / adv_eff

        half_spread = (out_df[abos_col].astype(float) / 10000.0) / 2.0
        temp_impact = alpha * out_df[vol_col].astype(float) * np.sqrt(pov.astype(float))

        # Correct additive structure:
        # Impact = spread/2 + alpha*Vol*sqrt(POV)
        return half_spread + temp_impact
