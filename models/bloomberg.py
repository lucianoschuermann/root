import numpy as np
import pandas as pd
from config import MIN_CALIBRATION_N, DAY_FRAC
from .base import MarketImpactModel


class BloombergModel(MarketImpactModel):
    """
    Bloomberg Market Impact Model (two-parameter variant).

    Model:
        MI = alpha * term_spread + beta * term_risk_size

    where
        term_spread    = 0.5 * (S / P)
        term_risk_size = sqrt( (sigma^2 / 3) / 250 ) * sqrt( V / (0.3 * EDV_eff) )
        EDV_eff        = ADV * DAY_FRAC

    Notes on units as in your implementation:
        - ABOS assumed in bps -> divide by 10000.0 to get decimal
        - volatility column assumed in percent -> divide by 100.0 to get decimal
    """

    name = "Bloomberg"

    # ==================================================
    # CALIBRATION
    # ==================================================

    def calibrate(self, calib_df: pd.DataFrame, window_type: str) -> dict:
        """
        @notice Calibrate Bloomberg market impact model with two parameters alpha and beta.
        @dev OLS without intercept:
             y ≈ alpha * term_spread + beta * term_risk_size
        """

        if len(calib_df) < MIN_CALIBRATION_N:
            return {"alpha": None, "beta": None}

        vol_col = "30d Vola Bloomberg"
        adv_col = "5d ADV Non Composite" if window_type == "5d" else "25d ADV Non Composite"
        abos_col = "5d ABOS" if window_type == "5d" else "25d ABOS"

        df = calib_df.dropna(
            subset=[
                abos_col,
                "Executed Price",
                vol_col,
                "Order Quantity",
                adv_col,
                "ImpactRealised",
            ]
        )

        if len(df) < MIN_CALIBRATION_N:
            return {"alpha": None, "beta": None}

        participation = 0.3
        EPS = 1e-12  # numerical safety
        EPS_Y = 1e-6

        # Effective expected volume in the interval (Variante A)
        edv_eff = df[adv_col] * DAY_FRAC
        edv_eff = np.maximum(edv_eff, EPS)

        qty = np.maximum(df["Order Quantity"], EPS)

        # Two regressors / terms (unchanged formulas from your code)
        term_spread = 0.5 * (df[abos_col] / 10000.0 / df["Executed Price"])
        term_risk_size = (
            np.sqrt((((df[vol_col] / 100.0) ** 2) / 3.0) / 250.0)
            * np.sqrt(qty / (participation * edv_eff))
        )

        # Target
        y = np.asarray(np.maximum(df["ImpactRealised"], EPS_Y), dtype=float)

        # Design matrix X (n x 2): columns are term_spread and term_risk_size
        X = np.column_stack([
            np.asarray(term_spread, dtype=float),
            np.asarray(term_risk_size, dtype=float),
        ])

        # Guard: ensure enough rows and finite data
        if X.shape[0] < MIN_CALIBRATION_N:
            return {"alpha": None, "beta": None}

        finite_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X = X[finite_mask]
        y = y[finite_mask]

        if len(y) < MIN_CALIBRATION_N:
            return {"alpha": None, "beta": None}

        # Solve least squares: minimize ||X @ theta - y||^2
        # theta = [alpha, beta]
        try:
            theta, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = float(theta[0]), float(theta[1])
        except Exception:
            return {"alpha": None, "beta": None}

        if not (np.isfinite(alpha) and np.isfinite(beta)):
            return {"alpha": None, "beta": None}

        return {"alpha": alpha, "beta": beta}

    # ==================================================
    # OUT-OF-SAMPLE CALCULATION
    # ==================================================

    def calculate(self, out_df: pd.DataFrame, window_type: str, params: dict) -> pd.Series:
        """
        @notice Compute Bloomberg market impact out-of-sample using alpha and beta.
        """

        alpha = params.get("alpha")
        beta = params.get("beta")

        if not (isinstance(alpha, (int, float)) and np.isfinite(alpha)):
            return pd.Series(np.nan, index=out_df.index)
        if not (isinstance(beta, (int, float)) and np.isfinite(beta)):
            return pd.Series(np.nan, index=out_df.index)

        vol_col = "30d Vola Bloomberg"
        adv_col = "5d ADV Non Composite" if window_type == "5d" else "25d ADV Non Composite"
        abos_col = "5d ABOS" if window_type == "5d" else "25d ABOS"

        participation = 0.3
        EPS = 1e-12

        edv_eff = out_df[adv_col] * DAY_FRAC
        edv_eff = np.maximum(edv_eff, EPS)

        qty = np.maximum(out_df["Order Quantity"], EPS)

        term_spread = 0.5 * (out_df[abos_col] / 10000.0 / out_df["Executed Price"])
        term_risk_size = (
            np.sqrt((((out_df[vol_col] / 100.0) ** 2) / 3.0) / 250.0)
            * np.sqrt(qty / (participation * edv_eff))
        )

        mi = alpha * term_spread + beta * term_risk_size
        return mi
    