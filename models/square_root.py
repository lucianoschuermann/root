import numpy as np
import pandas as pd

from config import MIN_CALIBRATION_N, DAY_FRAC
from .base import MarketImpactModel


class SquareRootModel(MarketImpactModel):
    name = "Sqrt"

    def calibrate(self, calib_df: pd.DataFrame, window_type: str) -> dict[str, float | None]:
        if len(calib_df) < MIN_CALIBRATION_N:
            return {"alpha": None}

        vol_col = "Tages Vola"
        adv_col = "5d ADV Non Composite" if window_type == "5d" else "25d ADV Non Composite"

        df = calib_df.dropna(
            subset=["Order Quantity", adv_col, vol_col, "ImpactRealised"]
        ).copy()

        if len(df) < MIN_CALIBRATION_N:
            return {"alpha": None}

        # Effective ADV
        df["adv_eff"] = pd.to_numeric(df[adv_col], errors="coerce") * DAY_FRAC
        df["qty_abs"] = pd.to_numeric(df["Order Quantity"], errors="coerce").abs()
        df["vol"] = pd.to_numeric(df[vol_col], errors="coerce")
        df["impact"] = pd.to_numeric(df["ImpactRealised"], errors="coerce")

        # Basic economic / numerical validity filters
        df = df[
            (df["adv_eff"] > 0)
            & (df["qty_abs"] >= 0)
            & np.isfinite(df["adv_eff"])
            & np.isfinite(df["qty_abs"])
            & np.isfinite(df["vol"])
            & np.isfinite(df["impact"])
        ].copy()

        if len(df) < MIN_CALIBRATION_N:
            return {"alpha": None}

        df["pov"] = df["qty_abs"] / df["adv_eff"]

        # POV must be finite and non-negative for sqrt
        df = df[np.isfinite(df["pov"]) & (df["pov"] >= 0)].copy()

        if len(df) < MIN_CALIBRATION_N:
            return {"alpha": None}

        df["x"] = df["vol"] * np.sqrt(df["pov"])
        df["y"] = df["impact"]

        # Final finite filter after transformation
        df = df[np.isfinite(df["x"]) & np.isfinite(df["y"])].copy()

        if len(df) < MIN_CALIBRATION_N:
            return {"alpha": None}

        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)

        denom = float(np.sum(x ** 2))
        num = float(np.sum(x * y))

        # robust denominator check
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            return {"alpha": None}

        if not np.isfinite(num):
            return {"alpha": None}

        return {"alpha": float(num / denom)}

    def calculate(self, out_df: pd.DataFrame, window_type: str, params: dict[str, float | None]) -> pd.Series:
        alpha = params.get("alpha")
        if alpha is None or not np.isfinite(alpha):
            return pd.Series(np.nan, index=out_df.index, dtype=float)

        vol_col = "Tages Vola"
        adv_col = "5d ADV Non Composite" if window_type == "5d" else "25d ADV Non Composite"

        d = out_df.copy()

        d["adv_eff"] = pd.to_numeric(d[adv_col], errors="coerce") * DAY_FRAC
        d["qty_abs"] = pd.to_numeric(d["Order Quantity"], errors="coerce").abs()
        d["vol"] = pd.to_numeric(d[vol_col], errors="coerce")

        # start with NaN output
        out = pd.Series(np.nan, index=d.index, dtype=float)

        valid = (
            np.isfinite(d["adv_eff"])
            & np.isfinite(d["qty_abs"])
            & np.isfinite(d["vol"])
            & (d["adv_eff"] > 0)
            & (d["qty_abs"] >= 0)
        )

        if not valid.any():
            return out

        pov = pd.Series(np.nan, index=d.index, dtype=float)
        pov.loc[valid] = d.loc[valid, "qty_abs"] / d.loc[valid, "adv_eff"]

        valid = valid & np.isfinite(pov) & (pov >= 0)

        if not valid.any():
            return out

        x = pd.Series(np.nan, index=d.index, dtype=float)
        x.loc[valid] = d.loc[valid, "vol"] * np.sqrt(pov.loc[valid])

        valid = valid & np.isfinite(x)

        if not valid.any():
            return out

        out.loc[valid] = float(alpha) * x.loc[valid]
        return out