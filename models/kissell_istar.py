import numpy as np
import pandas as pd
from config import MIN_CALIBRATION_N, DAY_FRAC
from .base import MarketImpactModel


class KissellIStarModel(MarketImpactModel):
    """
    I-Star Market Impact Model (as in provided slide):

        I*  = a1 * (S/ADV)^a2 * sigma^a3
        MI  = b1 * I* * POV^a4 + (1-b1) * I*
            = I* * ((1-b1) + b1 * POV^a4)

    We treat POV = S/ADV_eff where ADV_eff = ADV * DAY_FRAC.
    """

    name = "Kissell_IStar"

    def calibrate(self, calib_df: pd.DataFrame, window_type: str) -> dict:
        if len(calib_df) < MIN_CALIBRATION_N:
            return {"a1": None, "a2": None, "a3": None, "b1": None, "a4": None}

        vol_col = "30d Vola Bloomberg"
        adv_col = "5d ADV Non Composite" if window_type == "5d" else "25d ADV Non Composite"

        needed = ["Order Quantity", adv_col, vol_col, "ImpactRealised"]
        df = calib_df.dropna(subset=needed).copy()

        if len(df) < MIN_CALIBRATION_N:
            return {"a1": None, "a2": None, "a3": None, "b1": None, "a4": None}

        qty = df["Order Quantity"].astype(float)
        adv_eff = df[adv_col].astype(float) * float(DAY_FRAC)
        sigma = df[vol_col].astype(float) / 100
        mi_obs = df["ImpactRealised"].astype(float)

        # We calibrate on strictly positive MI and inputs (log-space).
        valid = (
            np.isfinite(qty) & np.isfinite(adv_eff) & np.isfinite(sigma) & np.isfinite(mi_obs)
            & (qty > 0.0) & (adv_eff > 0.0) & (sigma > 0.0) & (mi_obs > 0.0)
        )
        df = df.loc[valid].copy()
        qty = qty.loc[valid]
        adv_eff = adv_eff.loc[valid]
        sigma = sigma.loc[valid]
        mi_obs = mi_obs.loc[valid]

        if len(df) < MIN_CALIBRATION_N:
            return {"a1": None, "a2": None, "a3": None, "b1": None, "a4": None}

        pov = (qty / (qty + adv_eff)).astype(float)

        # Need POV > 0 for logs and powers
        valid2 = np.isfinite(pov) & (pov > 0.0)
        pov = pov.loc[valid2]
        sigma = sigma.loc[valid2]
        mi_obs = mi_obs.loc[valid2]

        if len(pov) < MIN_CALIBRATION_N:
            return {"a1": None, "a2": None, "a3": None, "b1": None, "a4": None}

        # Precompute logs
        y = np.log(mi_obs.to_numpy(dtype=np.float64))
        lpov = np.log(pov.to_numpy(dtype=np.float64))
        lsig = np.log(sigma.to_numpy(dtype=np.float64))

        # Final numeric mask
        m = np.isfinite(y) & np.isfinite(lpov) & np.isfinite(lsig)
        y, lpov, lsig = y[m], lpov[m], lsig[m]

        if y.size < MIN_CALIBRATION_N:
            return {"a1": None, "a2": None, "a3": None, "b1": None, "a4": None}

        # -----------------------------
        # Stable grid search for (b1, a4),
        # then OLS for (log a1, a2, a3)
        # -----------------------------
        # Typical sensible ranges; adjust if you want tighter/wider.
        b1_grid = np.linspace(0.0, 1.0, 21)     # 0.00, 0.05, ..., 1.00
        a4_grid = np.linspace(0.0, 2.0, 41)     # 0.00 ... 2.00 in 0.05 steps

        pov_vals = np.exp(lpov)  # back to POV in level for POV^a4

        best = None  # (sse, b1, a4, coef)
        lam = 1e-8   # ridge for stability in OLS solve

        for b1 in b1_grid:
            # If b1 == 0 => logT = log(1) = 0 regardless of a4; evaluate only once.
            if np.isclose(b1, 0.0):
                a4_candidates = [0.0]
            else:
                a4_candidates = a4_grid

            for a4 in a4_candidates:
                # T = (1-b1) + b1 * POV^a4  must be > 0
                T = (1.0 - b1) + b1 * np.power(pov_vals, a4)
                if not np.all(np.isfinite(T)) or np.any(T <= 0.0):
                    continue

                logT = np.log(T)

                # OLS: y_adj = y - logT = c0 + a2*lpov + a3*lsig
                y_adj = y - logT

                X = np.column_stack([np.ones_like(y_adj), lpov, lsig]).astype(np.float64)

                try:
                    coef = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y_adj)
                except np.linalg.LinAlgError:
                    coef = np.linalg.lstsq(X, y_adj, rcond=None)[0]

                # Compute fit error in log-space (robust)
                y_hat = X @ coef + logT
                resid = y - y_hat
                sse = float(np.sum(resid * resid))

                if best is None or sse < best[0]:
                    best = (sse, float(b1), float(a4), coef)

        if best is None:
            return {"a1": None, "a2": None, "a3": None, "b1": None, "a4": None}

        _, b1_hat, a4_hat, coef = best

        a1_hat = float(np.exp(coef[0]))
        a2_hat = float(coef[1])
        a3_hat = float(coef[2])

        if (not np.isfinite(a1_hat)) or a1_hat <= 0.0:
            return {"a1": None, "a2": None, "a3": None, "b1": None, "a4": None}

        return {"a1": a1_hat, "a2": a2_hat, "a3": a3_hat, "b1": b1_hat, "a4": a4_hat}

    def calculate(self, out_df: pd.DataFrame, window_type: str, params: dict) -> pd.Series:
        """
        Returns MI (market impact cost) as per slide.
        """

        a1 = params.get("a1")
        a2 = params.get("a2")
        a3 = params.get("a3")
        b1 = params.get("b1")
        a4 = params.get("a4")

        if not all(isinstance(x, (int, float)) for x in (a1, a2, a3, b1, a4)):
            return pd.Series(np.nan, index=out_df.index, name=f"{self.name}_MI")

        vol_col = "30d Vola Bloomberg"
        adv_col = "5d ADV Non Composite" if window_type == "5d" else "25d ADV Non Composite"

        mi = pd.Series(np.nan, index=out_df.index, name=f"{self.name}_MI")

        if any(c not in out_df.columns for c in ["Order Quantity", adv_col, vol_col]):
            return mi

        qty = out_df["Order Quantity"].astype(float)
        adv_eff = out_df[adv_col].astype(float) * float(DAY_FRAC)
        sigma = out_df[vol_col].astype(float) / 100

        valid = (
            np.isfinite(qty) & np.isfinite(adv_eff) & np.isfinite(sigma)
            & (qty > 0.0) & (adv_eff > 0.0) & (sigma > 0.0)
        )
        if not valid.any():
            return mi

        pov = qty[valid] / (qty[valid] + adv_eff[valid])
        valid2 = np.isfinite(pov) & (pov > 0.0)
        if not valid2.any():
            return mi

        idx = pov.index[valid2]
        pov2 = pov.loc[idx]
        sig2 = sigma.loc[idx]

        # I* = a1 * (S/ADV)^a2 * sigma^a3 = a1 * POV^a2 * sigma^a3
        i_star = float(a1) * (pov2 ** float(a2)) * (sig2 ** float(a3))

        # MI = I* * ((1-b1) + b1 * POV^a4)
        T = (1.0 - float(b1)) + float(b1) * (pov2 ** float(a4))
        mi.loc[idx] = i_star * T

        return mi
