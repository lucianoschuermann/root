import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any

from config import MIN_CALIBRATION_N, DAY_FRAC
from .base import MarketImpactModel


class JPMorganNoSpreadModel(MarketImpactModel):
    """
    JPMorgan model WITHOUT explicit spread term.

    MI_total = ω * I * (2*PoV/(1+PoV)) + (1-ω)*I
             = I * [1 + ω*(X-1)]
      where X = 2*PoV/(1+PoV)

    I = α * PoV^β * Vol^γ

    Calibration target:
        MI_target = ImpactRealised

    Design choices:
    - Magnitude model
    - Positive targets are FILTERED (not clipped)
    - Final model uses omega_unclipped directly (Variant 2)
    - omega_clipped is kept only as a diagnostic field
    """

    name = "JPMorgan_NoSpread"

    EPS = 1e-6
    DELTA = 0.10
    OMEGA_GRID_N = 301
    FINE_GRID_N = 401
    OMEGA_MIN = 0.001
    OMEGA_MAX = 0.999
    OMEGA_CLIP_MIN = 0.05
    OMEGA_CLIP_MAX = 0.95

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    @staticmethod
    def _is_finite_number(x: Any) -> bool:
        return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

    @staticmethod
    def _to_float_or_nan(x: Any) -> float:
        if isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x):
            return float(x)
        return np.nan

    def _null_result(self, note: str) -> dict:
        return {
            "alpha": None,
            "beta": None,
            "gamma": None,
            "omega": None,
            "omega_unclipped": None,
            "omega_clipped": None,
            "omega_best_loss": None,
            "omega_delta": float(self.DELTA),
            "omega_loss_left": None,
            "omega_loss_right": None,
            "omega_flatness_rel": None,
            "omega_best_source": None,
            "omega_n_used": None,
            "n_raw": 0,
            "n_after_dropna": 0,
            "n_after_basic_valid": 0,
            "n_positive_target": 0,
            "share_positive_target": None,
            "calibration_note": note,
        }

    def _empty_output(self, index: pd.Index) -> pd.Series:
        return pd.Series(np.nan, index=index, name=f"{self.name}_MI", dtype=float)

    @staticmethod
    def _ols_alpha_beta_gamma(
        lpov: np.ndarray,
        lvol: np.ndarray,
        lI: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Stable OLS:
            log(I) = c + beta*log(PoV) + gamma*log(Vol)
        """
        X = np.column_stack([np.ones(len(lI)), lpov, lvol]).astype(np.float64)
        y = lI.astype(np.float64)

        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            lam = 1e-6
            coef = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y)

        return float(np.exp(coef[0])), float(coef[1]), float(coef[2])

    @staticmethod
    def _sse(resid: np.ndarray) -> float:
        return float(np.sum(resid * resid))

    # ---------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------
    def calibrate(
        self,
        calib_df: pd.DataFrame,
        window_type: str,
        omega_fixed: Optional[float] = None,
    ) -> dict:
        null = self._null_result("too_few_rows_initial")

        n_raw = int(len(calib_df))
        null["n_raw"] = n_raw

        if n_raw < MIN_CALIBRATION_N:
            return null

        vol_col = "30d Vola Bloomberg"
        adv_col = "5d ADV Non Composite" if window_type == "5d" else "25d ADV Non Composite"

        required_cols = ["Order Quantity", adv_col, vol_col, "ImpactRealised"]
        missing_cols = [c for c in required_cols if c not in calib_df.columns]
        if missing_cols:
            null["calibration_note"] = f"missing_columns: {', '.join(missing_cols)}"
            return null

        df = calib_df.dropna(subset=required_cols).copy()
        n_after_dropna = int(len(df))
        null["n_after_dropna"] = n_after_dropna

        if n_after_dropna < MIN_CALIBRATION_N:
            null["calibration_note"] = "too_few_rows_after_dropna"
            return null

        adv_eff = df[adv_col].astype(float) * float(DAY_FRAC)
        pov = df["Order Quantity"].astype(float) / adv_eff
        vol = df[vol_col].astype(float) / 100.0
        mi_target = df["ImpactRealised"].astype(float)

        basic_valid = (
            (adv_eff > 0.0) & np.isfinite(adv_eff)
            & (pov > 0.0) & np.isfinite(pov)
            & (vol > 0.0) & np.isfinite(vol)
            & np.isfinite(mi_target)
        )

        df = df.loc[basic_valid].copy()
        pov = pov.loc[basic_valid]
        vol = vol.loc[basic_valid]
        mi_target = mi_target.loc[basic_valid]

        n_after_basic_valid = int(len(df))
        null["n_after_basic_valid"] = n_after_basic_valid

        if n_after_basic_valid < MIN_CALIBRATION_N:
            null["calibration_note"] = "too_few_rows_after_basic_valid_filter"
            return null

        positive_target = mi_target > self.EPS
        n_positive_target = int(positive_target.sum())
        null["n_positive_target"] = n_positive_target
        null["share_positive_target"] = (
            float(n_positive_target / n_after_basic_valid) if n_after_basic_valid > 0 else None
        )

        df = df.loc[positive_target].copy()
        pov = pov.loc[positive_target]
        vol = vol.loc[positive_target]
        mi_target = mi_target.loc[positive_target]

        if len(df) < MIN_CALIBRATION_N:
            null["calibration_note"] = "too_few_positive_targets"
            return null

        pov_np = pov.to_numpy(np.float64)
        vol_np = vol.to_numpy(np.float64)
        mi_np = mi_target.to_numpy(np.float64)

        lpov = np.log(pov_np)
        lvol = np.log(vol_np)
        lmi = np.log(mi_np)

        m = np.isfinite(lpov) & np.isfinite(lvol) & np.isfinite(lmi)
        lpov, lvol, lmi = lpov[m], lvol[m], lmi[m]
        pov_np, vol_np, mi_np = pov_np[m], vol_np[m], mi_np[m]

        n_used = int(lpov.size)
        if n_used < MIN_CALIBRATION_N:
            null["calibration_note"] = "too_few_rows_after_log_filter"
            return null

        X = (2.0 * pov_np) / (1.0 + pov_np)
        xm1 = X - 1.0

        def eval_loss_params(omega: float) -> Tuple[float, float, float, float]:
            denom = 1.0 + omega * xm1
            if np.any(~np.isfinite(denom)) or np.any(denom <= 0.0):
                return np.inf, np.nan, np.nan, np.nan

            I_imp = mi_np / denom
            valid_inner = np.isfinite(I_imp) & (I_imp > self.EPS)
            if valid_inner.sum() < MIN_CALIBRATION_N:
                return np.inf, np.nan, np.nan, np.nan

            lpov_i = lpov[valid_inner]
            lvol_i = lvol[valid_inner]
            lmi_i = lmi[valid_inner]
            pov_i = pov_np[valid_inner]
            vol_i = vol_np[valid_inner]
            denom_i = denom[valid_inner]
            I_imp_i = I_imp[valid_inner]

            lI = np.log(I_imp_i)
            alpha, beta, gamma = self._ols_alpha_beta_gamma(lpov_i, lvol_i, lI)

            I_hat = alpha * (pov_i ** beta) * (vol_i ** gamma)
            valid_hat = (
                np.isfinite(I_hat) & (I_hat > self.EPS)
                & np.isfinite(denom_i) & (denom_i > 0.0)
            )
            if valid_hat.sum() < MIN_CALIBRATION_N:
                return np.inf, np.nan, np.nan, np.nan

            lmi_hat = np.log(I_hat[valid_hat]) + np.log(denom_i[valid_hat])
            resid = lmi_i[valid_hat] - lmi_hat
            loss = self._sse(resid)

            if not np.isfinite(loss):
                return np.inf, np.nan, np.nan, np.nan

            return float(loss), float(alpha), float(beta), float(gamma)

        if self._is_finite_number(omega_fixed):
            omega0 = float(np.clip(float(omega_fixed), self.OMEGA_MIN, self.OMEGA_MAX))
            best_loss, alpha0, beta0, gamma0 = eval_loss_params(omega0)
            if not np.isfinite(best_loss):
                null["calibration_note"] = "omega_fixed_invalid_loss"
                return null
            source = "omega_fixed"
        else:
            best_loss = np.inf
            omega0 = None
            alpha0 = beta0 = gamma0 = np.nan

            for omega in np.linspace(self.OMEGA_MIN, self.OMEGA_MAX, self.OMEGA_GRID_N):
                loss, a, b, g = eval_loss_params(float(omega))
                if loss < best_loss:
                    best_loss = loss
                    omega0, alpha0, beta0, gamma0 = float(omega), a, b, g

            if omega0 is None or not np.isfinite(best_loss):
                null["calibration_note"] = "no_valid_omega_found"
                return null

            for omega in np.linspace(
                max(self.OMEGA_MIN, omega0 - 0.05),
                min(self.OMEGA_MAX, omega0 + 0.05),
                self.FINE_GRID_N,
            ):
                loss, a, b, g = eval_loss_params(float(omega))
                if loss < best_loss:
                    best_loss = loss
                    omega0, alpha0, beta0, gamma0 = float(omega), a, b, g

            source = "profile_scan"

        omega_left = float(max(self.OMEGA_MIN, omega0 - self.DELTA))
        omega_right = float(min(self.OMEGA_MAX, omega0 + self.DELTA))

        loss_left, _, _, _ = eval_loss_params(omega_left)
        loss_right, _, _, _ = eval_loss_params(omega_right)

        flatness_rel = (
            (0.5 * (loss_left + loss_right) - best_loss) / (best_loss + 1e-12)
            if np.isfinite(loss_left) and np.isfinite(loss_right) and np.isfinite(best_loss)
            else np.nan
        )

        omega_clipped = float(np.clip(omega0, self.OMEGA_CLIP_MIN, self.OMEGA_CLIP_MAX))

        return {
            "alpha": float(alpha0),
            "beta": float(beta0),
            "gamma": float(gamma0),
            "omega": float(omega0),  # Variant 2: use unclipped omega

            "omega_unclipped": float(omega0),
            "omega_clipped": float(omega_clipped),
            "omega_best_loss": float(best_loss),
            "omega_delta": float(self.DELTA),
            "omega_loss_left": float(loss_left) if np.isfinite(loss_left) else np.nan,
            "omega_loss_right": float(loss_right) if np.isfinite(loss_right) else np.nan,
            "omega_flatness_rel": float(flatness_rel) if np.isfinite(flatness_rel) else np.nan,
            "omega_best_source": source,
            "omega_n_used": n_used,

            "n_raw": n_raw,
            "n_after_dropna": n_after_dropna,
            "n_after_basic_valid": n_after_basic_valid,
            "n_positive_target": n_positive_target,
            "share_positive_target": (
                float(n_positive_target / n_after_basic_valid) if n_after_basic_valid > 0 else np.nan
            ),
            "calibration_note": "",
        }

    # ---------------------------------------------------------
    # Calculation
    # ---------------------------------------------------------
    def calculate(self, out_df: pd.DataFrame, window_type: str, params: dict) -> pd.Series:
        alpha = self._to_float_or_nan(params.get("alpha"))
        beta = self._to_float_or_nan(params.get("beta"))
        gamma = self._to_float_or_nan(params.get("gamma"))
        omega = self._to_float_or_nan(params.get("omega"))

        if not all(np.isfinite(x) for x in (alpha, beta, gamma, omega)):
            return self._empty_output(out_df.index)

        vol_col = "30d Vola Bloomberg"
        adv_col = "5d ADV Non Composite" if window_type == "5d" else "25d ADV Non Composite"

        required_cols = ["Order Quantity", adv_col, vol_col]
        if any(c not in out_df.columns for c in required_cols):
            return self._empty_output(out_df.index)

        d = out_df.copy()

        adv_eff = d[adv_col].astype(float) * float(DAY_FRAC)
        pov = d["Order Quantity"].astype(float) / adv_eff
        vol = d[vol_col].astype(float) / 100.0

        out = self._empty_output(d.index)

        valid = (
            (adv_eff > 0.0) & np.isfinite(adv_eff)
            & (pov > 0.0) & np.isfinite(pov)
            & (vol > 0.0) & np.isfinite(vol)
        )

        if not valid.any():
            return out

        pov_v = pov.loc[valid].to_numpy(dtype=float)
        vol_v = vol.loc[valid].to_numpy(dtype=float)

        I = alpha * (pov_v ** beta) * (vol_v ** gamma)
        MI = omega * I * (2.0 * pov_v / (1.0 + pov_v)) + (1.0 - omega) * I
        MI = np.where(np.isfinite(MI), MI, np.nan)

        out.loc[valid] = MI
        return out
