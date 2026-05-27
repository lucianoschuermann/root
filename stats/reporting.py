from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd

from .prepare import add_error_columns
from .metrics import summary_metrics, coverage_metrics
from .distributions import (
    distribution_moments,
    realised_moments,
    ks_tests,
    impact_moments_comparison,
)
from .regressions import predicted_vs_realised_regression
from .QQplots import run_qq_from_dataframe
from .plots import (
    plot_error_distribution,
    plot_abs_error_boxplot,
    plot_predicted_vs_realised,
    plot_predicted_and_realised_distributions,
    plot_predicted_and_realised_ecdf,
    plot_all_impacts_overlay,
)

# =========================================================
# Helpers
# =========================================================

def _normalise_reporting_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "CalibrationScope" not in df.columns:
        df["CalibrationScope"] = "global"

    if "WindowType" not in df.columns:
        df["WindowType"] = "ALL"

    df["CalibrationScope"] = df["CalibrationScope"].astype(str)
    df["WindowType"] = df["WindowType"].astype(str)

    return df


def _iter_scope_subsets(df: pd.DataFrame) -> Iterable[Tuple[str, pd.DataFrame]]:
    scopes = sorted(df["CalibrationScope"].dropna().unique())

    for scope in scopes:
        sub = df[df["CalibrationScope"] == scope].copy()
        if not sub.empty:
            yield str(scope), sub


# ✅ ✅ WICHTIGSTE FUNKTION (Model + SizeBucket integriert)
def _concat_scopewise(df: pd.DataFrame, func) -> pd.DataFrame:

    parts = []

    for scope, sub in _iter_scope_subsets(df):

        # =============================
        # ✅ Gruppierungslogik
        # =============================
        by_cols = []

        # Reihenfolge bewusst definiert
        if "Model" in sub.columns:
            by_cols.append("Model")

        if "SizeBucket" in sub.columns:
            by_cols.append("SizeBucket")

        if "WindowType" in sub.columns:
            by_cols.append("WindowType")

        if "Geography" in sub.columns:
            by_cols.append("Geography")

        # =============================
        # Metrics berechnen
        # =============================
        out = func(sub, by=by_cols)

        if out is None or not isinstance(out, pd.DataFrame):
            continue

        out = out.copy()

        # =============================
        # ✅ Spalten-Reihenfolge erzwingen
        # =============================
        ordered_cols = []

        for col in ["Model", "SizeBucket", "WindowType", "Geography"]:
            if col in out.columns:
                ordered_cols.append(col)

        remaining_cols = [c for c in out.columns if c not in ordered_cols]
        out = out[ordered_cols + remaining_cols]

        # Scope hinzufügen
        out["CalibrationScope"] = scope

        parts.append(out)

    if parts:
        return pd.concat(parts, ignore_index=True)
    else:
        return pd.DataFrame()


def _safe_plot(name: str, func):
    try:
        print(f"[OK start] {name}")
        func()
        print(f"[OK done ] {name}")
    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__} - {e}")


# =========================================================
# Plot Runner
# =========================================================

def _run_scope_plots(df: pd.DataFrame, output_dir: Path) -> None:

    scopes = list(_iter_scope_subsets(df))
    multi_scope = len(scopes) > 1

    for scope, sub_scope in scopes:

        scope_dir = output_dir / f"scope_{scope}" if multi_scope else output_dir
        scope_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Scope: {scope}")

        window_types = sorted(sub_scope["WindowType"].dropna().unique())

        for window_type in window_types:
            sub = sub_scope[sub_scope["WindowType"] == window_type].copy()

            if sub.empty:
                continue

            window_dir = scope_dir / f"window_{window_type}"
            window_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO]  -> WindowType: {window_type} ({len(sub)} rows)")

            _safe_plot("error_distribution",
                lambda: plot_error_distribution(sub, output_dir=window_dir)
            )

            _safe_plot("abs_error_boxplot",
                lambda: plot_abs_error_boxplot(sub, output_dir=window_dir)
            )

            _safe_plot("pred_vs_real",
                lambda: plot_predicted_vs_realised(sub, output_dir=window_dir)
            )

            _safe_plot("distribution_compare",
                lambda: plot_predicted_and_realised_distributions(sub, output_dir=window_dir)
            )

            _safe_plot("ecdf",
                lambda: plot_predicted_and_realised_ecdf(sub, output_dir=window_dir)
            )

            _safe_plot("overlay_impacts",
                lambda: plot_all_impacts_overlay(sub, output_dir=window_dir)
            )

            _safe_plot("qq_plot",
                lambda: run_qq_from_dataframe(sub, output_dir=window_dir)
            )

    print("[OK] Plot generation completed.")


# =========================================================
# Main Pipeline
# =========================================================

def run_full_statistics(evaluation_results: pd.DataFrame, output_dir: Path) -> None:

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Starting full statistics pipeline...")

    # Prepare
    df = _normalise_reporting_columns(evaluation_results)
    df = add_error_columns(df)

    # Metrics
    print("[INFO] Computing summary metrics...")
    sm = _concat_scopewise(df, summary_metrics)

    print("[INFO] Computing coverage metrics...")
    cm = _concat_scopewise(df, coverage_metrics)

    print("[INFO] Computing distributions...")
    dm = _concat_scopewise(df, distribution_moments)
    rm = _concat_scopewise(df, realised_moments)
    km = _concat_scopewise(df, ks_tests)
    im = _concat_scopewise(df, impact_moments_comparison)

    print("[INFO] Running regression...")
    reg = _concat_scopewise(df, predicted_vs_realised_regression)

    # Excel Output
    excel_path = output_dir / f"full_statistics_{timestamp}.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        if not sm.empty:
            sm.to_excel(writer, sheet_name="summary_metrics", index=False)

        if not cm.empty:
            cm.to_excel(writer, sheet_name="coverage_metrics", index=False)

        if not dm.empty:
            dm.to_excel(writer, sheet_name="distribution_moments", index=False)

        if not rm.empty:
            rm.to_excel(writer, sheet_name="realised_moments", index=False)

        if not km.empty:
            km.to_excel(writer, sheet_name="ks_tests", index=False)

        if not im.empty:
            im.to_excel(writer, sheet_name="impact_moments", index=False)

        if not reg.empty:
            reg.to_excel(writer, sheet_name="regression", index=False)

    print(f"[OK] Excel report saved: {excel_path}")

    # Plots
    print("[INFO] Generating plots...")
    _run_scope_plots(df, output_dir / "plots")

    print("[SUCCESS] Full statistics pipeline completed.")