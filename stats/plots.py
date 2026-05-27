from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from config import (
    PLOT_Q_LEFT,
    PLOT_Q_RIGHT,
    PLOT_Q_LEFT_ERROR,
    PLOT_Q_RIGHT_ERROR,
    PLOT_Q_LEFT_IMPACT,
    PLOT_Q_RIGHT_IMPACT,
    PLOT_MIN_SPAN,
    PLOT_FIXED_CAP_ERROR,
    PLOT_FIXED_CAP_IMPACT,
)

from .regressions import predicted_vs_realised_regression

BP = 10_000.0  # decimal -> basis points


# ----------------------------
# Helpers
# ----------------------------
def _q(v, default):
    return default if v is None else float(v)


def _quantile_caps(s: pd.Series, qL: float, qR: float) -> tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return -PLOT_MIN_SPAN, PLOT_MIN_SPAN

    low = float(np.quantile(x, qL))
    high = float(np.quantile(x, qR))

    if (high - low) < PLOT_MIN_SPAN:
        mid = 0.5 * (low + high)
        low = mid - 0.5 * PLOT_MIN_SPAN
        high = mid + 0.5 * PLOT_MIN_SPAN

    return low, high


def _impact_caps(df: pd.DataFrame, qL: float, qR: float) -> tuple[float, float]:
    combined = pd.concat(
        [
            pd.to_numeric(df["ImpactRealised"], errors="coerce"),
            pd.to_numeric(df["ModelOutput"], errors="coerce"),
        ],
        ignore_index=True,
    )
    return _quantile_caps(combined, qL, qR)


def _safe_move_legend(grid):
    """Seaborn version-safe legend move."""
    try:
        sns.move_legend(grid, "upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    except Exception:
        pass


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _prepare_facet_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add plotting labels:
      - ScopeLabel
      - FacetLabel

    Rules:
      - If no CalibrationScope exists: FacetLabel = Model
      - If exactly one scope exists:  FacetLabel = Model
      - If multiple scopes exist:     FacetLabel = Model | Scope
    """
    d = df.copy()

    if "CalibrationScope" not in d.columns:
        d["ScopeLabel"] = "by_geo_size"
        d["FacetLabel"] = d["Model"].astype(str)
        return d

    d["ScopeLabel"] = d["CalibrationScope"].astype(str)
    scope_n = d["ScopeLabel"].nunique(dropna=True)

    if scope_n <= 1:
        d["FacetLabel"] = d["Model"].astype(str)
    else:
        d["FacetLabel"] = d["Model"].astype(str) + " | " + d["ScopeLabel"].astype(str)

    return d


def _facet_group_cols(df: pd.DataFrame) -> list[str]:
    """
    Regression grouping for plotting:
      - multiple scopes -> Model + CalibrationScope
      - otherwise -> Model
    """
    if "CalibrationScope" in df.columns and df["CalibrationScope"].astype(str).nunique(dropna=True) > 1:
        return ["Model", "CalibrationScope"]
    return ["Model"]


def _make_regression_facet_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regression table matching FacetLabel.
    """
    d = _prepare_facet_labels(df)
    by = _facet_group_cols(d)

    reg_df = predicted_vs_realised_regression(d, by=by)

    if reg_df.empty:
        reg_df["FacetLabel"] = []
        return reg_df

    if "CalibrationScope" in reg_df.columns and d["ScopeLabel"].nunique(dropna=True) > 1:
        reg_df["FacetLabel"] = reg_df["Model"].astype(str) + " | " + reg_df["CalibrationScope"].astype(str)
    else:
        reg_df["FacetLabel"] = reg_df["Model"].astype(str)

    return reg_df


def _ks_gap_info(x_pred: pd.Series, x_real: pd.Series) -> dict[str, float] | None:
    """
    Compute two-sample KS distance information between
    predicted distribution and realised distribution.

    Returns
    -------
    dict with:
        D       : maximal vertical ECDF distance
        x_star  : x-location where the maximal distance occurs
        F_pred  : predicted ECDF at x_star
        F_real  : realised ECDF at x_star
        N_pred  : number of predicted observations
        N_real  : number of realised observations
    """
    a = pd.Series(x_pred).dropna().to_numpy(dtype=float)
    b = pd.Series(x_real).dropna().to_numpy(dtype=float)

    if len(a) == 0 or len(b) == 0:
        return None

    a_sorted = np.sort(a)
    b_sorted = np.sort(b)

    grid = np.sort(np.unique(np.concatenate([a_sorted, b_sorted])))
    if len(grid) == 0:
        return None

    f_a = np.searchsorted(a_sorted, grid, side="right") / len(a_sorted)
    f_b = np.searchsorted(b_sorted, grid, side="right") / len(b_sorted)

    diff = np.abs(f_a - f_b)
    idx = int(np.argmax(diff))

    return {
        "D": float(diff[idx]),
        "x_star": float(grid[idx]),
        "F_pred": float(f_a[idx]),
        "F_real": float(f_b[idx]),
        "N_pred": int(len(a_sorted)),
        "N_real": int(len(b_sorted)),
    }


# ============================================================
# 1) Error Distribution
# ============================================================
def plot_error_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    qL = _q(PLOT_Q_LEFT_ERROR, _q(PLOT_Q_LEFT, 0.005))
    qR = _q(PLOT_Q_RIGHT_ERROR, 0.95)

    d = _prepare_facet_labels(df)

    if PLOT_FIXED_CAP_ERROR is not None:
        low, high = map(float, PLOT_FIXED_CAP_ERROR)
    else:
        low, high = _quantile_caps(pd.to_numeric(d["Error"], errors="coerce"), qL, qR)

    d["Error_bp"] = pd.to_numeric(d["Error"], errors="coerce").clip(lower=low, upper=high) * BP

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=d, x="Error_bp", hue="FacetLabel", common_norm=False)
    plt.xlim(low * BP, high * BP)
    plt.xlabel("Error [bp]")
    plt.title(f"Error Distribution (clipped q=[{qL},{qR}])")
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png")
    plt.close()


# ============================================================
# 2) Abs Error Boxplot
# ============================================================
def plot_abs_error_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    qL = _q(PLOT_Q_LEFT_ERROR, _q(PLOT_Q_LEFT, 0.005))
    qR = _q(PLOT_Q_RIGHT_ERROR, 0.95)

    d = _prepare_facet_labels(df)

    if PLOT_FIXED_CAP_ERROR is not None:
        low, high = map(float, PLOT_FIXED_CAP_ERROR)
    else:
        low, high = _quantile_caps(pd.to_numeric(d["Error"], errors="coerce"), qL, qR)

    d["AbsError_bp"] = pd.to_numeric(d["AbsError"], errors="coerce").clip(lower=0.0, upper=high) * BP

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=d, x="FacetLabel", y="AbsError_bp")
    plt.ylim(0, high * BP)
    plt.ylabel("Absolute Error [bp]")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.title(f"Absolute Error (cap at qR={qR})")
    plt.tight_layout()
    plt.savefig(output_dir / "abs_error_boxplot.png")
    plt.close()


# ============================================================
# 3) Predicted vs Realised (zoom + super zoom) + OLS line
# ============================================================
def _facet_scatter_with_regression(
    df_bp: pd.DataFrame,
    reg_df: pd.DataFrame,
    output_path: Path,
    low_bp: float,
    high_bp: float,
    title: str,
) -> None:
    g = sns.FacetGrid(df_bp, col="FacetLabel", col_wrap=3, height=4, sharex=True, sharey=True)
    g.map_dataframe(
        sns.scatterplot,
        x="ModelOutput_bp",
        y="ImpactRealised_bp",
        alpha=0.25,
        s=12,
    )

    reg_map = reg_df.set_index("FacetLabel").to_dict(orient="index") if not reg_df.empty else {}
    x_line = np.array([low_bp, high_bp], dtype=float)

    for ax in g.axes.flat:
        title_text = ax.get_title()  # e.g. "FacetLabel = JPMorgan_Spread | global"
        facet = title_text.split("=", 1)[1].strip() if "=" in title_text else title_text.strip()

        # 45° line
        ax.plot(
            [low_bp, high_bp],
            [low_bp, high_bp],
            ls="--",
            c="black",
            lw=1.2,
        )

        # OLS line from full-sample regression
        if facet in reg_map:
            alpha = float(reg_map[facet]["Alpha"]) * BP if pd.notna(reg_map[facet]["Alpha"]) else np.nan
            beta = float(reg_map[facet]["Beta"]) if pd.notna(reg_map[facet]["Beta"]) else np.nan
            r2 = float(reg_map[facet]["R2"]) if pd.notna(reg_map[facet]["R2"]) else np.nan
            n = int(reg_map[facet]["N"]) if pd.notna(reg_map[facet]["N"]) else 0

            if np.isfinite(alpha) and np.isfinite(beta):
                y_line = alpha + beta * x_line
                ax.plot(
                    x_line,
                    y_line,
                    color="red",
                    lw=1.6,
                )

            ax.text(
                0.03,
                0.97,
                f"OLS: y = {alpha:.2f} + {beta:.2f}x\nR² = {r2:.3f}, N = {n}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="grey",
                ),
            )

        ax.set_xlim(low_bp, high_bp)
        ax.set_ylim(low_bp, high_bp)
        ax.set_xlabel("Predicted Impact [bp]")
        ax.set_ylabel("Realised Impact [bp]")

    handles = [
        Line2D([0], [0], color="black", lw=1.2, ls="--", label="45° line"),
        Line2D([0], [0], color="red", lw=1.6, label="OLS fit"),
    ]
    g.fig.legend(
        handles=handles,
        labels=["45° line", "OLS fit"],
        loc="upper right",
        title="",
    )

    g.fig.suptitle(title, y=1.02)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_predicted_vs_realised(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    d = _prepare_facet_labels(df)
    reg_df = _make_regression_facet_map(d)

    d = d.dropna(subset=["FacetLabel", "ModelOutput", "ImpactRealised"]).copy()
    d["ModelOutput_bp"] = pd.to_numeric(d["ModelOutput"], errors="coerce") * BP
    d["ImpactRealised_bp"] = pd.to_numeric(d["ImpactRealised"], errors="coerce") * BP
    d = d.dropna(subset=["ModelOutput_bp", "ImpactRealised_bp"])

    if d.empty:
        return

    if PLOT_FIXED_CAP_IMPACT is not None:
        low, high = map(float, PLOT_FIXED_CAP_IMPACT)
        _facet_scatter_with_regression(
            d,
            reg_df,
            output_dir / "predicted_vs_realised.png",
            low * BP,
            high * BP,
            "Predicted vs Realised Impact (bp) – fixed caps, OLS on full sample",
        )
        return

    # Zoom
    qL_zoom = _q(PLOT_Q_LEFT_IMPACT, 0.02)
    qR_zoom = _q(PLOT_Q_RIGHT_IMPACT, 0.98)
    low_z, high_z = _impact_caps(d, qL_zoom, qR_zoom)

    _facet_scatter_with_regression(
        d,
        reg_df,
        output_dir / "predicted_vs_realised_zoom.png",
        low_z * BP,
        high_z * BP,
        f"Predicted vs Realised Impact (bp) – Zoom q=[{qL_zoom},{qR_zoom}], OLS on full sample",
    )

    # Superzoom
    qL_sz, qR_sz = 0.05, 0.95
    low_sz, high_sz = _impact_caps(d, qL_sz, qR_sz)

    _facet_scatter_with_regression(
        d,
        reg_df,
        output_dir / "predicted_vs_realised_superzoom.png",
        low_sz * BP,
        high_sz * BP,
        f"Predicted vs Realised Impact (bp) – SuperZoom q=[{qL_sz},{qR_sz}], OLS on full sample",
    )


# ============================================================
# 4) Impact Distributions: Predicted vs Realised (by model / scope)
# ============================================================
def plot_predicted_and_realised_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    qL = _q(PLOT_Q_LEFT_IMPACT, _q(PLOT_Q_LEFT, 0.005))
    qR = _q(PLOT_Q_RIGHT_IMPACT, 0.95)

    d = _prepare_facet_labels(df)

    if PLOT_FIXED_CAP_IMPACT is not None:
        low, high = map(float, PLOT_FIXED_CAP_IMPACT)
    else:
        low, high = _impact_caps(d, qL, qR)

    d["Predicted_bp"] = pd.to_numeric(d["ModelOutput"], errors="coerce").clip(low, high) * BP
    d["Realised_bp"] = pd.to_numeric(d["ImpactRealised"], errors="coerce").clip(low, high) * BP

    parts = []
    for facet, g_facet in d.groupby("FacetLabel", dropna=False):
        pred = g_facet["Predicted_bp"].dropna()
        rea = g_facet["Realised_bp"].dropna()

        if len(pred) > 0:
            parts.append(pd.DataFrame({"FacetLabel": facet, "Series": "Predicted", "Value": pred}))
        if len(rea) > 0:
            parts.append(pd.DataFrame({"FacetLabel": facet, "Series": "Realised", "Value": rea}))

    if not parts:
        return

    long_df = pd.concat(parts, ignore_index=True)

    series_palette = {"Predicted": "#1f77b4", "Realised": "#ff7f0e"}

    g = sns.FacetGrid(long_df, col="FacetLabel", col_wrap=3, height=3.8, sharex=True, sharey=False)
    g.map_dataframe(
        sns.histplot,
        x="Value",
        hue="Series",
        palette=series_palette,
        stat="density",
        element="step",
        bins=60,
        common_norm=False,
        legend=False,
    )

    for ax in g.axes.flat:
        ax.set_xlim(low * BP, high * BP)
        ax.set_xlabel("Market Impact [bp]")
        ax.set_yscale("log")
        ax.set_ylabel("Density (log)")

    handles = [
        Patch(facecolor=series_palette["Predicted"], edgecolor=series_palette["Predicted"], alpha=0.25, label="Predicted"),
        Patch(facecolor=series_palette["Realised"], edgecolor=series_palette["Realised"], alpha=0.25, label="Realised"),
    ]
    g.fig.legend(
        handles=handles,
        labels=["Predicted", "Realised"],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="",
    )
    g.fig.suptitle(
        f"Impact Distributions: Predicted vs Realised (bp), q=[{qL},{qR}]",
        y=1.02,
    )

    plt.savefig(output_dir / "impact_distributions_pred_vs_realised_by_model.png", bbox_inches="tight")
    plt.close()


# ============================================================
# 4b) Impact ECDFs: Predicted vs Realised (by model / scope) + KS gap
# ============================================================
def plot_predicted_and_realised_ecdf(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    qL = _q(PLOT_Q_LEFT_IMPACT, _q(PLOT_Q_LEFT, 0.005))
    qR = _q(PLOT_Q_RIGHT_IMPACT, 0.95)

    d = _prepare_facet_labels(df)

    if PLOT_FIXED_CAP_IMPACT is not None:
        low, high = map(float, PLOT_FIXED_CAP_IMPACT)
    else:
        low, high = _impact_caps(d, qL, qR)

    d["Predicted_bp"] = pd.to_numeric(d["ModelOutput"], errors="coerce").clip(low, high) * BP
    d["Realised_bp"] = pd.to_numeric(d["ImpactRealised"], errors="coerce").clip(low, high) * BP

    parts = []
    ks_by_facet: dict[str, dict[str, float] | None] = {}

    for facet, g_facet in d.groupby("FacetLabel", dropna=False):
        pred = g_facet["Predicted_bp"].dropna()
        rea = g_facet["Realised_bp"].dropna()

        if len(pred) > 0:
            parts.append(pd.DataFrame({"FacetLabel": facet, "Series": "Predicted", "Value": pred}))
        if len(rea) > 0:
            parts.append(pd.DataFrame({"FacetLabel": facet, "Series": "Realised", "Value": rea}))

        ks_by_facet[facet] = _ks_gap_info(pred, rea)

    if not parts:
        return

    long_df = pd.concat(parts, ignore_index=True)

    series_palette = {"Predicted": "#1f77b4", "Realised": "#ff7f0e"}

    g = sns.FacetGrid(long_df, col="FacetLabel", col_wrap=3, height=3.8, sharex=True, sharey=True)
    g.map_dataframe(
        sns.ecdfplot,
        x="Value",
        hue="Series",
        palette=series_palette,
        legend=False,
    )

    for ax in g.axes.flat:
        title_text = ax.get_title()  # e.g. "FacetLabel = JPMorgan_Spread | global"
        facet = title_text.split("=", 1)[1].strip() if "=" in title_text else title_text.strip()

        ax.set_xlim(low * BP, high * BP)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Market Impact [bp]")
        ax.set_ylabel("ECDF")

        info = ks_by_facet.get(facet)
        if info is not None:
            x_star = float(info["x_star"])
            f_pred = float(info["F_pred"])
            f_real = float(info["F_real"])
            d_ks = float(info["D"])
            n_pred = int(info["N_pred"])
            n_real = int(info["N_real"])

            y_low = min(f_pred, f_real)
            y_high = max(f_pred, f_real)

            # location of maximal KS gap
            ax.axvline(
                x=x_star,
                color="grey",
                linestyle=":",
                linewidth=1.2,
                alpha=0.9,
            )

            # vertical KS gap
            ax.plot(
                [x_star, x_star],
                [y_low, y_high],
                color="purple",
                linewidth=2.2,
                alpha=0.95,
            )

            # markers at both ECDF values
            ax.plot([x_star], [f_pred], marker="o", markersize=4, color=series_palette["Predicted"])
            ax.plot([x_star], [f_real], marker="o", markersize=4, color=series_palette["Realised"])

            ax.text(
                0.5,
                0.5,
                f"KS D = {d_ks:.3f}\n"
                f"x* = {x_star:.2f} bp\n"
                f"Np = {n_pred}, Nr = {n_real}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.5,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="grey",
                ),
            )

    handles = [
        Line2D([0], [0], color=series_palette["Predicted"], lw=2, label="Predicted"),
        Line2D([0], [0], color=series_palette["Realised"], lw=2, label="Realised"),
        Line2D([0], [0], color="grey", lw=1.2, ls=":", label="KS location x*"),
        Line2D([0], [0], color="purple", lw=2.2, label="KS gap D"),
    ]
    g.fig.legend(
        handles=handles,
        labels=["Predicted", "Realised", "KS location x*", "KS gap D"],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="",
    )
    g.fig.suptitle(
        f"Impact ECDFs: Predicted vs Realised (bp), q=[{qL},{qR}] + KS gap on plotted basis",
        y=1.02,
    )

    plt.savefig(output_dir / "impact_ecdf_pred_vs_realised_by_model.png", bbox_inches="tight")
    plt.close()


# ============================================================
# 5) Overlay plot: Realised behind + model predicted overlays
# ============================================================
def plot_all_impacts_overlay(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    qL = _q(PLOT_Q_LEFT_IMPACT, _q(PLOT_Q_LEFT, 0.005))
    qR = _q(PLOT_Q_RIGHT_IMPACT, 0.95)

    d = _prepare_facet_labels(df)

    if PLOT_FIXED_CAP_IMPACT is not None:
        low, high = map(float, PLOT_FIXED_CAP_IMPACT)
    else:
        low, high = _impact_caps(d, qL, qR)

    d["Predicted_bp"] = pd.to_numeric(d["ModelOutput"], errors="coerce").clip(low, high) * BP
    d["Realised_bp"] = pd.to_numeric(d["ImpactRealised"], errors="coerce").clip(low, high) * BP

    fig, ax = plt.subplots(figsize=(9, 5))

    # Realised first, in background
    sns.histplot(
        data=d,
        x="Realised_bp",
        stat="density",
        element="step",
        bins=80,
        color="black",
        alpha=0.20,
        linewidth=2.0,
        ax=ax,
        label="Realised",
    )

    # Predicted by facet label (Model or Model|Scope)
    sns.histplot(
        data=d,
        x="Predicted_bp",
        hue="FacetLabel",
        stat="density",
        element="step",
        bins=80,
        common_norm=False,
        alpha=0.15,
        linewidth=1.5,
        ax=ax,
        legend=False,
    )

    ax.set_xlim(low * BP, high * BP)
    ax.set_yscale("log")
    ax.set_xlabel("Market Impact [bp]")
    ax.set_ylabel("Density (log)")
    ax.set_title(
        f"Predicted Impact Distributions + Realised (bp), q=[{qL},{qR}]"
    )

    facet_names = list(pd.unique(d["FacetLabel"]))
    palette = sns.color_palette(n_colors=len(facet_names))
    facet_handles = [
        Line2D([0], [0], color=palette[i], lw=2, label=facet_names[i])
        for i in range(len(facet_names))
    ]
    realised_handle = Line2D([0], [0], color="black", lw=3, label="Realised")

    handles = [realised_handle] + facet_handles
    labels = ["Realised"] + facet_names
    ax.legend(handles=handles, labels=labels, loc="upper right", title="")

    fig.tight_layout()
    fig.savefig(output_dir / "impact_distributions_overlay.png")
    plt.close(fig)