from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BP = 10_000.0  # decimal -> basis points


# ==================================================
# HELPER: NAMING / LABELS
# ==================================================
def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _prepare_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add plotting group labels:
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


# ==================================================
# HELPER: PREPARE DATA
# ==================================================
def _prepare_series(
    real: pd.Series,
    pred: pd.Series,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    df = pd.DataFrame(
        {
            "real": pd.to_numeric(real, errors="coerce"),
            "pred": pd.to_numeric(pred, errors="coerce"),
        }
    ).dropna()

    if len(df) < 10:
        return None, None

    real_sorted = np.sort(df["real"].to_numpy(dtype=float))
    pred_sorted = np.sort(df["pred"].to_numpy(dtype=float))

    return real_sorted, pred_sorted


# ==================================================
# CORE QQ PLOT FUNCTION
# ==================================================
def qq_plot(
    real: pd.Series,
    pred: pd.Series,
    plot_label: str,
    output_dir: str | Path = "statistics_output",
    tail_only: bool = False,
) -> None:
    real_sorted, pred_sorted = _prepare_series(real, pred)

    if real_sorted is None or pred_sorted is None:
        return

    # Tail filter: top 10%
    if tail_only:
        cutoff = int(0.9 * len(real_sorted))
        real_sorted = real_sorted[cutoff:]
        pred_sorted = pred_sorted[cutoff:]

        if len(real_sorted) < 5 or len(pred_sorted) < 5:
            return

    plt.figure(figsize=(6, 6))
    plt.scatter(real_sorted, pred_sorted, alpha=0.6)

    # 45° reference line
    min_val = float(min(real_sorted.min(), pred_sorted.min()))
    max_val = float(max(real_sorted.max(), pred_sorted.max()))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="black",
        linewidth=1.2,
    )

    suffix = " (Top 10% Tail)" if tail_only else ""
    plt.title(f"QQ Plot - {plot_label}{suffix}")
    plt.xlabel("Realised Quantiles [bp]")
    plt.ylabel("Predicted Quantiles [bp]")
    plt.grid(True, alpha=0.3)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"qq_{_safe_name(plot_label)}"
    if tail_only:
        filename += "_tail"
    filename += ".png"

    filepath = output_path / filename
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


# ==================================================
# MAIN ENTRY FUNCTION
# ==================================================
def run_qq_from_dataframe(
    df: pd.DataFrame,
    output_dir: str | Path = "statistics_output",
) -> None:
    """
    Complete QQ analysis in basis points (bp).

    Requires:
        df with columns:
            - Model
            - ModelOutput
            - ImpactRealised
        optional:
            - CalibrationScope

    Creates:
        - QQ plots (full distribution)
        - QQ plots (top 10% tail)
        - Saves everything to output_dir

    Notes:
        - Both series are scaled from decimal to bp via * 10,000
        - QQ compares distribution quantiles, not observation-by-observation fit
        - If multiple CalibrationScope values are present, QQ plots are created
          per (Model | Scope)
    """
    if df is None or len(df) == 0:
        print("QQ Analysis: dataframe empty")
        return

    required_cols = {"Model", "ModelOutput", "ImpactRealised"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"QQ Analysis: missing required columns: {missing}")
        return

    d = _prepare_groups(df)

    groups = list(d["FacetLabel"].dropna().unique())
    if len(groups) == 0:
        print("QQ Analysis: no groups found")
        return

    print(f"Running QQ analysis in bp for {len(groups)} model/scope groups...")

    for facet in groups:
        df_group = d[d["FacetLabel"] == facet].copy()

        if len(df_group) < 10:
            print(f"Skipping {facet} (too few data points)")
            continue

        # Scale to basis points
        real = pd.to_numeric(df_group["ImpactRealised"], errors="coerce") * BP
        pred = pd.to_numeric(df_group["ModelOutput"], errors="coerce") * BP

        # Standard QQ
        qq_plot(
            real=real,
            pred=pred,
            plot_label=str(facet),
            output_dir=output_dir,
            tail_only=False,
        )

        # Tail QQ (top 10%)
        qq_plot(
            real=real,
            pred=pred,
            plot_label=str(facet),
            output_dir=output_dir,
            tail_only=True,
        )

    print(f"QQ analysis complete -> saved to '{output_dir}'")