import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def run_parameter_analysis(calibration_file, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading calibration data...")

    # ✅ funktioniert mit DataFrame UND Excel
    if isinstance(calibration_file, pd.DataFrame):
        df = calibration_file.copy()
    else:
        df = pd.read_excel(calibration_file)

    # -------------------------
    # FILTER
    # -------------------------
    if "Model" not in df.columns or "CalibrationScope" not in df.columns:
        print("[WARNING] Required columns missing → skipping")
        return

    df = df[
        (df["Model"] == "JPMorgan Spread") &
        (df["CalibrationScope"] == "global")
    ].copy()

    if df.empty:
        print("[WARNING] No JPMorgan Spread global data found")
        return

    if "CalibrationMonth" not in df.columns:
        print("[WARNING] CalibrationMonth missing → skipping plot")
        return

    df = df.sort_values("CalibrationMonth")

    param_cols = ["alpha", "beta", "gamma", "omega"]

    # -------------------------
    # PLOT
    # -------------------------
    plt.figure(figsize=(10, 6))

    for col in param_cols:
        if col in df.columns:
            plt.plot(df["CalibrationMonth"], df[col], label=col)

    plt.title("Parameter Stability (JPMorgan Spread - Global)")
    plt.xlabel("Calibration Month")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)

    plot_path = output_dir / "parameter_stability.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"[OK] Plot saved: {plot_path}")

    # -------------------------
    # TABLE
    # -------------------------
    cols = ["CalibrationMonth"] + [c for c in param_cols if c in df.columns]
    table = df[cols]

    excel_path = output_dir / "parameter_table_global.xlsx"
    table.to_excel(excel_path, index=False)

    print(f"[OK] Table saved: {excel_path}")