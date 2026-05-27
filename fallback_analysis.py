import pandas as pd
from pathlib import Path


def run_fallback_analysis(evaluation_file, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading evaluation data...")

    if isinstance(evaluation_file, pd.DataFrame):
        df = evaluation_file.copy()
    else:
        df = pd.read_csv(evaluation_file)

    if "CalibrationScope" not in df.columns:
        print("[ERROR] CalibrationScope missing")
        return

    if "FallbackUsed" not in df.columns:
        print("[ERROR] FallbackUsed missing")
        return

    summary = (
        df.groupby("CalibrationScope")["FallbackUsed"]
        .agg(["count", "sum"])
        .rename(columns={"sum": "fallback_count"})
    )

    summary["fallback_rate"] = summary["fallback_count"] / summary["count"]
    summary = summary.reset_index()

    output_path = output_dir / "fallback_summary.xlsx"
    summary.to_excel(output_path, index=False)

    print(f"[OK] Saved fallback table: {output_path}")