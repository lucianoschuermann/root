from pathlib import Path

# =========================
# IMPORTS
# =========================
from data_loader import load_trades_from_excel
from calibration_pipeline import run_calibration_pipeline
from evaluation_pipeline import run_outsample_evaluation
from stats.reporting import run_full_statistics
from parameter_analysis import run_parameter_analysis
from fallback_analysis import run_fallback_analysis


def main():

    base_dir = Path(__file__).resolve().parent

    # ✅ EIN TIMESTAMP FÜR DEN GANZEN RUN
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[INFO] Run timestamp: {timestamp}")

    # -------------------------
    # 0. LOAD DATA
    # -------------------------
    print("[INFO] Loading trade data...")

    trades_file = base_dir / "trades.xlsx"
    print(f"[INFO] Using trades file: {trades_file}")

    trades = load_trades_from_excel(trades_file)
    print(f"[OK] Loaded {len(trades)} trades")

    # -------------------------
    # 1. CALIBRATION
    # -------------------------
    print("[INFO] Running calibration...")
    calibration_df = run_calibration_pipeline(trades)
    print(f"[OK] Calibration done ({len(calibration_df)} rows)")

    calibration_output_path = base_dir / f"calibration_results_{timestamp}.xlsx"
    calibration_df.to_excel(calibration_output_path, index=False)
    print(f"[OK] Calibration file saved: {calibration_output_path}")

    # -------------------------
    # 2. OUT-OF-SAMPLE
    # -------------------------
    print("[INFO] Running out-of-sample evaluation...")

    evaluation_df = run_outsample_evaluation(
        trades=trades,
        calibration_results=calibration_df
    )

    print(f"[OK] Evaluation done ({len(evaluation_df)} rows)")
    print("[DEBUG] Columns BEFORE fix:", evaluation_df.columns.tolist())

    # ✅ ✅ FIX 1: Geography
    if "Geography" not in evaluation_df.columns and "EvalGeography" in evaluation_df.columns:
        print("[INFO] Renaming EvalGeography → Geography...")
        evaluation_df = evaluation_df.rename(columns={"EvalGeography": "Geography"})

    # ✅ ✅ FIX 2: SizeBucket
    if "SizeBucket" not in evaluation_df.columns and "EvalSizeBucket" in evaluation_df.columns:
        print("[INFO] Renaming EvalSizeBucket → SizeBucket...")
        evaluation_df = evaluation_df.rename(columns={"EvalSizeBucket": "SizeBucket"})

    print("[DEBUG] Columns AFTER fix:", evaluation_df.columns.tolist())

    evaluation_output_path = base_dir / f"outsample_evaluation_{timestamp}.csv"
    evaluation_df.to_csv(evaluation_output_path, index=False)
    print(f"[OK] Evaluation file saved: {evaluation_output_path}")

    # -------------------------
    # 3. REPORTING
    # -------------------------
    print("[INFO] Running reporting...")

    statistics_output_dir = base_dir / f"statistics_output_{timestamp}"

    run_full_statistics(
        evaluation_results=evaluation_df,
        output_dir=statistics_output_dir,
    )

    # -------------------------
    # 4. EXTRA ANALYSIS
    # -------------------------
    print("[INFO] Running additional analyses...")

    extra_output = base_dir / f"analysis_outputs_{timestamp}"

    run_parameter_analysis(
        calibration_file=calibration_df,
        output_dir=extra_output / "parameters"
    )

    run_fallback_analysis(
        evaluation_file=evaluation_df,
        output_dir=extra_output / "fallback"
    )

    print("[SUCCESS] FULL PIPELINE COMPLETED ✅")


if __name__ == "__main__":
    main()
