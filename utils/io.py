from pathlib import Path
from datetime import datetime
import pandas as pd

EXCEL_MAX_ROWS = 1_048_576


def save_results_with_timestamp(
    df: pd.DataFrame,
    base_name: str,
    base_dir: Path,
) -> Path:
    """
    Save DataFrame with timestamp.

    Logic:
    - if DataFrame fits into one Excel sheet -> save as .xlsx
    - otherwise -> save as .csv

    Returns
    -------
    Path to the saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir.mkdir(parents=True, exist_ok=True)

    if len(df) <= EXCEL_MAX_ROWS:
        output_path = base_dir / f"{base_name}_{timestamp}.xlsx"
        df.to_excel(
            output_path,
            index=False,
            engine="openpyxl",
        )
    else:
        output_path = base_dir / f"{base_name}_{timestamp}.csv"
        df.to_csv(
            output_path,
            index=False,
        )

    return output_path