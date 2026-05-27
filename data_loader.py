import pandas as pd
from pathlib import Path


def load_trades_from_excel(file_path):

    print(f"[DEBUG] Reading Excel: {file_path}")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # ✅ stabiler Load
    df = pd.read_excel(
        file_path,
        engine="openpyxl"
    )

    print(f"[DEBUG] Finished reading Excel ({len(df)} rows)")

    return df