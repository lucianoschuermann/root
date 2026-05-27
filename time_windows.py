import pandas as pd
from typing import Iterable
from config import CALIBRATION_MONTHS

def prepare_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Arrival Time"] = pd.to_datetime(df["Arrival Time"])
    df["Month"] = df["Arrival Time"].dt.to_period("M").dt.to_timestamp("M")
    return df


def generate_rolling_windows(
    months: Iterable[pd.Timestamp],
):
    months_sorted = sorted(months)
    return [
        (
            months_sorted[i],
            months_sorted[i + CALIBRATION_MONTHS - 1],
            months_sorted[i + CALIBRATION_MONTHS],
        )
        for i in range(len(months_sorted) - CALIBRATION_MONTHS)
    ]