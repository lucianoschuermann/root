from abc import ABC, abstractmethod
import pandas as pd

class MarketImpactModel(ABC):
    name: str

    @abstractmethod
    def calibrate(self, calib_df: pd.DataFrame, window_type: str) -> dict:
        ...

    @abstractmethod
    def calculate(self, out_df: pd.DataFrame, window_type: str, params: dict) -> pd.Series:
        ...
