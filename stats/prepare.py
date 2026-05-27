import pandas as pd
import numpy as np


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add error-related columns used by all statistics functions.

    Expected columns:
        - ModelOutput
        - ImpactRealised

    Added columns:
        - Error
        - AbsError
        - RelError
        - SignedPctError
        - APE
    """
    out = df.copy()

    if "ModelOutput" not in out.columns:
        raise KeyError("add_error_columns: missing required column 'ModelOutput'")
    if "ImpactRealised" not in out.columns:
        raise KeyError("add_error_columns: missing required column 'ImpactRealised'")

    out["ModelOutput"] = pd.to_numeric(out["ModelOutput"], errors="coerce")
    out["ImpactRealised"] = pd.to_numeric(out["ImpactRealised"], errors="coerce")

    out["Error"] = out["ModelOutput"] - out["ImpactRealised"]
    out["AbsError"] = out["Error"].abs()

    realised_nonzero = out["ImpactRealised"].replace(0, np.nan)

    out["RelError"] = out["Error"] / realised_nonzero
    out["SignedPctError"] = 100.0 * out["RelError"]
    out["APE"] = out["RelError"].abs()

    return out
