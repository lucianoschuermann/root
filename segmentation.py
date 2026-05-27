import pandas as pd
from config import SIZE_BUCKETS

def assign_size_bucket(adv: float) -> str | None:
    for bucket, (low, high) in SIZE_BUCKETS.items():
        if low <= adv < high:
            return bucket
    return None


def assign_geography(region: str, isin: str) -> str:
    if region == "APAC":
        return "Asia"
    if region == "North America":
        return "Americas"
    if region == "Europe":
        return "CH" if isinstance(isin, str) and isin.startswith("CH") else "Europe ex CH"
    return "Unknown"


def prepare_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SizeBucket"] = df["MarketCapCHF"].apply(assign_size_bucket)
    # we use 25d non composite adv to determine the size bucket
    # we further use chf-converted values in order to segment in small, mid and large cap
    df["GeographyBucket"] = df.apply(
        lambda x: assign_geography(x["Region"], x["ISIN"]),
        axis=1,
    )
    return df
