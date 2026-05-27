from config import DAY_FRAC

def effective_volume(daily_volume: float) -> float:
    """
    Convert daily volume to effective execution-horizon volume.
    """
    return daily_volume * DAY_FRAC
