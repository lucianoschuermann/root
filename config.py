"""
@notice Global configuration for market impact pipeline.
"""

import numpy as np

# ==================================================
# Calibration / rolling setup
# ==================================================
WINDOW_TYPES = ["5d", "25d"]
CALIBRATION_MONTHS = 3
MIN_CALIBRATION_N = 30

# Separate threshold for pooled omega estimation (stability)
MIN_OMEGA_N = 2000

# Canonical window type used for pooled omega estimation
OMEGA_POOL_WINDOWTYPE = "25d"

SIZE_BUCKETS = {
    "Large": (50e6, np.inf),
    "Mid": (10e6, 50e6),
    "Small": (0, 10e6),
}

GEOGRAPHY_BUCKETS = [
    "CH",
    "Europe ex CH",
    "Americas",
    "Asia",
]

# ==================================================
# New: calibration scopes
# ==================================================
# These define where parameters are calibrated from.
#
# global
#   -> one parameter set over the full universe
#
# by_size
#   -> one parameter set per size bucket
#
# by_geo
#   -> one parameter set per geography bucket
#
# by_geo_size
#   -> one parameter set per geography x size segment
#
# Your previous pipeline was effectively equivalent to:
#   CALIBRATION_SCOPES = ["by_geo_size"]
#
CALIBRATION_SCOPES = [
    "global",
    "by_size",
    "by_geo",
    "by_geo_size",
]

DEFAULT_CALIBRATION_SCOPE = "by_geo_size"

# ==================================================
# New: fallback hierarchy if a requested scope
# does not have enough calibration observations
# ==================================================
# Example:
#   by_geo_size -> first try by_geo, then by_size, then global
#
CALIBRATION_SCOPE_FALLBACKS = {
    "global": [],
    "by_size": ["global"],
    "by_geo": ["global"],
    "by_geo_size": ["by_geo", "by_size", "global"],
}

# ==================================================
# New: minimum N per calibration scope
# Usually identical to MIN_CALIBRATION_N
# but can be overridden later if desired
# ==================================================
MIN_SCOPE_CALIBRATION_N = {
    "global": MIN_CALIBRATION_N,
    "by_size": MIN_CALIBRATION_N,
    "by_geo": MIN_CALIBRATION_N,
    "by_geo_size": MIN_CALIBRATION_N,
}

# ==================================================
# New: JPMorgan omega pooling rule by effective scope
# ==================================================
# Allowed pool values:
#   "global"
#   "geo"
#   "size"
#   "geo_size"
#
# Recommended defaults:
#   global      -> omega pooled globally
#   by_size     -> omega pooled by size
#   by_geo      -> omega pooled by geography
#   by_geo_size -> omega pooled by geography
#
# The last line preserves your current logic for
# JPM variants under segmented calibration.
#
JPM_OMEGA_POOL_SCOPE = {
    "global": "global",
    "by_size": "size",
    "by_geo": "geo",
    "by_geo_size": "geo",
}

# ==================================================
# Execution horizon assumption
# ==================================================
TRADING_DAY_MINUTES = 390.0          # US equity standard
EXECUTION_MINUTES = 10.0             # ex-ante assumption

DAY_FRAC = EXECUTION_MINUTES / TRADING_DAY_MINUTES

# ==================================================
# Plot / reporting (visualisation only)
# ==================================================

# Asymmetric quantiles (left/right) for plots (DECIMAL units)
PLOT_Q_LEFT = 0.10
PLOT_Q_RIGHT = 0.99

# Optional per-plot overrides; if None -> uses PLOT_Q_LEFT / PLOT_Q_RIGHT
PLOT_Q_LEFT_ERROR = None
PLOT_Q_RIGHT_ERROR = None
PLOT_Q_LEFT_IMPACT = None
PLOT_Q_RIGHT_IMPACT = None

# Minimum span to avoid degenerate axes
PLOT_MIN_SPAN = 1e-6

# Optional fixed caps (absolute values) – if set, override quantiles
# Use tuples (low, high) for asymmetric fixed caps in DECIMAL units.
PLOT_FIXED_CAP_ERROR = None   # e.g. (-0.01, 0.05)
PLOT_FIXED_CAP_IMPACT = None  # e.g. (-0.02, 0.08)