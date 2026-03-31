"""
Central configuration — single source of truth for all runtime parameters.

Every tunable value lives here. No hardcoded constants anywhere else.
Secrets are read from environment variables with safe local-dev defaults.

Stripped to only what the chart API and research scripts need.
"""

from __future__ import annotations

import os
from datetime import time
from zoneinfo import ZoneInfo

# ============================================================================
# TIMEZONE
# ============================================================================
IST = ZoneInfo("Asia/Kolkata")

# ============================================================================
# QUESTDB CONNECTION
# ============================================================================
QUESTDB_HOST: str = os.getenv("QUESTDB_HOST", "localhost")
QUESTDB_PORT: int = int(os.getenv("QUESTDB_PORT", "9000"))
QUESTDB_DB: str = os.getenv("QUESTDB_DB", "qdb")
QUESTDB_USER: str = os.getenv("QUESTDB_USER", "admin")
QUESTDB_PASSWORD: str = os.getenv("QUESTDB_PASSWORD", "quest")

# ============================================================================
# MARKET HOURS (IST)
# ============================================================================
MARKET_OPEN: time = time(9, 15)
MARKET_CLOSE: time = time(15, 30)

# ============================================================================
# ENSEMBLE / META-FILTER THRESHOLDS
# ============================================================================
ENSEMBLE_CONFIDENCE_MIN: float = 0.40
META_FILTER_PROB_MIN: float = 0.55

# ============================================================================
# META-FILTER MODEL SELECTION (2-model vs 3-model)
# ============================================================================
META_FILTER_MODE: str = os.getenv("META_FILTER_MODE", "2model")

# ============================================================================
# RISK MANAGEMENT (used by research scripts)
# ============================================================================
STARTING_CAPITAL: float = float(os.getenv("STARTING_CAPITAL", "500000"))
STOP_LOSS_ATR_MULT: float = 1.0
TARGET_ATR_MULT: float = 1.5

# ============================================================================
# LOT SIZES (standard exchange lots)
# ============================================================================
LOT_SIZES: dict[str, int] = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
}

# ============================================================================
# MODEL PATHS
# ============================================================================
MODELS_DIR: str = os.path.join(os.path.dirname(__file__), "models")

# ============================================================================
# MULTI-HORIZON WEIGHTS
# ============================================================================
WEIGHT_5M: float = 0.30
WEIGHT_15M: float = 0.70
WEIGHT_5M_MULTI: float = 0.25
WEIGHT_15M_MULTI: float = 0.55
WEIGHT_30M_MULTI: float = 0.20

# ============================================================================
# SYMBOLS
# ============================================================================
SYMBOLS: list[str] = ["NIFTY", "BANKNIFTY"]

# ============================================================================
# ASYMMETRIC MODEL THRESHOLDS (directional meta-filters)
# ============================================================================
META_FILTER_UP_THRESHOLD: float = 0.60
META_FILTER_DOWN_THRESHOLD: float = 0.52

# ============================================================================
# META-FEATURE COMPUTATION
# ============================================================================
SIGNAL_HISTORY_WINDOW: int = 20
