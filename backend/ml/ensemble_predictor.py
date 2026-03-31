"""
Triple-Timeframe Ensemble Predictor
====================================
Combines 5m (entry timing) + 15m (primary signal) models.
1h is EXCLUDED - performed below random (31.4% < 33.3%).

Architecture:
  15m model → 70% weight (better accuracy, balanced classes, 0% train-val gap)
  5m model  → 30% weight (entry timing context)

Veto Rules:
  - If 15m says NEUTRAL → HOLD regardless of 5m
  - Confidence < threshold → HOLD
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import add_features, MOVEMENT_FEATURES


# ============================================================================
# CONFIG
# ============================================================================

WEIGHT_15M = 0.70   # 15m is primary (better, balanced model)
WEIGHT_5M  = 0.30   # 5m is secondary (entry timing only)
MIN_CONFIDENCE = 0.40  # Minimum confidence to trade (0 = no filter)

DIRECTION_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
DIRECTION_MAP_INV = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EnsemblePrediction:
    direction: str          # "UP", "DOWN", "NEUTRAL"
    confidence: float       # 0.0 - 1.0
    should_trade: bool      # False if vetoed or low confidence

    prob_down: float        # P(DOWN)
    prob_neutral: float     # P(NEUTRAL)
    prob_up: float          # P(UP)

    pred_5m: str            # Individual 5m prediction
    pred_15m: str           # Individual 15m prediction
    conf_5m: float          # 5m confidence
    conf_15m: float         # 15m confidence
    
    probs_5m: Tuple[float, float, float]   # (down, neutral, up)
    probs_15m: Tuple[float, float, float]  # (down, neutral, up)

    veto_reason: Optional[str] = None  # Why trade was vetoed


# ============================================================================
# ENSEMBLE CLASS
# ============================================================================

class EnsemblePredictor:
    """
    Triple-Timeframe Ensemble Predictor.

    Usage:
        ensemble = EnsemblePredictor.load("backend/models")
        result = ensemble.predict(df_5m, df_15m)

        if result.should_trade:
            print(f"Trade: {result.direction} (confidence: {result.confidence:.2f})")
    """

    def __init__(
        self,
        model_5m: Any,
        model_15m: Any,
        weight_5m: float = WEIGHT_5M,
        weight_15m: float = WEIGHT_15M,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.model_5m = model_5m
        self.model_15m = model_15m
        self.weight_5m = weight_5m
        self.weight_15m = weight_15m
        self.min_confidence = min_confidence

    @classmethod
    def load(
        cls,
        model_dir: str = "backend/models",
        symbol: str = "NIFTY",
    ) -> "EnsemblePredictor":
        """Load both models from disk."""
        model_dir_path = Path(model_dir)

        path_5m  = model_dir_path / f"movement_predictor_{symbol.lower()}_5m.joblib"
        path_15m = model_dir_path / f"movement_predictor_{symbol.lower()}_15m.joblib"

        if not path_5m.exists():
            raise FileNotFoundError(f"5m model not found: {path_5m}")
        if not path_15m.exists():
            raise FileNotFoundError(f"15m model not found: {path_15m}")

        model_5m  = joblib.load(path_5m)
        model_15m = joblib.load(path_15m)

        print(f"✅ Loaded 5m model:  {path_5m}")
        print(f"✅ Loaded 15m model: {path_15m}")

        return cls(model_5m=model_5m, model_15m=model_15m)

    def _extract_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add features and return last row as single-row DataFrame."""
        if len(df) < 50:
            return None
        df_feat = add_features(df)
        df_feat = df_feat.dropna(subset=MOVEMENT_FEATURES)
        if len(df_feat) == 0:
            return None
        return df_feat[MOVEMENT_FEATURES].iloc[[-1]].astype(float)

    def predict(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> EnsemblePrediction:
        """
        Generate ensemble prediction from 5m and 15m data.

        Args:
            df_5m:  Recent 5-minute candles (need at least 50)
            df_15m: Recent 15-minute candles (need at least 50)

        Returns:
            EnsemblePrediction with direction, confidence, and veto status
        """
        # Extract features
        X_5m  = self._extract_features(df_5m)
        X_15m = self._extract_features(df_15m)

        if X_5m is None or X_15m is None:
            return EnsemblePrediction(
                direction="NEUTRAL", confidence=0.0, should_trade=False,
                prob_down=0.33, prob_neutral=0.34, prob_up=0.33,
                pred_5m="NEUTRAL", pred_15m="NEUTRAL",
                conf_5m=0.0, conf_15m=0.0,
                probs_5m=(0.33, 0.34, 0.33),
                probs_15m=(0.33, 0.34, 0.33),
                veto_reason="Insufficient data",
            )

        # Get probabilities from each model
        # Select only the features each model was trained on (may be a subset of MOVEMENT_FEATURES)
        feat_5m  = list(self.model_5m.get_booster().feature_names)
        feat_15m = list(self.model_15m.get_booster().feature_names)
        probs_5m  = self.model_5m.predict_proba(X_5m[feat_5m])[0]    # [P_down, P_neutral, P_up]
        probs_15m = self.model_15m.predict_proba(X_15m[feat_15m])[0]  # [P_down, P_neutral, P_up]

        # Individual predictions
        pred_5m_idx  = int(np.argmax(probs_5m))
        pred_15m_idx = int(np.argmax(probs_15m))
        pred_5m_str  = DIRECTION_MAP[pred_5m_idx]
        pred_15m_str = DIRECTION_MAP[pred_15m_idx]
        conf_5m      = float(probs_5m.max())
        conf_15m     = float(probs_15m.max())

        # Weighted ensemble
        ensemble_probs = (
            probs_5m  * self.weight_5m +
            probs_15m * self.weight_15m
        )

        prob_down    = float(ensemble_probs[0])
        prob_neutral = float(ensemble_probs[1])
        prob_up      = float(ensemble_probs[2])

        final_idx    = int(np.argmax(ensemble_probs))
        final_dir    = DIRECTION_MAP[final_idx]
        confidence   = float(ensemble_probs[final_idx])

        # ====================================================================
        # VETO LOGIC
        # ====================================================================
        should_trade = True
        veto_reason  = None

        # Veto 1: 15m says NEUTRAL → don't trade (15m has veto power)
        if pred_15m_str == "NEUTRAL":
            should_trade = False
            veto_reason  = "15m model says NEUTRAL (veto)"

        # Veto 2: Final ensemble says NEUTRAL
        elif final_dir == "NEUTRAL":
            should_trade = False
            veto_reason  = "Ensemble says NEUTRAL"

        # Veto 3: Low confidence
        elif confidence < self.min_confidence:
            should_trade = False
            veto_reason  = f"Confidence {confidence:.2f} below threshold {self.min_confidence:.2f}"

        # Veto 4: 5m and 15m completely disagree on direction
        elif (
            pred_5m_str  != "NEUTRAL"
            and pred_15m_str != "NEUTRAL"
            and pred_5m_str  != pred_15m_str
        ):
            should_trade = False
            veto_reason  = f"5m ({pred_5m_str}) and 15m ({pred_15m_str}) disagree"

        return EnsemblePrediction(
            direction=final_dir,
            confidence=confidence,
            should_trade=should_trade,
            prob_down=prob_down,
            prob_neutral=prob_neutral,
            prob_up=prob_up,
            pred_5m=pred_5m_str,
            pred_15m=pred_15m_str,
            conf_5m=conf_5m,
            conf_15m=conf_15m,
            probs_5m=(float(probs_5m[0]), float(probs_5m[1]), float(probs_5m[2])),
            probs_15m=(float(probs_15m[0]), float(probs_15m[1]), float(probs_15m[2])),
            veto_reason=veto_reason,
        )

    def predict_batch(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        min_lookback: int = 50,
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions for an entire dataframe (backtesting).

        Args:
            df_5m:  Full 5-minute dataframe
            df_15m: Full 15-minute dataframe (must align with 5m)

        Returns:
            DataFrame with prediction columns added
        """
        df_5m_feat  = add_features(df_5m).dropna(subset=MOVEMENT_FEATURES)
        df_15m_feat = add_features(df_15m).dropna(subset=MOVEMENT_FEATURES)

        # Get all probabilities at once using only features each model expects
        feat_5m  = list(self.model_5m.get_booster().feature_names)
        feat_15m = list(self.model_15m.get_booster().feature_names)
        X_5m  = df_5m_feat[feat_5m].astype(float)
        X_15m = df_15m_feat[feat_15m].astype(float)

        probs_5m  = self.model_5m.predict_proba(X_5m)
        probs_15m = self.model_15m.predict_proba(X_15m)

        # For backtesting: resample 15m probs to 5m frequency
        # Each 15m bar covers 3 x 5m bars → forward-fill
        if "timestamp" in df_15m_feat.columns:
            # Create a timestamp index mapping
            df_15m_feat = df_15m_feat.copy()
            df_15m_feat["prob_down_15m"]    = probs_15m[:, 0]
            df_15m_feat["prob_neutral_15m"] = probs_15m[:, 1]
            df_15m_feat["prob_up_15m"]      = probs_15m[:, 2]

            df_5m_feat = df_5m_feat.copy()
            df_5m_feat["prob_down_5m"]    = probs_5m[:, 0]
            df_5m_feat["prob_neutral_5m"] = probs_5m[:, 1]
            df_5m_feat["prob_up_5m"]      = probs_5m[:, 2]

            # Merge on timestamp (5m gets nearest previous 15m prediction)
            df_5m_ts  = pd.to_datetime(df_5m_feat["timestamp"])
            df_15m_ts = pd.to_datetime(df_15m_feat["timestamp"])

            df_5m_feat = df_5m_feat.sort_values("timestamp")
            df_15m_feat = df_15m_feat.sort_values("timestamp")

            merged = pd.merge_asof(
                df_5m_feat,
                df_15m_feat[["timestamp", "prob_down_15m", "prob_neutral_15m", "prob_up_15m"]],
                on="timestamp",
                direction="backward",
            )

            # Weighted ensemble
            merged["prob_down"]    = merged["prob_down_5m"]    * self.weight_5m + merged["prob_down_15m"]    * self.weight_15m
            merged["prob_neutral"] = merged["prob_neutral_5m"] * self.weight_5m + merged["prob_neutral_15m"] * self.weight_15m
            merged["prob_up"]      = merged["prob_up_5m"]      * self.weight_5m + merged["prob_up_15m"]      * self.weight_15m

            # Final prediction
            probs_matrix = merged[["prob_down", "prob_neutral", "prob_up"]].values
            merged["ensemble_pred"] = np.argmax(probs_matrix, axis=1)
            merged["ensemble_conf"] = np.max(probs_matrix, axis=1)
            merged["ensemble_dir"]  = merged["ensemble_pred"].map(DIRECTION_MAP)
            if len(df_15m_feat) > 0:
                ts_15m_arr = np.array(df_15m_feat["timestamp"].values, dtype="datetime64[ns]")
                ts_5m_arr  = np.array(merged["timestamp"].values,      dtype="datetime64[ns]")
                idx_15m    = np.searchsorted(ts_15m_arr, ts_5m_arr, side="right") - 1
                idx_15m    = np.clip(idx_15m, 0, len(probs_15m) - 1)
                merged["pred_15m"] = np.argmax(probs_15m[idx_15m], axis=1)
            else:
                merged["pred_15m"] = 1

            # Apply veto
            merged["should_trade"] = (
                (merged["ensemble_dir"] != "NEUTRAL") &
                (merged["ensemble_conf"] >= self.min_confidence)
            )

            return merged


# =============================================================================
# CHANGES
# =============================================================================
# Add Tuple to imports and expose probs_5m and probs_15m in EnsemblePrediction 
# to support JSD feature calculation in meta_filter_service.
# Handled per Section 4 handoff.
# Includes model impact details: Enables meta-features #20 and #21.
# Retraining is required: validate asymmetric UP/DOWN on vault first, then
# JSD and ratio features, then retrain only features that pass gates.
# Never retrain movement_predictor_*.joblib.
# =============================================================================

        # Fallback: just return 5m predictions if no timestamp
        results = pd.DataFrame({
            "prob_down":    probs_5m[:, 0] * self.weight_5m,
            "prob_neutral": probs_5m[:, 1] * self.weight_5m,
            "prob_up":      probs_5m[:, 2] * self.weight_5m,
        })
        results["ensemble_pred"] = np.argmax(results.values, axis=1)
        results["ensemble_conf"] = np.max(results.values, axis=1)
        results["ensemble_dir"]  = results["ensemble_pred"].map(DIRECTION_MAP)
        return results


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def load_ensemble(
    model_dir: str = "backend/models",
    symbol: str = "NIFTY",
) -> EnsemblePredictor:
    """Quick loader for use in agent/trading code."""
    return EnsemblePredictor.load(model_dir=model_dir, symbol=symbol)