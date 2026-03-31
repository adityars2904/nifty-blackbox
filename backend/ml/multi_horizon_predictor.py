"""
Multi-Horizon Ensemble Predictor  (Stage 1)

Drop-in replacement for ``EnsemblePredictor`` that adds a 30-minute
timeframe model.  Controlled by ``config.USE_MULTIHORIZON``.

Weights:
  5m  = 25%
  15m = 55%
  30m = 20%

Interface is identical to ``EnsemblePredictor`` so ``signal_service``
does not need to know which predictor it is using.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from ml.feature_engineering import add_features, MOVEMENT_FEATURES
from ml.ensemble_predictor import EnsemblePrediction, DIRECTION_MAP


# Default weights
WEIGHT_5M  = 0.25
WEIGHT_15M = 0.55
WEIGHT_30M = 0.20
MIN_CONFIDENCE = 0.40


class MultiHorizonPredictor:
    """
    3-timeframe ensemble: 5m + 15m + 30m.

    Usage is identical to ``EnsemblePredictor``::

        mhp = MultiHorizonPredictor.load("backend/models", "NIFTY")
        result = mhp.predict(df_5m, df_15m, df_30m)
    """

    def __init__(
        self,
        model_5m: Any,
        model_15m: Any,
        model_30m: Any,
        weight_5m: float = WEIGHT_5M,
        weight_15m: float = WEIGHT_15M,
        weight_30m: float = WEIGHT_30M,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.model_5m = model_5m
        self.model_15m = model_15m
        self.model_30m = model_30m
        self.weight_5m = weight_5m
        self.weight_15m = weight_15m
        self.weight_30m = weight_30m
        self.min_confidence = min_confidence

    @classmethod
    def load(
        cls,
        model_dir: str = "backend/models",
        symbol: str = "NIFTY",
    ) -> "MultiHorizonPredictor":
        """Load all three models from disk."""
        d = Path(model_dir)
        s = symbol.lower()

        for tf in ("5m", "15m", "30m"):
            p = d / f"movement_predictor_{s}_{tf}.joblib"
            if not p.exists():
                raise FileNotFoundError(f"Model not found: {p}")

        m5 = joblib.load(d / f"movement_predictor_{s}_5m.joblib")
        m15 = joblib.load(d / f"movement_predictor_{s}_15m.joblib")
        m30 = joblib.load(d / f"movement_predictor_{s}_30m.joblib")

        return cls(model_5m=m5, model_15m=m15, model_30m=m30)

    def _extract_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
        df_30m: Optional[pd.DataFrame] = None,
    ) -> EnsemblePrediction:
        """
        Generate 3-model ensemble prediction.

        If ``df_30m`` is None, falls back to 2-model behaviour.
        """
        X_5m = self._extract_features(df_5m)
        X_15m = self._extract_features(df_15m)
        X_30m = self._extract_features(df_30m) if df_30m is not None else None

        if X_5m is None or X_15m is None:
            return EnsemblePrediction(
                direction="NEUTRAL", confidence=0.0, should_trade=False,
                prob_down=0.33, prob_neutral=0.34, prob_up=0.33,
                pred_5m="NEUTRAL", pred_15m="NEUTRAL",
                conf_5m=0.0, conf_15m=0.0,
                veto_reason="Insufficient data",
            )

        probs_5m = self.model_5m.predict_proba(X_5m)[0]
        probs_15m = self.model_15m.predict_proba(X_15m)[0]

        pred_5m_str = DIRECTION_MAP[int(np.argmax(probs_5m))]
        pred_15m_str = DIRECTION_MAP[int(np.argmax(probs_15m))]
        conf_5m = float(probs_5m.max())
        conf_15m = float(probs_15m.max())

        if X_30m is not None:
            probs_30m = self.model_30m.predict_proba(X_30m)[0]
            ensemble_probs = (
                probs_5m * self.weight_5m
                + probs_15m * self.weight_15m
                + probs_30m * self.weight_30m
            )
        else:
            # Fallback to 2-model weights re-normalised
            w5 = self.weight_5m / (self.weight_5m + self.weight_15m)
            w15 = self.weight_15m / (self.weight_5m + self.weight_15m)
            ensemble_probs = probs_5m * w5 + probs_15m * w15

        prob_down = float(ensemble_probs[0])
        prob_neutral = float(ensemble_probs[1])
        prob_up = float(ensemble_probs[2])

        final_idx = int(np.argmax(ensemble_probs))
        final_dir = DIRECTION_MAP[final_idx]
        confidence = float(ensemble_probs[final_idx])

        # ── veto logic (same as EnsemblePredictor) ───────────────────────────
        should_trade = True
        veto_reason = None

        if pred_15m_str == "NEUTRAL":
            should_trade = False
            veto_reason = "15m model says NEUTRAL (veto)"
        elif final_dir == "NEUTRAL":
            should_trade = False
            veto_reason = "Ensemble says NEUTRAL"
        elif confidence < self.min_confidence:
            should_trade = False
            veto_reason = f"Confidence {confidence:.2f} below threshold {self.min_confidence:.2f}"
        elif (
            pred_5m_str != "NEUTRAL"
            and pred_15m_str != "NEUTRAL"
            and pred_5m_str != pred_15m_str
        ):
            should_trade = False
            veto_reason = f"5m ({pred_5m_str}) and 15m ({pred_15m_str}) disagree"

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
            veto_reason=veto_reason,
        )
