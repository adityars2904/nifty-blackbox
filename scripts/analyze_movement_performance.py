#!/usr/bin/env python3
"""
Analyze Movement Model Performance by Class

This script loads the trained movement model and checks accuracy
for each class (DOWN, NEUTRAL, UP) separately.

This will tell us if the model is good at directional prediction
but just bad at NEUTRAL, or if it's bad at everything.

Usage:
    cd /path/to/PYTHON_FinProj
    python scripts/analyze_movement_performance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add backend to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from backend.ml.data_loader import load_validation_data
from backend.ml.feature_engineering import add_features, MOVEMENT_FEATURES
from backend.ml.labels import label_movement


def main():
    print("=" * 70)
    print("MOVEMENT MODEL PERFORMANCE ANALYSIS")
    print("=" * 70)
    print()
    
    # ========================================================================
    # STEP 1: Load validation data
    # ========================================================================
    print("STEP 1: Loading validation data (2024-05-01 to 2024-12-31)...")
    val_df = load_validation_data(symbol="NIFTY", timeframe="5m")
    print(f"✅ Loaded {len(val_df):,} candles")
    print()
    
    # ========================================================================
    # STEP 2: Generate features and labels
    # ========================================================================
    print("STEP 2: Generating features and labels...")
    val_df_feat = add_features(val_df)
    movement_labels = label_movement(val_df_feat)
    
    # Add labels to dataframe
    val_df_feat["movement_label"] = movement_labels
    
    # Drop rows with NaN
    val_df_clean = val_df_feat.dropna(subset=MOVEMENT_FEATURES + ["movement_label"])
    
    print(f"✅ Generated features for {len(val_df_clean):,} samples")
    print()
    
    # ========================================================================
    # STEP 3: Load trained model
    # ========================================================================
    print("STEP 3: Loading trained movement model...")
    model_path = "backend/models/movement_predictor_nifty_5m.joblib"
    
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded from: {model_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: Model not found at {model_path}")
        print("   Please train the model first:")
        print("   python scripts/train_foundation_models.py --symbol NIFTY --timeframe 5m")
        return
    
    print()
    
    # ========================================================================
    # STEP 4: Get predictions
    # ========================================================================
    print("STEP 4: Generating predictions on validation set...")
    X_val = val_df_clean[MOVEMENT_FEATURES].astype(float)
    y_val = val_df_clean["movement_label"].astype(int)
    
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)
    
    print(f"✅ Generated {len(preds):,} predictions")
    print()
    
    # ========================================================================
    # STEP 5: Overall accuracy
    # ========================================================================
    print("=" * 70)
    print("OVERALL PERFORMANCE")
    print("=" * 70)
    
    overall_accuracy = (preds == y_val).mean()
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"Random Baseline:  0.3333 (33.33%)")
    print(f"Edge over random: {(overall_accuracy - 0.3333)*100:.2f}%")
    print()
    
    # ========================================================================
    # STEP 6: Per-class performance (THE KEY METRIC!)
    # ========================================================================
    print("=" * 70)
    print("PER-CLASS PERFORMANCE (🔥 THIS IS WHAT MATTERS!)")
    print("=" * 70)
    print()
    
    report = classification_report(
        y_val, 
        preds,
        target_names=['DOWN', 'NEUTRAL', 'UP'],
        digits=4
    )
    print(report)
    print()
    
    # ========================================================================
    # STEP 7: Confusion matrix
    # ========================================================================
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print()
    
    cm = confusion_matrix(y_val, preds)
    
    print("                Predicted")
    print("              DOWN  NEUTRAL  UP")
    print(f"Actual DOWN   {int(cm[0,0]):>4}  {int(cm[0,1]):>7}  {int(cm[0,2]):>4}")
    print(f"     NEUTRAL  {int(cm[1,0]):>4}  {int(cm[1,1]):>7}  {int(cm[1,2]):>4}")
    print(f"          UP  {int(cm[2,0]):>4}  {int(cm[2,1]):>7}  {int(cm[2,2]):>4}")
    print()
    
    # ========================================================================
    # STEP 8: Label distribution
    # ========================================================================
    print("=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)
    print()
    
    label_counts = pd.Series(y_val).value_counts().sort_index()
    label_pcts = label_counts / len(y_val) * 100
    
    print(f"DOWN (0):    {int(label_counts[0]):>6,} samples ({float(label_pcts[0]):>5.2f}%)")
    print(f"NEUTRAL (1): {int(label_counts[1]):>6,} samples ({float(label_pcts[1]):>5.2f}%)")
    print(f"UP (2):      {int(label_counts[2]):>6,} samples ({float(label_pcts[2]):>5.2f}%)")
    print()
    
    # ========================================================================
    # STEP 9: Directional-only performance
    # ========================================================================
    print("=" * 70)
    print("DIRECTIONAL-ONLY PERFORMANCE (Ignoring NEUTRAL)")
    print("=" * 70)
    print()
    
    # Filter to only UP/DOWN predictions and actuals
    directional_mask = (preds != 1) & (y_val != 1)
    
    if directional_mask.sum() > 0:
        directional_preds = preds[directional_mask]
        directional_actual = y_val.values[directional_mask]  # Convert to numpy array
        directional_accuracy = (directional_preds == directional_actual).mean()
        
        print(f"Total directional predictions: {int(directional_mask.sum()):,}")
        print(f"Directional accuracy: {directional_accuracy:.4f} ({directional_accuracy*100:.2f}%)")
        print(f"Random baseline: 0.5000 (50.00%)")
        print(f"Edge over random: {(directional_accuracy - 0.5)*100:.2f}%")
        print()
        
        # UP/DOWN breakdown
        up_mask = directional_preds == 2
        down_mask = directional_preds == 0
        
        if up_mask.sum() > 0:
            up_accuracy = (directional_preds[up_mask] == directional_actual[up_mask]).mean()
            print(f"UP predictions accuracy:   {up_accuracy:.4f} ({int(up_mask.sum()):,} predictions)")
        
        if down_mask.sum() > 0:
            down_accuracy = (directional_preds[down_mask] == directional_actual[down_mask]).mean()
            print(f"DOWN predictions accuracy: {down_accuracy:.4f} ({int(down_mask.sum()):,} predictions)")
    else:
        print("⚠️  No directional predictions found!")
    
    print()
    
    # ========================================================================
    # STEP 10: Analysis & Recommendation
    # ========================================================================
    print("=" * 70)
    print("ANALYSIS & RECOMMENDATION")
    print("=" * 70)
    print()
    
    # Get precision from classification report
    from sklearn.metrics import precision_score
    
    precision_per_class = np.array(precision_score(y_val, preds, average=None))
    up_precision = precision_per_class[2]
    down_precision = precision_per_class[0]
    
    avg_directional_precision = (up_precision + down_precision) / 2
    
    print("Key Metrics:")
    print(f"  UP precision:   {up_precision:.4f}")
    print(f"  DOWN precision: {down_precision:.4f}")
    print(f"  Average:        {avg_directional_precision:.4f}")
    print()
    
    if avg_directional_precision >= 0.42:
        print("✅ RECOMMENDATION: Move to Meta-Labeling NOW")
        print()
        print("Your model is ALREADY GOOD at directional prediction!")
        print(f"UP/DOWN precision of {avg_directional_precision:.1%} is solid.")
        print()
        print("The low overall accuracy (37%) is dragged down by NEUTRAL class.")
        print("Since we don't trade NEUTRAL anyway, this doesn't matter.")
        print()
        print("Next step: Build meta-filter to identify which UP/DOWN predictions to trust.")
        print("Expected result: 50-55% precision on traded signals.")
        
    elif avg_directional_precision >= 0.38:
        print("🟡 RECOMMENDATION: Try Triple-Timeframe (but cap at 1 day)")
        print()
        print(f"UP/DOWN precision of {avg_directional_precision:.1%} is borderline.")
        print("Triple-timeframe MIGHT push it to 42%+")
        print()
        print("BUT: Set a time limit of 1 day.")
        print("If it doesn't hit 42% by tomorrow, move to meta-labeling.")
        
    else:
        print("❌ ISSUE: Model struggling with directional prediction")
        print()
        print(f"UP/DOWN precision of {avg_directional_precision:.1%} is concerning.")
        print()
        print("Options:")
        print("1. Try triple-timeframe (might help)")
        print("2. Check feature quality (any NaN or constant features?)")
        print("3. Review label quality (are labels too noisy?)")
        print("4. Accept limitations and use conservative thresholds")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()