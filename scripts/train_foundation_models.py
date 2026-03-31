#!/usr/bin/env python3
"""
🔥 PHASE 1 & B: FOUNDATION REPAIR & MULTI-TIMEFRAME TRAINING

Data Split (Based on QuestDB availability):
    - Train: 2022-04-11 to 2024-04-30 (24 months)
    - Validation: 2024-05-01 to 2024-12-31 (8 months)
    - Test (VAULT): 2025-01-01 to present (DON'T TOUCH until final eval)

Usage:
    cd /path/to/PYTHON_FinProj
    python scripts/train_foundation_models.py --symbol NIFTY --timeframe 5m
    python scripts/train_foundation_models.py --symbol NIFTY --timeframe 15m
    python scripts/train_foundation_models.py --symbol NIFTY --timeframe 1h
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

# Add backend to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ml.data_loader import fetch_candles
from backend.ml.training_pipeline import train_models_sequential
import pandas as pd


def load_training_data(symbol: str, timeframe: str = "5m") -> pd.DataFrame:
    """
    Load training data: 2022-04-11 to 2024-04-30
    
    This is 24 months of data for model training.
    """
    print(f"📥 Loading TRAINING data for {symbol} ({timeframe})...")
    df = fetch_candles(
        symbol=symbol,
        timeframe=timeframe,
        start_date="2022-04-11 09:15:00",
        end_date="2024-04-30 16:30:00",
    )
    print(f"   ✅ Loaded {len(df):,} candles")
    if len(df) > 0:
        print(f"   📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    return df


def load_validation_data(symbol: str, timeframe: str = "5m") -> pd.DataFrame:
    """
    Load validation data: 2024-05-01 to 2024-12-31
    
    This is 8 months of OUT-OF-SAMPLE data for validation.
    """
    print(f"📥 Loading VALIDATION data for {symbol} ({timeframe})...")
    df = fetch_candles(
        symbol=symbol,
        timeframe=timeframe,
        start_date="2024-05-01 09:15:00",
        end_date="2024-12-31 16:30:00",
    )
    print(f"   ✅ Loaded {len(df):,} candles")
    if len(df) > 0:
        print(f"   📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train foundation models (Multi-Timeframe Support)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        choices=["NIFTY", "BANKNIFTY"],
        help="Trading symbol to train"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5m",
        choices=["5m", "15m", "1h"],
        help="Timeframe to train (5m, 15m, or 1h)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="backend/models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose training output"
    )
    
    args = parser.parse_args()
    
    # Dynamic thresholds based on timeframe
    min_train_samples = {"5m": 10000, "15m": 3000, "1h": 800}.get(args.timeframe, 10000)
    expected_train = {"5m": "~60,000", "15m": "~20,000", "1h": "~5,000"}.get(args.timeframe, "~60,000")
    
    min_val_samples = {"5m": 3000, "15m": 1000, "1h": 200}.get(args.timeframe, 3000)
    expected_val = {"5m": "~20,000", "15m": "~6,500", "1h": "~1,600"}.get(args.timeframe, "~20,000")

    # ========================================================================
    # Header
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"🔥 MODEL TRAINING PIPELINE - {args.symbol} {args.timeframe}")
    print(f"{'='*70}\n")
    print(f"Symbol:       {args.symbol}")
    print(f"Timeframe:    {args.timeframe}")
    print(f"Save Dir:     {args.save_dir}")
    print(f"Start Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'='*70}\n")
    
    # ========================================================================
    # STEP 1: Load training data
    # ========================================================================
    print("STEP 1: Loading Training Data (2022-04-11 to 2024-04-30)")
    print("-" * 70)
    
    try:
        train_df = load_training_data(args.symbol, args.timeframe)
        
        if len(train_df) == 0:
            print("❌ ERROR: No training data found!")
            print("   Check if QuestDB has data for this symbol and date range")
            return
        
        if len(train_df) < min_train_samples:
            print(f"⚠️  WARNING: Only {len(train_df)} training samples!")
            print(f"   Expected: {expected_train} candles for 24 months of {args.timeframe} data")
            response = input("\n   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("   Exiting...")
                return
            print()
    
    except Exception as e:
        print(f"❌ ERROR loading training data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # STEP 2: Load validation data
    # ========================================================================
    print("STEP 2: Loading Validation Data (2024-05-01 to 2024-12-31)")
    print("-" * 70)
    
    try:
        val_df = load_validation_data(args.symbol, args.timeframe)
        
        if len(val_df) == 0:
            print("❌ ERROR: No validation data found!")
            print("   Check if QuestDB has data for 2024-05-01 onwards")
            return
        
        if len(val_df) < min_val_samples:
            print(f"⚠️  WARNING: Only {len(val_df)} validation samples!")
            print(f"   Expected: {expected_val} candles for 8 months of {args.timeframe} data")
            response = input("\n   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("   Exiting...")
                return
            print()
    
    except Exception as e:
        print(f"❌ ERROR loading validation data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # STEP 3: Train models sequentially
    # ========================================================================
    print("STEP 3: Training Models (Sequential & Leak-Free)")
    print("-" * 70)
    print()
    
    try:
        movement_model, risk_model, metrics = train_models_sequential(
            train_df=train_df,
            val_df=val_df,
            save_dir=args.save_dir,
            symbol=args.symbol,
            timeframe=args.timeframe,
            verbose=not args.quiet,
        )
    
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # STEP 4: Save metrics to JSON
    # ========================================================================
    print("\nSTEP 4: Saving Training Metrics")
    print("-" * 70)
    
    metrics_dir = Path(args.save_dir) / "training_logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = metrics_dir / f"metrics_{args.symbol.lower()}_{args.timeframe}_{timestamp}.json"
    
    try:
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"✅ Metrics saved to: {metrics_file}\n")
    except Exception as e:
        print(f"⚠️  Could not save metrics: {e}\n")
    
    # ========================================================================
    # STEP 5: Final Summary & Success Criteria
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"🎉 TRAINING COMPLETE for {args.symbol} {args.timeframe}!")
    print(f"{'='*70}\n")
    
    print("📁 Models Saved:")
    print(f"  Movement: {args.save_dir}/movement_predictor_{args.symbol.lower()}_{args.timeframe}.joblib")
    print(f"  Risk:     {args.save_dir}/risk_assessor_{args.symbol.lower()}_{args.timeframe}.joblib")
    print()
    
    print("📊 Data Split:")
    print(f"  Training:   2022-04-11 to 2024-04-30 ({len(train_df):,} candles)")
    print(f"  Validation: 2024-05-01 to 2024-12-31 ({len(val_df):,} candles)")
    print(f"  Test:       2025-01-01+ (VAULT - untouched)")
    print()
    
    # Extract metrics
    movement_acc = metrics['movement']['val_accuracy']
    risk_rmse = metrics['risk']['val_rmse']
    risk_corr = metrics['risk']['val_spearman']
    
    print("📈 Validation Performance:")
    print(f"  Movement Accuracy:  {movement_acc:.4f} (Random baseline: 0.333)")
    print(f"  Risk RMSE:          {risk_rmse:.2f} (Lower is better)")
    print(f"  Risk Spearman:      {risk_corr:.4f} (Correlation: -1 to +1)")
    print()
    
    # ========================================================================
    # Target Criteria Check
    # ========================================================================
    print("🎯 Target Criteria:")
    print("-" * 70)
    
    success_count = 0
    total_checks = 3
    
    # Check 1: Movement Accuracy
    if movement_acc >= 0.42:
        print(f"  ✅ Movement Accuracy: {movement_acc:.4f} >= 0.42 (PASS)")
        success_count += 1
    else:
        print(f"  ❌ Movement Accuracy: {movement_acc:.4f} < 0.42 (FAIL)")
        print(f"     Current is only {(movement_acc - 0.333) / (0.42 - 0.333) * 100:.1f}% toward target")
    
    # Check 2: Risk RMSE
    if risk_rmse < 35:
        print(f"  ✅ Risk RMSE: {risk_rmse:.2f} < 35 (PASS)")
        success_count += 1
    else:
        print(f"  ⚠️  Risk RMSE: {risk_rmse:.2f} >= 35 (MARGINAL)")
    
    # Check 3: Risk Spearman
    if risk_corr > 0.12:
        print(f"  ✅ Risk Spearman: {risk_corr:.4f} > 0.12 (PASS)")
        success_count += 1
    else:
        print(f"  ❌ Risk Spearman: {risk_corr:.4f} <= 0.12 (FAIL)")
    
    print()
    print(f"Result: {success_count}/{total_checks} checks passed")
    print()
    
    print(f"\n{'='*70}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()