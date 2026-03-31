from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Any, List
import os
import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    classification_report,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import spearmanr
from .feature_engineering import MOVEMENT_FEATURES, RISK_FEATURES, add_features
from .labels import label_movement, label_risk_score


@dataclass
class TrainResults:
    """Container for trained model and metrics."""
    model: Any
    metrics: Dict[str, Any]


def build_datasets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Build feature datasets and labels for both models."""
    df_feat = add_features(df)
    movement_labels = label_movement(df_feat)
    risk_labels = label_risk_score(df_feat, movement_labels)

    df_feat = df_feat.copy()
    df_feat["movement_label"] = movement_labels
    df_feat["risk_score_label"] = risk_labels

    required_cols = MOVEMENT_FEATURES + ["movement_label", "risk_score_label"]
    df_feat = df_feat.dropna(subset=required_cols)

    X_movement = df_feat[MOVEMENT_FEATURES].astype(float)
    y_movement = df_feat["movement_label"].astype(int)

    X_risk = df_feat[RISK_FEATURES].astype(float).copy()
    X_risk["pred_direction"] = y_movement.astype(float)
    X_risk["pred_confidence"] = 1.0
    X_risk["drawdown"] = 0.0
    X_risk["daily_pnl_pct"] = 0.0
    X_risk["consecutive_losses"] = 0.0
    X_risk["position_size_pct"] = 0.0

    y_risk = df_feat["risk_score_label"].astype(float)

    return X_movement, y_movement, X_risk, y_risk


def _compute_sample_weights(y: pd.Series) -> npt.NDArray[np.float64]:
    """
    🔥 SOLUTION A: THE EQUALIZER
    
    Compute balanced sample weights to fix UP bias.
    
    If DOWN appears 30% of the time → weight 1/0.30 = 3.33x
    If UP appears 40% of the time → weight 1/0.40 = 2.5x
    
    This tells XGBoost: "Missing a DOWN move is a felony, not a minor oopsie."
    """
    classes = np.unique(y)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )
    
    weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    sample_weights = np.array([weight_dict[int(label)] for label in y])
    
    return sample_weights


def train_movement_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    verbose: bool = True,
) -> TrainResults:
    """
    Train movement prediction model with:
    1. Walk-forward cross-validation (no time leakage)
    2. Balanced class weights (fix UP bias)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics: Dict[str, Any] = {"folds": 0, "fold_metrics": []}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Movement Predictor (Walk-Forward CV)")
        print(f"{'='*60}")
        print(f"Total samples: {len(X)}")
        print(f"CV folds: {n_splits}")
        
        # Show class distribution
        class_counts = pd.Series(y).value_counts().sort_index()
        print(f"\nClass Distribution:")
        print(f"  DOWN (0):    {int(class_counts.get(0, 0)):>6,} ({class_counts.get(0, 0)/len(y)*100:.1f}%)")
        print(f"  NEUTRAL (1): {int(class_counts.get(1, 0)):>6,} ({class_counts.get(1, 0)/len(y)*100:.1f}%)")
        print(f"  UP (2):      {int(class_counts.get(2, 0)):>6,} ({class_counts.get(2, 0)/len(y)*100:.1f}%)")
        print()
    
    # Compute balanced sample weights for FULL dataset
    full_sample_weights = _compute_sample_weights(y)
    
    if verbose:
        weight_dict = {int(c): float(w) for c, w in zip(
            np.unique(y), 
            compute_class_weight('balanced', classes=np.unique(y), y=y)
        )}
        print(f"✅ Class Weights Applied (The Equalizer):")
        print(f"  DOWN weight:    {weight_dict.get(0, 1.0):.3f}x")
        print(f"  NEUTRAL weight: {weight_dict.get(1, 1.0):.3f}x")
        print(f"  UP weight:      {weight_dict.get(2, 1.0):.3f}x")
        print()

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        # Get sample weights for this fold's training data
        w_train = full_sample_weights[train_idx]

        if verbose:
            print(f"Fold {fold_num}:")
            print(f"  Train: {len(train_idx):,} samples")
            print(f"  Test:  {len(test_idx):,} samples")

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        
        # 🔥 THE KEY FIX: Pass balanced sample weights
        model.fit(X_train, y_train, sample_weight=w_train)

        preds = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, preds))

        if verbose:
            print(f"  Accuracy: {accuracy:.4f}")
            # Show per-class recall in CV
            from sklearn.metrics import recall_score
            recall_per_class = np.array(recall_score(y_test, preds, average=None))
            if len(recall_per_class) == 3:
                print(f"  DOWN Recall: {recall_per_class[0]:.3f} | NEUTRAL: {recall_per_class[1]:.3f} | UP: {recall_per_class[2]:.3f}")
            print()

        metrics["folds"] += 1
        metrics[f"accuracy_fold_{fold_num}"] = accuracy
        metrics["fold_metrics"].append({
            "fold": fold_num,
            "accuracy": accuracy,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        })

    accuracies: List[float] = [m["accuracy"] for m in metrics["fold_metrics"]]
    accuracies_array = np.array(accuracies)
    metrics["accuracy_mean"] = float(np.mean(accuracies_array))
    metrics["accuracy_std"] = float(np.std(accuracies_array))

    if verbose:
        print(f"{'='*60}")
        print(f"Final Movement Model Metrics:")
        print(f"  Mean Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
        print(f"{'='*60}\n")
        print("Training final model on full dataset...")

    # Train final model on ALL data with balanced weights
    final_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y, sample_weight=full_sample_weights)

    if verbose:
        print("✅ Movement model training complete!\n")

    return TrainResults(model=final_model, metrics=metrics)


def train_risk_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    verbose: bool = True,
) -> TrainResults:
    """Train risk assessment model with walk-forward cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics: Dict[str, Any] = {"folds": 0, "fold_metrics": []}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Risk Assessor (Walk-Forward CV)")
        print(f"{'='*60}")
        print(f"Total samples: {len(X)}")
        print(f"CV folds: {n_splits}\n")

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if verbose:
            print(f"Fold {fold_num}:")
            print(f"  Train: {len(train_idx):,} samples")
            print(f"  Test:  {len(test_idx):,} samples")

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.05,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse_float = float(mean_squared_error(y_test, preds))
        rmse = float(np.sqrt(mse_float))
        corr_result = spearmanr(y_test, preds)  # type: ignore[misc]
        corr = float(np.asarray(corr_result[0]))

        if verbose:
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Spearman: {corr:.4f}")
            print()

        metrics["folds"] += 1
        metrics[f"rmse_fold_{fold_num}"] = rmse
        metrics[f"spearman_fold_{fold_num}"] = corr
        metrics["fold_metrics"].append({
            "fold": fold_num,
            "rmse": rmse,
            "spearman": corr,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        })

    rmses: List[float] = [m["rmse"] for m in metrics["fold_metrics"]]
    spearmans: List[float] = [m["spearman"] for m in metrics["fold_metrics"]]

    metrics["rmse_mean"] = float(np.mean(np.array(rmses)))
    metrics["rmse_std"] = float(np.std(np.array(rmses)))
    metrics["spearman_mean"] = float(np.mean(np.array(spearmans)))
    metrics["spearman_std"] = float(np.std(np.array(spearmans)))

    if verbose:
        print(f"{'='*60}")
        print(f"Final Risk Model Metrics:")
        print(f"  Mean RMSE: {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
        print(f"  Mean Spearman: {metrics['spearman_mean']:.4f} ± {metrics['spearman_std']:.4f}")
        print(f"{'='*60}\n")
        print("Training final model on full dataset...")

    final_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.05,
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y)

    if verbose:
        print("✅ Risk model training complete!\n")

    return TrainResults(model=final_model, metrics=metrics)


def save_model(model: Any, path: str) -> None:
    joblib.dump(model, path)
    print(f"💾 Model saved to: {path}")


def load_model(path: str) -> Any:
    return joblib.load(path)


def train_models_sequential(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_dir: str,
    symbol: str,
    timeframe: str,
    verbose: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Train both models sequentially with proper leak prevention."""
    if verbose:
        print(f"\n{'#'*70}")
        print(f"# SEQUENTIAL MODEL TRAINING - {symbol} {timeframe}")
        print(f"{'#'*70}\n")
        print(f"Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"Val period:   {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
        print()

    if verbose:
        print("STEP 1: Building training datasets...")

    X_movement_train, y_movement_train, X_risk_train, y_risk_train = build_datasets(train_df)

    if verbose:
        print("\nSTEP 2: Training Movement Predictor...")

    movement_results = train_movement_model(
        X_movement_train, y_movement_train, n_splits=5, verbose=verbose,
    )
    movement_model = movement_results.model

    if verbose:
        print("\nSTEP 3: Getting OUT-OF-SAMPLE Movement predictions on validation set...")

    X_movement_val, y_movement_val, X_risk_val, y_risk_val = build_datasets(val_df)

    val_movement_preds = movement_model.predict(X_movement_val)
    val_movement_probs = movement_model.predict_proba(X_movement_val)
    val_movement_confidence = val_movement_probs.max(axis=1)
    val_accuracy = float(accuracy_score(y_movement_val, val_movement_preds))

    if verbose:
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"\n  📊 Per-Class Validation Performance:")
        report = classification_report(
            y_movement_val, val_movement_preds,
            target_names=['DOWN', 'NEUTRAL', 'UP'],
            digits=3
        )
        print(report)

    if verbose:
        print("STEP 4: Updating Risk model features with OOS predictions...")

    X_risk_train_updated = X_risk_train.copy()
    X_risk_train_updated["pred_direction"] = y_movement_train.astype(float)
    X_risk_train_updated["pred_confidence"] = 1.0

    X_risk_val_updated = X_risk_val.copy()
    X_risk_val_updated["pred_direction"] = val_movement_preds.astype(float)
    X_risk_val_updated["pred_confidence"] = val_movement_confidence

    if verbose:
        print("\nSTEP 5: Training Risk Assessor...")

    risk_results = train_risk_model(
        X_risk_train_updated, y_risk_train, n_splits=5, verbose=verbose,
    )
    risk_model = risk_results.model

    if verbose:
        print("\nSTEP 6: Evaluating Risk model on validation set...")

    val_risk_preds = risk_model.predict(X_risk_val_updated)
    val_risk_rmse = float(np.sqrt(float(mean_squared_error(y_risk_val, val_risk_preds))))
    corr_result = spearmanr(y_risk_val, val_risk_preds)  # type: ignore[misc]
    val_risk_corr = float(np.asarray(corr_result[0]))

    if verbose:
        print(f"  Validation RMSE: {val_risk_rmse:.4f}")
        print(f"  Validation Spearman: {val_risk_corr:.4f}\n")

    if verbose:
        print("STEP 7: Saving models...")

    os.makedirs(save_dir, exist_ok=True)

    movement_path = f"{save_dir}/movement_predictor_{symbol.lower()}_{timeframe}.joblib"
    risk_path = f"{save_dir}/risk_assessor_{symbol.lower()}_{timeframe}.joblib"

    save_model(movement_model, movement_path)
    save_model(risk_model, risk_path)

    final_metrics = {
        "symbol": symbol,
        "timeframe": timeframe,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "movement": {
            "train_cv_accuracy": movement_results.metrics["accuracy_mean"],
            "val_accuracy": val_accuracy,
        },
        "risk": {
            "train_cv_rmse": risk_results.metrics["rmse_mean"],
            "val_rmse": val_risk_rmse,
            "val_spearman": val_risk_corr,
        },
    }

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# TRAINING COMPLETE!")
        print(f"{'#'*70}")
        print(f"Final Metrics: {final_metrics}")

    return movement_model, risk_model, final_metrics