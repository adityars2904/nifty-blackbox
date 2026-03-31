from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.data_loader import load_and_process_questdb
from ml.training_pipeline import build_datasets, save_model, train_movement_model, train_risk_model


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to write parquet. Install pyarrow or fastparquet, "
            "then re-run the pipeline."
        ) from exc


def run(symbols: list[str], start: str | None, end: str | None) -> None:
    for symbol in symbols:
        print(f"[pipeline] Loading {symbol} from QuestDB...")
        df = load_and_process_questdb(
            symbol=symbol,
            start=start,
            end=end,
            table="candles",
            resample_rule="5min",
        )

        out_path = Path("data") / "processed" / f"{symbol.lower()}_5m.parquet"
        _save_parquet(df, out_path)
        print(f"[pipeline] Saved {symbol} -> {out_path}")

        print(f"[train] Building datasets for {symbol}...")
        X_movement, y_movement, X_risk, y_risk = build_datasets(df)

        print(f"[train] Training movement model for {symbol}...")
        movement_results = train_movement_model(X_movement, y_movement)
        movement_path = Path("backend") / "models" / f"movement_predictor_{symbol.lower()}_5m.joblib"
        movement_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(movement_results.model, str(movement_path))
        print(f"[train] Saved movement model -> {movement_path}")
        print(f"[train] Movement metrics: {movement_results.metrics}")

        print(f"[train] Training risk model for {symbol}...")
        risk_results = train_risk_model(X_risk, y_risk)
        risk_path = Path("backend") / "models" / f"risk_assessor_{symbol.lower()}_5m.joblib"
        save_model(risk_results.model, str(risk_path))
        print(f"[train] Saved risk model -> {risk_path}")
        print(f"[train] Risk metrics: {risk_results.metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data pipeline + train models.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["NIFTY", "BANKNIFTY"],
        help="Symbols to process",
    )
    parser.add_argument("--start", default="2025-01-01T00:00:00Z", help="Start timestamp (ISO)")
    parser.add_argument("--end", default=None, help="End timestamp (ISO)")
    args = parser.parse_args()

    run(symbols=args.symbols, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
