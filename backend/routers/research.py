from fastapi import APIRouter, HTTPException
import json
import pandas as pd
from pathlib import Path

router = APIRouter(tags=["Research Backtest Data"])

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "research" / "outputs"

@router.get("/api/research/metrics")
def get_backtest_metrics():
    """Serves the generated backtest metrics JSON."""
    json_path = OUTPUT_DIR / "backtest_metrics.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Backtest metrics JSON not found. Run 05_vectorbt_backtest.py first.")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

@router.get("/api/research/trades")
def get_backtest_trades(symbol: str = None, limit: int = 15000):
    """Serves the raw vault trades CSV as JSON."""
    csv_path = OUTPUT_DIR / "backtest_trades.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Backtest trades CSV not found. Run 05_vectorbt_backtest.py first.")
        
    df = pd.read_csv(csv_path)
    
    if symbol:
        df = df[df["symbol"] == symbol.upper()]
        
    df = df.sort_values(by="entry_ts").tail(limit)
    
    # Fill NA to prevent JSON serialization errors
    df = df.fillna("")
    
    return {"data": df.to_dict(orient="records")}
