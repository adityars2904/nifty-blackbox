"""
Health check router — simplified for chart-only mode.
"""

from fastapi import APIRouter
from datetime import datetime
import logging

from adapters import questdb_adapter
from config import IST, MARKET_OPEN, MARKET_CLOSE

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health_check():
    try:
        quest_connected = questdb_adapter.check_connection()

        now = datetime.now(IST)
        current_time = now.time()

        if current_time < MARKET_OPEN or current_time >= MARKET_CLOSE or now.weekday() >= 5:
            market_status = "MARKET_CLOSED"
        else:
            market_status = "MARKET_OPEN"

        return {
            "status": "ok",
            "questdb_connected": quest_connected,
            "market_status": market_status,
            "version": "2.0.0-chart-only",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "message": str(e),
        }
