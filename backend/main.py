"""
FastAPI application entry point — chart-only mode.

Serves historical candle data to the frontend chart.

Start with:
    cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adapters.questdb_adapter import init_pool, close_pool
from routers.health import router as health_router
from routers.candles import router as candles_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting NIFTY Chart API …")
    try:
        init_pool()
    except Exception:
        logger.exception("QuestDB pool init failed — continuing in degraded mode")
    yield
    close_pool()
    logger.info("Shutdown complete")


# ============================================================================
# APP
# ============================================================================

app = FastAPI(title="NIFTY Chart API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(candles_router)


@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0-chart-only"}
