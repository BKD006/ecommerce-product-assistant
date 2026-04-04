from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import shutil

from src.router.policy_ingest_router import router as policy_router
from src.router.product_ingest_router import router as product_router
from src.router.chat_router import router as chat_router
from src.router.health_router import router as health_router
from src.router.sqlite_ingest_router import router as sqlite_router


UPLOAD_DIR = "data/uploads"
TEMP_DIR = "temp_uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------------------------
    # STARTUP
    # ---------------------------
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    yield

    # ---------------------------
    # SHUTDOWN
    # ---------------------------
    try:
        for folder in [UPLOAD_DIR, TEMP_DIR]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                os.makedirs(folder, exist_ok=True)

    except Exception as e:
        print("Cleanup failed:", e)


app = FastAPI(
    title="NovaCart Product + Policy RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# =================================================
# ROUTERS
# =================================================

app.include_router(health_router, prefix="/health", tags=["Health"])

app.include_router(
    product_router,
    prefix="/api/products",
    tags=["Products"]
)

app.include_router(
    policy_router,
    prefix="/api/policies",
    tags=["Policies"]
)

app.include_router(
    chat_router,
    prefix="/api/chat",
    tags=["Chat"]
)

app.include_router(
    sqlite_router,
    prefix="/api/sqlite",
    tags=["SQLite Ingestion"]
)