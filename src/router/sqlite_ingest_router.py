from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid

from src.sql.sqlite_ingest import SQLiteIngestion
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException


router = APIRouter()

# Temp folder
TEMP_DIR = Path("temp_sql_uploads")
TEMP_DIR.mkdir(exist_ok=True)


# =====================================================
# INGEST CSV INTO SQLITE
# =====================================================

@router.post("/ingest")
async def ingest_sqlite(file: UploadFile = File(...)):

    temp_file = None

    try:
        # -----------------------------
        # VALIDATION
        # -----------------------------
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported",
            )

        log.info(
            "SQLite ingestion request received",
            filename=file.filename,
        )

        # -----------------------------
        # SAVE TEMP FILE (UNIQUE)
        # -----------------------------
        temp_file = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"

        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        log.info(
            "File saved",
            path=str(temp_file),
        )

        # -----------------------------
        # RUN INGESTION
        # -----------------------------
        ingestor = SQLiteIngestion(batch_size=500)
        ingestor.ingest_csv(str(temp_file))

        # -----------------------------
        # FETCH STATS
        # -----------------------------
        result = ingestor.db.execute_query(
            """
            SELECT 
                COUNT(*) as total_products,
                COUNT(DISTINCT category) as total_categories,
                COUNT(DISTINCT brand) as total_brands
            FROM products
            """
        )

        stats = result[0] if result else {}

        log.info(
            "SQLite ingestion completed",
            stats=stats
        )

        return {
            "status": "success",
            "message": "Data ingested into SQLite successfully",
            "stats": stats
        }

    except ProductAssistantException as e:

        log.error("SQLite ingestion failed", exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    except Exception as e:

        log.error("Unexpected ingestion error", exc_info=True)

        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )