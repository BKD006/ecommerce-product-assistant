from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid
import os

from src.product_ingestion.product_runner import ProductIngestionRunner
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log

router = APIRouter()

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "novacart_products")

runner = ProductIngestionRunner(index_name=PINECONE_INDEX)


# =================================================
# CSV INGESTION ENDPOINT
# =================================================

@router.post("/ingest")
async def ingest_products(file: UploadFile = File(...)):

    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported",
            )

        log.info(
            "Received product ingestion request",
            extra={"filename": file.filename},
        )

        # Create temp directory if not exists
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        # Unique file name
        temp_file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"

        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        log.info(
            "CSV file saved temporarily",
            extra={"path": str(temp_file_path)},
        )

        # Run ingestion
        result = runner.ingest_csv(str(temp_file_path))

        # Cleanup temp file
        temp_file_path.unlink(missing_ok=True)

        return result

    except ProductAssistantException as e:
        log.error("Product ingestion failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        log.error("Unexpected ingestion error", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )
