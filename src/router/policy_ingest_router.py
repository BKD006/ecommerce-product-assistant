import os
import shutil
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File

from src.policy_ingestion.policy_runner import PolicyIngestionRunner
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log

router = APIRouter()

# Temporary upload directory
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/ingest")
async def ingest_uploaded_files(files: List[UploadFile] = File(...)):

    try:
        log.info(
            "API request: multi-file ingestion",
            extra={"file_count": len(files)},
        )

        saved_files = []

        # ------------------------------------------
        # Save Uploaded Files
        # ------------------------------------------

        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"{file.filename} is not a PDF file",
                )

            file_path = os.path.join(UPLOAD_DIR, file.filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append(file.filename)

        # ------------------------------------------
        # Run Ingestion Per File
        # ------------------------------------------

        runner = PolicyIngestionRunner()

        for file_name in saved_files:
            runner.reingest_file(file_name)

        total_vectors = runner.store.count()

        log.info(
            "Multi-file ingestion completed",
            extra={"total_vectors": total_vectors},
        )

        return {
            "status": "success",
            "files_processed": saved_files,
            "vector_count": total_vectors,
        }

    except ProductAssistantException as e:
        log.error("Ingestion failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        log.error("Unexpected ingestion error", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")