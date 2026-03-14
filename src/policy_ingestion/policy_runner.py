"""
Policy Ingestion Runner

Responsible for orchestrating:
- Docling parsing
- Hierarchical chunking
- Qdrant storage
- Delete uploaded files after ingestion

Designed for API-based ingestion (FastAPI).
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.policy_ingestion.policy_parser import DoclingPolicyParser
from src.policy_ingestion.policy_chunker import HierarchicalChunker
from src.policy_ingestion.policy_vectorstore import PolicyVectorStore

from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log
from dotenv import load_dotenv

load_dotenv()


POLICY_DIR = os.getenv("POLICY_DIR", "data/uploads")

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "novacart_support_policies"
)


class PolicyIngestionRunner:

    def __init__(self):

        try:

            log.info(
                "Initializing PolicyIngestionRunner",
                policy_dir=POLICY_DIR,
                collection_name=COLLECTION_NAME,
            )

            self.policy_dir = POLICY_DIR
            self.collection_name = COLLECTION_NAME

            self.parser = DoclingPolicyParser(
                self.policy_dir
            )

            self.chunker = HierarchicalChunker()

            self.store = PolicyVectorStore(
                collection_name=self.collection_name
            )

            log.info(
                "PolicyIngestionRunner initialized"
            )

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing ingestion runner",
                e,
            )


    # =================================================
    # SINGLE FILE RE-INGESTION
    # =================================================

    def reingest_file(
        self,
        file_name: str,
    ) -> int:

        try:

            log.info(
                "Reingesting file",
                file_name=file_name,
            )

            file_path = (
                Path(self.policy_dir)
                / file_name
            )

            if not file_path.exists():

                raise ProductAssistantException(
                    f"File not found: {file_name}"
                )

            self.store.delete_by_source(
                file_name
            )

            section_docs = (
                self._parse_single_file(
                    file_name
                )
            )

            chunks = self.chunker.chunk(
                section_docs
            )

            self.store.add_documents(
                chunks
            )

            total_vectors = self.store.count()

            log.info(
                "Re-ingestion done",
                total_vectors=total_vectors,
            )

            # delete only this file
            self._delete_file(file_path)

            return total_vectors

        except Exception as e:

            log.error(
                "Re-ingestion failed",
                exc_info=True,
            )

            raise ProductAssistantException(
                "Re-ingestion failed",
                e,
            )

    # =================================================
    # PARSE SINGLE
    # =================================================

    def _parse_single_file(
        self,
        file_name: str,
    ) -> List[Document]:

        parser = DoclingPolicyParser(
            self.policy_dir
        )

        docs = parser.parse()

        return [
            d
            for d in docs
            if d.metadata.get("source")
            == file_name
        ]

    # =================================================
    # CLEANUP ALL FILES
    # =================================================

    def _cleanup_files(self):

        try:

            folder = Path(
                self.policy_dir
            )

            for file in folder.glob("*"):

                if file.is_file():

                    log.info(
                        "Deleting file",
                        file=str(file),
                    )

                    file.unlink()

        except Exception as e:

            log.warning(
                "Cleanup failed",
                error=str(e),
            )

    # =================================================
    # SAFE CLEANUP
    # =================================================

    def _safe_cleanup(self):

        try:
            self._cleanup_files()
        except Exception:
            pass

    # =================================================
    # DELETE SINGLE FILE
    # =================================================

    def _delete_file(
        self,
        file_path: Path,
    ):

        try:

            if file_path.exists():

                log.info(
                    "Deleting file",
                    file=str(file_path),
                )

                file_path.unlink()

        except Exception as e:

            log.warning(
                "Delete failed",
                error=str(e),
            )