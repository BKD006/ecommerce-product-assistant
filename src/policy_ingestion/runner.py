"""
Policy Ingestion Runner

Responsible for orchestrating:
- Docling parsing
- Hierarchical chunking
- Qdrant storage (Bedrock embeddings)

Designed for API-based ingestion (FastAPI).
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.ingestion.parser import DoclingPolicyParser
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.vectorstore import PolicyVectorStore

from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------
# Environment Configuration
# -------------------------------------------------

POLICY_DIR = os.getenv("POLICY_DIR", "data/uploads")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "support_policies")


class PolicyIngestionRunner:
    """
    High-level ingestion orchestrator.
    Intended to be used from API layer only.
    """

    def __init__(self):
        try:
            log.info(
                "Initializing PolicyIngestionRunner",
                extra={
                    "policy_dir": POLICY_DIR,
                    "collection_name": COLLECTION_NAME,
                },
            )

            self.policy_dir = POLICY_DIR
            self.collection_name = COLLECTION_NAME

            self.parser = DoclingPolicyParser(self.policy_dir)
            self.chunker = HierarchicalChunker()
            self.store = PolicyVectorStore(
                collection_name=self.collection_name
            )

            log.info("PolicyIngestionRunner initialized successfully")

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing ingestion runner", e
            )

    # -------------------------------------------------
    # FULL INGESTION
    # -------------------------------------------------

    def run_full_ingestion(self) -> int:
        """
        Runs ingestion for all policy documents.

        Returns:
            int: total vector count after ingestion
        """
        try:
            log.info("Starting full ingestion pipeline")

            section_docs = self.parser.parse()

            log.info(
                "Section parsing completed",
                extra={"section_count": len(section_docs)},
            )

            chunks = self.chunker.chunk(section_docs)

            log.info(
                "Hierarchical chunking completed",
                extra={"chunk_count": len(chunks)},
            )

            self.store.add_documents(chunks)

            total_vectors = self.store.count()

            log.info(
                "Full ingestion completed successfully",
                extra={"total_vectors": total_vectors},
            )

            return total_vectors

        except ProductAssistantException:
            raise
        except Exception as e:
            log.error("Full ingestion failed", exc_info=True)
            raise ProductAssistantException(
                "Full ingestion pipeline failed", e
            )

    # -------------------------------------------------
    # SINGLE FILE RE-INGESTION
    # -------------------------------------------------

    def reingest_file(self, file_name: str) -> int:
        """
        Re-ingests a single policy file.

        Returns:
            int: total vector count after re-ingestion
        """
        try:
            log.info(
                "Starting single-file re-ingestion",
                extra={"file_name": file_name},
            )

            file_path = Path(self.policy_dir) / file_name

            if not file_path.exists():
                raise ProductAssistantException(
                    f"File not found: {file_name}"
                )

            # Delete existing vectors
            self.store.delete_by_source(file_name)

            # Parse target file only
            section_docs = self._parse_single_file(file_name)

            log.info(
                "Section parsing completed",
                extra={"section_count": len(section_docs)},
            )

            chunks = self.chunker.chunk(section_docs)

            log.info(
                "Chunking completed",
                extra={"chunk_count": len(chunks)},
            )

            self.store.add_documents(chunks)

            total_vectors = self.store.count()

            log.info(
                "Re-ingestion completed successfully",
                extra={"total_vectors": total_vectors},
            )

            return total_vectors

        except ProductAssistantException:
            raise
        except Exception as e:
            log.error("Re-ingestion failed", exc_info=True)
            raise ProductAssistantException(
                "Re-ingestion pipeline failed", e
            )

    # -------------------------------------------------
    # INTERNAL HELPER
    # -------------------------------------------------

    def _parse_single_file(self, file_name: str) -> List[Document]:
        try:
            parser = DoclingPolicyParser(self.policy_dir)
            all_docs = parser.parse()

            return [
                doc for doc in all_docs
                if doc.metadata.get("source") == file_name
            ]

        except Exception as e:
            raise ProductAssistantException(
                f"Failed parsing file: {file_name}",
                e,
            )
