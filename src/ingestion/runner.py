"""
Policy Ingestion Runner

Responsible for orchestrating:
- Docling parsing
- Hierarchical chunking
- Qdrant storage (Bedrock embeddings)

Supports:
- Full ingestion
- Single-file re-ingestion
"""

import argparse
from pathlib import Path
from typing import List

from langchain.schema import Document

from src.ingestion.parser import DoclingPolicyParser
from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.vectorstore import PolicyVectorStore

from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log


POLICY_DIR = "data/policies"
COLLECTION_NAME = "support_policies"


class PolicyIngestionRunner:
    """
    High-level ingestion orchestrator.
    """

    def __init__(self):
        try:
            log.info("Initializing PolicyIngestionRunner")

            self.parser = DoclingPolicyParser(POLICY_DIR)
            self.chunker = HierarchicalChunker()
            self.store = PolicyVectorStore(
                collection_name=COLLECTION_NAME
            )

            log.info("PolicyIngestionRunner initialized successfully")

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing ingestion runner", e
            )

    # -------------------------------------------------
    # FULL INGESTION
    # -------------------------------------------------

    def run_full_ingestion(self) -> None:
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

        except ProductAssistantException:
            raise
        except Exception as e:
            log.error(
                "Full ingestion failed unexpectedly",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Full ingestion pipeline failed", e
            )

    # -------------------------------------------------
    # SINGLE FILE RE-INGESTION
    # -------------------------------------------------

    def reingest_file(self, file_name: str) -> None:
        try:
            log.info(
                "Starting single-file re-ingestion",
                extra={"file_name": file_name},
            )

            file_path = Path(POLICY_DIR) / file_name

            if not file_path.exists():
                raise ProductAssistantException(
                    f"File not found: {file_name}"
                )

            # Delete existing vectors
            self.store.delete_by_source(file_name)

            # Parse only target file
            section_docs = self._parse_single_file(file_name)

            log.info(
                "Section parsing for file completed",
                extra={"section_count": len(section_docs)},
            )

            chunks = self.chunker.chunk(section_docs)

            log.info(
                "Chunking for file completed",
                extra={"chunk_count": len(chunks)},
            )

            self.store.add_documents(chunks)

            log.info(
                "Re-ingestion completed successfully",
                extra={"file_name": file_name},
            )

        except ProductAssistantException:
            raise
        except Exception as e:
            log.error(
                "Single-file re-ingestion failed",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Re-ingestion pipeline failed", e
            )

    # -------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------

    def _parse_single_file(self, file_name: str) -> List[Document]:
        """
        Parse only a specific file without scanning entire directory.
        """
        try:
            parser = DoclingPolicyParser(POLICY_DIR)

            all_docs = parser.parse()

            filtered_docs = [
                doc for doc in all_docs
                if doc.metadata.get("source") == file_name
            ]

            return filtered_docs

        except Exception as e:
            raise ProductAssistantException(
                f"Failed parsing file during re-ingestion: {file_name}",
                e,
            )


# -------------------------------------------------
# CLI ENTRYPOINT
# -------------------------------------------------

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Policy Ingestion CLI"
    )

    arg_parser.add_argument(
        "--file",
        type=str,
        help="Re-ingest a specific policy file (e.g. refund_policy.pdf)",
    )

    args = arg_parser.parse_args()

    runner = PolicyIngestionRunner()

    try:
        if args.file:
            runner.reingest_file(args.file)
        else:
            runner.run_full_ingestion()

    except ProductAssistantException as e:
        log.critical(
            "Ingestion terminated due to critical error",
            exc_info=True,
        )
        raise
