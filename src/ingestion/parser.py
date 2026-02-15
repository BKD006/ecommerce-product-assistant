"""
Docling-Based Policy Parser

Responsible for:
- Loading PDF policy documents
- Extracting structured sections using Docling
- Converting them into LangChain Documents
- Attaching consistent metadata

This module is ingestion-only and does not perform chunking.
"""

from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter
from langchain.schema import Document

from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log


class DoclingPolicyParser:
    """
    Parses policy PDFs into structured section-level LangChain Documents.
    """

    def __init__(self, policy_dir: str):
        self.policy_dir = Path(policy_dir)
        self.converter = DocumentConverter()

        log.info(
            "DoclingPolicyParser initialized",
            extra={"policy_dir": str(self.policy_dir)},
        )

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def parse(self) -> List[Document]:
        """
        Parse all PDF files inside the policy directory.

        Returns:
            List[Document]: Section-level documents with metadata.
        """
        try:
            self._validate_directory()

            documents: List[Document] = []

            for pdf_path in self._get_pdf_files():
                log.info(
                    "Processing policy file",
                    extra={"file_name": pdf_path.name},
                )

                file_documents = self._parse_single_file(pdf_path)
                documents.extend(file_documents)

            log.info(
                "Policy parsing completed",
                extra={"total_sections": len(documents)},
            )

            return documents

        except ProductAssistantException:
            raise
        except Exception as e:
            log.error(
                "Unexpected failure during policy parsing",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Policy parsing pipeline failed", e
            )

    # -------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------

    def _validate_directory(self) -> None:
        if not self.policy_dir.exists():
            raise ProductAssistantException(
                f"Policy directory not found: {self.policy_dir}"
            )

    def _get_pdf_files(self) -> List[Path]:
        pdf_files = list(self.policy_dir.glob("*.pdf"))

        if not pdf_files:
            log.warning(
                "No PDF files found in policy directory",
                extra={"policy_dir": str(self.policy_dir)},
            )

        return pdf_files

    def _parse_single_file(self, pdf_path: Path) -> List[Document]:
        """
        Parse a single PDF into section-level Documents.
        """

        try:
            policy_type = pdf_path.stem.replace("_policy", "")

            result = self.converter.convert(pdf_path)
            structured_doc = result.document

            file_documents: List[Document] = []

            for section in structured_doc.sections:
                section_text = (section.text or "").strip()

                if not section_text:
                    continue

                file_documents.append(
                    Document(
                        page_content=section_text,
                        metadata={
                            "source": pdf_path.name,
                            "policy_type": policy_type,
                            "section_title": section.title or "Untitled Section",
                            "doc_type": "policy",
                            "file_path": str(pdf_path),
                        },
                    )
                )

            log.info(
                "File parsed successfully",
                extra={
                    "file_name": pdf_path.name,
                    "sections_extracted": len(file_documents),
                },
            )

            return file_documents

        except Exception as e:
            log.error(
                "Failed to parse individual policy file",
                extra={"file_name": pdf_path.name},
                exc_info=True,
            )
            raise ProductAssistantException(
                f"Failed parsing file: {pdf_path.name}", e
            )
