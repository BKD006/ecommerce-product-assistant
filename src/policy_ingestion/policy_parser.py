"""
Docling-Based Policy Parser

Responsible for:
- Loading PDF policy documents
- Extracting structured content using Docling
- Converting them into LangChain Documents
- Attaching consistent metadata

Uses markdown export for stable section extraction.
"""

from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log


class DoclingPolicyParser:
    """
    Parses policy PDFs into structured section-level LangChain Documents.
    Uses markdown export for reliable structure extraction.
    """

    def __init__(self, policy_dir: str):
        self.policy_dir = Path(policy_dir)
        self.converter = DocumentConverter()

        log.info(
            "DoclingPolicyParser initialized",
            policy_dir=str(self.policy_dir),
        )

    # ==========================================================
    # PUBLIC API
    # ==========================================================

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
                    file_name=pdf_path.name,
                )

                file_documents = self._parse_single_file(pdf_path)
                documents.extend(file_documents)

            log.info(
                "Policy parsing completed",
                total_sections=len(documents),
            )

            return documents

        except ProductAssistantException:
            raise
        except Exception as e:
            log.error("Policy parsing failed", exc_info=True)
            raise ProductAssistantException(
                "Policy parsing pipeline failed", e
            )

    # ==========================================================
    # INTERNAL HELPERS
    # ==========================================================

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
                policy_dir=str(self.policy_dir),
            )

        return pdf_files

    def _parse_single_file(self, pdf_path: Path) -> List[Document]:
        """
        Parse a single PDF into section-level Documents
        using markdown structure.
        """
        try:
            policy_type = pdf_path.stem.replace("_policy", "")

            result = self.converter.convert(pdf_path)
            structured_doc = result.document

            # Use markdown export (stable API)
            markdown_text = structured_doc.export_to_markdown()

            if not markdown_text:
                raise ProductAssistantException(
                    f"No content extracted from {pdf_path.name}"
                )

            sections = self._split_markdown_sections(markdown_text)

            file_documents: List[Document] = []

            for title, content in sections:

                content = content.strip()

                if not content:
                    continue

                file_documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": pdf_path.name,
                            "policy_type": policy_type,
                            "section_title": title,
                            "doc_type": "policy",
                            "file_path": str(pdf_path),
                        },
                    )
                )

            log.info(
                "File parsed successfully",
                file_name=pdf_path.name,
                sections_extracted=len(file_documents),
            )

            return file_documents

        except Exception as e:
            log.error(
                "Failed to parse individual policy file",
                file_name=pdf_path.name,
                exc_info=True,
            )
            raise ProductAssistantException(
                f"Failed parsing file: {pdf_path.name}", e
            )

    # ==========================================================
    # MARKDOWN SECTION SPLITTER
    # ==========================================================

    def _split_markdown_sections(self, markdown_text: str):
        """
        Splits markdown text into (title, content) tuples
        using heading markers (#, ##, ###).
        """

        sections = []
        current_title = "Introduction"
        current_content = []

        lines = markdown_text.splitlines()

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("#"):
                # Save previous section
                if current_content:
                    sections.append(
                        (current_title, "\n".join(current_content))
                    )

                # New section
                current_title = stripped.lstrip("#").strip()
                current_content = []

            else:
                current_content.append(line)

        # Add last section
        if current_content:
            sections.append(
                (current_title, "\n".join(current_content))
            )

        return sections
