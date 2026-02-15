"""
Hierarchical Chunker

Responsible for:
- Creating parent-child relationships
- Preserving section-level structure
- Producing retrieval-optimized chunks
- Attaching consistent metadata

Parent = Entire Section
Child  = Semantic Sub-chunk of Section
"""

from typing import List
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log


class HierarchicalChunker:
    """
    Converts section-level Documents into hierarchical
    child chunks suitable for advanced RAG retrieval.
    """

    def __init__(
        self,
        child_chunk_size: int = 400,
        child_overlap: int = 100,
    ):
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        log.info(
            "HierarchicalChunker initialized",
            extra={
                "child_chunk_size": child_chunk_size,
                "child_overlap": child_overlap,
            },
        )

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    def chunk(self, section_documents: List[Document]) -> List[Document]:
        """
        Convert section-level documents into hierarchical chunks.

        Returns:
            List[Document]: Child-level chunks enriched with metadata.
        """

        try:
            if not section_documents:
                log.warning("No section documents provided for chunking")
                return []

            hierarchical_chunks: List[Document] = []

            for section_doc in section_documents:
                section_chunks = self._process_section(section_doc)
                hierarchical_chunks.extend(section_chunks)

            log.info(
                "Hierarchical chunking completed",
                extra={"total_chunks": len(hierarchical_chunks)},
            )

            return hierarchical_chunks

        except ProductAssistantException:
            raise
        except Exception as e:
            log.error(
                "Unexpected error during hierarchical chunking",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Hierarchical chunking pipeline failed", e
            )

    # -------------------------------------------------
    # Internal Section Processor
    # -------------------------------------------------

    def _process_section(self, section_doc: Document) -> List[Document]:
        """
        Process a single section and generate child chunks.
        """

        try:
            if not section_doc.page_content.strip():
                return []

            parent_id = str(uuid4())

            base_metadata = self._build_parent_metadata(
                section_doc.metadata,
                parent_id,
            )

            parent_document = Document(
                page_content=section_doc.page_content,
                metadata=base_metadata,
            )

            child_docs = self.child_splitter.split_documents([parent_document])

            return self._attach_child_metadata(child_docs, parent_id)

        except Exception as e:
            log.error(
                "Failed processing section during chunking",
                extra={"section_title": section_doc.metadata.get("section_title")},
                exc_info=True,
            )
            raise ProductAssistantException(
                "Section-level chunking failed", e
            )

    # -------------------------------------------------
    # Metadata Builders
    # -------------------------------------------------

    def _build_parent_metadata(self, original_metadata: dict, parent_id: str) -> dict:
        metadata = original_metadata.copy()

        metadata.update(
            {
                "parent_id": parent_id,
                "chunk_level": "parent",
            }
        )

        return metadata

    def _attach_child_metadata(
        self,
        child_docs: List[Document],
        parent_id: str,
    ) -> List[Document]:

        enriched_children: List[Document] = []

        for child in child_docs:
            child.metadata.update(
                {
                    "parent_id": parent_id,
                    "chunk_level": "child",
                }
            )
            enriched_children.append(child)

        return enriched_children
