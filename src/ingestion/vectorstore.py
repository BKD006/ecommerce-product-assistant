"""
Policy Vector Store

Uses:
- AWS Bedrock (Titan) embeddings
- Qdrant vector database
- Full CRUD support
- Delete restricted to source file only
"""

from typing import List, Optional, Dict

from langchain.schema import Document
from langchain_aws import BedrockEmbeddings
from src.utils.model_loader import ModelLoader
from langchain.vectorstores import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log


class PolicyVectorStore:
    """
    Vector store abstraction layer for policy documents.
    """

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        vector_size: int = 1024,  # Titan v2 dimension
    ):
        try:
            self.collection_name = collection_name
            self.vector_size = vector_size
            log.info(
                "Initializing PolicyVectorStore",
                extra={"collection_name": collection_name},
            )
            # Bedrock Embeddings
            self.embeddings = ModelLoader().load_embeddings()
            # Qdrant Client
            self.client = QdrantClient(host=host, port=port)
            # Ensure collection exists
            self._ensure_collection()
            # LangChain wrapper
            self.vectordb = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings,
            )
            log.info("PolicyVectorStore initialized successfully")

        except Exception as e:
            log.error(
                "Failed to initialize PolicyVectorStore",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Vector store initialization failed", e
            )

    # -------------------------------------------------
    # Collection Setup
    # -------------------------------------------------

    def _ensure_collection(self) -> None:
        try:
            existing = [
                col.name
                for col in self.client.get_collections().collections
            ]

            if self.collection_name not in existing:
                log.info(
                    "Creating new Qdrant collection",
                    extra={"collection_name": self.collection_name},
                )

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as e:
            raise ProductAssistantException(
                "Failed ensuring Qdrant collection", e
            )

    # -------------------------------------------------
    # CREATE
    # -------------------------------------------------

    def add_documents(self, documents: List[Document]) -> None:
        try:
            if not documents:
                log.warning("No documents provided for insertion")
                return

            log.info(
                "Adding documents to vector store",
                extra={"document_count": len(documents)},
            )

            self.vectordb.add_documents(documents)

            log.info("Documents inserted successfully")

        except Exception as e:
            log.error(
                "Failed inserting documents into vector store",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Vector insertion failed", e
            )

    # -------------------------------------------------
    # READ
    # -------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Document]:

        try:
            log.info(
                "Performing vector search",
                extra={"top_k": k, "filters": filters},
            )

            results = self.vectordb.similarity_search(
                query=query,
                k=k,
                filter=filters,
            )

            return results

        except Exception as e:
            log.error(
                "Vector search failed",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Vector search operation failed", e
            )

    # -------------------------------------------------
    # DELETE (Only by source file)
    # -------------------------------------------------

    def delete_by_source(self, source_file: str) -> None:
        try:
            log.info(
                "Deleting vectors by source file",
                extra={"source_file": source_file},
            )

            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_file),
                    )
                ]
            )

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter,
            )

            log.info("Deletion completed")

        except Exception as e:
            log.error(
                "Vector deletion failed",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Failed deleting vectors by source", e
            )

    # -------------------------------------------------
    # UPDATE
    # -------------------------------------------------

    def update_policy(
        self,
        source_file: str,
        new_documents: List[Document],
    ) -> None:

        try:
            log.info(
                "Updating policy in vector store",
                extra={"source_file": source_file},
            )

            self.delete_by_source(source_file)
            self.add_documents(new_documents)

            log.info("Policy update completed")

        except Exception as e:
            log.error(
                "Policy update failed",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Policy update operation failed", e
            )

    # -------------------------------------------------
    # ADMIN
    # -------------------------------------------------

    def count(self) -> int:
        try:
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            return collection_info.points_count
        except Exception as e:
            raise ProductAssistantException(
                "Failed retrieving vector count", e
            )
