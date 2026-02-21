"""
Policy Vector Store

Uses:
- AWS Bedrock (Titan) embeddings
- Qdrant Cloud vector database
- Full CRUD support
- Safe payload indexing for filtering
"""

from typing import List, Optional, Dict
import os

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from src.utils.model_loader import ModelLoader
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log
from dotenv import load_dotenv

load_dotenv()


class PolicyVectorStore:
    """
    Production-ready Qdrant Cloud vector store wrapper.
    """

    def __init__(
        self,
        collection_name: str,
        vector_size: int = 1024,  # Titan v2 dimension
    ):
        try:
            self.collection_name = collection_name
            self.vector_size = vector_size

            log.info(
                "Initializing PolicyVectorStore",
                collection_name=collection_name,
            )

            # Load embeddings
            self.embeddings = ModelLoader().load_embeddings()

            # Qdrant Cloud client
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
            )

            # Ensure collection + indexes
            self._ensure_collection()
            self._ensure_payload_indexes()

            # LangChain wrapper
            self.vectordb = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

            log.info("PolicyVectorStore initialized successfully")

        except Exception as e:
            log.error("Vector store initialization failed", exc_info=True)
            raise ProductAssistantException(
                "Vector store initialization failed", e
            )

    # ==========================================================
    # COLLECTION SETUP
    # ==========================================================

    def _ensure_collection(self) -> None:
        try:
            existing_collections = [
                col.name
                for col in self.client.get_collections().collections
            ]

            if self.collection_name not in existing_collections:
                log.info(
                    "Creating Qdrant collection",
                    collection_name=self.collection_name,
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

    def _ensure_payload_indexes(self) -> None:
        """
        Ensures all filterable fields are indexed.
        Safe to call multiple times.
        """
        fields_to_index = [
            "source",
            "policy_type",
            "section_title",
            "chunk_level",
            "parent_id",
        ]

        for field in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                log.info(f"Payload index ensured for field: {field}")
            except Exception:
                # Index already exists — safe to ignore
                pass

    # ==========================================================
    # CREATE
    # ==========================================================

    def add_documents(self, documents: List[Document]) -> None:
        try:
            if not documents:
                log.warning("No documents provided for insertion")
                return

            log.info(
                "Adding documents to vector store",
                document_count=len(documents),
            )

            self.vectordb.add_documents(documents)

            log.info("Documents inserted successfully")

        except Exception as e:
            log.error("Vector insertion failed", exc_info=True)
            raise ProductAssistantException(
                "Vector insertion failed", e
            )

    # ==========================================================
    # READ
    # ==========================================================

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Document]:

        try:
            log.info(
                "Performing vector search",
                top_k=k,
                filters=filters or {},
            )

            return self.vectordb.similarity_search(
                query=query,
                k=k,
                filter=filters,
            )

        except Exception as e:
            log.error("Vector search failed", exc_info=True)
            raise ProductAssistantException(
                "Vector search operation failed", e
            )

    # ==========================================================
    # DELETE
    # ==========================================================

    def delete_by_source(self, source_file: str) -> None:
        try:
            log.info(
                "Deleting vectors by source file",
                source_file=source_file,
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

            log.info("Deletion completed successfully")

        except Exception as e:
            log.error("Vector deletion failed", exc_info=True)
            raise ProductAssistantException(
                "Failed deleting vectors by source", e
            )

    # ==========================================================
    # UPDATE
    # ==========================================================

    def update_policy(
        self,
        source_file: str,
        new_documents: List[Document],
    ) -> None:

        try:
            log.info(
                "Updating policy in vector store",
                source_file=source_file,
            )

            self.delete_by_source(source_file)
            self.add_documents(new_documents)

            log.info("Policy update completed successfully")

        except Exception as e:
            log.error("Policy update failed", exc_info=True)
            raise ProductAssistantException(
                "Policy update operation failed", e
            )

    # ==========================================================
    # ADMIN
    # ==========================================================

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
