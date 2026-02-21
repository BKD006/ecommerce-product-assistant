"""
Product Vector Store (Pinecone)

Uses:
- AWS Bedrock (Titan) embeddings
- Pinecone vector database
- Full CRUD support
- Metadata-aware filtering
"""

from typing import List, Dict, Optional, Any
import os
from pinecone import Pinecone, ServerlessSpec

from src.utils.model_loader import ModelLoader
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log

from dotenv import load_dotenv
load_dotenv()


class ProductVectorStore:
    """
    Pinecone vector store abstraction layer
    for product catalog search.
    """

    def __init__(
        self,
        index_name: str,
        dimension: int = 1024,  # Titan v2 embedding size
    ):
        try:
            log.info(
                "Initializing ProductVectorStore",
                index_name=index_name,
            )

            self.index_name = index_name
            self.dimension = dimension

            # Load embeddings
            self.embeddings = ModelLoader().load_embeddings("product")

            # Initialize Pinecone
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

            # Ensure index exists
            self._ensure_index()

            # Connect to index
            self.index = self.pc.Index(self.index_name)

            log.info("ProductVectorStore initialized successfully")

        except Exception as e:
            log.error("ProductVectorStore initialization failed", exc_info=True)
            raise ProductAssistantException(
                "Failed initializing ProductVectorStore", e
            )

    # -------------------------------------------------
    # INDEX SETUP
    # -------------------------------------------------

    def _ensure_index(self) -> None:
        existing_indexes = [i["name"] for i in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            log.info(
                "Creating Pinecone index",
                index_name=self.index_name,
            )

            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

    # -------------------------------------------------
    # CREATE (UPSERT)
    # -------------------------------------------------

    def upsert_products(
        self,
        products: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """
        products = [
            {
                "id": str,
                "embedding_text": str,
                "metadata": dict
            }
        ]
        """

        try:
            log.info(
                "Upserting product vectors",
                product_count=len(products),
            )

            for i in range(0, len(products), batch_size):
                batch = products[i : i + batch_size]

                vectors = []

                texts = [p["embedding_text"] for p in batch]
                embeddings = self.embeddings.embed_documents(texts)

                for product, vector in zip(batch, embeddings):
                    vectors.append(
                        {
                            "id": product["id"],
                            "values": vector,
                            "metadata": product["metadata"],
                        }
                    )

                self.index.upsert(vectors=vectors)

            log.info("Product upsert completed")

        except Exception as e:
            log.error("Product upsert failed", exc_info=True)
            raise ProductAssistantException(
                "Failed upserting products", e
            )

    # -------------------------------------------------
    # READ (SEARCH)
    # -------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Example filters:
        {
            "category": "Electronics",
            "brand": "Dell",
            "price": {"$lt": 60000}
        }
        """

        try:
            query_vector = self.embeddings.embed_query(query)

            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filters,
            )

            return results.get("matches", [])

        except Exception as e:
            log.error("Product search failed", exc_info=True)
            raise ProductAssistantException(
                "Product search failed", e
            )

    # -------------------------------------------------
    # DELETE
    # -------------------------------------------------

    def delete_by_product_id(self, product_id: str) -> None:
        try:
            self.index.delete(ids=[product_id])
            log.info(
                "Deleted product vector",
                product_id=product_id,
            )
        except Exception as e:
            raise ProductAssistantException(
                "Failed deleting product", e
            )

    def delete_by_filter(self, filters: Dict) -> None:
        """
        Delete by metadata filter.
        Example:
        {"category": "Clothing"}
        """
        try:
            self.index.delete(filter=filters)
            log.info("Deleted products by filter")
        except Exception as e:
            raise ProductAssistantException(
                "Failed deleting by filter", e
            )

    # -------------------------------------------------
    # ADMIN
    # -------------------------------------------------

    def describe_index(self) -> Dict:
        return self.index.describe_index_stats()
