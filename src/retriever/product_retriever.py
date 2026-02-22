"""
Hybrid Product Retriever (Dense + BM25 + RRF)

Reranking temporarily disabled.
"""

from typing import List, Optional, Dict
from collections import defaultdict
import asyncio

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.product_ingestion.product_vectorstore import ProductVectorStore
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log


class HybridProductRetriever:

    def __init__(
        self,
        index_name: str,
        rrf_k: int = 60,
    ):
        """
        Initialize the HybridProductRetriever.

        Args:
            index_name (str): Name of the Pinecone index storing products.
            rrf_k (int): Reciprocal Rank Fusion constant used to
                        balance ranking influence between dense and sparse results.

        Initializes:
            - Product vector store connection
            - RRF fusion configuration
        """
        try:
            log.info("Initializing HybridProductRetriever")

            self.store = ProductVectorStore(index_name=index_name)
            self.rrf_k = rrf_k

            log.info("HybridProductRetriever initialized successfully")

        except Exception as e:
            raise ProductAssistantException(
                "Hybrid product retriever initialization failed", e
            )

    # =================================================
    # SYNC WRAPPER
    # =================================================

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
    ):
        """
        Synchronous wrapper around the async retrieval method.

        This method allows the retriever to be used in non-async
        contexts (e.g., FastAPI routes, tool execution).

        Args:
            query (str): User search query.
            k (int): Number of top results to return.
            filters (Optional[Dict]): Optional metadata filters such as:
                - category
                - brand
                - min_price
                - max_price
                - min_rating

        Returns:
            List[Document]: Top-k ranked product documents.
        """
        filters = filters or {}

        return asyncio.run(
            self.retrieve_async(
                query=query,
                k=k,
                category=filters.get("category"),
                brand=filters.get("brand"),
                min_price=filters.get("min_price"),
                max_price=filters.get("max_price"),
                min_rating=filters.get("min_rating"),
            )
        )

    # =================================================
    # ASYNC HYBRID RETRIEVE
    # =================================================

    async def retrieve_async(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
    ):
        """
        Performs hybrid product retrieval using:

            1. Dense vector search (Pinecone)
            2. Sparse lexical ranking (BM25)
            3. Reciprocal Rank Fusion (RRF)

        Args:
            query (str): User search query.
            k (int): Number of final results to return.
            category (Optional[str]): Filter by product category.
            brand (Optional[str]): Filter by brand name.
            min_price (Optional[float]): Minimum price constraint.
            max_price (Optional[float]): Maximum price constraint.
            min_rating (Optional[float]): Minimum rating constraint.

        Returns:
            List[Document]: Ranked product documents.

        Raises:
            ProductAssistantException: If retrieval execution fails.
        """

        try:
            if not query.strip():
                raise ProductAssistantException("Query cannot be empty")

            metadata_filter = self._build_filter(
                category, brand, min_price, max_price, min_rating
            )

            log.info("Executing hybrid product retrieval")

            # -------------------------------------------------
            # Dense Retrieval (Pinecone)
            # -------------------------------------------------
            dense_matches = self.store.search(
                query=query,
                top_k=10,
                filters=metadata_filter,
            )

            if not dense_matches:
                return []

            dense_documents = [
                Document(
                    page_content=self._build_sparse_text(m["metadata"]),
                    metadata=m["metadata"],
                )
                for m in dense_matches
            ]

            # -------------------------------------------------
            # Sparse BM25 Retrieval
            # -------------------------------------------------
            sparse_documents = self._bm25_search(
                query=query,
                documents=dense_documents,
                k=50,
            )

            # -------------------------------------------------
            # RRF Fusion
            # -------------------------------------------------
            fused = self._rrf_fusion(
                dense_documents,
                sparse_documents,
            )

            # -------------------------------------------------
            # RETURN
            # -------------------------------------------------
            return fused[:k]

        except Exception as e:
            log.error("Hybrid product retrieval failed", exc_info=True)
            raise ProductAssistantException(
                "Hybrid product retrieval execution failed", e
            )

    # =================================================
    # SPARSE TEXT BUILDER
    # =================================================

    def _build_sparse_text(self, metadata: Dict) -> str:
        """
        Constructs a structured text representation of product metadata
        for BM25 sparse ranking.

        Args:
            metadata (Dict): Product metadata dictionary.

        Returns:
            str: Concatenated textual representation of product attributes.
        """
        return f"""
        {metadata.get('category', '')}
        {metadata.get('sub_category', '')}
        {metadata.get('brand', '')}
        {metadata.get('price', '')}
        {metadata.get('rating', '')}
        """

    # =================================================
    # BM25 SEARCH
    # =================================================

    def _bm25_search(
        self,
        query: str,
        documents: List[Document],
        k: int,
    ) -> List[Document]:
        """
        Performs sparse lexical search using BM25 over
        dense-retrieved candidate documents.

        Args:
            query (str): User search query.
            documents (List[Document]): Candidate documents from dense retrieval.
            k (int): Number of top results to return.

        Returns:
            List[Document]: BM25-ranked documents.
        """

        tokenized_corpus = [
            doc.page_content.split()
            for doc in documents
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        top_indices = ranked_indices[:k]

        return [documents[i] for i in top_indices]

    # =================================================
    # RRF FUSION
    # =================================================

    def _rrf_fusion(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
    ) -> List[Document]:
        """
        Combines dense and sparse results using Reciprocal Rank Fusion (RRF).

        RRF Formula:
            score += 1 / (rrf_k + rank_position)

        Args:
            dense_docs (List[Document]): Results from dense vector search.
            sparse_docs (List[Document]): Results from sparse BM25 ranking.

        Returns:
            List[Document]: Documents sorted by fused ranking score.
        """

        score_dict = defaultdict(float)

        for rank, doc in enumerate(dense_docs):
            score_dict[id(doc)] += 1 / (self.rrf_k + rank)

        for rank, doc in enumerate(sparse_docs):
            score_dict[id(doc)] += 1 / (self.rrf_k + rank)

        unique_docs = {id(doc): doc for doc in dense_docs + sparse_docs}

        sorted_docs = sorted(
            unique_docs.values(),
            key=lambda doc: score_dict[id(doc)],
            reverse=True,
        )

        return sorted_docs

    # =================================================
    # FILTER BUILDER
    # =================================================

    def _build_filter(
        self,
        category: Optional[str],
        brand: Optional[str],
        min_price: Optional[float],
        max_price: Optional[float],
        min_rating: Optional[float],
    ) -> Optional[Dict]:
        """
        Combines dense and sparse results using Reciprocal Rank Fusion (RRF).

        RRF Formula:
            score += 1 / (rrf_k + rank_position)

        Args:
            dense_docs (List[Document]): Results from dense vector search.
            sparse_docs (List[Document]): Results from sparse BM25 ranking.

        Returns:
            List[Document]: Documents sorted by fused ranking score.
        """

        filter_dict = {}

        if category:
            filter_dict["category"] = category

        if brand:
            filter_dict["brand"] = brand

        if min_price is not None or max_price is not None:
            price_filter = {}
            if min_price is not None:
                price_filter["$gte"] = min_price
            if max_price is not None:
                price_filter["$lte"] = max_price

            filter_dict["price"] = price_filter

        if min_rating is not None:
            filter_dict["rating"] = {"$gte": min_rating}

        return filter_dict if filter_dict else None