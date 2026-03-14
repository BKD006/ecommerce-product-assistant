"""
Improved Hybrid Product Retriever
(Dense + BM25 + RRF + Metadata Aggregation support)
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
        try:
            log.info("Initializing HybridProductRetriever")

            self.store = ProductVectorStore(index_name=index_name)
            self.rrf_k = rrf_k

            log.info("HybridProductRetriever initialized")

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
        dense_k: int = 25,
    ):

        filters = filters or {}

        try:
            return asyncio.run(
                self.retrieve_async(
                    query=query,
                    k=k,
                    dense_k=dense_k,
                    category=filters.get("category"),
                    brand=filters.get("brand"),
                    min_price=filters.get("min_price"),
                    max_price=filters.get("max_price"),
                    min_rating=filters.get("min_rating"),
                )
            )
        except RuntimeError:
            return asyncio.get_event_loop().run_until_complete(
                self.retrieve_async(
                    query=query,
                    k=k,
                    dense_k=dense_k,
                    category=filters.get("category"),
                    brand=filters.get("brand"),
                    min_price=filters.get("min_price"),
                    max_price=filters.get("max_price"),
                    min_rating=filters.get("min_rating"),
                )
            )

    # =================================================
    # HYBRID RETRIEVAL
    # =================================================

    async def retrieve_async(
        self,
        query: str,
        k: int = 5,
        dense_k: int = 25,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
    ):

        try:

            if not query.strip():
                raise ProductAssistantException("Query cannot be empty")

            metadata_filter = self._build_filter(
                category,
                brand,
                min_price,
                max_price,
                min_rating,
            )

            log.info(
                "Executing hybrid product retrieval",
                query=query,
                metadata_filter=metadata_filter,
            )

            # -------------------------------
            # Dense Retrieval
            # -------------------------------

            dense_matches = self.store.search(
                query=query,
                top_k=dense_k,
                filters=metadata_filter,
            )

            if not dense_matches:
                return []

            dense_documents = [
                Document(
                    page_content=self._build_sparse_text(
                        m["metadata"]
                    ),
                    metadata=m["metadata"],
                )
                for m in dense_matches
            ]

            # -------------------------------
            # BM25
            # -------------------------------

            sparse_documents = self._bm25_search(
                query=query,
                documents=dense_documents,
                k=dense_k,
            )

            # -------------------------------
            # RRF
            # -------------------------------

            fused_docs = self._rrf_fusion(
                dense_documents,
                sparse_documents,
            )

            return fused_docs[:k]

        except Exception as e:
            log.error(
                "Hybrid product retrieval failed",
                exc_info=True,
            )

            raise ProductAssistantException(
                "Hybrid product retrieval execution failed",
                e,
            )

    # =================================================
    # NEW — FETCH MANY (FOR BRANDS / CATALOG)
    # =================================================

    def retrieve_many(
        self,
        k: int = 200,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:

        filters = filters or {}

        log.info(
            "Retrieving many products",
            k=k,
            filters=filters,
        )

        matches = self.store.search(
            query="*",
            top_k=k,
            filters=filters,
        )

        return matches

    # =================================================
    # NEW — GET UNIQUE METADATA VALUES
    # =================================================

    def get_unique_metadata(
        self,
        field: str,
        k: int = 200,
    ) -> List[str]:

        matches = self.retrieve_many(k=k)

        values = set()

        for m in matches:
            meta = m.get("metadata", {})
            val = meta.get(field)

            if val:
                values.add(str(val))

        return sorted(list(values))

    # =================================================
    # SPARSE TEXT
    # =================================================

    def _build_sparse_text(
        self,
        metadata: Dict,
    ) -> str:

        return f"""
        {metadata.get('category', '')}
        {metadata.get('sub_category', '')}
        {metadata.get('brand', '')}
        {metadata.get('price', '')}
        {metadata.get('rating', '')}
        """

    # =================================================
    # BM25
    # =================================================

    def _bm25_search(
        self,
        query: str,
        documents: List[Document],
        k: int,
    ) -> List[Document]:

        tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query.lower().split()

        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        top_indices = ranked_indices[:k]

        return [documents[i] for i in top_indices]

    # =================================================
    # RRF
    # =================================================

    def _rrf_fusion(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
    ) -> List[Document]:

        score_dict = defaultdict(float)

        for rank, doc in enumerate(dense_docs):
            score_dict[id(doc)] += 1 / (self.rrf_k + rank)

        for rank, doc in enumerate(sparse_docs):
            score_dict[id(doc)] += 1 / (self.rrf_k + rank)

        unique_docs = {
            id(doc): doc
            for doc in dense_docs + sparse_docs
        }

        sorted_docs = sorted(
            unique_docs.values(),
            key=lambda doc: score_dict[id(doc)],
            reverse=True,
        )

        return sorted_docs

    # =================================================
    # FILTER
    # =================================================

    def _build_filter(
        self,
        category,
        brand,
        min_price,
        max_price,
        min_rating,
    ) -> Optional[Dict]:

        filter_dict = {}

        if category:
            filter_dict["category"] = {"$eq": category}

        if brand:
            filter_dict["brand"] = {"$eq": brand}

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