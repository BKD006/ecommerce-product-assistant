from typing import List, Optional, Dict
from collections import defaultdict
import asyncio

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.product_ingestion.product_vectorstore import ProductVectorStore
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log


class HybridProductRetrieverV2:

    def __init__(
        self,
        index_name: str,
        rrf_k: int = 60,
        category_boost: float = 0.15,
    ):
        try:
            log.info("Initializing HybridProductRetrieverV2")

            self.store = ProductVectorStore(index_name=index_name)
            self.rrf_k = rrf_k
            self.category_boost = category_boost

            log.info("HybridProductRetrieverV2 initialized")

        except Exception as e:
            raise ProductAssistantException(
                "Hybrid product retriever V2 initialization failed", e
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
    # MAIN RETRIEVAL
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

            # -----------------------------------
            # STEP 1: Detect category
            # -----------------------------------
            detected_category = None

            if not category:
                detected_category = await self._detect_category(query)

            log.info(
                "Category detection",
                detected_category=detected_category,
                user_category=category,
            )

            # -----------------------------------
            # STEP 2: Build filter (ONLY if user provided)
            # -----------------------------------
            metadata_filter = self._build_filter(
                category, brand, min_price, max_price, min_rating
            )

            # -----------------------------------
            # STEP 3: Dense Retrieval
            # -----------------------------------
            dense_matches = self.store.search(
                query=query,
                top_k=dense_k,
                filters=metadata_filter,   # only user filters
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

            # -----------------------------------
            # STEP 4: BM25
            # -----------------------------------
            sparse_documents = self._bm25_search(
                query=query,
                documents=dense_documents,
                k=dense_k,
            )

            # -----------------------------------
            # STEP 5: RRF + CATEGORY BOOST
            # -----------------------------------
            fused_docs = self._rrf_fusion(
                dense_documents,
                sparse_documents,
                detected_category or category,
            )

            return fused_docs[:k]

        except Exception as e:
            log.error("Hybrid V2 retrieval failed", exc_info=True)

            raise ProductAssistantException(
                "Hybrid V2 retrieval execution failed",
                e,
            )

    # =================================================
    # CATEGORY DETECTION
    # =================================================

    async def _detect_category(self, query: str) -> Optional[str]:
        try:
            query = query.lower()

            categories = self.get_unique_metadata("category", k=200)

            for cat in categories:
                if cat.lower() in query:
                    return cat

            return None

        except Exception:
            return None

    # =================================================
    # FETCH METADATA VALUES
    # =================================================

    def get_unique_metadata(self, field: str, k: int = 200) -> List[str]:
        matches = self.store.search(query="*", top_k=k)

        values = set()

        for m in matches:
            val = m.get("metadata", {}).get(field)
            if val:
                values.add(str(val))

        return list(values)

    # =================================================
    # BETTER SPARSE TEXT
    # =================================================

    def _build_sparse_text(self, metadata: Dict) -> str:
        return f"""
        category: {metadata.get('category', '')}
        subcategory: {metadata.get('sub_category', '')}
        brand: {metadata.get('brand', '')}
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
    # RRF + CATEGORY BOOST
    # =================================================

    def _rrf_fusion(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
        detected_category: Optional[str] = None,
    ) -> List[Document]:

        score_dict = defaultdict(float)

        # Dense scores
        for rank, doc in enumerate(dense_docs):
            score_dict[id(doc)] += 1 / (self.rrf_k + rank)

        # Sparse scores
        for rank, doc in enumerate(sparse_docs):
            score_dict[id(doc)] += 1 / (self.rrf_k + rank)

        # CATEGORY BOOST
        if detected_category:
            for doc in dense_docs:
                if doc.metadata.get("category") == detected_category:
                    score_dict[id(doc)] += self.category_boost

        unique_docs = {
            id(doc): doc for doc in dense_docs + sparse_docs
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

        # -----------------------------
        # CATEGORY
        # -----------------------------
        if category:
            filter_dict["category"] = {"$eq": category}

        # -----------------------------
        # BRAND
        # -----------------------------
        if brand:
            filter_dict["brand"] = {"$eq": brand}

        # -----------------------------
        # PRICE (VECTOR FILTER)
        # -----------------------------
        if min_price is not None or max_price is not None:

            price_filter = {}

            if min_price is not None:
                price_filter["$gte"] = float(min_price)

            if max_price is not None:
                price_filter["$lte"] = float(max_price)

            if price_filter:
                filter_dict["price"] = price_filter

        # -----------------------------
        # RATING
        # -----------------------------
        if min_rating is not None:
            filter_dict["rating"] = {"$gte": float(min_rating)}

        return filter_dict if filter_dict else None