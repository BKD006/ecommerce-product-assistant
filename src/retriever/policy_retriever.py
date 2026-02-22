from typing import List, Optional, Dict
from collections import defaultdict
import asyncio

from langchain_core.documents import Document
from src.utils.model_loader import ModelLoader
from rank_bm25 import BM25Okapi

from src.policy_ingestion.policy_vectorstore import PolicyVectorStore
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log


class HybridPolicyRetriever:

    def __init__(
        self,
        collection_name: str,
        rrf_k: int = 60,
        max_concurrency: int = 5,
    ):
        """
        Initialize the HybridPolicyRetriever.

        Args:
            collection_name (str): Name of the Qdrant collection.
            rrf_k (int): Reciprocal Rank Fusion constant.
                        Higher values reduce rank impact.
            max_concurrency (int): (Legacy parameter) Previously used
                                for async reranking concurrency.

        Initializes:
            - Policy vector store connection
            - RRF fusion configuration
            - Internal BM25 state
        """
        try:
            log.info("Initializing HybridPolicyRetriever")

            self.store = PolicyVectorStore(
                collection_name=collection_name
            )

            self.rrf_k = rrf_k
            self.max_concurrency = max_concurrency

            self._bm25_index = None
            self._bm25_documents: List[Document] = []

            log.info("HybridPolicyRetriever initialized successfully")

        except Exception as e:
            raise ProductAssistantException(
                "Hybrid retriever initialization failed", e
            )

    # =================================================
    # PUBLIC SYNC WRAPPER
    # =================================================

    def retrieve(
        self,
        query: str,
        k: int = 5,
        policy_type: Optional[str] = None,
        section_title: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Synchronous public interface for hybrid retrieval.

        Wraps the async retrieval method for compatibility
        with non-async code.

        Args:
            query (str): User query string.
            k (int): Number of top documents to return.
            policy_type (Optional[str]): Filter by policy type.
            section_title (Optional[str]): Filter by section title.
            filters (Optional[Dict]): Additional metadata filters.

        Returns:
            List[Document]: Top-k fused documents.
        """

        return asyncio.run(
            self.retrieve_async(
                query=query,
                k=k,
                policy_type=policy_type,
                section_title=section_title,
                filters=filters,
            )
        )

    # =================================================
    # ASYNC HYBRID RETRIEVE
    # =================================================

    async def retrieve_async(
        self,
        query: str,
        k: int = 5,
        policy_type: Optional[str] = None,
        section_title: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Performs hybrid retrieval using:

            1. Dense vector search (Qdrant)
            2. Sparse lexical search (BM25)
            3. Reciprocal Rank Fusion (RRF)

        Args:
            query (str): User search query.
            k (int): Number of top results to retrieve.
            policy_type (Optional[str]): Filter by policy category.
            section_title (Optional[str]): Filter by section name.
            filters (Optional[Dict]): Additional metadata constraints.

        Returns:
            List[Document]: Ranked documents after fusion.

        Raises:
            ProductAssistantException: If retrieval fails.
        """

        try:
            if not query.strip():
                raise ProductAssistantException("Query cannot be empty")

            metadata_filter = self._build_filter(
                policy_type, section_title, filters
            )

            log.info(
                "Executing hybrid retrieval",
                query=query,
                top_k=k,
            )

            # Step 1: Dense Retrieval
            dense_results = self.store.search(
                query=query,
                k=k * 4,
                filters=metadata_filter,
            )

            # Step 2: Sparse Retrieval
            sparse_results = self._bm25_search(
                query=query,
                k=k * 4,
                metadata_filter=metadata_filter,
            )

            # Step 3: RRF Fusion
            fused = self._rrf_fusion(dense_results, sparse_results)

            final_results = fused[: k * 3]

            log.info(
                "Hybrid retrieval completed",
                returned_count=len(final_results),
            )

            return final_results

        except ProductAssistantException:
            raise
        except Exception as e:
            log.error("Hybrid retrieval failed", exc_info=True)
            raise ProductAssistantException(
                "Hybrid retrieval execution failed", e
            )

    # =================================================
    # BM25 SEARCH
    # =================================================

    def _build_bm25_index(self, documents: List[Document]) -> None:
        """
        Builds a BM25 index over provided documents.

        Args:
            documents (List[Document]): Documents to index.

        Stores:
            - Tokenized corpus
            - Document references for ranking
        """
        tokenized_corpus = [
            doc.page_content.split() for doc in documents
        ]

        self._bm25_index = BM25Okapi(tokenized_corpus)
        self._bm25_documents = documents

    def _bm25_search(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[Dict],
    ) -> List[Document]:
        """
        Performs sparse lexical search using BM25.

        Dense results are first retrieved from Qdrant,
        then ranked using BM25 scoring.

        Args:
            query (str): Search query.
            k (int): Number of top results to return.
            metadata_filter (Optional[Dict]): Metadata constraints.

        Returns:
            List[Document]: Top-k BM25-ranked documents.
        """

        docs = self.store.search(
            query=query,
            k=100,
            filters=metadata_filter,
        )

        if not docs:
            return []

        self._build_bm25_index(docs)

        tokenized_query = query.split()
        scores = self._bm25_index.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        top_indices = ranked_indices[:k]

        return [self._bm25_documents[i] for i in top_indices]

    # =================================================
    # RRF FUSION
    # =================================================

    def _rrf_fusion(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
    ) -> List[Document]:
        """
        Applies Reciprocal Rank Fusion (RRF) to combine
        dense and sparse retrieval rankings.

        RRF Formula:
            score += 1 / (rrf_k + rank)

        Args:
            dense_results (List[Document]): Dense vector results.
            sparse_results (List[Document]): Sparse BM25 results.

        Returns:
            List[Document]: Documents sorted by fused score.
        """
        score_dict = defaultdict(float)

        for rank, doc in enumerate(dense_results):
            score_dict[id(doc)] += 1 / (self.rrf_k + rank)

        for rank, doc in enumerate(sparse_results):
            score_dict[id(doc)] += 1 / (self.rrf_k + rank)

        unique_docs = {id(doc): doc for doc in dense_results + sparse_results}

        sorted_docs = sorted(
            unique_docs.values(),
            key=lambda doc: score_dict[id(doc)],
            reverse=True,
        )

        return sorted_docs

    # =================================================
    # METADATA FILTER
    # =================================================

    def _build_filter(
        self,
        policy_type: Optional[str],
        section_title: Optional[str],
        additional_filters: Optional[Dict],
    ) -> Optional[Dict]:
        """
        Constructs metadata filter dictionary for Qdrant search.

        Args:
            policy_type (Optional[str]): Policy category filter.
            section_title (Optional[str]): Section name filter.
            additional_filters (Optional[Dict]): Extra metadata filters.

        Returns:
            Optional[Dict]: Combined filter dictionary,
                            or None if no filters applied.
        """
        filter_dict: Dict = {}

        if policy_type:
            filter_dict["policy_type"] = policy_type

        if section_title:
            filter_dict["section_title"] = section_title

        if additional_filters:
            filter_dict.update(additional_filters)

        return filter_dict if filter_dict else None
