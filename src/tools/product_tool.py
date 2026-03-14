from typing import Dict, Any, List, Optional

from src.retriever.product_retriever import HybridProductRetriever
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log
from src.utils.config_loader import load_config


class ProductTool:
    """
    Tool wrapper around HybridProductRetriever.

    Supports:
    - product search
    - brand listing
    - category listing
    - subcategory listing
    """

    def __init__(self):

        try:
            log.info("Initializing ProductTool")

            config = load_config()
            index_name = config["product_index"]["name"]

            self.retriever = HybridProductRetriever(
                index_name=index_name
            )

            log.info("ProductTool initialized")

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing ProductTool", e
            )

    # -------------------------------------------------
    # TOOL EXECUTION
    # -------------------------------------------------

    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        try:

            if not query or not query.strip():
                raise ProductAssistantException(
                    "Product query cannot be empty"
                )

            filters = filters or {}

            q = query.lower()

            log.info(
                "ProductTool execution started",
                query=query,
                filters=filters,
                top_k=top_k,
            )

            # ============================================
            # BRAND LIST QUERY
            # ============================================

            if "brand" in q and (
                "available" in q
                or "list" in q
                or "all" in q
            ):

                brands = self.retriever.get_unique_metadata(
                    field="brand",
                    k=300,
                )

                return [
                    {
                        "type": "brand_list",
                        "values": brands[:50],
                    }
                ]

            # ============================================
            # CATEGORY LIST QUERY
            # ============================================

            if "category" in q:

                categories = self.retriever.get_unique_metadata(
                    field="category",
                    k=300,
                )

                return [
                    {
                        "type": "category_list",
                        "values": categories[:50],
                    }
                ]

            # ============================================
            # SUBCATEGORY LIST
            # ============================================

            if "sub" in q and "category" in q:

                subs = self.retriever.get_unique_metadata(
                    field="sub_category",
                    k=300,
                )

                return [
                    {
                        "type": "sub_category_list",
                        "values": subs[:50],
                    }
                ]

            # ============================================
            # NORMAL PRODUCT SEARCH
            # ============================================

            documents = self.retriever.retrieve(
                query=query,
                k=top_k,
                filters=filters,
            )

            if not documents:

                log.warning(
                    "ProductTool returned no results",
                    query=query,
                    filters=filters,
                )

                return []

            structured_results = self._format_results(
                documents
            )

            log.info(
                "ProductTool execution completed",
                query=query,
                result_count=len(structured_results),
            )

            return structured_results

        except Exception as e:

            log.error(
                "ProductTool execution failed",
                query=query,
                exc_info=True,
            )

            raise ProductAssistantException(
                "Product tool execution failed",
                e,
            )

    # -------------------------------------------------
    # FORMAT RESULTS
    # -------------------------------------------------

    def _format_results(
        self,
        documents,
    ) -> List[Dict[str, Any]]:

        results = []

        for idx, doc in enumerate(
            documents,
            start=1,
        ):

            metadata = getattr(
                doc,
                "metadata",
                {},
            ) or {}

            content = getattr(
                doc,
                "page_content",
                "",
            )

            results.append(
                {
                    "source_id": idx,
                    "product_id": metadata.get(
                        "product_id"
                    ),
                    "category": metadata.get(
                        "category"
                    ),
                    "sub_category": metadata.get(
                        "sub_category"
                    ),
                    "brand": metadata.get(
                        "brand"
                    ),
                    "price": metadata.get(
                        "price"
                    ),
                    "rating": metadata.get(
                        "rating"
                    ),
                    "content": content[:300],
                }
            )

        return results