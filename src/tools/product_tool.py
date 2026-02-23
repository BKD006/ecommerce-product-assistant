from typing import Dict, Any, List, Optional

from src.retriever.product_retriever import HybridProductRetriever
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log
from src.utils.config_loader import load_config

class ProductTool:
    """
    Tool wrapper around HybridProductRetriever.

    This is used by the agent to:
    - Search products
    - Retrieve relevant product data
    - Return structured results to the agent state
    """

    def __init__(self):
        try:
            log.info("Initializing ProductTool")
            
            config = load_config()
            index_name = config["product_index"]["name"]
            self.retriever = HybridProductRetriever(index_name=index_name)

            log.info("ProductTool initialized successfully")

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
        """
        Executes product search.

        Returns structured product results.
        """

        try:
            log.info(
                "ProductTool execution started",
                query=query,
                filters=filters,
            )

            documents = self.retriever.retrieve(
                query=query,
                k=top_k,
                filters=filters,
            )

            structured_results = self._format_results(documents)

            log.info(
                "ProductTool execution completed",
                result_count=len(structured_results),
            )

            return structured_results

        except Exception as e:
            log.error(
                "ProductTool execution failed",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Product tool execution failed", e
            )

    # -------------------------------------------------
    # FORMAT RESULTS
    # -------------------------------------------------

    def _format_results(
        self,
        documents,
    ) -> List[Dict[str, Any]]:
        """
        Convert LangChain Documents into structured dictionaries.
        """

        results = []

        for doc in documents:
            metadata = doc.metadata or {}

            results.append(
                {
                    "product_id": metadata.get("product_id"),
                    "category": metadata.get("category"),
                    "sub_category": metadata.get("sub_category"),
                    "brand": metadata.get("brand"),
                    "price": metadata.get("price"),
                    "rating": metadata.get("rating"),
                    "content": doc.page_content[:500],  # trimmed context
                }
            )

        return results