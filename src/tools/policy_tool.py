from typing import Dict, Any, List, Optional
import asyncio

from src.retriever.policy_retriever import HybridPolicyRetriever
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log


class PolicyTool:
    """
    Tool wrapper around HybridPolicyRetriever.

    Used by the agent to:
    - Search policy documents
    - Retrieve relevant policy sections
    - Return structured results for reasoning
    """

    def __init__(self):
        try:
            log.info("Initializing PolicyTool")

            self.retriever = HybridPolicyRetriever(
                collection_name="support_policies"
            )

            log.info("PolicyTool initialized successfully")

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing PolicyTool", e
            )

    # -------------------------------------------------
    # TOOL EXECUTION
    # -------------------------------------------------

    def run(
        self,
        query: str,
        policy_type: Optional[str] = None,
        section_title: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Executes policy search.

        Returns structured policy results.
        """

        try:
            log.info(
                "PolicyTool execution started",
                query=query,
                policy_type=policy_type,
            )

            # If retriever is async-based
            results = asyncio.run(
                self.retriever.retrieve_async(
                    query=query,
                    k=top_k,
                    policy_type=policy_type,
                    section_title=section_title,
                    filters=filters,
                )
            )

            structured_results = self._format_results(results)

            log.info(
                "PolicyTool execution completed",
                result_count=len(structured_results),
            )

            return structured_results

        except Exception as e:
            log.error(
                "PolicyTool execution failed",
                exc_info=True,
            )
            raise ProductAssistantException(
                "Policy tool execution failed", e
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
                    "source": metadata.get("source"),
                    "policy_type": metadata.get("policy_type"),
                    "section_title": metadata.get("section_title"),
                    "content": doc.page_content[:800],  # trimmed
                }
            )

        return results