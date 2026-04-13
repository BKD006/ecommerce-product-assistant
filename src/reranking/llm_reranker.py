from typing import List
from langchain_core.documents import Document

from src.utils.model_loader import ModelLoader
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException


class LLMReranker:

    def __init__(self):
        try:
            self.llm = ModelLoader().load_reranker_llm()
            log.info("LLM Reranker initialized")
        except Exception as e:
            raise ProductAssistantException(
                "Failed to initialize reranker", e
            )

    # =================================================
    # MAIN
    # =================================================

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:

        try:

            if not documents:
                return []

            # Limit docs (VERY IMPORTANT)
            docs = documents[:15]

            prompt = self._build_prompt(query, docs)

            response = self.llm.invoke(prompt)

            ranked_indices = self._parse_response(
                response.content,
                len(docs)
            )

            reranked = [docs[i] for i in ranked_indices]

            return reranked[:top_k]

        except Exception:
            log.error("Reranking failed", exc_info=True)
            return documents[:top_k]

    # =================================================
    # PROMPT
    # =================================================

    def _build_prompt(self, query, docs):

        text = ""

        for i, d in enumerate(docs):
            meta = d.metadata or {}

            text += f"""
                    [{i}]
                    Category: {meta.get('category')}
                    Brand: {meta.get('brand')}
                    Price: {meta.get('price')}
                    Content: {d.page_content[:200]}
                    """

        return f"""
                You are a ranking system.

                User Query:
                {query}

                Rank the products from MOST relevant to LEAST relevant.

                Return ONLY a JSON list of indices.

                Example:
                [2,0,1]

                Products:
                {text}
                """

    # =================================================
    # PARSER
    # =================================================

    def _parse_response(self, text, max_len):

        import json

        try:
            indices = json.loads(text.strip())

            return [
                i for i in indices
                if isinstance(i, int) and i < max_len
            ]

        except Exception:
            return list(range(max_len))