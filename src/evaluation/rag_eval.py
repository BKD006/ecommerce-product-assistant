import json
import numpy as np
from typing import Dict, Any
from src.agent.prompts import evaluation_prompt

class RAGEvaluator:

    def __init__(
        self,
        embedding_model=None,
        llm=None
    ):
        self.embedding_model = embedding_model
        self.llm = llm

    # =====================================================
    # CONTEXT BUILDER
    # =====================================================

    def _build_context(self, state: Dict[str, Any]) -> str:

        chunks = []

        for item in (state.get("product_results") or []):
            if isinstance(item, dict) and item.get("content"):
                chunks.append(item["content"])

        for item in (state.get("policy_results") or []):
            if isinstance(item, dict) and item.get("content"):
                chunks.append(item["content"])

        return " ".join(chunks).strip()

    # =====================================================
    # HEURISTIC GROUNDING
    # =====================================================

    def _grounding_score(self, answer: str, context: str):

        if not context:
            return 0.0, True

        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())

        overlap = answer_tokens.intersection(context_tokens)
        coverage = len(overlap) / max(len(answer_tokens), 1)

        return round(coverage, 3), coverage < 0.2

    # =====================================================
    # SEMANTIC SIMILARITY
    # =====================================================

    def _semantic_similarity(self, answer: str, context: str):

        if not self.embedding_model or not context:
            return 0.0

        try:
            a = np.array(self.embedding_model.embed_query(answer))
            c = np.array(self.embedding_model.embed_query(context))

            sim = np.dot(a, c) / (
                np.linalg.norm(a) * np.linalg.norm(c)
            )

            return float(sim)

        except Exception:
            return 0.0

    # =====================================================
    # LLM-AS-JUDGE
    # =====================================================

    def _llm_judge(self, query: str, answer: str, context: str):

        if not self.llm or not context:
            return {
                "llm_score": 0.0,
                "llm_verdict": "skipped",
                "llm_reason": "LLM not available or no context",
            }

        prompt = evaluation_prompt(query, answer, context)

        try:
            response = self.llm.invoke(prompt, temperature=0)
            parsed = json.loads(response.content.strip())

            return {
                "llm_score": parsed.get("score", 0.0),
                "llm_verdict": parsed.get("verdict", "unknown"),
                "llm_reason": parsed.get("reason", ""),
            }

        except Exception:
            return {
                "llm_score": 0.0,
                "llm_verdict": "error",
                "llm_reason": "Failed to parse LLM response",
            }

    # =====================================================
    # MAIN EVALUATE
    # =====================================================

    def evaluate(
        self,
        state: Dict[str, Any],
        query: str,
        answer: str,
        use_llm: bool = False
    ) -> Dict[str, Any]:

        context = self._build_context(state)

        grounding_score, hallucination_flag = self._grounding_score(
            answer, context
        )

        similarity = self._semantic_similarity(
            answer, context
        )

        result = {
            "grounding_score": grounding_score,
            "hallucination_flag": hallucination_flag,
            "semantic_similarity": round(similarity, 3),
            "context_length": len(context),
        }

        if use_llm:
            result.update(
                self._llm_judge(query, answer, context)
            )

        return result